"""On-disk store for goldens and evaluations (survives a server restart)."""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
import uuid
from pathlib import Path

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _slug(text: str) -> str:
    return _SAFE.sub("-", text).strip("-")[:40] or "golden"


# How long to keep retrying a file operation that fails with PermissionError.
# On Windows, replacing or opening a file fails with WinError 5/32 while any
# other handle has it open without FILE_SHARE_DELETE -- a concurrent status poll
# from this server, or Windows Defender / the search indexer scanning a file
# that was just written. Those handles close within milliseconds, so retrying
# briefly succeeds; a genuine permission problem still fails after this window.
SHARING_RETRY_SECONDS = 5.0


def _retry_on_sharing_violation(op):
    deadline = time.monotonic() + SHARING_RETRY_SECONDS
    delay = 0.01
    while True:
        try:
            return op()
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 0.25)


def _atomic_write_text(path: Path, text: str) -> None:
    """Replace `path` so a concurrent reader sees either the old or the new
    content, never a truncated file. Path.write_text() truncates first, so a
    status poll landing mid-write used to read "" (JSONDecodeError). The temp
    file lives in the same directory (os.replace is only atomic within one
    filesystem) and ends in .tmp, so it never matches the *.json globs. The
    swap is retried on PermissionError (see SHARING_RETRY_SECONDS)."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        _retry_on_sharing_violation(lambda: os.replace(tmp, path))
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


class Store:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.goldens = self.root / "goldens"
        self.evaluations = self.root / "evaluations"
        self.goldens.mkdir(parents=True, exist_ok=True)
        self.evaluations.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ---- goldens ----
    def save_golden(self, name: str, turns: list[dict]) -> dict:
        golden_id = f"{_slug(name)}-{uuid.uuid4().hex[:6]}"
        doc = {"golden_id": golden_id, "name": name, "created_at": _now(),
               "conversations": sorted({t["conversation_id"] for t in turns}), "turns": turns}
        _atomic_write_text(self.goldens / f"{golden_id}.json", json.dumps(doc))
        return {k: v for k, v in doc.items() if k != "turns"}

    def load_golden(self, golden_id: str) -> dict | None:
        path = self.goldens / f"{_slug(golden_id)}.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def list_goldens(self) -> list[dict]:
        out = []
        for p in sorted(self.goldens.glob("*.json")):
            doc = json.loads(p.read_text(encoding="utf-8"))
            out.append({k: v for k, v in doc.items() if k != "turns"})
        return out

    # ---- evaluations ----
    def new_evaluation(self) -> tuple[str, Path]:
        eval_id = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
        path = self.evaluations / eval_id
        path.mkdir(parents=True)
        return eval_id, path

    def eval_dir(self, eval_id: str) -> Path | None:
        path = self.evaluations / _slug(eval_id)
        return path if path.is_dir() else None

    def write_status(self, eval_dir: Path, status: str, **extra) -> None:
        with self._lock:
            _atomic_write_text(eval_dir / "status.json",
                               json.dumps({"status": status, "updated_at": _now(), **extra}))

    def fail_interrupted(self) -> list[str]:
        """Mark every evaluation still `queued`/`running` as `failed`. Called
        once at server start: jobs live in the server process's thread pool,
        so after a restart nothing will ever pick those evaluations up again
        and a client polling them would wait forever."""
        failed = []
        for d in sorted(p for p in self.evaluations.iterdir() if p.is_dir()):
            try:
                status = self.read_status(d).get("status")
            except (OSError, ValueError):
                continue
            if status in ("queued", "running"):
                self.write_status(d, "failed", error="the atf-eval server restarted before this evaluation "
                                  "finished; submit it again")
                failed.append(d.name)
        return failed

    def read_status(self, eval_dir: Path) -> dict:
        p = eval_dir / "status.json"
        # Same lock as write_status: on Windows an open read handle makes the
        # writer's os.replace fail, so this server's own status polls must never
        # overlap its own writes (a busy poller could otherwise starve them).
        # The retry covers handles held by other processes (Defender, indexer).
        with self._lock:
            if not p.exists():
                return {"status": "unknown"}
            return json.loads(_retry_on_sharing_violation(lambda: p.read_text(encoding="utf-8")))


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
