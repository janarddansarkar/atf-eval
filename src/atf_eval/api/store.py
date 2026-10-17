"""On-disk store for goldens and evaluations (survives a server restart)."""
from __future__ import annotations

import json
import re
import threading
import time
import uuid
from pathlib import Path

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _slug(text: str) -> str:
    return _SAFE.sub("-", text).strip("-")[:40] or "golden"


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
        (self.goldens / f"{golden_id}.json").write_text(json.dumps(doc), encoding="utf-8")
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
            (eval_dir / "status.json").write_text(
                json.dumps({"status": status, "updated_at": _now(), **extra}), encoding="utf-8")

    def read_status(self, eval_dir: Path) -> dict:
        p = eval_dir / "status.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"status": "unknown"}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
