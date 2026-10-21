"""api.store: files that are read while being rewritten must never be seen
half-written. Regression test for the status.json race: GET /v1/evaluations/{id}
could read status.json between the writer truncating it and writing it."""
from __future__ import annotations

import json
import threading
from pathlib import Path

from atf_eval.api.store import Store


def _read_while_writing(write, read, writes: int) -> list[Exception]:
    """Run `writes` calls of write(i) in a thread while calling read() in a
    loop until the writer finishes; return every exception read() raised."""
    errors: list[Exception] = []
    t = threading.Thread(target=lambda: [write(i) for i in range(writes)])
    t.start()
    while t.is_alive():
        try:
            read()
        except Exception as e:  # noqa: BLE001 - collected and asserted below
            errors.append(e)
    t.join()
    return errors


def test_status_is_never_read_half_written(tmp_path):
    store = Store(tmp_path)
    _, eval_dir = store.new_evaluation()
    store.write_status(eval_dir, "queued")
    padding = "x" * 20_000  # a larger payload widens the truncate-then-write window

    errors = _read_while_writing(
        lambda i: store.write_status(eval_dir, "running", step=i, detail=padding),
        lambda: store.read_status(eval_dir),
        writes=2_000,
    )
    assert errors == []
    assert store.read_status(eval_dir) | {"detail": None} == {
        "status": "running", "updated_at": store.read_status(eval_dir)["updated_at"], "step": 1_999, "detail": None,
    }


def test_goldens_are_never_listed_half_written(tmp_path):
    store = Store(tmp_path)
    turns = [{"conversation_id": f"c{i}", "turn_id": 1, "pad": "x" * 5_000} for i in range(20)]

    errors = _read_while_writing(lambda i: store.save_golden(f"g{i}", turns), store.list_goldens, writes=150)
    assert errors == []
    assert len(store.list_goldens()) == 150


def test_atomic_writes_leave_no_temp_files_behind(tmp_path):
    store = Store(tmp_path)
    _, eval_dir = store.new_evaluation()
    for status in ("queued", "running", "done"):
        store.write_status(eval_dir, status, conversations=2)
    store.save_golden("demo", [{"conversation_id": "c", "turn_id": 1}])
    assert [p.name for p in eval_dir.iterdir()] == ["status.json"]
    assert [p.suffix for p in (tmp_path / "goldens").iterdir()] == [".json"]
    assert json.loads((eval_dir / "status.json").read_text()) | {"updated_at": None} == {
        "status": "done", "updated_at": None, "conversations": 2,
    }


# --- Windows sharing violations ---------------------------------------------------
# On Windows os.replace (MoveFileEx) and open() fail with PermissionError
# [WinError 5/32] while another handle -- a concurrent status poll, Windows
# Defender, the search indexer -- has the file open. Linux allows both, so these
# tests make os.replace / the read raise exactly that error on demand.

import os  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402

from atf_eval.api import store as store_mod  # noqa: E402


def _win_error():
    return PermissionError(13, "Access is denied")


@pytest.fixture
def fast_retry(monkeypatch):
    monkeypatch.setattr(store_mod, "SHARING_RETRY_SECONDS", 1.0)


def test_status_write_retries_while_the_file_is_held_open(tmp_path, monkeypatch, fast_retry):
    store = Store(tmp_path)
    _, eval_dir = store.new_evaluation()
    store.write_status(eval_dir, "queued")

    held_until = time.monotonic() + 0.3  # e.g. Defender scanning the file just written
    real_replace = os.replace
    attempts = []

    def windows_replace(src, dst):
        attempts.append(dst)
        if time.monotonic() < held_until:
            raise _win_error()
        return real_replace(src, dst)

    monkeypatch.setattr(store_mod.os, "replace", windows_replace)
    store.write_status(eval_dir, "running")
    assert len(attempts) > 1
    assert store.read_status(eval_dir)["status"] == "running"
    assert [p.name for p in eval_dir.iterdir()] == ["status.json"]  # temp file swapped in, none left


def test_status_write_gives_up_after_the_window_and_cleans_up(tmp_path, monkeypatch, fast_retry):
    store = Store(tmp_path)
    _, eval_dir = store.new_evaluation()
    store.write_status(eval_dir, "queued")

    def always_denied(src, dst):
        raise _win_error()

    monkeypatch.setattr(store_mod.os, "replace", always_denied)
    start = time.monotonic()
    with pytest.raises(PermissionError):
        store.write_status(eval_dir, "running")
    assert 0.9 <= time.monotonic() - start < 3.0  # retried for the window, then a real error
    assert [p.name for p in eval_dir.iterdir()] == ["status.json"]
    assert store.read_status(eval_dir)["status"] == "queued"  # the old content is intact


def test_status_read_retries_while_the_file_is_being_swapped(tmp_path, monkeypatch, fast_retry):
    store = Store(tmp_path)
    _, eval_dir = store.new_evaluation()
    store.write_status(eval_dir, "running")

    real_read = Path.read_text
    calls = []

    def windows_read(self, *a, **kw):
        if self.name == "status.json":
            calls.append(1)
            if len(calls) <= 2:
                raise _win_error()
        return real_read(self, *a, **kw)

    monkeypatch.setattr(Path, "read_text", windows_read)
    assert store.read_status(eval_dir)["status"] == "running"
    assert len(calls) == 3


def test_concurrent_polling_with_windows_sharing_rules(tmp_path, monkeypatch):
    """A reader holding status.json open blocks os.replace, as on Windows; the
    writer must still get every update through and the reader must never fail."""
    store = Store(tmp_path)
    _, eval_dir = store.new_evaluation()
    store.write_status(eval_dir, "queued")
    target = eval_dir / "status.json"

    open_handles = []
    handles_lock = threading.Lock()
    real_replace, real_read = os.replace, Path.read_text

    def windows_replace(src, dst):
        with handles_lock:
            if Path(dst) == target and open_handles:
                raise _win_error()
            return real_replace(src, dst)

    def windows_read(self, *a, **kw):
        if self != target:
            return real_read(self, *a, **kw)
        with handles_lock:
            open_handles.append(1)
        try:
            time.sleep(0.002)  # the read keeps the handle open for a moment
            return real_read(self, *a, **kw)
        finally:
            with handles_lock:
                open_handles.pop()

    monkeypatch.setattr(store_mod.os, "replace", windows_replace)
    monkeypatch.setattr(Path, "read_text", windows_read)

    errors = _read_while_writing(
        lambda i: store.write_status(eval_dir, "running", step=i), lambda: store.read_status(eval_dir), writes=200
    )
    assert errors == []
    assert store.read_status(eval_dir)["step"] == 199
