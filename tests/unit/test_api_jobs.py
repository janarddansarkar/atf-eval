"""Evaluation job lifecycle on the API server: an evaluation must never be left
`queued`/`running` forever. Covers the worker pool being busy (visible queue
position, then progress), a job crashing before/while it runs, and evaluations
orphaned by a server restart. Offline: `evaluate` is stubbed or the LLM metrics
are off."""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import atf_eval.api.app as app_mod  # noqa: E402
from atf_eval.api.store import Store  # noqa: E402
from atf_eval.client import AtfClient, collect  # noqa: E402
from atf_eval.dataset import load_dataset  # noqa: E402
from tests.fixtures.mock_adapter import MockTrajectoryAgent  # noqa: E402

DATASET = str(ROOT / "tests" / "fixtures" / "mock_golden_dataset.jsonl")
OFFLINE = {"llm_evals": False}


def _client(tmp_path, workers=2) -> AtfClient:
    c = AtfClient("http://testserver")
    c._http = TestClient(app_mod.create_app(tmp_path, workers=workers))
    return c


def _submit(c: AtfClient) -> str:
    golden = load_dataset(DATASET)
    return c.submit(collect(MockTrajectoryAgent(), golden), golden=golden, options=OFFLINE)


def _status(c: AtfClient, eval_id: str) -> dict:
    return c._http.get(f"/v1/evaluations/{eval_id}").json()


def _until(predicate, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def blocked_evaluate(monkeypatch):
    """Stand-in for slow judge calls: evaluate() waits until released."""
    release = threading.Event()
    real = app_mod.evaluate

    def slow(*args, **kwargs):
        assert release.wait(30), "test never released the evaluation"
        return real(*args, **kwargs)

    monkeypatch.setattr(app_mod, "evaluate", slow)
    yield release
    release.set()


def test_busy_workers_show_queue_position_and_the_queue_drains(tmp_path, blocked_evaluate):
    c = _client(tmp_path, workers=1)
    first, second, third = _submit(c), _submit(c), _submit(c)
    assert _until(lambda: _status(c, first)["status"] == "running")
    s2, s3 = _status(c, second), _status(c, third)
    assert (s2["status"], s2["queue_position"], s2["running"], s2["workers"]) == ("queued", 1, 1, 1)
    assert (s3["status"], s3["queue_position"]) == ("queued", 2)

    blocked_evaluate.set()
    for eval_id in (first, second, third):
        assert c.wait(eval_id, poll=0.02, timeout=20)["status"] == "done"
    assert "queue_position" not in _status(c, third)


def test_more_workers_run_evaluations_in_parallel(tmp_path, blocked_evaluate):
    c = _client(tmp_path, workers=3)
    ids = [_submit(c) for _ in range(3)]
    assert _until(lambda: all(_status(c, i)["status"] == "running" for i in ids))
    blocked_evaluate.set()
    assert all(c.wait(i, poll=0.02, timeout=20)["status"] == "done" for i in ids)


def test_wait_reports_each_status_change(tmp_path, blocked_evaluate):
    c = _client(tmp_path, workers=1)
    first, second = _submit(c), _submit(c)
    assert _until(lambda: _status(c, first)["status"] == "running")
    seen = []
    threading.Timer(0.3, blocked_evaluate.set).start()
    c.wait(second, poll=0.02, timeout=20, on_status=lambda d: seen.append((d["status"], d.get("queue_position"))))
    assert seen[0] == ("queued", 1) and seen[-1] == ("done", None)
    assert len(seen) == len(set(seen))  # one callback per change, not per poll


def test_a_failing_running_status_write_marks_the_evaluation_failed(tmp_path, monkeypatch):
    # Before the fix this exception vanished into the pool's Future and the
    # evaluation stayed `queued` forever with no error anywhere.
    real = Store.write_status

    def flaky(self, eval_dir, status, **extra):
        if status == "running":
            raise OSError("disk full")
        return real(self, eval_dir, status, **extra)

    monkeypatch.setattr(Store, "write_status", flaky)
    c = _client(tmp_path)
    eval_id = _submit(c)
    assert _until(lambda: _status(c, eval_id)["status"] == "failed")
    assert _status(c, eval_id)["error"] == "disk full"


def test_a_crashing_evaluation_is_marked_failed_and_frees_its_worker(tmp_path, monkeypatch):
    calls = []

    def boom(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("judge exploded")

    monkeypatch.setattr(app_mod, "evaluate", boom)
    c = _client(tmp_path, workers=1)
    ids = [_submit(c) for _ in range(3)]
    for eval_id in ids:  # one worker: each failure must release it for the next
        assert _until(lambda: _status(c, eval_id)["status"] == "failed")
        assert _status(c, eval_id)["error"] == "judge exploded"
    assert len(calls) == 3


def test_evaluations_orphaned_by_a_restart_are_marked_failed(tmp_path):
    store = Store(tmp_path)
    ids = {}
    for status in ("queued", "running", "done", "failed"):
        eval_id, d = store.new_evaluation()
        store.write_status(d, status)
        ids[status] = eval_id

    c = _client(tmp_path)  # a new server process over the same data dir
    for status in ("queued", "running"):
        doc = _status(c, ids[status])
        assert doc["status"] == "failed" and "server restarted" in doc["error"]
    assert _status(c, ids["done"])["status"] == "done"
    assert _status(c, ids["failed"])["status"] == "failed" and "error" not in _status(c, ids["failed"])


def test_create_app_rejects_zero_workers(tmp_path):
    with pytest.raises(ValueError, match="at least 1"):
        app_mod.create_app(tmp_path, workers=0)


def test_serve_validates_workers(monkeypatch, capsys, tmp_path):
    from atf_eval import cli

    started = []
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: started.append(app))
    parse = cli._build_parser().parse_args

    assert cli.cmd_serve(parse(["serve", "--workers", "0"])) == 2
    assert "--workers must be at least 1" in capsys.readouterr().err
    monkeypatch.setenv("ATF_SERVER_WORKERS", "lots")
    assert cli.cmd_serve(parse(["serve"])) == 2
    assert "ATF_SERVER_WORKERS must be an integer" in capsys.readouterr().err
    monkeypatch.setenv("ATF_SERVER_WORKERS", "4")
    assert cli.cmd_serve(parse(["serve", "--data-dir", str(tmp_path)])) == 0
    assert "up to 4 evaluation(s)" in capsys.readouterr().err and len(started) == 1


# --- Windows: status.json held open by another handle (WinError 5) -------------------

def _windows_replace(monkeypatch, fail_while):
    """os.replace raising PermissionError [WinError 5] while fail_while() is true."""
    import os

    from atf_eval.api import store as store_mod

    real = os.replace

    def replace(src, dst):
        if fail_while():
            raise PermissionError(13, "Access is denied")
        return real(src, dst)

    monkeypatch.setattr(store_mod.os, "replace", replace)
    monkeypatch.setattr(store_mod, "SHARING_RETRY_SECONDS", 1.0)


def test_evaluation_completes_when_status_json_is_briefly_locked(tmp_path, monkeypatch):
    # the user's case: the `running` write hit WinError 5 right after `queued` was written
    c = _client(tmp_path)
    held_until = []
    _windows_replace(monkeypatch, lambda: held_until and time.monotonic() < held_until[0])

    real_ws = Store.write_status

    def lock_after_queued(self, eval_dir, status, **extra):
        real_ws(self, eval_dir, status, **extra)
        if status == "queued":  # e.g. Defender scans the file that was just written
            held_until[:] = [time.monotonic() + 0.3]

    monkeypatch.setattr(Store, "write_status", lock_after_queued)
    eval_id = _submit(c)
    assert c.wait(eval_id, poll=0.02, timeout=20)["status"] == "done"


def test_unsaveable_status_is_still_reported_and_the_client_stops_waiting(tmp_path, monkeypatch):
    """status.json can be written as `queued` but never replaced afterwards
    (a permanent WinError 5): the failure must still reach the client."""
    from atf_eval.client import AtfApiError

    job_started = threading.Event()
    real_ws = Store.write_status

    def mark_job_start(self, eval_dir, status, **extra):
        if status != "queued":
            job_started.set()
        return real_ws(self, eval_dir, status, **extra)

    monkeypatch.setattr(Store, "write_status", mark_job_start)
    _windows_replace(monkeypatch, job_started.is_set)
    c = _client(tmp_path)
    eval_id = _submit(c)

    start = time.monotonic()
    with pytest.raises(AtfApiError, match="failed: .*Access is denied"):
        c.wait(eval_id, poll=0.05, timeout=30)
    assert time.monotonic() - start < 10  # reported, not an endless `queued`
    doc = _status(c, eval_id)
    assert doc["status"] == "failed" and "could not save this status" in doc["detail"]
