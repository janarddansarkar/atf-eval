"""The atf-eval HTTP API.

    GET  /health
    POST /v1/goldens                       store a golden dataset          -> {golden_id}
    GET  /v1/goldens, /v1/goldens/{id}
    POST /v1/evaluations                   submit observed trajectories    -> 202 {evaluation_id}
    GET  /v1/evaluations/{id}              status; scores once finished
    GET  /v1/evaluations/{id}/dashboard    the HTML dashboard
    GET  /v1/evaluations/{id}/files/{name} metrics / components / llm CSVs, settings JSON

An evaluation runs in a background thread (the LLM judges take a while); poll the
status endpoint. Judge API keys live on the server (env or .env), never in a request.
The server never imports or calls the agent: the agent side runs the agent and
submits what it did (see atf_eval.client).
"""
from __future__ import annotations

import json
import os
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from atf_eval.api.service import EvalOptions, evaluate
from atf_eval.api.store import Store
from atf_eval.dataset import GoldenTurn
from atf_eval.normalized import NormalizedTurn

DEFAULT_DATA_DIR = "atf_server_data"


class GoldenUpload(BaseModel):
    name: str
    turns: list[GoldenTurn]


class EvaluationRequest(BaseModel):
    golden_id: str | None = None
    golden_turns: list[GoldenTurn] | None = None  # alternative to golden_id: send the golden inline
    observed: dict[str, list[NormalizedTurn]]  # conversation_id -> the agent's normalized turns
    availability: dict[str, str] = Field(default_factory=dict)  # evidence the agent's traces never expose
    options: EvalOptions = Field(default_factory=EvalOptions)
    label: str = "api submission"


DEFAULT_WORKERS = 2


def _log(message: str) -> None:
    print(f"[atf-eval] {message}", file=sys.stderr, flush=True)


def create_app(data_dir: str | Path | None = None, workers: int = DEFAULT_WORKERS) -> FastAPI:
    if workers < 1:
        raise ValueError(f"workers must be at least 1, got {workers}")
    store = Store(data_dir or os.environ.get("ATF_SERVER_DATA", DEFAULT_DATA_DIR))
    for eval_id in store.fail_interrupted():
        _log(f"evaluation {eval_id} was left unfinished by a previous server process; marked failed")
    pool = ThreadPoolExecutor(max_workers=workers)
    # evaluation ids accepted but not yet started, in submission order, so a
    # `queued` status can say how many evaluations are ahead of it
    waiting: list[str] = []
    running: set[str] = set()
    # last-resort record of a job's outcome when status.json could not be
    # written even after retries, so GET still reports it instead of a stale
    # `queued`/`running` that would leave the client polling forever
    unsaved_outcome: dict[str, dict] = {}
    queue_lock = threading.Lock()
    app = FastAPI(title="atf-eval", version="0.1.0",
                  description="Agent Trajectory Fidelity evaluation as a service.")
    app.state.store = store

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        """FastAPI's default 422, unless the rejected input can't be echoed back
        as JSON (e.g. a NaN/Infinity weight) -- then the same 422 without the
        echoed input instead of a 500."""
        try:
            return await request_validation_exception_handler(request, exc)
        except ValueError:
            detail = [{k: e[k] for k in ("type", "loc", "msg") if k in e} for e in exc.errors()]
            return JSONResponse(status_code=422, content={"detail": detail})

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    # ---- goldens ----
    @app.post("/v1/goldens", status_code=201)
    def upload_golden(body: GoldenUpload) -> dict:
        if not body.turns:
            raise HTTPException(422, "a golden needs at least one turn")
        return store.save_golden(body.name, [t.model_dump(mode="json") for t in body.turns])

    @app.get("/v1/goldens")
    def list_goldens() -> list[dict]:
        return store.list_goldens()

    @app.get("/v1/goldens/{golden_id}")
    def get_golden(golden_id: str) -> dict:
        doc = store.load_golden(golden_id)
        if doc is None:
            raise HTTPException(404, f"no golden {golden_id!r}")
        return doc

    # ---- evaluations ----
    @app.post("/v1/evaluations", status_code=202)
    def submit(body: EvaluationRequest) -> dict:
        if body.golden_turns:
            golden = body.golden_turns
        elif body.golden_id:
            doc = store.load_golden(body.golden_id)
            if doc is None:
                raise HTTPException(404, f"no golden {body.golden_id!r}")
            golden = [GoldenTurn.model_validate(t) for t in doc["turns"]]
        else:
            raise HTTPException(422, "send golden_id or golden_turns")
        if not body.observed:
            raise HTTPException(422, "observed is empty")

        eval_id, eval_dir = store.new_evaluation()
        store.write_status(eval_dir, "queued", conversations=len(body.observed))

        def job() -> None:
            # Every step, status writes included, is inside the try: an
            # exception escaping here would vanish into the pool's Future and
            # leave the evaluation `queued`/`running` forever.
            with queue_lock:
                waiting.remove(eval_id)
                running.add(eval_id)
            try:
                store.write_status(eval_dir, "running", conversations=len(body.observed))
                _log(f"evaluation {eval_id} running ({len(body.observed)} conversation(s))")
                result = evaluate(golden, body.observed, body.availability, body.options, eval_dir, body.label)
                (eval_dir / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
                store.write_status(eval_dir, "done", conversations=result["summary"]["conversations"])
                _log(f"evaluation {eval_id} done")
            except Exception as e:  # noqa: BLE001 - surface any failure as the job's status
                _log(f"evaluation {eval_id} failed: {e}\n{traceback.format_exc()}")
                try:
                    store.write_status(eval_dir, "failed", error=str(e), detail=traceback.format_exc(limit=3))
                except Exception as e2:  # noqa: BLE001 - keep the outcome in memory instead
                    _log(f"evaluation {eval_id}: could not record the failure in status.json: {e2}")
                    with queue_lock:
                        unsaved_outcome[eval_id] = {
                            "status": "failed", "error": str(e),
                            "detail": f"also could not save this status to status.json: {e2}",
                        }
            finally:
                with queue_lock:
                    running.discard(eval_id)

        with queue_lock:
            waiting.append(eval_id)
            ahead = len(waiting) - 1 + len(running)
        pool.submit(job)
        if ahead >= workers:
            _log(f"evaluation {eval_id} queued behind {ahead} evaluation(s); {workers} run at a time "
                 "(raise with `atf-eval serve --workers N`)")
        return {"evaluation_id": eval_id, "status": "queued", "url": f"/v1/evaluations/{eval_id}"}

    def _dir(eval_id: str) -> Path:
        d = store.eval_dir(eval_id)
        if d is None:
            raise HTTPException(404, f"no evaluation {eval_id!r}")
        return d

    @app.get("/v1/evaluations/{eval_id}")
    def get_evaluation(eval_id: str) -> dict:
        d = _dir(eval_id)
        with queue_lock:
            unsaved = unsaved_outcome.get(eval_id)
        doc = {"evaluation_id": eval_id, **(unsaved or store.read_status(d))}
        if doc["status"] == "queued":
            with queue_lock:
                if eval_id in waiting:
                    # 1 = next to start once a worker frees up
                    doc["queue_position"] = waiting.index(eval_id) + 1
                    doc["running"] = len(running)
                    doc["workers"] = workers
        result = d / "result.json"
        if doc["status"] == "done" and result.exists():
            doc.update(json.loads(result.read_text(encoding="utf-8")))
            doc["links"] = {**{k: f"/v1/evaluations/{eval_id}/files/{v}" for k, v in doc["files"].items()},
                            "dashboard": f"/v1/evaluations/{eval_id}/dashboard"}
        return doc

    @app.get("/v1/evaluations/{eval_id}/dashboard", response_class=HTMLResponse)
    def dashboard(eval_id: str) -> HTMLResponse:
        files = list(_dir(eval_id).glob("atf_dashboard_*.html"))
        if not files:
            raise HTTPException(404, "dashboard not ready (evaluation not finished?)")
        return HTMLResponse(files[0].read_text(encoding="utf-8"))

    @app.get("/v1/evaluations/{eval_id}/files/{name}")
    def file(eval_id: str, name: str) -> FileResponse:
        d = _dir(eval_id)
        path = (d / name).resolve()
        if path.parent != d.resolve() or not path.is_file():
            raise HTTPException(404, f"no file {name!r}")
        return FileResponse(path)

    return app

