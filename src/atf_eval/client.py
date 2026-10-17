"""Agent-side client for the atf-eval API.

The agent owner writes only an adapter (a `TrajectoryAgent`: `run_turn()` returning a
`NormalizedTurn`). This client runs that adapter locally over the golden dataset's
customer inputs, submits what the agent did to an atf-eval server, waits for the
evaluation and downloads the results:

    from atf_eval.client import AtfClient
    result = AtfClient("http://localhost:8000").evaluate(MyAdapter(), "golden_dataset.jsonl", out_dir="results")

or, with no code at all:  atf-eval submit --adapter my_adapter:MyAdapter --dataset golden_dataset.jsonl

The server never calls the agent; it only receives trajectories. Judge API keys stay on the server.
"""
from __future__ import annotations

import dataclasses
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from atf_eval.adapter import TrajectoryAgent
from atf_eval.dataset import GoldenTurn, group_by_conversation, load_dataset
from atf_eval.normalized import NormalizedTurn

DEFAULT_SERVER = "http://127.0.0.1:8000"


class AtfApiError(RuntimeError):
    pass


@dataclass
class Collected:
    observed: dict[str, list[NormalizedTurn]] = field(default_factory=dict)
    errors: list[dict] = field(default_factory=list)  # agent failures: {"conversation_id", "turn_id", "error"}
    availability: dict[str, str] = field(default_factory=dict)


def collect(
    adapter: TrajectoryAgent,
    golden: list[GoldenTurn],
    concurrency: int = 1,
    limit: int | None = None,
) -> Collected:
    """Drive the agent through every golden conversation's customer inputs. A turn that
    raises ends that conversation (a stateful agent's context may be corrupt), as in
    `atf-eval run`; the turns up to then are still submitted."""
    conversations = list(group_by_conversation(golden).items())[:limit]

    def one(item) -> tuple[str, list[NormalizedTurn], dict | None]:
        cid, turns = item
        history: list[NormalizedTurn] = []
        for gt in turns:
            try:
                history.append(adapter.run_turn(cid, gt.turn_id, gt.user_input, list(history)))
            except Exception as e:  # noqa: BLE001 - any agent failure is data, not a crash
                return cid, history, {"conversation_id": cid, "turn_id": gt.turn_id, "error": repr(e)}
        return cid, history, None

    if concurrency <= 1:
        results = [one(c) for c in conversations]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(pool.map(one, conversations))

    out = Collected(availability=dict(getattr(adapter, "availability", None) or {}))
    for cid, turns, err in results:
        out.observed[cid] = turns
        if err:
            out.errors.append(err)
    return out


def _payload(turns: list[NormalizedTurn]) -> list[dict]:
    return json.loads(json.dumps([dataclasses.asdict(t) for t in turns], default=str))


class AtfClient:
    def __init__(self, base_url: str | None = None, timeout: float = 60.0) -> None:
        try:
            import httpx
        except ImportError as e:  # pragma: no cover
            raise ImportError("the atf-eval client needs httpx: pip install 'atf-eval[client]'") from e
        self.base_url = (base_url or os.environ.get("ATF_SERVER") or DEFAULT_SERVER).rstrip("/")
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout)

    def _check(self, r) -> dict:
        if r.status_code >= 400:
            raise AtfApiError(f"{r.request.method} {r.request.url.path} -> {r.status_code}: {r.text[:500]}")
        return r.json()

    def health(self) -> dict:
        return self._check(self._http.get("/health"))

    def upload_golden(self, name: str, golden: list[GoldenTurn]) -> str:
        body = {"name": name, "turns": [t.model_dump(mode="json") for t in golden]}
        return self._check(self._http.post("/v1/goldens", json=body))["golden_id"]

    def submit(
        self,
        collected: Collected,
        *,
        golden_id: str | None = None,
        golden: list[GoldenTurn] | None = None,
        options: dict | None = None,
        label: str = "atf-eval client",
    ) -> str:
        body = {
            "observed": {cid: _payload(turns) for cid, turns in collected.observed.items()},
            "availability": collected.availability,
            "options": options or {},
            "label": label,
        }
        if golden_id:
            body["golden_id"] = golden_id
        else:
            body["golden_turns"] = [t.model_dump(mode="json") for t in golden or []]
        return self._check(self._http.post("/v1/evaluations", json=body))["evaluation_id"]

    def wait(self, evaluation_id: str, poll: float = 2.0, timeout: float = 3600.0) -> dict:
        deadline = time.time() + timeout
        while True:
            doc = self._check(self._http.get(f"/v1/evaluations/{evaluation_id}"))
            if doc["status"] == "done":
                return doc
            if doc["status"] == "failed":
                raise AtfApiError(f"evaluation {evaluation_id} failed: {doc.get('error')}")
            if time.time() > deadline:
                raise AtfApiError(f"timed out waiting for evaluation {evaluation_id} ({doc['status']})")
            time.sleep(poll)

    def download(self, result: dict, out_dir: str | Path) -> dict[str, Path]:
        """Save the dashboard and every CSV/JSON artefact into out_dir."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        saved = {}
        for name, link in result["links"].items():
            r = self._http.get(link)
            if r.status_code >= 400:
                continue
            fname = result["files"].get(name, f"{name}.html")
            (out / fname).write_bytes(r.content)
            saved[name] = out / fname
        (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        saved["result"] = out / "result.json"
        return saved

    def evaluate(
        self,
        adapter: TrajectoryAgent,
        golden: str | Path | list[GoldenTurn],
        *,
        golden_id: str | None = None,
        options: dict | None = None,
        out_dir: str | Path | None = None,
        concurrency: int = 1,
        limit: int | None = None,
        poll: float = 2.0,
        timeout: float = 3600.0,
    ) -> dict:
        """Run the agent, submit, wait, (optionally) download. Returns the finished result document,
        with `agent_errors` (turns where the agent raised) and `saved` (downloaded files) added."""
        turns = load_dataset(str(golden)) if isinstance(golden, (str, Path)) else golden
        collected = collect(adapter, turns, concurrency=concurrency, limit=limit)
        eval_id = self.submit(collected, golden_id=golden_id, golden=None if golden_id else turns, options=options)
        result = self.wait(eval_id, poll=poll, timeout=timeout)
        result["agent_errors"] = collected.errors
        if out_dir:
            result["saved"] = {k: str(v) for k, v in self.download(result, out_dir).items()}
        return result
