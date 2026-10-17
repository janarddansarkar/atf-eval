"""The atf-eval HTTP API end to end, in-process (FastAPI TestClient), judges off so it is offline."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from atf_eval.api.app import create_app  # noqa: E402
from atf_eval.client import AtfApiError, AtfClient, collect  # noqa: E402
from atf_eval.dataset import load_dataset  # noqa: E402
from atf_eval.normalized import NormalizedNode, NormalizedTurn  # noqa: E402
from tests.fixtures.mock_adapter import MockTrajectoryAgent  # noqa: E402

DATASET = str(ROOT / "tests" / "fixtures" / "mock_golden_dataset.jsonl")
OFFLINE = {"routing_judge": False, "llm_evals": False}


@pytest.fixture
def client(tmp_path):
    c = AtfClient("http://testserver")
    c._http = TestClient(create_app(tmp_path))
    return c


def test_health_and_golden_upload(client):
    assert client.health() == {"status": "ok"}
    gid = client.upload_golden("demo", load_dataset(DATASET))
    assert gid.startswith("demo-")
    listed = client._http.get("/v1/goldens").json()
    assert listed[0]["conversations"] == ["demo_conv_01", "demo_conv_02"]


def test_full_round_trip_scores_perfect_and_saves_artifacts(client, tmp_path):
    result = client.evaluate(MockTrajectoryAgent(), DATASET, options=OFFLINE, out_dir=tmp_path / "out", poll=0.05)
    assert result["status"] == "done" and result["summary"]["conversations"] == 2
    for run in result["runs"]:
        assert run["nts"] == run["sts"] == run["tis"] == run["os"] == 1.0
        assert run["rs"] is None  # judge off -> RS is N/A, never 0
        assert "nts_coverage" in run["components"]
    saved = {p.name for p in (tmp_path / "out").iterdir()}
    assert any(n.startswith("atf_dashboard_") and n.endswith(".html") for n in saved)
    assert any(n.startswith("atf_metrics_components_") for n in saved) and "result.json" in saved
    assert "ATF Trajectory Metrics" in client._http.get(result["links"]["dashboard"]).text


def test_a_worse_agent_scores_lower_and_golden_can_be_stored_once(client):
    class SkipsANode(MockTrajectoryAgent):
        def run_turn(self, conversation_id, turn_id, user_input, history):
            t = super().run_turn(conversation_id, turn_id, user_input, history)
            return NormalizedTurn(**{**t.__dict__, "nodes": [NormalizedNode(node_id="wrong_node")]})

    golden = load_dataset(DATASET)
    gid = client.upload_golden("demo", golden)
    result = client.wait(client.submit(collect(SkipsANode(), golden), golden_id=gid, options=OFFLINE), poll=0.05)
    assert all(r["nts"] < 0.5 for r in result["runs"])


def test_unknown_conversations_are_skipped_and_errors_surface(client):
    golden = load_dataset(DATASET)
    collected = collect(MockTrajectoryAgent(), golden)
    collected.observed["not_in_golden"] = collected.observed["demo_conv_01"]
    result = client.wait(client.submit(collected, golden=golden, options=OFFLINE), poll=0.05)
    assert [s["conversation_id"] for s in result["skipped"]] == ["not_in_golden"]

    with pytest.raises(AtfApiError):
        client.submit(collected, golden_id="nope", options=OFFLINE)


def test_a_failing_agent_does_not_crash_collection():
    class Boom:
        def run_turn(self, *a):
            raise RuntimeError("down")

    c = collect(Boom(), load_dataset(DATASET))
    assert {e["conversation_id"] for e in c.errors} == {"demo_conv_01", "demo_conv_02"}
    assert all(turns == [] for turns in c.observed.values())
