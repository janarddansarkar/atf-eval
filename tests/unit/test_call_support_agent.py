"""The LangGraph call-support agent (agents/call_support), run offline with
keyword rules (no LLM) and scored against its golden dataset."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("langgraph")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agents.call_support.adapter import CallSupportAdapter  # noqa: E402
from agents.call_support.graph import build_graph  # noqa: E402
from atf_eval.aggregate import DEFAULT_WEIGHTS  # noqa: E402
from atf_eval.dataset import group_by_conversation, load_dataset  # noqa: E402
from atf_eval.runner import run_evaluation  # noqa: E402


def _agent():
    return CallSupportAdapter(build_graph(use_llm=False), save_traces=False)


def test_unverified_caller_is_asked_for_an_account_then_issue_is_remembered():
    a = _agent()
    t1 = a.run_turn("c", 1, "my internet is slow", [])
    assert [n.node_id for n in t1.nodes] == ["identify_customer"] and t1.routing.path == "ask_account"
    t2 = a.run_turn("c", 2, "ACC-1003", [t1])
    # the earlier complaint, not "ACC-1003", was classified; no outage in S2 -> ticket
    assert [n.node_id for n in t2.nodes] == ["identify_customer", "classify_intent", "outage_node", "escalate_node"]
    assert t2.outcome.id == "ticket_created"


def test_tool_calls_are_attached_to_the_node_that_made_them():
    t = _agent().run_turn("c", 1, "why is my bill so high? ACC-1002", [])
    by_node = {n.node_id: [c.tool_id for c in n.tool_calls] for n in t.nodes}
    assert by_node == {"identify_customer": ["lookup_account"], "classify_intent": [], "billing_node": ["get_bill"]}


def test_unknown_account_is_not_authenticated():
    t = _agent().run_turn("c", 1, "internet down, ACC-9999", [])
    assert t.routing.path == "ask_account" and not t.state_changes


def test_golden_dataset_scores_perfectly_offline():
    results = run_evaluation(
        group_by_conversation(load_dataset(str(ROOT / "agents/call_support/golden_dataset.jsonl"))),
        _agent(),
        weights=dict(DEFAULT_WEIGHTS),
    )
    assert len(results) == 3 and all(r.atf == pytest.approx(1.0) for r in results)


def test_agent_saves_an_atf_ready_trace_and_raw_events_as_it_runs(tmp_path):
    import json

    a = CallSupportAdapter(build_graph(use_llm=False), trace_dir=tmp_path)
    a.annotate_trace("t1", golden_scenario="support_billing_dispute")
    t1 = a.run_turn("t1", 1, "why is my bill so high? ACC-1002", [])
    a.run_turn("t1", 2, "I want to dispute it with a human", [t1])

    doc = json.loads((tmp_path / "t1.json").read_text())
    assert doc["golden_scenario"] == "support_billing_dispute" and doc["trace_type"] == "observed"
    assert [t["turn_id"] for t in doc["turns"]] == [1, 2]
    assert doc["turns"][0]["nodes"][2]["tool_calls"][0]["tool_name"] == "get_bill"
    assert doc["turns"][0]["timing"]["response_latency_ms"] >= 0
    assert doc["outcome"]["id"] == "ticket_created"

    events = [json.loads(l) for l in (tmp_path / "raw" / "t1.jsonl").read_text().splitlines()]
    assert {"node_start", "tool_start", "tool_end"} <= {e["event"] for e in events}
    assert not any(e.get("node", "").startswith("__") for e in events)

    # starting the same conversation id again begins a fresh trace
    a.run_turn("t1", 1, "internet down ACC-1001", [])
    assert [t["turn_id"] for t in json.loads((tmp_path / "t1.json").read_text())["turns"]] == [1]


def test_saved_traces_load_and_score_with_the_dashboard(tmp_path):
    from atf_eval.dashboard.generate import load_fixture_turns

    a = CallSupportAdapter(build_graph(use_llm=False), trace_dir=tmp_path)
    t1 = a.run_turn("t2", 1, "internet down ACC-1001", [])
    a.run_turn("t2", 2, "that's all", [t1])
    turns = load_fixture_turns(tmp_path / "t2.json")
    assert [n.node_id for n in turns[1].nodes] == ["classify_intent", "close_node"]
    assert turns[-1].outcome.id == "issue_resolved"
