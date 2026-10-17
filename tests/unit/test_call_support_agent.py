"""The ATF side of the call-support agent: the adapter that bridges the standalone agent
(call_support_agent/) into atf_eval, and its golden dataset. Offline: keyword rules, no LLM."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("call_support")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agents.call_support.adapter import CallSupportAdapter  # noqa: E402
from atf_eval.aggregate import DEFAULT_WEIGHTS  # noqa: E402
from atf_eval.dashboard.generate import load_fixture_turns  # noqa: E402
from atf_eval.dataset import group_by_conversation, load_dataset  # noqa: E402
from atf_eval.runner import run_evaluation  # noqa: E402
from call_support import CallSupportAgent  # noqa: E402
from call_support.graph import build_graph  # noqa: E402


def _adapter(**kw):
    return CallSupportAdapter(CallSupportAgent(build_graph(use_llm=False), **kw))


def test_adapter_normalizes_nodes_state_tools_routing_and_outcome():
    t = _adapter(save_traces=False).run_turn("c", 1, "why is my bill so high? ACC-1002", [])
    assert [n.node_id for n in t.nodes] == ["identify_customer", "classify_intent", "billing_node"]
    assert {n.node_id: [c.tool_id for c in n.tool_calls] for n in t.nodes}["billing_node"] == ["get_bill"]
    assert [(c.key, c.old, c.new) for c in t.state_changes] == [("authenticated", False, True), ("intent", None, "billing")]
    assert (t.routing.path, t.routing.target) == ("billing", "self_service") and t.outcome is None
    assert t.customer_input and t.response


def test_golden_dataset_scores_perfectly_offline():
    results = run_evaluation(
        group_by_conversation(load_dataset(str(ROOT / "agents/call_support/golden_dataset.jsonl"))),
        _adapter(save_traces=False),
        weights=dict(DEFAULT_WEIGHTS),
    )
    assert len(results) == 3 and all(r.atf == pytest.approx(1.0) for r in results)


def test_traces_the_agent_saves_load_into_the_dashboard(tmp_path):
    a = _adapter(trace_dir=tmp_path)
    t1 = a.run_turn("t2", 1, "internet down ACC-1001", [])
    a.run_turn("t2", 2, "that's all", [t1])
    turns = load_fixture_turns(tmp_path / "t2.json")
    assert [n.node_id for n in turns[1].nodes] == ["classify_intent", "close_node"]
    assert turns[-1].outcome.id == "issue_resolved"
