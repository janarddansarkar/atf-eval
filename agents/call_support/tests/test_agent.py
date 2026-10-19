"""The call-support agent, offline (keyword rules instead of the LLM)."""
from __future__ import annotations

import json

from call_support import CallSupportAgent
from call_support.graph import build_graph


def _agent(**kw):
    return CallSupportAgent(build_graph(use_llm=False), **kw)


def test_unverified_caller_is_asked_for_an_account_then_the_issue_is_remembered():
    a = _agent(save_traces=False)
    t1 = a.chat("c", "my internet is slow")
    assert [n.node_id for n in t1.nodes] == ["identify_customer"] and t1.route["path"] == "ask_account"
    t2 = a.chat("c", "ACC-1003")
    # the earlier complaint, not "ACC-1003", was classified; no outage in S2 -> ticket
    assert [n.node_id for n in t2.nodes] == ["identify_customer", "classify_intent", "outage_node", "escalate_node"]
    assert t2.outcome["id"] == "ticket_created" and t2.turn_id == 2


def test_tool_calls_are_attached_to_the_node_that_made_them():
    t = _agent(save_traces=False).chat("c", "why is my bill so high? ACC-1002")
    assert {n.node_id: [c.tool for c in n.tool_calls] for n in t.nodes} == {
        "identify_customer": ["lookup_account"], "classify_intent": [], "billing_node": ["get_bill"]}


def test_state_changes_are_attributed_to_the_node_that_set_them():
    t = _agent(save_traces=False).chat("c", "internet down, ACC-1001")
    by_node = {n.node_id: n.state_changes for n in t.nodes}
    assert by_node["identify_customer"] == [{"key": "authenticated", "old": False, "new": True}]
    assert by_node["classify_intent"] == [{"key": "intent", "old": None, "new": "outage"}]


def test_unknown_account_is_not_authenticated():
    t = _agent(save_traces=False).chat("c", "internet down, ACC-9999")
    assert t.route["path"] == "ask_account" and not any(n.state_changes for n in t.nodes)


def test_conversations_are_independent():
    a = _agent(save_traces=False)
    a.chat("one", "internet down, ACC-1001")
    assert a.chat("two", "hello").route["path"] == "ask_account"  # "two" is not verified


def test_every_run_saves_a_trace_and_raw_events(tmp_path):
    a = _agent(trace_dir=tmp_path)
    a.annotate_trace("t1", scenario="billing")
    a.chat("t1", "why is my bill so high? ACC-1002")
    a.chat("t1", "I want to dispute it with a human")

    doc = json.loads((tmp_path / "t1.json").read_text())
    assert doc["scenario"] == "billing" and [t["turn_id"] for t in doc["turns"]] == [1, 2]
    assert doc["turns"][0]["nodes"][2]["tool_calls"][0]["tool_name"] == "get_bill"
    assert doc["turns"][0]["timing"]["response_latency_ms"] >= 0
    assert doc["outcome"]["id"] == "ticket_created"

    events = [json.loads(l) for l in (tmp_path / "raw" / "t1.jsonl").read_text().splitlines()]
    assert {"node_start", "tool_start", "tool_end"} <= {e["event"] for e in events}
    assert not any(e.get("node", "").startswith("__") for e in events)

    a2 = _agent(trace_dir=tmp_path)  # a new process reusing the id starts a fresh trace
    a2.chat("t1", "internet down ACC-1001")
    assert [t["turn_id"] for t in json.loads((tmp_path / "t1.json").read_text())["turns"]] == [1]
