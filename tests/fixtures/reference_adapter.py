"""Test-only reference normalizer for agent-eval-main's own fixtures.

Not part of src/atf_eval. Two distinct shapes:

  - agent-eval-main/golden/*.json is already canonical (matches
    normalized_trajectory.schema.json exactly) -- straight field mapping.
  - agent-eval-main/tests/fixtures/scenarios/*.json are pre-Adapter, raw-ish
    observed traces (per METRICS.md §12's documented Adapter/fixture split):
    they use `sequence_index`/`tool_name` instead of the canonical
    `sequence`/`tool_id`, and inconsistently key their turns under `turns`
    (a full-conversation array) or `turn` (a single-turn object). This module
    normalizes either shape into the same NormalizedTurn list atf_eval's
    metric functions operate on, so agent-eval-main's own fixtures can be
    used as an end-to-end conformance check.
"""
from __future__ import annotations

import json
from pathlib import Path

from atf_eval.normalized import (
    InterruptionEvent,
    NormalizedNode,
    NormalizedTurn,
    Outcome,
    Routing,
    StateChange,
    ToolCall,
    TurnTiming,
)


def _state_changes(raw: list[dict]) -> list[StateChange]:
    return [StateChange(key=c["key"], old=c.get("old"), new=c.get("new")) for c in raw]


def _routing(raw_turn: dict) -> Routing | None:
    """Turn-level `routing` block (canonical golden and observed fixtures now
    both carry it) -> Routing, so RS's judge and routing-order similarity
    have an expected path/target to compare against."""
    raw = raw_turn.get("routing")
    if not raw:
        return None
    return Routing(path=raw.get("path"), target=raw.get("target"))


def _timing(raw_turn: dict) -> TurnTiming | None:
    """Turn-level `timing` block (structured timestamps/response-latency/
    interruption events, no audio required) -> TurnTiming, so Group 5's
    timing-tier LLM metrics have evidence to score against."""
    raw = raw_turn.get("timing")
    if not raw:
        return None
    return TurnTiming(
        customer_start_ms=raw.get("customer_start_ms"),
        customer_end_ms=raw.get("customer_end_ms"),
        agent_start_ms=raw.get("agent_start_ms"),
        agent_end_ms=raw.get("agent_end_ms"),
        response_latency_ms=raw.get("response_latency_ms"),
        interruptions=[
            InterruptionEvent(by=i["by"], at_ms=i["at_ms"], note=i.get("note"))
            for i in raw.get("interruptions", [])
        ],
    )


def _canonical_tool_calls(raw: list[dict]) -> list[ToolCall]:
    return [
        ToolCall(
            tool_id=tc["tool_id"],
            arguments=tc.get("input", {}),
            result=tc.get("output"),
            sequence=tc.get("sequence"),
            status=tc.get("status", "unknown"),
        )
        for tc in raw
    ]


def _raw_fixture_tool_calls(raw: list[dict]) -> list[ToolCall]:
    return [
        ToolCall(
            tool_id=tc["tool_name"],
            arguments=tc.get("input", {}),
            result=tc.get("output"),
            sequence=tc.get("sequence_index"),
            status="success",
        )
        for tc in raw
    ]


def load_golden_turns(path: Path) -> list[NormalizedTurn]:
    """agent-eval-main/golden/*.json -> NormalizedTurn list. Already
    canonical; the trace-level `outcome` is attached to the last turn since
    atf_eval's metric functions read outcome from NormalizedTurn.outcome,
    not from a separate trace-level field."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    turns = []
    for t in doc["turns"]:
        nodes = [
            NormalizedNode(
                node_id=n["node_id"],
                sequence=n.get("sequence"),
                output=n.get("output", {}),
                state_changes=_state_changes(n.get("state_changes", [])),
                tool_calls=_canonical_tool_calls(n.get("tool_calls", [])),
            )
            for n in t.get("nodes", [])
        ]
        # a flattened copy of the node-level calls; the metrics de-duplicate
        # it against the node-level events (atf_eval.metrics.evidence)
        flat_tool_calls = [tc for n in nodes for tc in n.tool_calls]
        turns.append(
            NormalizedTurn(
                conversation_id=doc["trace_id"],
                turn_id=t["turn_id"],
                nodes=nodes,
                tool_calls=flat_tool_calls,
                routing=_routing(t),
                response=t.get("output", {}).get("agent"),
                customer_input=t.get("input", {}).get("customer"),
                timing=_timing(t),
            )
        )

    raw_outcome = doc.get("outcome")
    if raw_outcome is not None and turns:
        turns[-1].outcome = Outcome(id=raw_outcome["id"], attributes=raw_outcome.get("attributes", {}))
    return turns


def _normalize_raw_turn(raw_turn: dict) -> NormalizedTurn:
    nodes = [
        NormalizedNode(
            node_id=n["node_id"],
            sequence=n.get("sequence_index"),
            state_changes=_state_changes(n.get("state_changes", [])),
            tool_calls=_raw_fixture_tool_calls(n.get("tool_calls", [])),
        )
        for n in raw_turn.get("nodes", [])
    ]
    flat_tool_calls = [tc for n in nodes for tc in n.tool_calls]
    return NormalizedTurn(
        conversation_id="_",
        turn_id=raw_turn["turn_id"],
        nodes=nodes,
        tool_calls=flat_tool_calls,
        routing=_routing(raw_turn),
        response=raw_turn.get("conversation", {}).get("agent"),
        customer_input=raw_turn.get("conversation", {}).get("customer"),
        timing=_timing(raw_turn),
    )


def load_fixture_turns(path: Path) -> list[NormalizedTurn]:
    """agent-eval-main/tests/fixtures/scenarios/*.json -> NormalizedTurn
    list. Handles both the full-conversation `"turns": [...]` shape and the
    single-turn `"turn": {...}` shape these fixtures inconsistently use."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    if "turns" in doc:
        turns = [_normalize_raw_turn(t) for t in doc["turns"]]
    else:
        turns = [_normalize_raw_turn(doc["turn"])]
    # Some observed fixtures carry a trace-level `outcome`, same convention
    # as golden -- attach it to the last turn so OS can see it.
    raw_outcome = doc.get("outcome")
    if raw_outcome is not None and turns:
        turns[-1].outcome = Outcome(id=raw_outcome["id"], attributes=raw_outcome.get("attributes", {}))
    return turns
