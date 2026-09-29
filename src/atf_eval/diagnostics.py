"""Turns ATF's raw scores into the specific "what went wrong, where" callouts
a scorecard needs -- the diagnostic layer the spec insists on (§17: "never
collapse the evaluation into ATF alone").
"""
from __future__ import annotations

from dataclasses import dataclass

from atf_eval.metrics.evidence import turn_tool_calls, turn_transitions
from atf_eval.metrics.state import sts_turn
from atf_eval.normalized import NormalizedTurn


@dataclass
class StateDeviation:
    turn_id: int
    key: str
    expected_old: object
    expected_new: object
    observed_new: object | None
    reason: str


@dataclass
class ToolDeviation:
    turn_id: int
    expected_tool: str
    observed: str


def _reason(expected_key: str, observed_change) -> str:
    if observed_change is None:
        return f"Agent did not update '{expected_key}'."
    return f"Agent set '{expected_key}' to '{observed_change.new}' instead of the expected value."


def find_key_state_deviation(
    expected_turns: list[NormalizedTurn], observed_turns: list[NormalizedTurn]
) -> StateDeviation | None:
    """The state mismatch on the worst-scoring STS turn, i.e. the single
    deviation most responsible for dragging STS down."""
    worst_pair = None
    worst_score = None
    for expected, observed in zip(expected_turns, observed_turns):
        score = sts_turn(expected, observed)
        if score is None:
            continue
        if worst_score is None or score < worst_score:
            worst_score = score
            worst_pair = (expected, observed)

    if worst_pair is None:
        return None
    expected, observed = worst_pair

    observed_by_key = {c.key: c for c in turn_transitions(observed)}
    for ec in turn_transitions(expected):
        oc = observed_by_key.get(ec.key)
        if oc is None or oc.old != ec.old or oc.new != ec.new:
            return StateDeviation(
                turn_id=expected.turn_id,
                key=ec.key,
                expected_old=ec.old,
                expected_new=ec.new,
                observed_new=oc.new if oc else None,
                reason=_reason(ec.key, oc),
            )
    return None


def find_key_tool_deviation(
    expected_turns: list[NormalizedTurn], observed_turns: list[NormalizedTurn]
) -> ToolDeviation | None:
    """First expected tool call that never shows up in the observed trace."""
    for expected, observed in zip(expected_turns, observed_turns):
        observed_names = {t.tool_id for t in turn_tool_calls(observed)}
        for tc in turn_tool_calls(expected):
            if tc.tool_id not in observed_names:
                return ToolDeviation(
                    turn_id=expected.turn_id, expected_tool=tc.tool_id, observed="Not invoked"
                )
    return None
