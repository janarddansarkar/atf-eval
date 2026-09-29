"""Small convenience readers for writing PolicyRule.check functions -- not
required, just cuts boilerplate for the common cases."""
from __future__ import annotations

from atf_eval.metrics.evidence import turn_state_changes, turn_tool_calls
from atf_eval.normalized import NormalizedTurn, ToolCall


def tool_calls_named(turn: NormalizedTurn, tool_id: str) -> list[ToolCall]:
    return [t for t in turn_tool_calls(turn) if t.tool_id == tool_id]


def state_value(turn: NormalizedTurn, key: str, default=None):
    for change in reversed(turn_state_changes(turn)):
        if change.key == key:
            return change.new
    return default


def latest_state_value(history: list[NormalizedTurn], turn: NormalizedTurn, key: str, default=None):
    """Most recent value of `key` as of and including this turn."""
    value = state_value(turn, key, default=None)
    if value is not None:
        return value
    for past_turn in reversed(history):
        value = state_value(past_turn, key, default=None)
        if value is not None:
            return value
    return default
