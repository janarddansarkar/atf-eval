"""Shared evidence helpers for the component metrics (METRICS.md §2-§4).

A NormalizedTurn can carry the same tool call twice: once on the node it was
observed in and once in the turn-level list (some loaders flatten node events
into the turn). The helpers return each event exactly once and report whether
a trajectory associates its tool calls with nodes -- the condition METRICS.md
§4 uses to choose node-level vs. turn-level tool alignment. Nothing here
invents a node association the trace did not contain (Frozen Rule 7).
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

from atf_eval import lcs
from atf_eval.matching import value_key, values_match
from atf_eval.normalized import NormalizedNode, NormalizedTurn, StateChange, ToolCall

T = TypeVar("T")


def _unattributed(turn_level: Sequence[T], node_level: Sequence[T]) -> list[T]:
    """Turn-level events that are not duplicates of a node-level event
    (matched one-for-one, by identity first and then by equality)."""
    remaining = list(node_level)
    leftover: list[T] = []
    for item in turn_level:
        idx = next((i for i, n in enumerate(remaining) if n is item), None)
        if idx is None:
            idx = next((i for i, n in enumerate(remaining) if n == item), None)
        if idx is None:
            leftover.append(item)
        else:
            remaining.pop(idx)
    return leftover


def order_deviations(expected: Sequence[T], observed: Sequence[T]) -> list[T]:
    """Elements present on both sides whose relative order differs: in the
    multiset intersection but outside the LCS (METRICS.md §8 "order
    deviations")."""
    from collections import Counter

    in_lcs = Counter(expected[i] for i, j in lcs.align(expected, observed) if i is not None and j is not None)
    return list(((Counter(expected) & Counter(observed)) - in_lcs).elements())


# --- nodes -----------------------------------------------------------------

def aligned_node_pairs(
    expected: list[NormalizedNode], observed: list[NormalizedNode]
) -> list[tuple[NormalizedNode | None, NormalizedNode | None]]:
    """The corresponding-node alignment (METRICS.md §3/§4): the same exact,
    position-aware sequential (LCS) alignment NTS uses. Unaligned nodes come
    back as (node, None) or (None, node)."""
    return [
        (expected[i] if i is not None else None, observed[j] if j is not None else None)
        for i, j in lcs.align([n.node_id for n in expected], [n.node_id for n in observed])
    ]


# --- state -----------------------------------------------------------------

def is_transition(change: StateChange) -> bool:
    # METRICS.md §3: unchanged state variables are not represented as transitions.
    return not values_match(change.old, change.new)


def node_transitions(node: NormalizedNode) -> list[StateChange]:
    return [c for c in node.state_changes if is_transition(c)]


def exposes_nodes(*sides: list[NormalizedTurn]) -> bool:
    return any(t.nodes for turns in sides for t in turns)


def trajectory_transitions(turns: list[NormalizedTurn], node_level: bool) -> list[StateChange]:
    """State is evaluated from node-level state_changes[] (Frozen Rule 6).
    Only when neither compared trajectory exposes any nodes (`node_level`
    False) are the turn-level state changes used instead."""
    if node_level:
        return [c for t in turns for n in t.nodes for c in node_transitions(n)]
    return [c for t in turns for c in t.state_changes if is_transition(c)]


def turn_transitions(turn: NormalizedTurn) -> list[StateChange]:
    """One turn's transitions under the same rule (for evidence text and the
    scorecard): node-level when the turn has nodes, else turn-level."""
    return trajectory_transitions([turn], bool(turn.nodes))


def turn_state_changes(turn: NormalizedTurn) -> list[StateChange]:
    """Every recorded state change of a turn, once, including no-op ones
    (for readers such as policy rules and LLM context; metrics use
    turn_transitions)."""
    node_level = [c for n in turn.nodes for c in n.state_changes]
    return node_level + _unattributed(turn.state_changes, node_level)


def transition_key(change: StateChange) -> tuple[str, str, str]:
    """A transition (key, old, new) as a hashable token, so the transition
    sequence compared by Transition Order Similarity is the sequence of full
    transitions (METRICS.md §3)."""
    return (change.key, value_key(change.old), value_key(change.new))


# --- tools -----------------------------------------------------------------

def unattributed_tool_calls(turn: NormalizedTurn) -> list[ToolCall]:
    node_level = [tc for n in turn.nodes for tc in n.tool_calls]
    return _unattributed(turn.tool_calls, node_level)


def turn_tool_calls(turn: NormalizedTurn) -> list[ToolCall]:
    return [tc for n in turn.nodes for tc in n.tool_calls] + unattributed_tool_calls(turn)


def tools_node_associated(turns: list[NormalizedTurn]) -> bool:
    """Node association is supported by the trace when its tool calls are
    attached to nodes and none are left unassociated."""
    has_node_calls = any(n.tool_calls for t in turns for n in t.nodes)
    has_loose_calls = any(unattributed_tool_calls(t) for t in turns)
    return has_node_calls and not has_loose_calls


def turns_by_id(turns: list[NormalizedTurn]) -> dict[int, NormalizedTurn]:
    return {t.turn_id: t for t in turns}
