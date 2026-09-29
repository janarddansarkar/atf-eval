"""Tool Invocation Similarity (TIS) — METRICS.md §4.

Alignment: tools are aligned within corresponding nodes -- the node pairs of
the exact, position-aware sequential (LCS) node alignment -- when node
association is supported by the normalized trace (both trajectories attach
their tool calls to nodes); otherwise tools are compared at turn level
(turns matched by turn_id). Node-tool relationships are never invented.

Within each scope the expected and observed tool sequences are aligned by
minimum edit distance (METRICS.md fixes the scope, not the within-scope
algorithm; ties prefer a match, then a substitution). This yields matched
pairs (same tool), substituted pairs (a different tool in an expected slot),
missing and extra calls.

    Coverage  = Matched Expected Tools / Expected Tools
    Precision = Matched Observed Tools / Observed Tools
    Identity  = Correct Tool Matches / Aligned Tool Pairs      (matched + substituted)
    Input     = Average similarity of matched tool inputs      (same-tool pairs)
    Order     = LCS(Expected Tool Sequence, Observed Tool Sequence)
                / max(Expected Tool Count, Observed Tool Count)
    TIS = 0.25 C + 0.20 P + 0.25 I + 0.20 Input + 0.10 Order

Input similarity compares expected arguments with observed arguments and can
be partial: the fraction of expected arguments present with matching values
(1.0 when none are expected). No tool information means TIS is N/A.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from atf_eval import lcs
from atf_eval.aggregate import subweights_for, weighted_composite
from atf_eval.matching import values_match
from atf_eval.metric_result import MetricResult, make_result
from atf_eval.metrics.evidence import (
    aligned_node_pairs,
    order_deviations,
    tools_node_associated,
    turn_tool_calls,
    turns_by_id,
)
from atf_eval.normalized import NormalizedTurn, ToolCall


@dataclass
class ToolAlignment:
    pairs: list[tuple[ToolCall, ToolCall]] = field(default_factory=list)  # matched + substituted
    missing: list[ToolCall] = field(default_factory=list)
    extra: list[ToolCall] = field(default_factory=list)
    scope: str = "turn"

    @property
    def same_tool_pairs(self) -> list[tuple[ToolCall, ToolCall]]:
        return [(e, o) for e, o in self.pairs if e.tool_id == o.tool_id]


def argument_match(expected_args: dict, observed_args: dict, tolerance: float = 0.0) -> float:
    """sim_arg: fraction of expected arguments present with matching values;
    1.0 when nothing was expected of the call."""
    if not expected_args:
        return 1.0
    matched = sum(
        1
        for k, v in expected_args.items()
        if k in observed_args and values_match(v, observed_args[k], tolerance)
    )
    return matched / len(expected_args)


def align_tool_sequence(
    expected: list[ToolCall], observed: list[ToolCall]
) -> list[tuple[ToolCall | None, ToolCall | None]]:
    """Minimum-edit-distance alignment. Returns (expected, observed) pairs in
    order; (e, None) is a deletion, (None, o) an insertion.

    Ties between equal-cost alignments are broken globally in favour of more
    matches, then more substitutions: the DP minimizes the
    lexicographic cost (edits, -matches, -substitutions), so e.g. (a, b) vs.
    (b, a) aligns b↔b with a deleted and re-inserted, not as two wrong tools."""
    n, m = len(expected), len(observed)
    # cell = (edits, -matches, -substitutions)
    cost: list[list[tuple[int, int, int]]] = [[(0, 0, 0)] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        cost[i][0] = (i, 0, 0)
    for j in range(m + 1):
        cost[0][j] = (j, 0, 0)

    def diag(i: int, j: int) -> tuple[int, int, int]:
        e, neg_m, neg_s = cost[i - 1][j - 1]
        if expected[i - 1].tool_id == observed[j - 1].tool_id:
            return (e, neg_m - 1, neg_s)
        return (e + 1, neg_m, neg_s - 1)

    def indel(prev: tuple[int, int, int]) -> tuple[int, int, int]:
        return (prev[0] + 1, prev[1], prev[2])

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i][j] = min(diag(i, j), indel(cost[i - 1][j]), indel(cost[i][j - 1]))

    out: list[tuple[ToolCall | None, ToolCall | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and cost[i][j] == diag(i, j):
            out.append((expected[i - 1], observed[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and cost[i][j] == indel(cost[i - 1][j]):
            out.append((expected[i - 1], None))
            i -= 1
        else:
            out.append((None, observed[j - 1]))
            j -= 1
    out.reverse()
    return out


def _accumulate(alignment: ToolAlignment, expected: list[ToolCall], observed: list[ToolCall]) -> None:
    for e, o in align_tool_sequence(expected, observed):
        if e is not None and o is not None:
            alignment.pairs.append((e, o))
        elif e is not None:
            alignment.missing.append(e)
        else:
            alignment.extra.append(o)


def align_tools(expected_turns: list[NormalizedTurn], observed_turns: list[NormalizedTurn]) -> ToolAlignment:
    alignment = ToolAlignment()

    if tools_node_associated(expected_turns) and tools_node_associated(observed_turns):
        alignment.scope = "node"
        exp_nodes = [n for t in expected_turns for n in t.nodes]
        obs_nodes = [n for t in observed_turns for n in t.nodes]
        for e_node, o_node in aligned_node_pairs(exp_nodes, obs_nodes):
            _accumulate(
                alignment,
                e_node.tool_calls if e_node is not None else [],
                o_node.tool_calls if o_node is not None else [],
            )
        return alignment

    if len(expected_turns) == 1 and len(observed_turns) == 1:
        observed_by_id = {expected_turns[0].turn_id: observed_turns[0]}
    else:
        observed_by_id = turns_by_id(observed_turns)
    used: set[int] = set()
    for et in expected_turns:
        ot = observed_by_id.get(et.turn_id)
        if ot is not None:
            used.add(id(ot))
        _accumulate(alignment, turn_tool_calls(et), turn_tool_calls(ot) if ot else [])
    for ot in observed_turns:
        if id(ot) not in used:
            alignment.extra += turn_tool_calls(ot)
    return alignment


def _names(turns: list[NormalizedTurn]) -> list[str]:
    return [tc.tool_id for t in turns for tc in turn_tool_calls(t)]


def tool_coverage(expected_turns, observed_turns) -> float | None:
    expected_count = len(_names(expected_turns))
    if not expected_count:
        return None
    return len(align_tools(expected_turns, observed_turns).same_tool_pairs) / expected_count


def tool_precision(expected_turns, observed_turns) -> float | None:
    observed_count = len(_names(observed_turns))
    if not observed_count:
        return None
    return len(align_tools(expected_turns, observed_turns).same_tool_pairs) / observed_count


def tool_identity_accuracy(expected_turns, observed_turns) -> float | None:
    alignment = align_tools(expected_turns, observed_turns)
    if not alignment.pairs:
        return None
    return len(alignment.same_tool_pairs) / len(alignment.pairs)


def tool_input_similarity(expected_turns, observed_turns, tolerance: float = 0.0) -> float | None:
    same = align_tools(expected_turns, observed_turns).same_tool_pairs
    if not same:
        return None
    return sum(argument_match(e.arguments, o.arguments, tolerance) for e, o in same) / len(same)


def tool_order_similarity(expected_turns, observed_turns) -> float | None:
    return lcs.order_similarity(_names(expected_turns), _names(observed_turns))


def _tis(expected_turns, observed_turns, tolerance: float, subweights) -> float | None:
    components = {
        "coverage": tool_coverage(expected_turns, observed_turns),
        "precision": tool_precision(expected_turns, observed_turns),
        "identity": tool_identity_accuracy(expected_turns, observed_turns),
        "input": tool_input_similarity(expected_turns, observed_turns, tolerance),
        "order": tool_order_similarity(expected_turns, observed_turns),
    }
    return weighted_composite(components, subweights_for("tis", subweights))


def tis_turn(
    expected: NormalizedTurn,
    observed: NormalizedTurn,
    tolerance: float = 0.0,
    subweights: dict | None = None,
) -> float | None:
    """Turn-level diagnostic score."""
    return _tis([expected], [observed], tolerance, subweights)


def tis_conversation(
    expected_turns: list[NormalizedTurn],
    observed_turns: list[NormalizedTurn],
    tolerance: float = 0.0,
    subweights: dict | None = None,
    available: bool = True,
) -> float | None:
    """Overall TIS from the complete trajectory, not a turn-score average."""
    if not available:
        return None
    return _tis(expected_turns, observed_turns, tolerance, subweights)


def tool_diagnostics(expected_turns, observed_turns, tolerance: float = 0.0) -> dict:
    alignment = align_tools(expected_turns, observed_turns)
    return {
        "scope": alignment.scope,
        "matched": [e.tool_id for e, _ in alignment.same_tool_pairs],
        "order_deviations": order_deviations(_names(expected_turns), _names(observed_turns)),
        "missing": [tc.tool_id for tc in alignment.missing],
        "extra": [tc.tool_id for tc in alignment.extra],
        "substituted": [
            {"expected": e.tool_id, "observed": o.tool_id}
            for e, o in alignment.pairs
            if e.tool_id != o.tool_id
        ],
        "argument_mismatches": [
            {
                "tool": e.tool_id,
                "expected": e.arguments,
                "observed": o.arguments,
                "similarity": argument_match(e.arguments, o.arguments, tolerance),
            }
            for e, o in alignment.same_tool_pairs
            if argument_match(e.arguments, o.arguments, tolerance) < 1.0
        ],
        # observed arguments the Golden call does not specify (diagnostic only;
        # Input compares the expected arguments)
        "unexpected_arguments": [
            {"tool": e.tool_id, "arguments": {k: v for k, v in o.arguments.items() if k not in e.arguments}}
            for e, o in alignment.same_tool_pairs
            if any(k not in e.arguments for k in o.arguments)
        ],
    }


def tis_conversation_result(
    expected_turns: list[NormalizedTurn],
    observed_turns: list[NormalizedTurn],
    tolerance: float = 0.0,
    availability_status: str = "available",
    subweights: dict | None = None,
) -> MetricResult:
    score = tis_conversation(
        expected_turns, observed_turns, tolerance, subweights, available=availability_status != "unavailable"
    )
    return make_result(
        "tis", score, availability_status, diagnostics=tool_diagnostics(expected_turns, observed_turns, tolerance)
    )
