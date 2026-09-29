"""State Transition Similarity (STS) — METRICS.md §3.

A state transition is (key, old, new). State is evaluated from node-level
`state_changes[]` (Frozen Rule 6). Alignment uses the corresponding node
alignment -- the exact, position-aware sequential (LCS) node alignment NTS
uses -- and state changes are compared within each aligned node pair, paired
by key in occurrence order. Only when neither side exposes any nodes are
transitions paired by key across the trajectory. Unchanged state variables
are not represented as transitions.

    Transition Accuracy = Matched Complete Transitions
                          / max(Expected Transitions, Observed Transitions)
    Transition Order    = LCS(Expected Transition Sequence, Observed Transition Sequence)
                          / max(Expected Transition Count, Observed Transition Count)
    STS = 0.70 × Transition Accuracy + 0.30 × Transition Order

A complete match agrees on key, old value and new value; the transition
sequence is the sequence of full (key, old, new) transitions. Key/old/new-
value accuracy are diagnostics. If state information is unavailable, STS is
N/A (as it is when neither side has any transition).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from atf_eval import lcs
from atf_eval.aggregate import subweights_for, weighted_composite
from atf_eval.matching import values_match
from atf_eval.metric_result import MetricResult, make_result
from atf_eval.metrics.evidence import (
    aligned_node_pairs,
    exposes_nodes,
    order_deviations,
    node_transitions,
    trajectory_transitions,
    transition_key,
)
from atf_eval.normalized import NormalizedTurn, StateChange


@dataclass
class TransitionAlignment:
    pairs: list[tuple[StateChange, StateChange | None]] = field(default_factory=list)
    extra: list[StateChange] = field(default_factory=list)
    expected_count: int = 0
    observed_count: int = 0


def _pair_by_key(
    expected: list[StateChange], observed: list[StateChange]
) -> tuple[list[tuple[StateChange, StateChange | None]], list[StateChange]]:
    observed_by_key: dict[str, list[StateChange]] = defaultdict(list)
    for oc in observed:
        observed_by_key[oc.key].append(oc)
    used: dict[str, int] = defaultdict(int)

    pairs: list[tuple[StateChange, StateChange | None]] = []
    for ec in expected:
        candidates = observed_by_key[ec.key]
        idx = used[ec.key]
        if idx < len(candidates):
            pairs.append((ec, candidates[idx]))
            used[ec.key] += 1
        else:
            pairs.append((ec, None))
    extra = [oc for key, ocs in observed_by_key.items() for oc in ocs[used[key]:]]
    return pairs, extra


def align_transitions(
    expected_turns: list[NormalizedTurn], observed_turns: list[NormalizedTurn]
) -> TransitionAlignment:
    node_level = exposes_nodes(expected_turns, observed_turns)
    expected = trajectory_transitions(expected_turns, node_level)
    observed = trajectory_transitions(observed_turns, node_level)
    result = TransitionAlignment(expected_count=len(expected), observed_count=len(observed))

    if not node_level:
        result.pairs, result.extra = _pair_by_key(expected, observed)
        return result

    exp_nodes = [n for t in expected_turns for n in t.nodes]
    obs_nodes = [n for t in observed_turns for n in t.nodes]

    for e_node, o_node in aligned_node_pairs(exp_nodes, obs_nodes):
        pairs, extra = _pair_by_key(
            node_transitions(e_node) if e_node is not None else [],
            node_transitions(o_node) if o_node is not None else [],
        )
        result.pairs += pairs
        result.extra += extra
    return result


def _is_complete(e: StateChange, o: StateChange | None) -> bool:
    return o is not None and values_match(e.old, o.old) and values_match(e.new, o.new)


def _fraction_of_expected(alignment: TransitionAlignment, predicate) -> float | None:
    if not alignment.pairs:
        return None
    return sum(1 for e, o in alignment.pairs if predicate(e, o)) / len(alignment.pairs)


def state_key_accuracy(expected_turns, observed_turns) -> float | None:
    """Diagnostic: fraction of expected transitions whose key was found."""
    return _fraction_of_expected(align_transitions(expected_turns, observed_turns), lambda e, o: o is not None)


def old_state_accuracy(expected_turns, observed_turns) -> float | None:
    """Diagnostic: fraction of expected transitions whose old value matched."""
    return _fraction_of_expected(
        align_transitions(expected_turns, observed_turns), lambda e, o: o is not None and values_match(e.old, o.old)
    )


def new_state_accuracy(expected_turns, observed_turns) -> float | None:
    """Diagnostic: fraction of expected transitions whose new value matched."""
    return _fraction_of_expected(
        align_transitions(expected_turns, observed_turns), lambda e, o: o is not None and values_match(e.new, o.new)
    )


def transition_accuracy(expected_turns, observed_turns) -> float | None:
    alignment = align_transitions(expected_turns, observed_turns)
    denom = max(alignment.expected_count, alignment.observed_count)
    if denom == 0:
        return None
    return sum(1 for e, o in alignment.pairs if _is_complete(e, o)) / denom


def transition_order_similarity(expected_turns, observed_turns) -> float | None:
    node_level = exposes_nodes(expected_turns, observed_turns)
    exp_seq = [transition_key(c) for c in trajectory_transitions(expected_turns, node_level)]
    obs_seq = [transition_key(c) for c in trajectory_transitions(observed_turns, node_level)]
    return lcs.order_similarity(exp_seq, obs_seq)


def _sts(expected_turns, observed_turns, subweights) -> float | None:
    components = {
        "transition_accuracy": transition_accuracy(expected_turns, observed_turns),
        "order": transition_order_similarity(expected_turns, observed_turns),
    }
    return weighted_composite(components, subweights_for("sts", subweights))


def sts_turn(
    expected: NormalizedTurn, observed: NormalizedTurn, subweights: dict | None = None
) -> float | None:
    """Turn-level diagnostic score (METRICS.md §3 "Levels")."""
    return _sts([expected], [observed], subweights)


def sts_conversation(
    expected_turns: list[NormalizedTurn],
    observed_turns: list[NormalizedTurn],
    subweights: dict | None = None,
    available: bool = True,
) -> float | None:
    """Overall STS from the complete trajectory, not a turn-score average
    (Frozen Rule 10)."""
    if not available:
        return None
    return _sts(expected_turns, observed_turns, subweights)


def _fmt_change(c: StateChange) -> dict:
    return {"key": c.key, "old": c.old, "new": c.new}


def state_diagnostics(expected_turns, observed_turns) -> dict:
    alignment = align_transitions(expected_turns, observed_turns)
    node_level = exposes_nodes(expected_turns, observed_turns)
    exp_changes = trajectory_transitions(expected_turns, node_level)
    obs_changes = trajectory_transitions(observed_turns, node_level)
    by_key = {transition_key(c): c for c in exp_changes}
    return {
        "matched": [_fmt_change(e) for e, o in alignment.pairs if _is_complete(e, o)],
        "order_deviations": [
            _fmt_change(by_key[k])
            for k in order_deviations([transition_key(c) for c in exp_changes], [transition_key(c) for c in obs_changes])
        ],
        "missing": [_fmt_change(e) for e, o in alignment.pairs if o is None],
        "mismatched": [
            {
                "key": e.key,
                "expected_old": e.old,
                "expected_new": e.new,
                "observed_old": o.old,
                "observed_new": o.new,
            }
            for e, o in alignment.pairs
            if o is not None and not _is_complete(e, o)
        ],
        "extra": [_fmt_change(c) for c in alignment.extra],
    }


def sts_conversation_result(
    expected_turns: list[NormalizedTurn],
    observed_turns: list[NormalizedTurn],
    availability_status: str = "available",
    subweights: dict | None = None,
) -> MetricResult:
    score = sts_conversation(
        expected_turns, observed_turns, subweights, available=availability_status != "unavailable"
    )
    return make_result(
        "sts", score, availability_status, diagnostics=state_diagnostics(expected_turns, observed_turns)
    )
