"""Node Traversal Similarity (NTS) — METRICS.md §2.

Alignment is exact node matching with position-aware sequential (LCS)
alignment, which identifies missing, extra and reordered nodes without
cascading later nodes into failures. "Matched" nodes in Coverage/Precision
are the nodes this alignment pairs:

    Coverage  = Matched Expected Nodes / Expected Nodes
    Precision = Matched Observed Nodes / Observed Nodes
    Recall    = Coverage (diagnostic only)
    Order     = LCS / max(Expected Node Count, Observed Node Count)
    NTS       = 0.40 × Coverage + 0.30 × Precision + 0.30 × Order
"""
from __future__ import annotations

from collections import Counter

from atf_eval import lcs
from atf_eval.aggregate import subweights_for, weighted_composite
from atf_eval.metric_result import MetricResult, make_result
from atf_eval.normalized import NormalizedTurn


def _matched_count(expected: list[str], observed: list[str]) -> int:
    pairs = lcs.align(expected, observed)
    return sum(1 for i, j in pairs if i is not None and j is not None)


def node_coverage(expected: list[str], observed: list[str]) -> float | None:
    if not expected:
        return None
    return _matched_count(expected, observed) / len(expected)


def node_precision(expected: list[str], observed: list[str]) -> float | None:
    if not observed:
        return None
    return _matched_count(expected, observed) / len(observed)


def node_recall(expected: list[str], observed: list[str]) -> float | None:
    """Diagnostic only -- identical to coverage under the frozen definitions."""
    return node_coverage(expected, observed)


def node_order_similarity(expected: list[str], observed: list[str]) -> float | None:
    return lcs.order_similarity(expected, observed)


def node_diagnostics(expected: list[str], observed: list[str]) -> dict:
    """Matched / missing / extra from the alignment (Frozen Rule 5). A node
    that is both missing at its expected position and extra at another
    position -- the same id on both sides of the alignment -- is reported as
    reordered."""
    pairs = lcs.align(expected, observed)
    missing = [expected[i] for i, j in pairs if i is not None and j is None]
    extra = [observed[j] for i, j in pairs if i is None and j is not None]
    reordered = Counter(missing) & Counter(extra)
    return {
        "matched": [expected[i] for i, j in pairs if i is not None and j is not None],
        "missing": list((Counter(missing) - reordered).elements()),
        "extra": list((Counter(extra) - reordered).elements()),
        "reordered": list(reordered.elements()),
    }


def _node_ids(turns: list[NormalizedTurn]) -> list[str]:
    return [n.node_id for t in turns for n in t.nodes]


def _nts(expected_ids: list[str], observed_ids: list[str], weights: dict[str, float]) -> float | None:
    components = {
        "coverage": node_coverage(expected_ids, observed_ids),
        "precision": node_precision(expected_ids, observed_ids),
        "order": node_order_similarity(expected_ids, observed_ids),
    }
    return weighted_composite(components, weights)


def nts_turn(
    expected: NormalizedTurn,
    observed: NormalizedTurn,
    subweights: dict | None = None,
) -> float | None:
    """Turn-level diagnostic score (METRICS.md §2 "Levels")."""
    return _nts(_node_ids([expected]), _node_ids([observed]), subweights_for("nts", subweights))


def nts_conversation(
    expected_turns: list[NormalizedTurn],
    observed_turns: list[NormalizedTurn],
    subweights: dict | None = None,
    available: bool = True,
) -> float | None:
    """Overall NTS from the complete concatenated trajectory, not a turn-score
    average (Frozen Rule 10). N/A when node evidence is unavailable."""
    if not available:
        return None
    return _nts(_node_ids(expected_turns), _node_ids(observed_turns), subweights_for("nts", subweights))


def nts_conversation_result(
    expected_turns: list[NormalizedTurn],
    observed_turns: list[NormalizedTurn],
    availability_status: str = "available",
    subweights: dict | None = None,
) -> MetricResult:
    score = nts_conversation(
        expected_turns, observed_turns, subweights, available=availability_status != "unavailable"
    )
    return make_result(
        "nts", score, availability_status,
        diagnostics=node_diagnostics(_node_ids(expected_turns), _node_ids(observed_turns)),
    )
