"""Shared N/A-aware weighting rules, reused at every level the spec calls
"Both" or "Overall" (turn composites, turn->conversation rollups, and the
top-level ATF formula in METRICS.md §7) rather than duplicated per metric
group.

The subcomponent weights are the frozen METRICS.md §2-§6 formulas; the ATF
component weights default to METRICS.md §7 and may be overridden with a
weights file (the pre-existing `--weights-file` option).
"""
from __future__ import annotations

import copy
import math

DEFAULT_WEIGHTS = {"nts": 0.30, "sts": 0.30, "tis": 0.15, "rs": 0.10, "os": 0.15}

DEFAULT_SUBWEIGHTS: dict[str, dict[str, float]] = {
    "nts": {"coverage": 0.40, "precision": 0.30, "order": 0.30},
    "sts": {"transition_accuracy": 0.70, "order": 0.30},
    "tis": {"coverage": 0.25, "precision": 0.20, "identity": 0.25, "input": 0.20, "order": 0.10},
    "rs": {"semantic": 0.70, "order": 0.30},
    "os": {"identity": 0.50, "attribute": 0.30, "completion": 0.20},
}


def validate_weights(weights: dict[str, float], expected_keys: set[str], label: str) -> None:
    """Weights must cover exactly the expected keys, be non-negative, and sum
    to one, as the METRICS.md §7 weights do."""
    keys = set(weights)
    if keys != expected_keys:
        raise ValueError(
            f"{label} weights must have exactly the keys {sorted(expected_keys)}, got {sorted(keys)}"
        )
    if any(w < 0 for w in weights.values()):
        raise ValueError(f"{label} weights must be non-negative: {weights}")
    total = sum(weights.values())
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ValueError(f"{label} weights must sum to 1, got {total}")


def resolve_weights(config: dict | None) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """(ATF weights, subcomponent weights). `config` optionally overrides the
    ATF component weights ({"nts": .3, "sts": .3, ...}); the subcomponent
    weights are always the frozen METRICS.md formulas."""
    weights = dict(config) if config else dict(DEFAULT_WEIGHTS)
    validate_weights(weights, set(DEFAULT_WEIGHTS), "ATF")
    return weights, copy.deepcopy(DEFAULT_SUBWEIGHTS)


def subweights_for(component: str, subweights: dict[str, dict[str, float]] | None) -> dict[str, float]:
    return (subweights or DEFAULT_SUBWEIGHTS).get(component, DEFAULT_SUBWEIGHTS[component])


def weighted_composite(components: dict[str, float | None], weights: dict[str, float]) -> float | None:
    """Σ(weight*score) / Σ(weight) over components whose score is not None.

    None components are excluded from both numerator and denominator (the
    remaining weights are implicitly renormalized) rather than treated as 0.
    Returns None if every component is None.
    """
    total_weight = 0.0
    total_score = 0.0
    for key, score in components.items():
        if score is None:
            continue
        weight = weights[key]
        total_weight += weight
        total_score += weight * score

    if total_weight == 0.0:
        return None
    return total_score / total_weight


def aggregate_turns(turn_scores: list[float | None], turn_weight: float = 1.0) -> float | None:
    """Σ(score*turn_weight) / Σ(turn_weight) over non-None turn scores (spec §7.5)."""
    applicable = [s for s in turn_scores if s is not None]
    if not applicable:
        return None
    return sum(s * turn_weight for s in applicable) / (len(applicable) * turn_weight)


def atf_score(
    group_scores: dict[str, float | None], weights: dict[str, float]
) -> tuple[float | None, float]:
    """METRICS.md §7: ATF weighted mean over applicable components (N/A
    weights removed and the rest renormalized), plus the metric coverage --
    the applicable share of the total weight -- reported alongside it."""
    total_weight = sum(weights.values())
    applicable_weight = sum(
        weights[key] for key, score in group_scores.items() if score is not None
    )
    metric_coverage = applicable_weight / total_weight if total_weight else 0.0
    atf = weighted_composite(group_scores, weights)
    return atf, metric_coverage
