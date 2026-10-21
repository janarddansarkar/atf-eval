"""The evaluation settings recorded with every result for auditability
(METRICS.md §8, §23): the argument value-matching rule, the outcome-identity
rule, and the weights.
"""
from __future__ import annotations

from atf_eval.aggregate import DEFAULT_SUBWEIGHTS, DEFAULT_WEIGHTS


def evaluation_settings(
    tolerance: float = 0.0,
    weights: dict | None = None,
    subweights: dict | None = None,
) -> dict:
    return {
        "argument_matching": (
            "exact equality; numeric values match within ±" + f"{tolerance:g}"
            if tolerance
            else "exact equality (numeric tolerance 0)"
        ),
        "outcome_identity_evaluator": "exact match of the outcome id (METRICS.md §6)",
        "weights": weights or DEFAULT_WEIGHTS,
        "subweights": subweights or DEFAULT_SUBWEIGHTS,
    }
