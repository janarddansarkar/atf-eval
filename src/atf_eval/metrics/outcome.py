"""Outcome Similarity (OS) — METRICS.md §6. Conversation-level only.

    Outcome Identity   = 1 if identity matches, else 0
    Attribute Accuracy = Correct Expected Attributes / Expected Attributes
    Outcome Completion = Completed Required Outcome / Required Outcome
    OS = 0.50 × Identity + 0.30 × Attribute Accuracy + 0.20 × Completion

Only attributes defined in the Golden dataset are evaluated. The required
outcome is the Golden outcome's `required_conditions` -- the names of the
outcome attributes that must be achieved; a condition is completed when the
observed outcome carries that attribute with the Golden value. Completion is
N/A when the Golden declares no required conditions.

Missing outcome information means OS is N/A: when the Golden has no outcome,
or the observed trace does not expose outcome evidence (availability
`unavailable`). When outcome evidence is available but the agent produced no
outcome, every component is 0 (METRICS.md §21: evidence exists + behaviour
wrong -> failure).

The expected/observed outcome for a conversation is the last non-null
`Outcome` across its turns.
"""
from __future__ import annotations

from atf_eval.aggregate import subweights_for, weighted_composite
from atf_eval.matching import values_match
from atf_eval.metric_result import MetricResult, make_result
from atf_eval.normalized import NormalizedTurn, Outcome


def last_outcome(turns: list[NormalizedTurn]) -> Outcome | None:
    for turn in reversed(turns):
        if turn.outcome is not None:
            return turn.outcome
    return None


def outcome_identity_accuracy(expected: Outcome | None, observed: Outcome | None) -> float | None:
    if expected is None or expected.id is None:
        return None
    if observed is None:
        return 0.0
    return 1.0 if observed.id == expected.id else 0.0


def _attribute_correct(key: str, expected: Outcome, observed: Outcome, tolerance: float) -> bool:
    return (
        key in expected.attributes
        and key in observed.attributes
        and values_match(expected.attributes[key], observed.attributes[key], tolerance)
    )


def outcome_attribute_accuracy(
    expected: Outcome | None, observed: Outcome | None, tolerance: float = 0.0
) -> float | None:
    if expected is None or not expected.attributes:
        return None
    if observed is None:
        return 0.0
    matched = sum(1 for k in expected.attributes if _attribute_correct(k, expected, observed, tolerance))
    return matched / len(expected.attributes)


def outcome_completion(
    expected: Outcome | None, observed: Outcome | None, tolerance: float = 0.0
) -> float | None:
    if expected is None or not expected.required_conditions:
        return None
    if observed is None:
        return 0.0
    satisfied = sum(
        1 for key in expected.required_conditions if _attribute_correct(key, expected, observed, tolerance)
    )
    return satisfied / len(expected.required_conditions)


def os_conversation(
    expected_turns: list[NormalizedTurn],
    observed_turns: list[NormalizedTurn],
    tolerance: float = 0.0,
    subweights: dict | None = None,
    available: bool = True,
) -> float | None:
    if not available:
        return None
    expected_outcome = last_outcome(expected_turns)
    observed_outcome = last_outcome(observed_turns)
    components = {
        "identity": outcome_identity_accuracy(expected_outcome, observed_outcome),
        "attribute": outcome_attribute_accuracy(expected_outcome, observed_outcome, tolerance),
        "completion": outcome_completion(expected_outcome, observed_outcome, tolerance),
    }
    return weighted_composite(components, subweights_for("os", subweights))


def outcome_diagnostics(expected_turns, observed_turns, tolerance: float = 0.0) -> dict:
    expected = last_outcome(expected_turns)
    observed = last_outcome(observed_turns)
    diag: dict = {
        "expected_id": expected.id if expected else None,
        "observed_id": observed.id if observed else None,
        "incorrect_attributes": [],
        "unmet_conditions": [],
    }
    if expected is None:
        return diag
    observed_attrs = observed.attributes if observed else {}
    diag["incorrect_attributes"] = [
        {"key": k, "expected": v, "observed": observed_attrs.get(k)}
        for k, v in expected.attributes.items()
        if observed is None or not _attribute_correct(k, expected, observed, tolerance)
    ]
    diag["unmet_conditions"] = [
        k for k in expected.required_conditions
        if observed is None or not _attribute_correct(k, expected, observed, tolerance)
    ]
    return diag


def os_conversation_result(
    expected_turns: list[NormalizedTurn],
    observed_turns: list[NormalizedTurn],
    tolerance: float = 0.0,
    availability_status: str = "available",
    subweights: dict | None = None,
) -> MetricResult:
    score = os_conversation(
        expected_turns, observed_turns, tolerance, subweights, available=availability_status != "unavailable"
    )
    return make_result(
        "os", score, availability_status, diagnostics=outcome_diagnostics(expected_turns, observed_turns, tolerance)
    )
