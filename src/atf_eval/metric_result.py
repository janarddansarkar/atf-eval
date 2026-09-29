"""Uniform per-metric result shape (METRICS.md §8): every metric result must
expose metric_id, score, status, applicability, coverage, diagnostics.

This wraps the existing (unchanged) scoring functions in each metrics module
-- it does not replace the bare-float functions runner.py's TurnResult and
diagnostics.py already consume directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetricResult:
    metric_id: str
    score: float | None
    status: str  # "available" | "unavailable" | "not_applicable"
    applicability: bool
    coverage: float | None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def make_result(
    metric_id: str,
    score: float | None,
    availability_status: str,
    diagnostics: dict[str, Any] | None = None,
) -> MetricResult:
    """Build a MetricResult from a computed score plus the conversation's
    already-computed `availability` status for this dimension. `score is
    None` (the existing, correct N/A signal from the metric functions) always
    wins over `availability_status` for `applicability`/`coverage` -- a
    dimension can be `available` at the conversation level yet still be N/A
    for a particular aggregation (e.g. this golden turn had no expected state
    changes), and that must not be masked."""
    applicable = score is not None
    return MetricResult(
        metric_id=metric_id,
        score=score,
        # N/A because the evidence was declared unavailable vs. N/A because
        # there was nothing to compare (e.g. no expected or observed events)
        status=availability_status if applicable or availability_status != "available" else "not_applicable",
        applicability=applicable,
        coverage=1.0 if applicable else 0.0,
        diagnostics=diagnostics or {},
    )
