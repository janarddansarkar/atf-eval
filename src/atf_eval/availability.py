"""Per-conversation evidence `availability` for the five ATF dimensions
(nodes/state/tools/routing/outcome -> available | unavailable | not_applicable).

The canonical normalized trajectory declares `availability` (a required
field of agent-eval's schema), and METRICS.md makes "the trace does not
expose enough information to evaluate a metric" an N/A, not a failure (§11
"Insufficient evidence", §21). Availability is therefore taken from what the
trace/Adapter declares, not inferred from the absence of observed events --
inferring it would turn an agent that skips every expected state update into
an N/A. A dimension declared `unavailable` makes its metric N/A; an
undeclared dimension defaults to `available`.

Routing is additionally `not_applicable` when no routing judge is configured:
RS is an LLM-based semantic evaluation (METRICS.md §5, Frozen Rule 8).
"""
from __future__ import annotations

DIMENSIONS = ("nodes", "state", "tools", "routing", "outcome")
_STATUSES = ("available", "unavailable", "not_applicable")

# metric id -> the evidence dimension it depends on
METRIC_DIMENSION = {"nts": "nodes", "sts": "state", "tis": "tools", "rs": "routing", "os": "outcome"}


def resolve_availability(
    declared: dict[str, str] | None = None,
    judge_configured: bool = False,
) -> dict[str, str]:
    availability = {dim: "available" for dim in DIMENSIONS}
    for dim, status in (declared or {}).items():
        if dim not in availability:
            raise ValueError(f"unknown availability dimension {dim!r}; expected one of {DIMENSIONS}")
        if status not in _STATUSES:
            raise ValueError(f"availability[{dim!r}] must be one of {_STATUSES}, got {status!r}")
        availability[dim] = status
    if not judge_configured and availability["routing"] == "available":
        availability["routing"] = "not_applicable"
    return availability


def is_scorable(availability: dict[str, str], metric_id: str) -> bool:
    return availability[METRIC_DIMENSION[metric_id]] == "available"
