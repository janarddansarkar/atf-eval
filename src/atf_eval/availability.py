"""Per-conversation evidence `availability` for the trace's five evidence dimensions
(nodes/state/tools/routing/outcome -> available | unavailable | not_applicable).

The canonical normalized trajectory declares `availability` (a required
field of agent-eval's schema), and METRICS.md makes "the trace does not
expose enough information to evaluate a metric" an N/A, not a failure (§11
"Insufficient evidence", §21). Availability is therefore taken from what the
trace/Adapter declares, not inferred from the absence of observed events --
inferring it would turn an agent that skips every expected state update into
an N/A. A dimension declared `unavailable` makes its metric N/A; an
undeclared dimension defaults to `available`.

`routing` remains a declarable dimension of the trace schema, but no ATF
metric depends on it (RS has been removed from ATF).
"""
from __future__ import annotations

DIMENSIONS = ("nodes", "state", "tools", "routing", "outcome")
_STATUSES = ("available", "unavailable", "not_applicable")

# metric id -> the evidence dimension it depends on
METRIC_DIMENSION = {"nts": "nodes", "sts": "state", "tis": "tools", "os": "outcome"}


def resolve_availability(declared: dict[str, str] | None = None) -> dict[str, str]:
    availability = {dim: "available" for dim in DIMENSIONS}
    for dim, status in (declared or {}).items():
        if dim not in availability:
            raise ValueError(f"unknown availability dimension {dim!r}; expected one of {DIMENSIONS}")
        if status not in _STATUSES:
            raise ValueError(f"availability[{dim!r}] must be one of {_STATUSES}, got {status!r}")
        availability[dim] = status
    return availability
