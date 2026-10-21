"""Value comparison shared by every metric (STS transition values, TIS
arguments, OS attributes and completion conditions), so the same pair of
values never matches under one metric term and mismatches under another.

Rules: numbers compare by value (3000 == 3000.0) and may use a numeric
tolerance; booleans are never equal to numbers (True != 1); strings, None,
lists and dicts compare structurally under these same rules.
"""
from __future__ import annotations

import json
from numbers import Real


def _is_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def canonical(value: object) -> object:
    """Normalize a JSON-ish value so structural equality follows the rules
    above: numbers become floats, booleans stay booleans, containers recurse."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if _is_number(value):
        return float(value)
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    return str(value)


def value_key(value: object) -> str:
    """Hashable, order-stable token; two values have the same key iff they
    are equal under the rules above (with zero tolerance)."""
    return json.dumps(canonical(value), sort_keys=True)


def values_match(expected_val: object, observed_val: object, tolerance: float = 0.0) -> bool:
    if _is_number(expected_val) and _is_number(observed_val):
        return abs(float(expected_val) - float(observed_val)) <= tolerance
    return value_key(expected_val) == value_key(observed_val)
