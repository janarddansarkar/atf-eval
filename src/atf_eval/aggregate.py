"""Shared N/A-aware weighting rules, reused at every level the spec calls
"Both" or "Overall" (turn composites, turn->conversation rollups, and the
top-level ATF score) rather than duplicated per metric group. Subcomponent
composites are weighted arithmetic means; ATF itself is a weighted geometric
mean of NTS, STS, TIS and OS (see atf_score).

The subcomponent weights are the frozen METRICS.md §2-§6 formulas; the ATF
component weights default to DEFAULT_WEIGHTS (the single shared default) and
may be overridden per run -- `--weights` / `--weights-file` on the CLI,
`options.weights` in the API -- always through resolve_weights().
"""
from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from pathlib import Path

# ATF component weights. RS was removed from ATF; its former weight (0.10) is
# redistributed proportionally over the remaining components, i.e. each of the
# original weights NTS .30 / STS .30 / TIS .15 / OS .15 is divided by their sum
# (0.90), preserving their 2:2:1:1 relative importance.
DEFAULT_WEIGHTS = {"nts": 0.30 / 0.90, "sts": 0.30 / 0.90, "tis": 0.15 / 0.90, "os": 0.15 / 0.90}

# Floor applied to a component score before taking its logarithm in the ATF
# geometric mean, so a component score of 0 does not produce log(0).
ATF_EPSILON = 1e-6

DEFAULT_SUBWEIGHTS: dict[str, dict[str, float]] = {
    "nts": {"coverage": 0.40, "precision": 0.30, "order": 0.30},
    "sts": {"transition_accuracy": 0.70, "order": 0.30},
    "tis": {"coverage": 0.25, "precision": 0.20, "identity": 0.25, "input": 0.20, "order": 0.10},
    "os": {"identity": 0.50, "attribute": 0.30, "completion": 0.20},
}


def validate_weights(weights, expected_keys: set[str], label: str) -> None:
    """Weights must be a mapping with exactly the expected keys, each value a
    finite, non-negative number (booleans and strings are rejected), summing
    to 1 (within 1e-6). Raises ValueError with a message naming the problem."""
    if not isinstance(weights, Mapping):
        raise ValueError(f"{label} weights must be an object mapping component to weight, got {type(weights).__name__}")
    keys = set(weights)
    if keys != expected_keys:
        problems = []
        if missing := sorted(expected_keys - keys):
            problems.append(f"missing {missing}")
        if unexpected := sorted(str(k) for k in keys - expected_keys):
            problems.append(f"unexpected {unexpected}")
        raise ValueError(
            f"{label} weights must have exactly the keys {sorted(expected_keys)} ({'; '.join(problems)})"
        )
    for key in sorted(expected_keys):
        w = weights[key]
        if isinstance(w, bool) or not isinstance(w, (int, float)):
            raise ValueError(f"{label} weight {key!r} must be a number, got {w!r}")
        if not math.isfinite(w):
            raise ValueError(f"{label} weight {key!r} must be finite, got {w!r}")
        if w < 0:
            raise ValueError(f"{label} weight {key!r} must be non-negative, got {w!r}")
    total = sum(weights.values())
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ValueError(f"{label} weights must sum to 1, got {total:g}")


def resolve_weights(config: Mapping | None) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """(ATF weights, subcomponent weights) -- the one validation path shared by
    the CLI, the API and the runner. `config=None` means "use DEFAULT_WEIGHTS";
    anything else must pass validate_weights() (all four components, finite,
    non-negative, summing to 1). The subcomponent weights are always the frozen
    METRICS.md formulas."""
    if config is None:
        weights = dict(DEFAULT_WEIGHTS)
    else:
        validate_weights(config, set(DEFAULT_WEIGHTS), "ATF")
        weights = {key: float(config[key]) for key in DEFAULT_WEIGHTS}
    return weights, copy.deepcopy(DEFAULT_SUBWEIGHTS)


def parse_weights_spec(spec: str) -> dict[str, float]:
    """Parse the CLI form "nts=0.4,sts=0.3,tis=0.2,os=0.1" into a dict (keys
    case-insensitive, whitespace ignored). Only the syntax is checked here;
    pass the result to resolve_weights() for the validation rules."""
    weights: dict[str, float] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        key, sep, value = part.partition("=")
        key, value = key.strip().lower(), value.strip()
        if not sep or not key or not value:
            raise ValueError(f"invalid --weights entry {part!r}; expected component=number, e.g. nts=0.4")
        if key in weights:
            raise ValueError(f"--weights lists {key!r} more than once")
        try:
            weights[key] = float(value)
        except ValueError:
            raise ValueError(f"ATF weight {key!r} must be a number, got {value!r}") from None
    if not weights:
        raise ValueError('--weights is empty; expected e.g. "nts=0.4,sts=0.3,tis=0.2,os=0.1"')
    return weights


def load_weights_file(path: str | Path) -> dict:
    """Read a --weights-file (a JSON object). Only I/O and JSON syntax are
    checked here; pass the result to resolve_weights() for the rules."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as e:
        raise ValueError(f"cannot read --weights-file {str(path)!r}: {e.strerror or e}") from None
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"--weights-file {str(path)!r} is not valid JSON: {e}") from None


def weights_from_cli(spec: str | None, path: str | Path | None) -> dict[str, float] | None:
    """The raw CLI weights override: from --weights, from --weights-file, or
    None (use the defaults) when neither is given. Both at once is an error."""
    if spec is not None and path is not None:
        raise ValueError("pass either --weights or --weights-file, not both")
    if spec is not None:
        return parse_weights_spec(spec)
    if path is not None:
        return load_weights_file(path)
    return None


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


def atf_score(
    group_scores: dict[str, float | None], weights: dict[str, float]
) -> tuple[float | None, float]:
    """ATF as a weighted geometric mean over the applicable components
    (NTS, STS, TIS, OS):

        ATF = exp( Σᵢ wᵢ · log(max(Sᵢ, ε)) / Σᵢ wᵢ )

    N/A (None) components are excluded from both sums, so the remaining
    weights are renormalized rather than the component counting as 0. A score
    of 0 is floored at ε (ATF_EPSILON) before the logarithm. Returns None when
    no component is applicable (or the applicable weight is 0). The metric
    coverage -- the applicable share of the total weight -- is reported
    alongside it."""
    total_weight = sum(weights.values())
    applicable = {key: score for key, score in group_scores.items() if score is not None}
    applicable_weight = sum(weights[key] for key in applicable)
    metric_coverage = applicable_weight / total_weight if total_weight else 0.0
    if applicable_weight == 0.0:
        return None, metric_coverage
    log_sum = sum(weights[key] * math.log(max(score, ATF_EPSILON)) for key, score in applicable.items())
    atf = math.exp(log_sum / applicable_weight)
    return atf, metric_coverage
