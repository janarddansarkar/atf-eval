"""Common result shape + scoring-type handling for the LLM/multimodal
evaluation metrics (METRICS.md Part B, Groups 1-5).

METRICS.md §19 freezes *native* per-metric scoring (1-5, Pass/Fail/N/A,
-2..+2, counts, categorical+confidence) and explicitly defers cross-metric
normalization. We keep the native score as the authoritative value and only
attach an optional `normalized` 0-1 projection as a display aid -- never the
other way round.

Every result exposes the METRICS.md §23 fields:
    metric_id, score, status, applicability, evidence/reason, diagnostics
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ScoringType(str, Enum):
    ORDINAL_1_5 = "ordinal_1_5"          # 1..5 rubric
    ORDINAL_1_5_NA = "ordinal_1_5_na"    # 1..5 rubric, N/A allowed as a value
    PASS_FAIL = "pass_fail"              # PASS / FAIL / N/A
    SENTIMENT = "sentiment"              # -2..+2
    COUNT = "count"                      # integer >= 0
    COUNT_SEVERITY = "count_severity"    # integer + NONE/LOW/MEDIUM/HIGH
    CATEGORICAL_CONFIDENCE = "categorical_confidence"  # label + 0..1 confidence


class Level(str, Enum):
    TURN = "turn"
    OVERALL = "overall"
    BOTH = "both"


# status values, mirrors the deterministic side + METRICS.md §8/§21
AVAILABLE = "available"
UNAVAILABLE = "unavailable"
NOT_APPLICABLE = "not_applicable"


def normalize_score(scoring_type: ScoringType, native: Any) -> float | None:
    """Optional 0-1 projection for dashboards only. Returns None for scoring
    types that have no meaningful 0-1 mapping (counts, categoricals) or when
    the value is N/A."""
    if native is None:
        return None
    if scoring_type in (ScoringType.ORDINAL_1_5, ScoringType.ORDINAL_1_5_NA):
        if not isinstance(native, (int, float)):
            return None
        return (float(native) - 1.0) / 4.0
    if scoring_type is ScoringType.PASS_FAIL:
        s = str(native).upper()
        if s == "PASS":
            return 1.0
        if s == "FAIL":
            return 0.0
        return None
    if scoring_type is ScoringType.SENTIMENT:
        if not isinstance(native, (int, float)):
            return None
        return (float(native) + 2.0) / 4.0
    if scoring_type is ScoringType.COUNT_SEVERITY:
        # severity is what carries the judgement; map it, not the raw count
        sev = None
        if isinstance(native, dict):
            sev = str(native.get("severity", "")).upper()
        return {"NONE": 1.0, "LOW": 0.75, "MEDIUM": 0.4, "HIGH": 0.0}.get(sev)
    return None


@dataclass
class LLMMetricResult:
    metric_id: str
    metric_name: str
    group: int
    level: str                 # "turn" | "overall"
    scoring_type: str
    score: Any                 # native score (int / "PASS" / dict / ...), or None for N/A
    status: str                # AVAILABLE | UNAVAILABLE | NOT_APPLICABLE
    applicability: bool
    normalized: float | None   # optional 0-1 projection, display only
    reason: str                # LLM rationale / evidence (auditable, not authoritative)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    turn_id: int | None = None

    @classmethod
    def na(
        cls,
        spec: "MetricSpec",
        level: str,
        reason: str,
        *,
        turn_id: int | None = None,
        status: str = NOT_APPLICABLE,
    ) -> "LLMMetricResult":
        return cls(
            metric_id=spec.metric_id,
            metric_name=spec.name,
            group=spec.group,
            level=level,
            scoring_type=spec.scoring_type.value,
            score=None,
            status=status,
            applicability=False,
            normalized=None,
            reason=reason,
            turn_id=turn_id,
        )

    @classmethod
    def scored(
        cls,
        spec: "MetricSpec",
        level: str,
        native: Any,
        reason: str,
        *,
        turn_id: int | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> "LLMMetricResult":
        return cls(
            metric_id=spec.metric_id,
            metric_name=spec.name,
            group=spec.group,
            level=level,
            scoring_type=spec.scoring_type.value,
            score=native,
            status=AVAILABLE,
            applicability=True,
            normalized=normalize_score(spec.scoring_type, native),
            reason=reason,
            diagnostics=diagnostics or {},
            turn_id=turn_id,
        )


@dataclass(frozen=True)
class MetricSpec:
    """Frozen description of one LLM metric, straight from METRICS.md's
    per-metric table + rubric. `overall_from_turns` says how turn results
    roll up (METRICS.md §20: defined per metric, never a blind average)."""
    metric_id: str
    name: str
    group: int
    level: Level
    scoring_type: ScoringType
    definition: str
    rubric: dict[str, str]          # score-value -> meaning
    na_rule: str
    overlap_boundary: str
    required_evidence: tuple[str, ...] = ("transcript",)
    # how a list of per-turn native scores becomes the overall native score
    overall_from_turns: str = "mean"   # mean | min | worst_pass_fail | sum | max | none
