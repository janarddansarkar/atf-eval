"""Structured-output models the judge returns, one per scoring family.

Each carries `applicable` so the judge can signal METRICS.md's N/A condition
directly (insufficient evidence / not-applicable), plus a short `reason`
string for auditability (METRICS.md §23 -- rationale supports auditing but is
not itself authoritative).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class OrdinalVerdict(BaseModel):
    applicable: bool = Field(description="false when the metric's N/A rule applies")
    score: int | None = Field(default=None, description="integer 1-5 per the rubric, or null when not applicable")
    reason: str


class PassFailVerdict(BaseModel):
    applicable: bool = Field(description="false when there is no applicable rule/evidence to judge")
    verdict: Literal["pass", "fail"] | None = None
    reason: str


class SentimentVerdict(BaseModel):
    applicable: bool
    score: int | None = Field(default=None, description="integer -2..+2, or null when not applicable")
    reason: str


class RepetitionVerdict(BaseModel):
    applicable: bool
    count: int = Field(default=0, description="number of meaningful customer repetitions")
    severity: Literal["none", "low", "medium", "high"] = "none"
    reason: str


class CountVerdict(BaseModel):
    applicable: bool
    count: int = 0
    reason: str


class CategoricalConfidenceVerdict(BaseModel):
    applicable: bool
    category: str | None = None
    confidence: float | None = Field(default=None, description="0.0-1.0")
    reason: str
