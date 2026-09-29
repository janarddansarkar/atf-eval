"""The evaluation settings recorded with every result for auditability
(METRICS.md §8, §23): the routing judge's model, decoding settings and prompt
identity, the argument value-matching rule, the outcome-identity rule, and
the weights.
"""
from __future__ import annotations

import hashlib

from atf_eval.aggregate import DEFAULT_SUBWEIGHTS, DEFAULT_WEIGHTS
from atf_eval.judge import ROUTING_JUDGE_MAX_TOKENS, ROUTING_JUDGE_SYSTEM_PROMPT


def _prompt_id(prompt: str) -> str:
    return "sha256:" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def evaluation_settings(
    judge_model: str | None,
    judge_effort: str | None = None,
    tolerance: float = 0.0,
    weights: dict | None = None,
    subweights: dict | None = None,
) -> dict:
    """`judge_model=None` records that no routing judge was configured."""
    return {
        "routing_judge": (
            {
                "configured": True,
                "model": judge_model,
                "effort": judge_effort,
                "max_tokens": ROUTING_JUDGE_MAX_TOKENS,
                "temperature": "API default (not set)",
                "output": "structured {verdict: correct|partial|incorrect, reasoning}",
                "system_prompt_id": _prompt_id(ROUTING_JUDGE_SYSTEM_PROMPT),
                "system_prompt": ROUTING_JUDGE_SYSTEM_PROMPT,
            }
            if judge_model
            else {"configured": False, "effect": "RS is N/A (judge score is required)"}
        ),
        "argument_matching": (
            "exact equality; numeric values match within ±" + f"{tolerance:g}"
            if tolerance
            else "exact equality (numeric tolerance 0)"
        ),
        "outcome_identity_evaluator": "exact match of the outcome id (METRICS.md §6)",
        "weights": weights or DEFAULT_WEIGHTS,
        "subweights": subweights or DEFAULT_SUBWEIGHTS,
    }
