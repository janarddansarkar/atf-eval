"""Sage -- Kestrel with an LLM doing the intent classification.

Same nodes, tools, state and routing as Kestrel; the only difference is that
the intent comes from an LLM (through atf_eval's provider layer, so it uses
whichever of `--provider`/ATF_LLM_PROVIDER is configured) instead of keyword
rules. With no API key, or if a call fails, it falls back to Kestrel's
keyword rules so a run never breaks.

    ANTHROPIC_API_KEY=... atf-eval run --dataset examples/golden_dataset.jsonl \
        --adapter agents.sage.agent:Sage --no-routing-judge
"""
from __future__ import annotations

import sys
from typing import Literal

from pydantic import BaseModel

from agents.kestrel.agent import INTENTS, Kestrel
from atf_eval.llm_provider import get_provider

SYSTEM = (
    "You route messages for a loan-collections agent. Pick the customer's intent:\n"
    "- hardship: cannot pay because of financial difficulty (job loss, illness, ...)\n"
    "- pay_now: wants to make a payment right now\n"
    "- unclear: anything else"
)


class IntentChoice(BaseModel):
    intent: Literal["hardship", "pay_now", "unclear"]


class Sage(Kestrel):
    def __init__(self, provider=None, model: str | None = None) -> None:
        self._provider = provider if provider is not None else get_provider()
        self._model = model

    def classify(self, text: str) -> str | None:
        if self._provider is None:
            return super().classify(text)
        try:
            choice = self._provider.parse_structured(
                model=self._model or self._provider.default_model,
                system=SYSTEM,
                user=text,
                output_model=IntentChoice,
                max_tokens=64,
            )
        except Exception as e:  # noqa: BLE001 - never fail the run on a classifier hiccup
            print(f"[WARN] Sage LLM classify failed, using keyword rules: {e}", file=sys.stderr)
            return super().classify(text)
        return choice.intent if choice.intent in INTENTS else None
