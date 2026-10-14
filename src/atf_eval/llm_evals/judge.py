"""LLM judge plumbing shared by every Group 1-5 metric.

One entry point, `judge_structured()`, that:
  - returns a cached response when one exists (cache.py), otherwise makes
    exactly one structured-output call through the configured LLM provider
    (atf_eval.llm_provider) and records it;
  - raises `JudgeUnavailable` (never a bare score) when there is no client or
    the call fails, so a metric can degrade to N/A per METRICS.md §21 rather
    than to a false 0.
"""
from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from atf_eval.llm_evals import cache
from atf_eval.llm_provider import LLMProvider, get_provider

T = TypeVar("T", bound=BaseModel)


class JudgeUnavailable(RuntimeError):
    """No usable judge client, or the judge call failed. The caller turns
    this into an N/A metric result, not a zero."""


def get_judge_client(provider: str | None = None) -> LLMProvider | None:
    """The configured LLM provider if its API key is reachable, else None
    (LLM metrics then report N/A). See atf_eval.llm_provider.get_provider."""
    return get_provider(provider)


def judge_structured(
    client: LLMProvider | None,
    *,
    system: str,
    user: str,
    output_model: type[T],
    model: str | None = None,
    effort: str | None = None,
    max_tokens: int = 1024,
) -> T:
    """`model=None` uses the provider's default model."""
    if model is None and client is not None:
        model = client.default_model
    schema_name = f"{output_model.__module__}.{output_model.__qualname__}"
    ck = cache.key(model=model, effort=effort, system=system, user=user, schema=schema_name)

    hit = cache.load(ck)
    if hit is not None:
        return output_model.model_validate(hit["parsed"])

    if client is None:
        raise JudgeUnavailable("no judge client (provider API key not set)")

    try:
        parsed = client.parse_structured(
            model=model,
            system=system,
            user=user,
            output_model=output_model,
            max_tokens=max_tokens,
            effort=effort,
        )
    except Exception as e:  # noqa: BLE001 - any transport/parse error -> N/A, not 0
        raise JudgeUnavailable(f"judge call failed: {e}") from e

    cache.store(ck, {"parsed": parsed.model_dump(mode="json"), "schema": schema_name})
    return parsed
