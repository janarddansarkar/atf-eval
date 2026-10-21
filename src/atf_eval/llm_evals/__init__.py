"""LLM / multimodal evaluation metrics -- METRICS.md Part B, Groups 1-5.

These are independent of the deterministic ATF pipeline (NTS/STS/TIS/OS):
they read the same NormalizedTurn evidence but score conversation-quality,
grounding, safety and voice dimensions via an LLM judge, and preserve native
per-metric scoring (METRICS.md §19 -- no cross-metric normalization).

Entry point:

    from atf_eval.llm_evals import build_context, evaluate_conversation, get_judge_client

    client = get_judge_client()                 # None -> every metric N/A, no calls
    ctx = build_context(cid, golden_turns, observed_turns)
    report = evaluate_conversation(ctx, client)
"""
from atf_eval.llm_evals.base import LLMMetricResult, Level, MetricSpec, ScoringType, normalize_score
from atf_eval.llm_evals.context import ConversationContext, TurnContext, build_context
from atf_eval.llm_evals.evaluator import ConversationLLMReport, evaluate_conversation
from atf_eval.llm_evals.judge import JudgeUnavailable, get_judge_client
from atf_eval.llm_evals.registry import ALL_METRICS, BY_GROUP, BY_ID

__all__ = [
    "ALL_METRICS",
    "BY_GROUP",
    "BY_ID",
    "ConversationContext",
    "ConversationLLMReport",
    "JudgeUnavailable",
    "LLMMetricResult",
    "Level",
    "MetricSpec",
    "ScoringType",
    "TurnContext",
    "build_context",
    "evaluate_conversation",
    "get_judge_client",
    "normalize_score",
]
