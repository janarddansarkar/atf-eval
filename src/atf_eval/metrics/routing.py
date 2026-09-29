"""Routing Similarity (RS) — METRICS.md §5.

RS is an LLM-based semantic evaluation: given the customer input, the agent
response, the expected path/context and the available normalized trajectory
evidence, a judge classifies each applicable routing turn as Correct (1.0),
Partially correct (0.5) or Incorrect (0.0). A turn is applicable when the
Golden trajectory declares an expected path or target for it.

    Semantic Routing Score = mean judge score across applicable routing turns
    Routing Order          = LCS(Expected Path Decisions, Observed Path Decisions)
                             / max(Expected Decision Count, Observed Decision Count)
    RS = 0.70 × Semantic Routing Score + 0.30 × Routing Order

Path decisions are the `routing.path` values on applicable turns; the order
term is N/A (and RS is the semantic score alone) when the observed trace
exposes no routing field, since RS does not require one. Each
judge verdict and its reasoning is kept in the diagnostics (METRICS.md §8,
"routing judge results").

If routing cannot meaningfully be evaluated, RS is N/A: because RS *is* the
semantic judgment, a conversation with no valid judge verdict (no judge
configured, no applicable turn, or every judge call failed) reports RS as
N/A rather than an order-only score. A turn whose judge call fails is
excluded from the semantic score -- a judge outage is never the agent's
failure.

The judge is called at most once per turn; runner.py threads the resulting
verdicts into both the turn-level and the conversation-level results.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import anthropic

from atf_eval import lcs
from atf_eval.aggregate import subweights_for, weighted_composite
from atf_eval.judge import run_routing_judge, verdict_to_score
from atf_eval.metric_result import MetricResult, make_result
from atf_eval.metrics.evidence import turn_tool_calls, turn_transitions
from atf_eval.normalized import NormalizedTurn


@dataclass
class RoutingVerdict:
    turn_id: int
    verdict: str
    score: float
    rationale: str


def _trajectory_evidence(observed: NormalizedTurn) -> str:
    lines: list[str] = []
    if observed.nodes:
        lines.append("nodes visited: " + ", ".join(n.node_id for n in observed.nodes))
    tool_calls = turn_tool_calls(observed)
    if tool_calls:
        lines.append("tools called: " + ", ".join(t.tool_id for t in tool_calls))
    transitions = turn_transitions(observed)
    if transitions:
        lines.append(
            "state changes: " + "; ".join(f"{c.key}: {c.old!r} -> {c.new!r}" for c in transitions)
        )
    if observed.routing:
        lines.append(
            f"observed routing: path={observed.routing.path!r} target={observed.routing.target!r}"
        )
    return "\n".join(lines) if lines else "(no additional trajectory evidence available)"


def is_applicable(expected: NormalizedTurn) -> bool:
    return expected.routing is not None and (
        expected.routing.path is not None or expected.routing.target is not None
    )


def routing_verdict(
    customer_input: str,
    expected: NormalizedTurn,
    observed: NormalizedTurn,
    judge_client: anthropic.Anthropic | None,
    judge_model: str = "claude-opus-5",
    judge_effort: str | None = None,
) -> RoutingVerdict | None:
    """One judge call for one turn. None if there's no client, the turn is
    not applicable, or the judge fails to return a valid verdict."""
    if judge_client is None or not is_applicable(expected):
        return None
    try:
        result = run_routing_judge(
            judge_client,
            judge_model,
            customer_input,
            observed.response,
            expected.routing.path,
            expected.routing.target,
            _trajectory_evidence(observed),
            effort=judge_effort,
        )
    except Exception as e:  # noqa: BLE001 - a judge outage makes this turn N/A, not 0
        print(f"[WARN] routing judge failed: {e}", file=sys.stderr)
        return None
    return RoutingVerdict(
        turn_id=observed.turn_id,
        verdict=result.verdict,
        score=verdict_to_score(result.verdict),
        rationale=result.reasoning,
    )


def semantic_routing_score(
    customer_input: str,
    expected: NormalizedTurn,
    observed: NormalizedTurn,
    judge_client: anthropic.Anthropic | None,
    judge_model: str = "claude-opus-5",
    judge_effort: str | None = None,
) -> float | None:
    """Score-only convenience wrapper around routing_verdict()."""
    v = routing_verdict(customer_input, expected, observed, judge_client, judge_model, judge_effort)
    return v.score if v is not None else None


def rs_turn(semantic_score: float | None) -> float | None:
    """Turn-level score: order is only defined across the conversation, so
    this is the judge score itself (None when the turn was not judged)."""
    return semantic_score


def _decision(turn: NormalizedTurn) -> str | None:
    """A path decision: the turn's routing.path."""
    return turn.routing.path if turn.routing is not None else None


def routing_order_similarity(
    expected_turns: list[NormalizedTurn], observed_turns: list[NormalizedTurn]
) -> float | None:
    """Routing Order Similarity over the path decisions of applicable turns.
    N/A when the observed trace exposes no routing field at all: RS "does not
    require every source trace to expose a dedicated routing field"
    (METRICS.md §5), so absent routing evidence is not an order failure."""
    if not any(t.routing is not None for t in observed_turns):
        return None
    applicable_ids = [t.turn_id for t in expected_turns if is_applicable(t)]
    observed_by_id = {t.turn_id: t for t in observed_turns}
    expected_seq = [d for t in expected_turns if is_applicable(t) and (d := _decision(t)) is not None]
    observed_seq = [
        d for tid in applicable_ids
        if (ot := observed_by_id.get(tid)) is not None and (d := _decision(ot)) is not None
    ]
    return lcs.order_similarity(expected_seq, observed_seq)


def semantic_mean(per_turn_semantic_scores: list[float | None]) -> float | None:
    applicable = [s for s in per_turn_semantic_scores if s is not None]
    return sum(applicable) / len(applicable) if applicable else None


def rs_conversation(
    expected_turns: list[NormalizedTurn],
    observed_turns: list[NormalizedTurn],
    per_turn_semantic_scores: list[float | None],
    subweights: dict | None = None,
    available: bool = True,
) -> float | None:
    """RS = 0.70 × semantic + 0.30 × order, or N/A when there is no valid
    judge verdict (routing cannot meaningfully be evaluated)."""
    if not available:
        return None
    semantic = semantic_mean(per_turn_semantic_scores)
    if semantic is None:
        return None
    components = {
        "semantic": semantic,
        "order": routing_order_similarity(expected_turns, observed_turns),
    }
    return weighted_composite(components, subweights_for("rs", subweights))


def rs_conversation_result(
    expected_turns: list[NormalizedTurn],
    observed_turns: list[NormalizedTurn],
    verdicts: list[RoutingVerdict | None],
    availability_status: str = "available",
    subweights: dict | None = None,
) -> MetricResult:
    scores = [v.score if v is not None else None for v in verdicts]
    score = rs_conversation(
        expected_turns,
        observed_turns,
        scores,
        subweights,
        available=availability_status not in ("unavailable", "not_applicable"),
    )
    return make_result(
        "rs",
        score,
        availability_status,
        diagnostics={
            "applicable_turns": sum(1 for t in expected_turns if is_applicable(t)),
            "judged_turns": sum(1 for v in verdicts if v is not None),
            "order_similarity": routing_order_similarity(expected_turns, observed_turns),
            "verdicts": [
                {"turn_id": v.turn_id, "verdict": v.verdict, "rationale": v.rationale}
                for v in verdicts
                if v is not None
            ],
        },
    )
