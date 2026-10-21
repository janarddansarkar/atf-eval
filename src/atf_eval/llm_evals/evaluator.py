"""Runs the Group 1-5 metric specs against a ConversationContext.

Per METRICS.md:
  - a metric is scored only where its required evidence exists, else N/A
    (§21) -- Group 5 needs audio/timing the canonical trace lacks, so every
    Group 5 metric returns N/A on a text-only trace;
  - turn-level and overall are distinct evaluations (§20). BOTH-level
    metrics get a dedicated conversation-level judge call for the overall
    rather than a blind turn average; TURN-only metrics roll up via the
    spec's `overall_from_turns` rule; OVERALL-only metrics get one call.
"""
from __future__ import annotations

import statistics
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from atf_eval.llm_evals.base import (
    LLMMetricResult,
    Level,
    MetricSpec,
    ScoringType,
)
from atf_eval.llm_evals.context import ConversationContext, TurnContext
from atf_eval.llm_evals.judge import JudgeUnavailable, judge_structured
from atf_eval.llm_evals.registry import ALL_METRICS
from atf_eval.llm_evals.verdicts import (
    CategoricalConfidenceVerdict,
    CountVerdict,
    OrdinalVerdict,
    PassFailVerdict,
    RepetitionVerdict,
    SentimentVerdict,
)
from atf_eval.llm_provider import LLMProvider

_SCALE_DESC = {
    ScoringType.ORDINAL_1_5: "integer 1-5",
    ScoringType.ORDINAL_1_5_NA: "integer 1-5, or not applicable",
    ScoringType.PASS_FAIL: "pass or fail",
    ScoringType.SENTIMENT: "integer -2 (strongly negative) to +2 (strongly positive)",
    ScoringType.COUNT: "a non-negative integer count",
    ScoringType.COUNT_SEVERITY: "a count plus a severity of none / low / medium / high",
    ScoringType.CATEGORICAL_CONFIDENCE: "a category label plus a confidence from 0.0 to 1.0",
}

_VERDICT_MODEL = {
    ScoringType.ORDINAL_1_5: OrdinalVerdict,
    ScoringType.ORDINAL_1_5_NA: OrdinalVerdict,
    ScoringType.PASS_FAIL: PassFailVerdict,
    ScoringType.SENTIMENT: SentimentVerdict,
    ScoringType.COUNT: CountVerdict,
    ScoringType.COUNT_SEVERITY: RepetitionVerdict,
    ScoringType.CATEGORICAL_CONFIDENCE: CategoricalConfidenceVerdict,
}


@dataclass
class ConversationLLMReport:
    conversation_id: str
    turn_results: dict[str, list[LLMMetricResult]] = field(default_factory=dict)   # metric_id -> per-turn
    overall_results: dict[str, LLMMetricResult] = field(default_factory=dict)      # metric_id -> overall

    def all_results(self) -> list[LLMMetricResult]:
        out: list[LLMMetricResult] = []
        for rs in self.turn_results.values():
            out.extend(rs)
        out.extend(self.overall_results.values())
        return out


# ---------------------------------------------------------------------------
# prompt construction
# ---------------------------------------------------------------------------


def _rubric_lines(spec: MetricSpec) -> str:
    return "\n".join(f"  {k}: {v}" for k, v in spec.rubric.items())


def _system(spec: MetricSpec, level_word: str) -> str:
    return (
        f"You are a strict, impartial evaluator scoring exactly ONE metric for {level_word} "
        f"of a customer-service (motor insurance) conversation.\n\n"
        f"Metric: {spec.name}\n"
        f"Definition: {spec.definition}\n"
        f"Scope boundary: {spec.overlap_boundary}\n"
        f"Score ONLY this metric on its own terms. Do not let another quality dimension move this score.\n\n"
        f"Scoring scale ({_SCALE_DESC[spec.scoring_type]}):\n{_rubric_lines(spec)}\n\n"
        f"N/A rule: set applicable=false when {spec.na_rule} "
        f"N/A means the metric cannot be meaningfully evaluated - it is never a zero or a failure.\n\n"
        f"Give a one-to-two sentence reason citing the specific evidence for your verdict."
    )


def _prior_block(turn: TurnContext) -> str:
    if not turn.prior_exchanges:
        return "(this is the first evaluated turn)"
    lines = []
    for i, (c, a) in enumerate(turn.prior_exchanges, start=1):
        lines.append(f"  {i}. customer: {c}")
        lines.append(f"     agent: {a}")
    return "\n".join(lines)


def _turn_user(spec: MetricSpec, turn: TurnContext) -> str:
    return (
        f"<prior_exchanges>\n{_prior_block(turn)}\n</prior_exchanges>\n\n"
        f"<current_turn id=\"{turn.turn_id}\">\n"
        f"customer: {turn.customer_input or '(not captured)'}\n"
        f"agent: {turn.agent_response or '(no agent response)'}\n"
        f"</current_turn>\n\n"
        f"<trajectory_evidence>\n{turn.evidence_block()}\n</trajectory_evidence>\n\n"
        f"Score {spec.name} for this turn only."
    )


def _overall_user(spec: MetricSpec, ctx: ConversationContext) -> str:
    ref = ctx.reference_outcome or {}
    obs = ctx.observed_outcome or {}
    return (
        f"<full_transcript>\n{ctx.transcript()}\n</full_transcript>\n\n"
        f"<trajectory_evidence>\n{ctx.trajectory_evidence()}\n</trajectory_evidence>\n\n"
        f"<outcome_evidence>\n"
        f"reference (golden) outcome attributes: {ref or '(none)'}\n"
        f"observed outcome attributes: {obs or '(none)'}\n"
        f"</outcome_evidence>\n\n"
        f"Score {spec.name} for the conversation as a whole."
    )


# ---------------------------------------------------------------------------
# verdict -> result
# ---------------------------------------------------------------------------


def _native_from_verdict(spec: MetricSpec, v: object):
    if isinstance(v, OrdinalVerdict):
        return v.score if v.applicable else None
    if isinstance(v, PassFailVerdict):
        if not v.applicable or v.verdict is None:
            return None
        return v.verdict.upper()
    if isinstance(v, SentimentVerdict):
        return v.score if v.applicable else None
    if isinstance(v, RepetitionVerdict):
        return {"count": v.count, "severity": v.severity.upper()} if v.applicable else None
    if isinstance(v, CountVerdict):
        return v.count if v.applicable else None
    if isinstance(v, CategoricalConfidenceVerdict):
        return {"category": v.category, "confidence": v.confidence} if v.applicable else None
    return None


def _judge_one(
    spec: MetricSpec,
    client: LLMProvider | None,
    level: str,
    system: str,
    user: str,
    model: str | None,
    effort: str | None,
    turn_id: int | None,
) -> LLMMetricResult:
    verdict_model = _VERDICT_MODEL[spec.scoring_type]
    try:
        v = judge_structured(
            client,
            system=system,
            user=user,
            output_model=verdict_model,
            model=model,
            effort=effort,
        )
    except JudgeUnavailable as e:
        return LLMMetricResult.na(
            spec, level, f"judge unavailable: {e}", turn_id=turn_id, status="unavailable"
        )

    native = _native_from_verdict(spec, v)
    if native is None:
        return LLMMetricResult.na(spec, level, v.reason, turn_id=turn_id)
    diagnostics = {}
    if isinstance(v, (RepetitionVerdict, CategoricalConfidenceVerdict, CountVerdict)):
        diagnostics = v.model_dump(mode="json")
    return LLMMetricResult.scored(spec, level, native, v.reason, turn_id=turn_id, diagnostics=diagnostics)


# ---------------------------------------------------------------------------
# aggregation for TURN-only metrics (METRICS.md §20)
# ---------------------------------------------------------------------------


def _aggregate_turns(spec: MetricSpec, turn_results: list[LLMMetricResult]) -> LLMMetricResult:
    scored = [r for r in turn_results if r.applicability and r.score is not None]
    if not scored:
        return LLMMetricResult.na(spec, "overall", "no turn had sufficient evidence to score")

    rule = spec.overall_from_turns
    if rule == "worst_pass_fail":
        native = "FAIL" if any(r.score == "FAIL" for r in scored) else "PASS"
        reason = f"worst of {len(scored)} scored turn(s)"
    elif rule in ("mean", "max", "min", "sum"):
        vals = [float(r.score) for r in scored]
        agg = {
            "mean": statistics.mean,
            "max": max,
            "min": min,
            "sum": sum,
        }[rule](vals)
        native = round(agg, 3) if rule == "mean" else agg
        reason = f"{rule} of {len(scored)} scored turn(s): {[r.score for r in scored]}"
    else:  # "none" - no defined rollup
        return LLMMetricResult.na(spec, "overall", "metric defines no turn->overall rollup")

    return LLMMetricResult.scored(spec, "overall", native, reason,
                                  diagnostics={"aggregated_from": [r.turn_id for r in scored]})


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------


def _evidence_present(spec: MetricSpec, ctx: ConversationContext) -> tuple[bool, str]:
    req = set(spec.required_evidence)
    if "audio" in req and not ctx.has_audio:
        return False, "no audio evidence in the trace (text-only); Group 5 metric is N/A"
    if "timing" in req and not ctx.has_timing:
        return False, "no timing/interruption evidence in the trace; Group 5 metric is N/A"
    if not any((t.customer_input or t.agent_response) for t in ctx.turns):
        return False, "transcript has no usable customer/agent text"
    return True, ""


def evaluate_conversation(
    ctx: ConversationContext,
    client: LLMProvider | None,
    *,
    metrics: list[MetricSpec] | None = None,
    model: str | None = None,
    effort: str | None = None,
    concurrency: int = 8,
) -> ConversationLLMReport:
    """Every independent judge call for this conversation is submitted to a
    thread pool (the SDK call is blocking I/O). Cache hits resolve instantly,
    so a re-run collapses to a fast serial pass regardless of `concurrency`.
    TURN-only overall rollups are computed after their turn calls resolve."""
    metrics = metrics or ALL_METRICS
    report = ConversationLLMReport(conversation_id=ctx.conversation_id)

    # 1. plan every judge call; short-circuit N/A-by-evidence up front.
    jobs: list[tuple] = []   # (spec, level, turn_id_or_None, system, user)
    rollup_specs: list[MetricSpec] = []

    for spec in metrics:
        ok, why = _evidence_present(spec, ctx)
        if not ok:
            report.overall_results[spec.metric_id] = LLMMetricResult.na(spec, "overall", why)
            if spec.level in (Level.TURN, Level.BOTH):
                report.turn_results[spec.metric_id] = [
                    LLMMetricResult.na(spec, "turn", why, turn_id=t.turn_id) for t in ctx.turns
                ]
            continue

        if spec.level in (Level.TURN, Level.BOTH):
            report.turn_results.setdefault(spec.metric_id, [])
            for turn in ctx.turns:
                if spec.metric_id == "question_quality" and not turn.asks_question():
                    report.turn_results[spec.metric_id].append(
                        LLMMetricResult.na(spec, "turn", "agent asked no question this turn", turn_id=turn.turn_id)
                    )
                    continue
                jobs.append((spec, "turn", turn.turn_id,
                             _system(spec, f"turn {turn.turn_id}"), _turn_user(spec, turn)))

        if spec.level in (Level.OVERALL, Level.BOTH):
            jobs.append((spec, "overall", None,
                         _system(spec, "the whole conversation"), _overall_user(spec, ctx)))
        else:
            rollup_specs.append(spec)

    # 2. run the calls.
    def _do(job):
        spec, level, turn_id, system, user = job
        return job, _judge_one(spec, client, level, system, user, model, effort, turn_id)

    if concurrency > 1 and len(jobs) > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            done = list(pool.map(_do, jobs))
    else:
        done = [_do(j) for j in jobs]

    # 3. file results.
    for (spec, level, turn_id, _s, _u), result in done:
        if level == "turn":
            report.turn_results[spec.metric_id].append(result)
        else:
            report.overall_results[spec.metric_id] = result

    for rs in report.turn_results.values():
        rs.sort(key=lambda r: (r.turn_id is None, r.turn_id))

    # 4. TURN-only overall rollups (need their turn results first).
    for spec in rollup_specs:
        report.overall_results[spec.metric_id] = _aggregate_turns(
            spec, report.turn_results.get(spec.metric_id, [])
        )

    return report
