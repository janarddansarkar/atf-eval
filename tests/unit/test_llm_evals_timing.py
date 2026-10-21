"""Group 5's timing tier (Interruption Count/Recovery/Understanding,
Turn-taking Quality, Perceived Response Latency) -- METRICS.md §18 says these
need audio *or* reliable timing evidence, never an inferred score from
transcript alone. These tests check the evidence gate only (network-free):
that build_context() correctly derives has_timing from a turn's TurnTiming,
and that the 5 timing-tier specs' N/A-by-evidence check opens once it does --
not real judge scoring, which needs a live client.
"""
from atf_eval.llm_evals.context import build_context
from atf_eval.llm_evals.evaluator import _evidence_present
from atf_eval.llm_evals.registry import BY_ID
from atf_eval.normalized import InterruptionEvent, NormalizedTurn, TurnTiming

_TIMING_TIER = [
    "interruption_count",
    "interruption_recovery",
    "interruption_understanding",
    "turn_taking_quality",
    "perceived_response_latency",
]

_ACOUSTIC_TIER = [
    "speech_naturalness",
    "prosody_tone",
    "pronunciation_clarity",
    "vocal_anger_frustration",
    "vocal_sarcasm",
    "customer_boredom_frustration_detection",
    "customer_disconnection_reason",
    "disconnection_handling",
]


def _turn(turn_id: int, timing: TurnTiming | None = None) -> NormalizedTurn:
    return NormalizedTurn(
        conversation_id="c",
        turn_id=turn_id,
        customer_input="hello",
        response="hi there",
        timing=timing,
    )


def test_has_timing_false_when_no_turn_carries_timing():
    turns = [_turn(1), _turn(2)]
    ctx = build_context("c", turns, turns)
    assert ctx.has_timing is False


def test_has_timing_true_when_any_observed_turn_carries_timing():
    timed = _turn(2, timing=TurnTiming(customer_end_ms=1000, agent_start_ms=1700, response_latency_ms=700))
    turns = [_turn(1), timed]
    ctx = build_context("c", turns, turns)
    assert ctx.has_timing is True


def test_timing_tier_specs_are_na_by_evidence_without_timing():
    turns = [_turn(1)]
    ctx = build_context("c", turns, turns)
    for metric_id in _TIMING_TIER:
        ok, why = _evidence_present(BY_ID[metric_id], ctx)
        assert ok is False, f"{metric_id} should be evidence-gated without timing"
        assert why


def test_timing_tier_specs_open_once_timing_is_present():
    timed = _turn(1, timing=TurnTiming(customer_end_ms=1000, agent_start_ms=1700, response_latency_ms=700))
    turns = [timed]
    ctx = build_context("c", turns, turns)
    for metric_id in _TIMING_TIER:
        ok, _ = _evidence_present(BY_ID[metric_id], ctx)
        assert ok is True, f"{metric_id} should open once has_timing is True"


def test_acoustic_tier_specs_stay_na_even_with_timing_present():
    """Timing evidence must not leak into the audio-gated tier -- those 8
    metrics need has_audio, which nothing in this framework sets yet."""
    timed = _turn(1, timing=TurnTiming(customer_end_ms=1000, agent_start_ms=1700, response_latency_ms=700))
    turns = [timed]
    ctx = build_context("c", turns, turns)
    assert ctx.has_audio is False
    for metric_id in _ACOUSTIC_TIER:
        ok, why = _evidence_present(BY_ID[metric_id], ctx)
        assert ok is False, f"{metric_id} must stay N/A -- no audio evidence exists"
        assert why


def test_evidence_block_reports_latency_and_interruptions():
    timing = TurnTiming(
        customer_end_ms=1000,
        agent_start_ms=1700,
        response_latency_ms=700,
        interruptions=[InterruptionEvent(by="customer", at_ms=2200, note="cuts in")],
    )
    turns = [_turn(1, timing=timing)]
    ctx = build_context("c", turns, turns)
    block = ctx.turns[0].evidence_block()
    assert "700ms" in block
    assert "interruption: customer interrupted at 2200ms (cuts in)" in block


def test_conversation_trajectory_evidence_includes_turn_timing():
    """The overall-level judge prompt (Interruption Count, Turn-taking
    Quality) reads trajectory_evidence(), not evidence_block() directly --
    confirm timing survives that rollup too."""
    timing = TurnTiming(customer_end_ms=1000, agent_start_ms=1700, response_latency_ms=700)
    turns = [_turn(1, timing=timing)]
    ctx = build_context("c", turns, turns)
    assert "700ms" in ctx.trajectory_evidence()
