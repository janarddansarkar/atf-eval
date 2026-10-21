from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from atf_eval.adapter import TrajectoryAgent
from atf_eval.aggregate import DEFAULT_SUBWEIGHTS, atf_score, resolve_weights
from atf_eval.availability import resolve_availability
from atf_eval.dataset import GoldenTurn
from atf_eval.metric_result import MetricResult
from atf_eval.metrics import nodes, outcome, state, tools
from atf_eval.normalized import NormalizedTurn


@dataclass
class TurnResult:
    conversation_id: str
    turn_id: int
    status: str  # "ok" | "agent_error" | "skipped_after_prior_error"
    error: str | None = None
    nts: float | None = None
    sts: float | None = None
    tis: float | None = None
    latency_ms: float | None = None


@dataclass
class ConversationResult:
    conversation_id: str
    turns: list[TurnResult]
    nts: float | None
    sts: float | None
    tis: float | None
    os: float | None
    atf: float | None
    metric_coverage: float
    expected_turns: list[NormalizedTurn]
    observed_turns: list[NormalizedTurn]
    availability: dict[str, str]
    metric_results: dict[str, MetricResult]


@dataclass
class TrajectoryScore:
    """Conversation-level ATF for one (Golden, Observed) pair -- the pure
    scoring half of an evaluation, with no agent invocation."""
    nts: float | None
    sts: float | None
    tis: float | None
    os: float | None
    atf: float | None
    metric_coverage: float
    availability: dict[str, str]
    metric_results: dict[str, MetricResult]

    @property
    def components(self) -> dict[str, float | None]:
        return {"nts": self.nts, "sts": self.sts, "tis": self.tis, "os": self.os}


def score_trajectory(
    expected_turns: list[NormalizedTurn],
    observed_turns: list[NormalizedTurn],
    availability: dict[str, str] | None = None,
    weights: dict[str, float] | None = None,
    subweights: dict[str, dict[str, float]] | None = None,
    tolerance: float = 0.0,
) -> TrajectoryScore:
    """Score one observed trajectory against one Golden trajectory.

    `availability` is the already-resolved per-dimension availability (see
    availability.resolve_availability); a dimension that is not `available`
    makes its metric N/A."""
    weights = resolve_weights(weights)[0]  # None -> DEFAULT_WEIGHTS; otherwise validated
    subweights = subweights or DEFAULT_SUBWEIGHTS
    availability = availability or resolve_availability()

    metric_results = {
        "nts": nodes.nts_conversation_result(expected_turns, observed_turns, availability["nodes"], subweights),
        "sts": state.sts_conversation_result(expected_turns, observed_turns, availability["state"], subweights),
        "tis": tools.tis_conversation_result(
            expected_turns, observed_turns, tolerance, availability["tools"], subweights
        ),
        "os": outcome.os_conversation_result(
            expected_turns, observed_turns, tolerance, availability["outcome"], subweights
        ),
    }
    group_scores = {k: r.score for k, r in metric_results.items()}
    atf, coverage = atf_score(group_scores, weights)
    return TrajectoryScore(
        **group_scores,
        atf=atf,
        metric_coverage=coverage,
        availability=availability,
        metric_results=metric_results,
    )


def _evaluate_conversation(
    conversation_id: str,
    golden_turns: list[GoldenTurn],
    adapter: TrajectoryAgent,
    weights: dict[str, float],
    tolerance: float = 0.0,
    subweights: dict[str, dict[str, float]] | None = None,
) -> ConversationResult:
    history: list[NormalizedTurn] = []
    expected_turns: list[NormalizedTurn] = []
    observed_turns: list[NormalizedTurn] = []
    turn_results: list[TurnResult] = []
    broken = False

    for golden_turn in golden_turns:
        expected_turn = golden_turn.to_normalized()
        expected_turns.append(expected_turn)

        if broken:
            observed_turns.append(
                NormalizedTurn(conversation_id=conversation_id, turn_id=golden_turn.turn_id)
            )
            turn_results.append(
                TurnResult(
                    conversation_id=conversation_id,
                    turn_id=golden_turn.turn_id,
                    status="skipped_after_prior_error",
                )
            )
            continue

        start = time.monotonic()
        try:
            observed_turn = adapter.run_turn(
                conversation_id, golden_turn.turn_id, golden_turn.user_input, list(history)
            )
        except Exception as e:  # noqa: BLE001 - isolate a broken turn, don't crash the whole run
            print(
                f"[WARN] adapter failed on {conversation_id}/{golden_turn.turn_id}: {e}",
                file=sys.stderr,
            )
            observed_turns.append(
                NormalizedTurn(conversation_id=conversation_id, turn_id=golden_turn.turn_id)
            )
            turn_results.append(
                TurnResult(
                    conversation_id=conversation_id,
                    turn_id=golden_turn.turn_id,
                    status="agent_error",
                    error=str(e),
                    latency_ms=(time.monotonic() - start) * 1000,
                )
            )
            broken = True  # a broken turn can leave a stateful agent's context corrupted
            continue

        latency_ms = (time.monotonic() - start) * 1000
        observed_turns.append(observed_turn)
        history.append(observed_turn)

        turn_results.append(
            TurnResult(
                conversation_id=conversation_id,
                turn_id=golden_turn.turn_id,
                status="ok",
                nts=nodes.nts_turn(expected_turn, observed_turn, subweights),
                sts=state.sts_turn(expected_turn, observed_turn, subweights),
                tis=tools.tis_turn(expected_turn, observed_turn, tolerance, subweights),
                latency_ms=latency_ms,
            )
        )

    availability = resolve_availability(getattr(adapter, "availability", None))
    # overall scores are computed from the complete trajectory, not by
    # averaging turn scores (Frozen Rule 10)
    scored = score_trajectory(
        expected_turns,
        observed_turns,
        availability=availability,
        weights=weights,
        subweights=subweights,
        tolerance=tolerance,
    )

    return ConversationResult(
        conversation_id=conversation_id,
        turns=turn_results,
        nts=scored.nts,
        sts=scored.sts,
        tis=scored.tis,
        os=scored.os,
        atf=scored.atf,
        metric_coverage=scored.metric_coverage,
        expected_turns=expected_turns,
        observed_turns=observed_turns,
        availability=availability,
        metric_results=scored.metric_results,
    )


def run_evaluation(
    conversations: dict[str, list[GoldenTurn]],
    adapter: TrajectoryAgent,
    weights: dict[str, float] | None = None,
    concurrency: int = 1,
    tolerance: float = 0.0,
    subweights: dict[str, dict[str, float]] | None = None,
) -> list[ConversationResult]:
    weights = resolve_weights(weights)[0]  # None -> DEFAULT_WEIGHTS; otherwise validated
    items = list(conversations.items())

    def _run(cid: str, turns: list[GoldenTurn]) -> ConversationResult:
        return _evaluate_conversation(
            cid, turns, adapter, weights, tolerance, subweights
        )

    if concurrency <= 1:
        return [_run(cid, turns) for cid, turns in items]

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        return list(executor.map(lambda kv: _run(kv[0], kv[1]), items))
