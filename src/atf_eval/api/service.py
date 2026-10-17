"""The evaluation itself, independent of HTTP: golden + observed trajectories in,
scores and the dashboard/CSV artefacts out. Reuses the `atf-eval dashboard`
pipeline (dashboard.generate.build_outputs), so the API and the CLI always agree.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from atf_eval.aggregate import resolve_weights
from atf_eval.dashboard.generate import build_outputs, turns_compared_label
from atf_eval.dataset import GoldenTurn, group_by_conversation
from atf_eval.normalized import NormalizedTurn

SCORE_KEYS = ("nts", "sts", "tis", "rs", "os", "atf")


class EvalOptions(BaseModel):
    routing_judge: bool = True  # RS's LLM routing judge; off -> RS is N/A
    llm_evals: bool = True  # the Group 1-5 LLM/multimodal metrics; off -> skipped
    provider: str | None = None  # anthropic | nvidia (default: server's ATF_LLM_PROVIDER / anthropic)
    judge_model: str | None = None  # default: the provider's default model
    effort: str = "low"
    weights: dict[str, float] | None = None  # ATF component weights {"nts", "sts", "tis", "rs", "os"}


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def build_runs(
    golden: list[GoldenTurn],
    observed: dict[str, list[NormalizedTurn]],
    availability: dict[str, str] | None,
) -> tuple[list[dict], list[dict]]:
    """One run per observed conversation, paired with the golden conversation of
    the same id and sliced to the turns the agent actually produced. Conversations
    with no golden counterpart are returned in `skipped`, not scored."""
    golden_by_conv = group_by_conversation(golden)
    runs, skipped = [], []
    for cid, turns in observed.items():
        g_turns = golden_by_conv.get(cid)
        if g_turns is None:
            skipped.append({"conversation_id": cid, "reason": "no golden conversation with this id"})
            continue
        if not turns:
            skipped.append({"conversation_id": cid, "reason": "no observed turns"})
            continue
        golden_norm = [t.to_normalized() for t in g_turns]
        ids = {t.turn_id for t in turns}
        expected = [t for t in golden_norm if t.turn_id in ids] or golden_norm
        runs.append({
            "scenario": cid,
            "deviation_type": "agent_run",
            "description": f"Agent run, {len(turns)} turn(s) scored against golden '{cid}'.",
            "expected": expected,
            "observed": sorted(turns, key=lambda t: t.turn_id),
            "turns_compared": turns_compared_label(expected, golden_norm),
            "doc_meta": {},
            "golden_key": cid,
            "availability": dict(availability or {}),
        })
    return runs, skipped


def _run_summary(run: dict) -> dict:
    m = run["metrics"]
    out = {
        "conversation_id": run["scenario"],
        "turns_compared": run["turns_compared"],
        **{k: m[f"{k}_overall"] if k != "atf" else m["atf"] for k in SCORE_KEYS},
        "metric_coverage": m["metric_coverage"],
        "availability": m["_availability"],
        "components": {k: v for k, v in m.items() if not k.startswith("_") and isinstance(v, (int, float, type(None)))},
        "diagnostics": _jsonable(m["_diagnostics"]),
    }
    report = run.get("llm")
    if report is not None:
        out["llm"] = {
            mid: {"score": _jsonable(r.score), "normalized": r.normalized, "status": r.status, "reason": r.reason}
            for mid, r in report.overall_results.items()
        }
    return out


def _mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def evaluate(
    golden: list[GoldenTurn],
    observed: dict[str, list[NormalizedTurn]],
    availability: dict[str, str] | None,
    options: EvalOptions,
    out_dir: Path,
    label: str,
) -> dict:
    """Score everything and write the artefacts into `out_dir`. Returns the result document."""
    runs, skipped = build_runs(golden, observed, availability)
    if not runs:
        raise ValueError("nothing to score: no observed conversation matches a golden conversation id")

    judge_client = None
    if options.routing_judge or options.llm_evals:
        from atf_eval.llm_evals import get_judge_client

        judge_client = get_judge_client(options.provider)
    judge_model = (options.judge_model or judge_client.default_model) if judge_client else None
    judge_effort = options.effort if judge_client and judge_client.supports_effort else None
    weights, subweights = resolve_weights(options.weights)

    built = build_outputs(
        runs,
        golden_summary=f"{len({t.conversation_id for t in golden})} golden conversation(s)",
        source_label=label,
        output_dir=out_dir,
        judge_client=judge_client,
        rs_client=judge_client if options.routing_judge else None,
        judge_model=judge_model,
        judge_effort=judge_effort,
        weights=weights,
        subweights=subweights,
        llm_evals=options.llm_evals,
    )
    summaries = [_run_summary(r) for r in built["runs"]]
    return {
        "summary": {
            "conversations": len(summaries),
            **{f"{k}_mean": _mean([s[k] for s in summaries]) for k in SCORE_KEYS},
            "judge": (f"{judge_client.name}:{judge_model}" if judge_client else None),
        },
        "runs": summaries,
        "skipped": skipped,
        "settings": built["settings"],
        "files": {name: p.name for name, p in built["paths"].items()},
    }
