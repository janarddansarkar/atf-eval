from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from atf_eval.runner import ConversationResult

_GROUP_KEYS = ("nts", "sts", "tis", "rs", "os")


def _availability_summary(results: list[ConversationResult]) -> dict[str, str]:
    """Per-dimension availability across the whole run: 'available' if any
    conversation reported it available, else the first non-available status
    seen (or 'unavailable' if there were no conversations)."""
    summary: dict[str, str] = {}
    for dimension in ("nodes", "state", "tools", "routing", "outcome"):
        statuses = [r.availability.get(dimension, "unavailable") for r in results]
        if "available" in statuses:
            summary[dimension] = "available"
        elif statuses:
            summary[dimension] = statuses[0]
        else:
            summary[dimension] = "unavailable"
    return summary


def _metrics_summary(results: list[ConversationResult]) -> dict[str, dict]:
    """Averages each conversation's per-key MetricResult into one
    METRICS.md §8-shaped result per metric group, for the run as a whole."""
    out: dict[str, dict] = {}
    for key in _GROUP_KEYS:
        mrs = [r.metric_results[key] for r in results if key in r.metric_results]
        applicable = [mr for mr in mrs if mr.applicability]
        score = sum(mr.score for mr in applicable) / len(applicable) if applicable else None
        statuses = {mr.status for mr in mrs}
        status = "available" if "available" in statuses else (next(iter(statuses), "unavailable"))
        out[key] = {
            "metric_id": key,
            "score": score,
            "status": status,
            "applicability": bool(applicable),
            "coverage": len(applicable) / len(mrs) if mrs else 0.0,
            "diagnostics": {"per_conversation": [asdict(mr) for mr in mrs]},
        }
    return out


def _fmt_score(v: float | None) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "N/A"


def _fmt_pct(v: float | None) -> str:
    return f"{v:.1%}" if isinstance(v, (int, float)) else "N/A"


def render_results_table(conversations: list[dict]) -> str:
    """One row per conversation: ATF + coverage + all 5 metric-group scores.
    Takes the same `conversations` list shape write_reports() writes into the
    JSON report (conversation_id/atf/metric_coverage/nts/sts/tis/rs/os), so
    it works both on a just-computed run and on a report.json loaded back
    from disk later."""
    if not conversations:
        return "(no conversations)"

    id_width = max(len("Conversation"), max(len(str(c["conversation_id"])) for c in conversations))
    header = (
        f"{'Conversation':<{id_width}} {'ATF':>7} {'Coverage':>9} "
        f"{'NTS':>7} {'STS':>7} {'TIS':>7} {'RS':>7} {'OS':>7}"
    )
    divider = "-" * len(header)
    lines = [header, divider]
    for c in conversations:
        lines.append(
            f"{str(c['conversation_id']):<{id_width}} "
            f"{_fmt_score(c.get('atf')):>7} {_fmt_pct(c.get('metric_coverage')):>9} "
            f"{_fmt_score(c.get('nts')):>7} {_fmt_score(c.get('sts')):>7} "
            f"{_fmt_score(c.get('tis')):>7} {_fmt_score(c.get('rs')):>7} {_fmt_score(c.get('os')):>7}"
        )
    return "\n".join(lines)


def _mean_of(values: list[float | None]) -> float | None:
    applicable = [v for v in values if v is not None]
    if not applicable:
        return None
    return sum(applicable) / len(applicable)


def aggregate(results: list[ConversationResult]) -> dict:
    groups = {}
    for key in _GROUP_KEYS:
        mean = _mean_of([getattr(r, key) for r in results])
        groups[key] = {"score": mean, "status": "available" if mean is not None else "unavailable"}

    turns = {}
    for r in results:
        for t in r.turns:
            turns[f"{r.conversation_id}/{t.turn_id}"] = {
                "status": t.status,
                "nts": t.nts,
                "sts": t.sts,
                "tis": t.tis,
                "rs": t.rs,
            }

    return {
        "atf": {
            "score": _mean_of([r.atf for r in results]),
            "coverage": _mean_of([r.metric_coverage for r in results]) or 0.0,
        },
        "groups": groups,
        "turns": turns,
        "total_conversations": len(results),
        "total_turns": sum(len(r.turns) for r in results),
    }


def write_reports(
    results: list[ConversationResult], output_dir: str, settings: dict | None = None
) -> tuple[Path, Path]:
    """`settings` (see atf_eval.settings.evaluation_settings) is written into
    the JSON report so the judge configuration, matching rules and weights
    are auditable alongside the scores (METRICS.md §8, §23)."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    evaluation_id = f"EVAL_{ts}"
    all_metrics_applicable = all(
        any(r.metric_results[k].applicability for r in results if k in r.metric_results)
        for k in _GROUP_KEYS
    ) if results else False

    json_path = output_path / f"report_{ts}.json"
    json_path.write_text(
        json.dumps(
            {
                "evaluation_id": evaluation_id,
                "status": "completed" if results else "no_conversations",
                "atf": {
                    "score": _mean_of([r.atf for r in results]),
                    "coverage": _mean_of([r.metric_coverage for r in results]) or 0.0,
                },
                "metrics": _metrics_summary(results),
                "coverage": _mean_of([r.metric_coverage for r in results]) or 0.0,
                "diagnostics": {"availability": _availability_summary(results)},
                "metadata": {"total_conversations": len(results), "all_metrics_applicable": all_metrics_applicable},
                "settings": settings or {},
                "summary": aggregate(results),
                "conversations": [
                    {
                        "conversation_id": r.conversation_id,
                        "nts": r.nts,
                        "sts": r.sts,
                        "tis": r.tis,
                        "rs": r.rs,
                        "os": r.os,
                        "atf": r.atf,
                        "metric_coverage": r.metric_coverage,
                        "availability": r.availability,
                        "diagnostics": {k: mr.diagnostics for k, mr in r.metric_results.items()},
                        "turns": [asdict(t) for t in r.turns],
                    }
                    for r in results
                ],
            },
            indent=2,
            default=str,  # diagnostics carry arbitrary state/argument values
        ),
        encoding="utf-8",
    )

    csv_path = output_path / f"report_{ts}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "conversation_id",
                "turn_id",
                "status",
                "nts",
                "sts",
                "tis",
                "rs",
                "os",
                "atf",
                "metric_coverage",
                "latency_ms",
                "error",
            ]
        )
        for r in results:
            for t in r.turns:
                writer.writerow(
                    [r.conversation_id, t.turn_id, t.status, t.nts, t.sts, t.tis, t.rs, "", "", "", t.latency_ms, t.error or ""]
                )
            writer.writerow(
                [r.conversation_id, "__summary__", "", r.nts, r.sts, r.tis, r.rs, r.os, r.atf, r.metric_coverage, "", ""]
            )

    return json_path, csv_path
