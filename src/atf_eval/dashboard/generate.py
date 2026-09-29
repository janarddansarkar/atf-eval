"""Generate ATF metrics (group-level + full component-level) and an
interactive HTML dashboard for a golden dataset vs. its observed-scenario
fixtures -- deterministic NTS/STS/TIS/RS/OS/ATF plus the Group 1-5
LLM/multimodal metrics (METRICS.md Part B), all in one page.

Supports multiple golden scenarios at once: every *.json file in
--golden-dir is its own golden trajectory, and each observed fixture is
routed to the golden it belongs to via its own `golden_scenario` field
(falling back to the sole golden when there's only one).

Self-contained: normalizes the golden/fixture JSON shapes itself (no
dependency on any adapter), computes every NTS/STS/TIS/RS/OS sub-component
via atf_eval's real metric functions and every LLM metric via
atf_eval.llm_evals (never re-derives a formula), and renders CSVs + a
single-file HTML dashboard with no external JS (radio-driven CSS tabs +
native <details>).

Usage (as the installed CLI, from your project root -- defaults assume
`golden/` and `tests/fixtures/scenarios/` relative to the current directory):
    atf-eval dashboard
    atf-eval dashboard --golden-dir path/to/golden --scenarios path/to/fixtures \\
        --output-dir results/

Or as a library:
    from atf_eval.dashboard.generate import add_arguments, run
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import sys
from html import escape as html_escape
from pathlib import Path

from atf_eval.aggregate import resolve_weights
from atf_eval.availability import resolve_availability
from atf_eval.metrics.nodes import (
    node_coverage,
    node_order_similarity,
    node_precision,
    node_recall,
)
from atf_eval.metrics.outcome import (
    last_outcome,
    outcome_attribute_accuracy,
    outcome_completion,
    outcome_identity_accuracy,
)
from atf_eval.metrics.routing import routing_order_similarity, semantic_mean
from atf_eval.metrics.state import (
    new_state_accuracy,
    old_state_accuracy,
    state_key_accuracy,
    transition_accuracy,
    transition_order_similarity,
)
from atf_eval.metrics.tools import (
    tool_coverage,
    tool_identity_accuracy,
    tool_input_similarity,
    tool_order_similarity,
    tool_precision,
)
from atf_eval.runner import score_trajectory
from atf_eval.normalized import (
    InterruptionEvent,
    NormalizedNode,
    NormalizedTurn,
    Outcome,
    Routing,
    StateChange,
    ToolCall,
    TurnTiming,
)

# ---------------------------------------------------------------------------
# Fixture normalization -- kept self-contained here (no dependency on any
# particular adapter) so this works against any project's golden/fixture
# JSON, as long as it matches schema/normalized_trajectory.schema.json
# (golden) or the same shape plus a `golden_scenario` field (fixtures).
# ---------------------------------------------------------------------------


def _state_changes(raw: list[dict]) -> list[StateChange]:
    return [StateChange(key=c["key"], old=c.get("old"), new=c.get("new")) for c in raw]


def _routing(raw_turn: dict) -> Routing | None:
    raw = raw_turn.get("routing")
    if not raw:
        return None
    return Routing(path=raw.get("path"), target=raw.get("target"))


def _timing(raw_turn: dict) -> TurnTiming | None:
    raw = raw_turn.get("timing")
    if not raw:
        return None
    return TurnTiming(
        customer_start_ms=raw.get("customer_start_ms"),
        customer_end_ms=raw.get("customer_end_ms"),
        agent_start_ms=raw.get("agent_start_ms"),
        agent_end_ms=raw.get("agent_end_ms"),
        response_latency_ms=raw.get("response_latency_ms"),
        interruptions=[
            InterruptionEvent(by=i["by"], at_ms=i["at_ms"], note=i.get("note"))
            for i in raw.get("interruptions", [])
        ],
    )


def _canonical_tool_calls(raw: list[dict]) -> list[ToolCall]:
    return [
        ToolCall(
            tool_id=tc["tool_id"],
            arguments=tc.get("input", {}),
            result=tc.get("output"),
            sequence=tc.get("sequence"),
            status=tc.get("status", "unknown"),
        )
        for tc in raw
    ]


def _raw_fixture_tool_calls(raw: list[dict]) -> list[ToolCall]:
    return [
        ToolCall(
            tool_id=tc["tool_name"],
            arguments=tc.get("input", {}),
            result=tc.get("output"),
            sequence=tc.get("sequence_index"),
            status="success",
        )
        for tc in raw
    ]


def load_golden_turns(path: Path) -> list[NormalizedTurn]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    turns = []
    for t in doc["turns"]:
        nodes = [
            NormalizedNode(
                node_id=n["node_id"],
                sequence=n.get("sequence"),
                output=n.get("output", {}),
                state_changes=_state_changes(n.get("state_changes", [])),
                tool_calls=_canonical_tool_calls(n.get("tool_calls", [])),
            )
            for n in t.get("nodes", [])
        ]
        turns.append(
            NormalizedTurn(
                conversation_id=doc["trace_id"],
                turn_id=t["turn_id"],
                nodes=nodes,
                tool_calls=_canonical_tool_calls(t.get("unassociated_tool_calls", [])),
                routing=_routing(t),
                response=t.get("output", {}).get("agent"),
                customer_input=t.get("input", {}).get("customer"),
                timing=_timing(t),
            )
        )
    _attach_trace_outcome(doc, turns)
    return turns


def _attach_trace_outcome(doc: dict, turns: list[NormalizedTurn]) -> None:
    """A trace-level `outcome` belongs to the conversation's last turn (same
    convention for golden and observed)."""
    if not turns:
        return
    raw_outcome = doc.get("outcome")
    if raw_outcome is not None:
        turns[-1].outcome = Outcome(
            id=raw_outcome.get("id"),
            attributes=raw_outcome.get("attributes", {}),
            required_conditions=raw_outcome.get("required_conditions", []),
        )


def declared_availability(doc: dict) -> dict[str, str]:
    """The canonical trace's declared `availability` (a required field of the
    agent-eval schema) wins; for a raw fixture without one, the fixture
    "adapter" declares a dimension available
    iff the raw trace carries that field at all. A field that is present but
    empty means "instrumented, nothing happened" (a scored miss); a field
    that is absent means "not recorded" (N/A)."""
    if doc.get("availability"):
        return dict(doc["availability"])
    raw_turns = doc.get("turns") or ([doc["turn"]] if "turn" in doc else [])
    raw_nodes = [n for t in raw_turns for n in t.get("nodes", [])]

    def has(field: str) -> bool:
        return any(field in t for t in raw_turns) or any(field in n for n in raw_nodes)

    return {
        "nodes": "available" if has("nodes") else "unavailable",
        "state": "available" if has("state_changes") else "unavailable",
        "tools": "available" if has("tool_calls") else "unavailable",
        "routing": "available",
        "outcome": "available" if "outcome" in doc else "unavailable",
    }


def _normalize_raw_turn(raw_turn: dict) -> NormalizedTurn:
    nodes = [
        NormalizedNode(
            node_id=n["node_id"],
            sequence=n.get("sequence_index"),
            state_changes=_state_changes(n.get("state_changes", [])),
            tool_calls=_raw_fixture_tool_calls(n.get("tool_calls", [])),
        )
        for n in raw_turn.get("nodes", [])
    ]
    # Raw fixtures repeat every node's tool calls / state changes in a
    # turn-level list; the metrics de-duplicate those against the node-level
    # events (atf_eval.metrics.evidence), so either copy may be passed.
    return NormalizedTurn(
        conversation_id="_",
        turn_id=raw_turn["turn_id"],
        nodes=nodes,
        tool_calls=_raw_fixture_tool_calls(raw_turn.get("tool_calls", [])),
        state_changes=_state_changes(raw_turn.get("state_changes", [])),
        routing=_routing(raw_turn),
        response=raw_turn.get("conversation", {}).get("agent"),
        customer_input=raw_turn.get("conversation", {}).get("customer"),
        timing=_timing(raw_turn),
    )


def load_fixture_turns(path: Path) -> list[NormalizedTurn]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if "turns" in doc:
        turns = [_normalize_raw_turn(t) for t in doc["turns"]]
    else:
        turns = [_normalize_raw_turn(doc["turn"])]
    _attach_trace_outcome(doc, turns)
    return turns


# ---------------------------------------------------------------------------
# Metric computation -- every sub-component, via atf_eval's real functions.
# ---------------------------------------------------------------------------


def _routing_verdicts(
    expected: list[NormalizedTurn],
    observed: list[NormalizedTurn],
    judge_client,
    judge_model: str,
    judge_effort: str | None,
) -> list:
    """One RS routing-judge verdict per observed turn (METRICS.md §5), aligned to
    the golden turn with the same turn_id; None for any turn with no judge
    client, no expected routing, or a judge-call failure -- never 0. Turn
    calls run concurrently; a cache hit resolves instantly."""
    from concurrent.futures import ThreadPoolExecutor

    from atf_eval.metrics.routing import routing_verdict

    ref_by_id = {t.turn_id: t for t in expected}

    def _one(obs: NormalizedTurn):
        ref = ref_by_id.get(obs.turn_id)
        if ref is None:
            return None
        return routing_verdict(
            obs.customer_input or ref.customer_input or "",
            ref, obs, judge_client, judge_model, judge_effort,
        )

    if judge_client is None or len(observed) <= 1:
        return [_one(o) for o in observed]
    with ThreadPoolExecutor(max_workers=min(8, len(observed))) as pool:
        return list(pool.map(_one, observed))


def score_components(
    expected: list[NormalizedTurn],
    observed: list[NormalizedTurn],
    judge_client=None,
    judge_model: str = "claude-opus-5",
    judge_effort: str | None = None,
    declared: dict[str, str] | None = None,
    weights: dict | None = None,
    subweights: dict | None = None,
) -> dict:
    """Every group score and subcomponent for one run, via atf_eval's own
    scoring (runner.score_trajectory) -- no formula is re-derived here.
    Subcomponents of a dimension declared unavailable are reported N/A,
    matching the group score."""
    availability = resolve_availability(declared, judge_configured=judge_client is not None)
    verdicts = _routing_verdicts(expected, observed, judge_client, judge_model, judge_effort)
    scored = score_trajectory(
        expected, observed, verdicts=verdicts, availability=availability, weights=weights, subweights=subweights
    )

    def gate(dimension: str, value):
        return value if availability[dimension] == "available" else None

    exp_node_ids = [n.node_id for t in expected for n in t.nodes]
    obs_node_ids = [n.node_id for t in observed for n in t.nodes]
    exp_outcome = last_outcome(expected)
    obs_outcome = last_outcome(observed)

    return {
        "nts_coverage": gate("nodes", node_coverage(exp_node_ids, obs_node_ids)),
        "nts_precision": gate("nodes", node_precision(exp_node_ids, obs_node_ids)),
        "nts_recall_diag": gate("nodes", node_recall(exp_node_ids, obs_node_ids)),
        "nts_order": gate("nodes", node_order_similarity(exp_node_ids, obs_node_ids)),
        "nts_overall": scored.nts,
        "sts_transition_accuracy": gate("state", transition_accuracy(expected, observed)),
        "sts_order": gate("state", transition_order_similarity(expected, observed)),
        "sts_key_accuracy_diag": gate("state", state_key_accuracy(expected, observed)),
        "sts_old_value_accuracy_diag": gate("state", old_state_accuracy(expected, observed)),
        "sts_new_value_accuracy_diag": gate("state", new_state_accuracy(expected, observed)),
        "sts_overall": scored.sts,
        "tis_coverage": gate("tools", tool_coverage(expected, observed)),
        "tis_precision": gate("tools", tool_precision(expected, observed)),
        "tis_identity": gate("tools", tool_identity_accuracy(expected, observed)),
        "tis_input_similarity": gate("tools", tool_input_similarity(expected, observed)),
        "tis_order": gate("tools", tool_order_similarity(expected, observed)),
        "tis_overall": scored.tis,
        # mean judge verdict over applicable turns; without any verdict RS
        # is N/A and order is shown as a diagnostic (METRICS.md §5)
        "rs_semantic": gate("routing", semantic_mean([v.score if v else None for v in verdicts])),
        "rs_order": routing_order_similarity(expected, observed),
        "rs_overall": scored.rs,
        "os_identity": gate("outcome", outcome_identity_accuracy(exp_outcome, obs_outcome)),
        "os_attribute": gate("outcome", outcome_attribute_accuracy(exp_outcome, obs_outcome)),
        "os_completion": gate("outcome", outcome_completion(exp_outcome, obs_outcome)),
        "os_overall": scored.os,
        "atf": scored.atf,
        "metric_coverage": scored.metric_coverage,
        "_exp_outcome": exp_outcome,
        "_obs_outcome": obs_outcome,
        "_availability": availability,
        "_diagnostics": {k: r.diagnostics for k, r in scored.metric_results.items()},
    }


def turns_compared_label(expected: list[NormalizedTurn], golden_turns: list[NormalizedTurn]) -> str:
    if len(expected) == len(golden_turns):
        return f"{len(expected)} (full conversation)"
    ids = sorted(t.turn_id for t in expected)
    if len(ids) == 1:
        return f"1 (turn {ids[0]} only)"
    return f"{len(ids)} (turns {ids[0]}-{ids[-1]})"


# ---------------------------------------------------------------------------
# Run discovery + orchestration
# ---------------------------------------------------------------------------


def discover_goldens(golden_dir: Path) -> list[dict]:
    """Every *.json golden file in golden_dir. Each is keyed by its own
    metadata.scenario_id (falling back to trace_id) -- that's the same key
    an observed fixture names in its `golden_scenario` field, so fixtures can
    be routed to the correct golden regardless of how many golden scenarios
    are in play."""
    goldens = []
    for path in sorted(golden_dir.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        key = doc.get("metadata", {}).get("scenario_id") or doc["trace_id"]
        goldens.append({
            "key": key,
            "path": path,
            "turns": load_golden_turns(path),
            "availability": declared_availability(doc),
        })
    return goldens


def discover_runs(goldens: list[dict], scenarios_dir: Path) -> list[dict]:
    """A golden-vs-itself baseline per golden scenario, plus every *.json
    fixture found in scenarios_dir, each routed to its matching golden via
    its own `golden_scenario` field (falling back to the sole golden when
    there's only one, for older fixtures that don't set the field). The
    matching golden slice for each fixture is whatever golden turn_ids the
    fixture itself covers -- this generalizes correctly whether a fixture is
    a full conversation or a single turn. Runs are grouped by golden -- each
    golden's baseline immediately followed by its own matching fixtures --
    so the dashboard's tab order reads as one block per golden scenario."""
    golden_by_key = {g["key"]: g for g in goldens}
    fixture_docs = []  # (path, doc_meta, matched_golden_or_None), matched once per fixture
    for path in sorted(scenarios_dir.glob("*.json")):
        doc_meta = json.loads(path.read_text(encoding="utf-8"))
        golden_key = doc_meta.get("golden_scenario")
        if golden_key and golden_key in golden_by_key:
            g = golden_by_key[golden_key]
        elif len(goldens) == 1:
            g = goldens[0]
        else:
            print(
                f"[WARN] {path.name}: golden_scenario={golden_key!r} matches no golden "
                f"file in {scenarios_dir} -- skipping", file=sys.stderr,
            )
            g = None
        fixture_docs.append((path, doc_meta, g))

    runs = []
    for g in goldens:
        short = g["key"].split("_", 1)[0] if "_" in g["key"] else g["key"]
        runs.append(
            {
                "scenario": f"{short}_golden_baseline",
                "deviation_type": "none",
                "description": f"Golden trajectory ({g['key']}) scored against itself (sanity baseline).",
                "expected": g["turns"],
                "observed": g["turns"],
                "turns_compared": turns_compared_label(g["turns"], g["turns"]),
                "doc_meta": {},
                "golden_key": g["key"],
                "availability": g["availability"],
            }
        )
        for path, doc_meta, matched in fixture_docs:
            if matched is not g:
                continue
            golden_turns = g["turns"]
            observed = load_fixture_turns(path)
            observed_ids = {t.turn_id for t in observed}
            expected = [t for t in golden_turns if t.turn_id in observed_ids] or golden_turns
            runs.append(
                {
                    "scenario": path.stem,
                    "deviation_type": doc_meta.get("deviation_type", "unknown"),
                    "description": doc_meta.get("deviation", {}).get("description", ""),
                    "expected": expected,
                    "observed": observed,
                    "turns_compared": turns_compared_label(expected, golden_turns),
                    "doc_meta": doc_meta,
                    "golden_key": g["key"],
                    "availability": declared_availability(doc_meta),
                }
            )
    return runs


def band(score: float | None) -> str:
    if score is None:
        return "na"
    if score >= 0.95:
        return "good"
    if score >= 0.80:
        return "warning"
    if score >= 0.50:
        return "serious"
    return "critical"


def fmt(score: float | None) -> str:
    return "N/A" if score is None else f"{score:.3f}"


def pct(width: float | None) -> str:
    return "100%" if width is None else f"{width * 100:.1f}%"


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

GROUP_FIELDS = [
    "golden", "scenario", "deviation_type", "description", "turns_compared",
    "nts", "sts", "tis", "rs", "os", "atf", "metric_coverage",
]

COMPONENT_FIELDS = [
    "golden", "scenario", "deviation_type", "description", "turns_compared",
    "nts_coverage", "nts_precision", "nts_recall_diag", "nts_order", "nts_overall",
    "sts_transition_accuracy", "sts_order", "sts_key_accuracy_diag",
    "sts_old_value_accuracy_diag", "sts_new_value_accuracy_diag", "sts_overall",
    "tis_coverage", "tis_precision", "tis_identity", "tis_input_similarity", "tis_order", "tis_overall",
    "rs_semantic", "rs_order", "rs_overall",
    "os_identity", "os_attribute", "os_completion", "os_overall",
    "atf", "metric_coverage",
]


def write_csvs(runs: list[dict], output_dir: Path, timestamp: str) -> tuple[Path, Path]:
    group_path = output_dir / f"atf_metrics_{timestamp}.csv"
    component_path = output_dir / f"atf_metrics_components_{timestamp}.csv"

    with group_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=GROUP_FIELDS)
        writer.writeheader()
        for run in runs:
            m = run["metrics"]
            writer.writerow({
                "golden": run.get("golden_key", ""),
                "scenario": run["scenario"], "deviation_type": run["deviation_type"],
                "description": run["description"], "turns_compared": run["turns_compared"],
                "nts": m["nts_overall"], "sts": m["sts_overall"], "tis": m["tis_overall"],
                "rs": m["rs_overall"], "os": m["os_overall"],
                "atf": m["atf"], "metric_coverage": m["metric_coverage"],
            })

    with component_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COMPONENT_FIELDS)
        writer.writeheader()
        for run in runs:
            m = run["metrics"]
            row = {k: v for k, v in m.items() if not k.startswith("_")}
            row.update({
                "golden": run.get("golden_key", ""),
                "scenario": run["scenario"], "deviation_type": run["deviation_type"],
                "description": run["description"], "turns_compared": run["turns_compared"],
            })
            writer.writerow(row)

    return group_path, component_path


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

PAGE_SHELL = Path(__file__).with_name("dashboard_shell.html").read_text(encoding="utf-8")

GLOSSARY_LABELS = ["NTS", "STS", "TIS", "RS", "OS"]

GROUP_META = [
    ("nts", "NTS", "30%", [
        ("Coverage", "nts_coverage", False),
        ("Precision", "nts_precision", False),
        ("Order", "nts_order", False),
        ("Recall", "nts_recall_diag", True),
    ]),
    ("sts", "STS", "30%", [
        ("Transition accuracy", "sts_transition_accuracy", False),
        ("Order", "sts_order", False),
        ("Key accuracy", "sts_key_accuracy_diag", True),
        ("Old-value accuracy", "sts_old_value_accuracy_diag", True),
        ("New-value accuracy", "sts_new_value_accuracy_diag", True),
    ]),
    ("tis", "TIS", "15%", [
        ("Coverage", "tis_coverage", False),
        ("Precision", "tis_precision", False),
        ("Identity", "tis_identity", False),
        ("Input similarity", "tis_input_similarity", False),
        ("Order", "tis_order", False),
    ]),
    ("rs", "RS", "10%", [
        ("Semantic (LLM judge)", "rs_semantic", False),
        ("Order", "rs_order", False),
    ]),
    ("os", "OS", "15%", [
        ("Identity", "os_identity", False),
        ("Attribute", "os_attribute", False),
        ("Completion", "os_completion", False),
    ]),
]


def render_pill(score: float | None, flagged: bool = False) -> str:
    b = band(score)
    flag = "&nbsp;&#9888;" if flagged else ""
    return f'<span class="pill {b}">{fmt(score)}{flag}</span>'


def _llm_coverage_str(run: dict) -> str:
    """Plain (non-pill) coverage count, not a quality score -- how many of the
    19 Group 1-4 metrics actually scored vs. N/A for this run. Group 5 (13
    voice metrics) is excluded from the denominator: it is expected to be
    N/A without audio/timing evidence, so folding it in would make a
    perfectly correct run look like it's missing coverage."""
    report = run.get("llm")
    if report is None:
        return '<span style="color:var(--text-faint)">off</span>'
    from atf_eval.llm_evals import BY_ID

    g14_ids = [mid for mid, spec in BY_ID.items() if spec.group != 5]
    scored = sum(
        1 for mid in g14_ids
        if (r := report.overall_results.get(mid)) is not None and r.status == "available"
    )
    return f'<span style="color:var(--text-faint)">{scored}/{len(g14_ids)}</span>'


def render_glance_row(run: dict, baseline: bool) -> str:
    m = run["metrics"]
    row_class = ' class="baseline"' if baseline else ""
    flag_os = m.get("_os_flag", False)
    return f"""
          <tr{row_class}>
            <td class="scenario-cell"><span class="name">{run['scenario']}</span><span class="dev">{run['description'][:60]}</span></td>
            <td class="num">{render_pill(m['nts_overall'])}</td>
            <td class="num">{render_pill(m['sts_overall'])}</td>
            <td class="num">{render_pill(m['tis_overall'])}</td>
            <td class="num">{render_pill(m['rs_overall'])}</td>
            <td class="num">{render_pill(m['os_overall'], flagged=flag_os)}</td>
            <td class="num">{render_pill(m['atf'], flagged=flag_os)}</td>
            <td class="num">{_llm_coverage_str(run)}</td>
          </tr>"""


def render_component_row(label: str, score: float | None, diag: bool) -> str:
    b = band(score)
    diag_tag = '<span class="diag">diagnostic</span>' if diag else ""
    color = "" if score is not None else f' style="color:var(--na)"'
    return f"""
              <div class="component-row"><span class="c-label">{label}{diag_tag}</span><span class="c-track"><span class="c-fill {b}" style="width:{pct(score)}"></span></span><span class="c-score"{color}>{fmt(score)}</span></div>"""


def render_group_meter(key: str, label: str, weight: str, components: list, m: dict, open_default: bool) -> str:
    score = m[f"{key}_overall"]
    b = band(score)
    open_attr = " open" if open_default else ""
    rows = "".join(render_component_row(c_label, m[c_key], diag) for c_label, c_key, diag in components)
    evidence = render_evidence(key, m)
    return f"""
          <details class="group-meter"{open_attr}>
            <summary>
              <span class="g-label">{label}<span class="weight">{weight}</span></span>
              <span class="meter-track"><span class="meter-fill {b}" style="width:{pct(score)}"></span></span>
              <span class="g-score" style="color:var(--{b})">{fmt(score)}</span>
            </summary>
            <div class="component-list">{rows}
            </div>{evidence}
          </details>"""


def _esc(value) -> str:
    return html_escape(value if isinstance(value, str) else json.dumps(value, default=str))


def _evidence_items(key: str, m: dict) -> list[str]:
    """The evidence responsible for a component's score (METRICS.md §8)."""
    availability = m.get("_availability", {})
    dimension = {"nts": "nodes", "sts": "state", "tis": "tools", "rs": "routing", "os": "outcome"}[key]
    status = availability.get(dimension, "available")
    if key == "os" and m.get("_exp_outcome") is None:
        return ["N/A: the compared golden turns define no outcome."]
    if status == "unavailable":
        return [f"N/A: the trace declares <code>{dimension}</code> evidence unavailable, so this is not scored as a failure."]
    d = m.get("_diagnostics", {}).get(key, {})
    items: list[str] = []
    if key == "nts":
        for label, field in (("Missing", "missing"), ("Extra", "extra"), ("Reordered", "reordered")):
            if d.get(field):
                items.append(f"{label} nodes: " + ", ".join(f"<code>{_esc(n)}</code>" for n in d[field]))
    elif key == "sts":
        if d.get("missing"):
            items.append("Missing transitions: " + ", ".join(
                f"<code>{_esc(c['key'])}: {_esc(c['old'])} &rarr; {_esc(c['new'])}</code>" for c in d["missing"]))
        for c in d.get("mismatched", []):
            items.append(
                f"Wrong transition <code>{_esc(c['key'])}</code>: expected "
                f"<code>{_esc(c['expected_old'])} &rarr; {_esc(c['expected_new'])}</code>, observed "
                f"<code>{_esc(c['observed_old'])} &rarr; {_esc(c['observed_new'])}</code>")
        if d.get("order_deviations"):
            items.append("Transitions out of expected order: " + ", ".join(
                f"<code>{_esc(c['key'])}: {_esc(c['old'])} &rarr; {_esc(c['new'])}</code>" for c in d["order_deviations"]))
        if d.get("extra"):
            items.append("Unexpected transitions: " + ", ".join(
                f"<code>{_esc(c['key'])}: {_esc(c['old'])} &rarr; {_esc(c['new'])}</code>" for c in d["extra"]))
    elif key == "tis":
        if d.get("missing"):
            items.append("Missing tools: " + ", ".join(f"<code>{_esc(t)}</code>" for t in d["missing"]))
        if d.get("extra"):
            items.append("Extra tools: " + ", ".join(f"<code>{_esc(t)}</code>" for t in d["extra"]))
        if d.get("order_deviations"):
            items.append("Tools out of expected order: " + ", ".join(f"<code>{_esc(t)}</code>" for t in d["order_deviations"]))
        for sub in d.get("substituted", []):
            items.append(f"Wrong tool: expected <code>{_esc(sub['expected'])}</code>, called <code>{_esc(sub['observed'])}</code>")
        for a in d.get("unexpected_arguments", []):
            items.append(
                f"Arguments passed to <code>{_esc(a['tool'])}</code> that golden does not specify "
                f"(input similarity compares only the expected arguments): <code>{_esc(a['arguments'])}</code>")
        for a in d.get("argument_mismatches", []):
            items.append(
                f"Arguments of <code>{_esc(a['tool'])}</code> ({a['similarity']:.2f}): expected "
                f"<code>{_esc(a['expected'])}</code>, observed <code>{_esc(a['observed'])}</code>")
    elif key == "rs":
        if status == "not_applicable":
            items.append("N/A: no routing judge configured. The judge score is a required part of RS, "
                         f"so routing order ({fmt(m.get('rs_order'))}) is shown as a diagnostic only.")
        else:
            items.append(f"Judged {d.get('judged_turns', 0)} of {d.get('applicable_turns', 0)} applicable turns.")
            for v in d.get("verdicts", []):
                items.append(f"Turn {v['turn_id']}: <strong>{_esc(v['verdict'])}</strong> &mdash; {_esc(v['rationale'])}")
    elif key == "os":
        for a in d.get("incorrect_attributes", []):
            items.append(f"Attribute <code>{_esc(a['key'])}</code>: expected <code>{_esc(a['expected'])}</code>, "
                         f"observed <code>{_esc(a['observed'])}</code>")
        for c in d.get("unmet_conditions", []):
            items.append(f"Unmet completion condition: <code>{_esc(c)}</code>")
    return items


def render_evidence(key: str, m: dict) -> str:
    items = _evidence_items(key, m)
    if not items:
        return ""
    lis = "".join(f"<li>{i}</li>" for i in items)
    return f'''
            <div class="evidence"><span class="evidence-label">Evidence</span><ul>{lis}</ul></div>'''


def primary_metric_for(deviation_type: str) -> str | None:
    """METRICS.md §11's deviation-taxonomy -> primary-metric mapping, for the
    deviation_type values a fixture's `deviation_type` field commonly uses."""
    taxonomy = {
        "correct_answer_wrong_trajectory": "NTS",
        "wrong_node_executed": "NTS",
        "node_skipped": "NTS",
        "unexpected_node_executed": "NTS",
        "wrong_node_order": "NTS",
        "wrong_state_transition": "STS",
        "missing_state_transition": "STS",
        "wrong_state_progression": "STS",
        "missing_tool_call": "TIS",
        "unexpected_tool_call": "TIS",
        "wrong_tool": "TIS",
        "wrong_tool_input": "TIS",
        "wrong_tool_sequence": "TIS",
        "wrong_conversational_path": "RS",
        "partial_incorrect_outcome": "OS",
        "correct_trajectory_wrong_outcome": "OS",
        "trajectory_drift": "NTS / RS",
        # Not a literal METRICS.md §11 row -- closest official bucket is
        # "Trajectory drift across turns", same NTS / RS mapping.
        "trajectory_drift_customer_disconnection": "NTS / RS",
    }
    return taxonomy.get(deviation_type)


def render_panel(idx: int, run: dict) -> str:
    m = run["metrics"]
    atf_band = band(m["atf"])
    chips = f'<span class="chip">{run["turns_compared"]}</span>'
    if run["deviation_type"] != "none":
        chips += f'<span class="chip">deviation: {run["deviation_type"]}</span>'
        primary = primary_metric_for(run["deviation_type"])
        if primary:
            chips += f'<span class="chip primary">METRICS.md &sect;11 primary metric: {primary}</span>'

    flag_os = m.get("_os_flag", False)
    atf_flag = "&nbsp;&#9888;" if flag_os else ""

    open_flags = {key: False for key, *_ in GROUP_META}
    # Auto-open whichever groups are not a clean 1.000 / N/A, so the reader's
    # eye lands on what actually moved for this run.
    for key, *_ in GROUP_META:
        s = m[f"{key}_overall"]
        if s is not None and s < 0.999:
            open_flags[key] = True
    if not any(open_flags.values()):
        open_flags["nts"] = True  # baseline run: open something by default

    meters = "".join(
        render_group_meter(key, label, weight, components, m, open_flags[key])
        for key, label, weight, components in GROUP_META
    )

    note = m.get("_note", "")
    note_html = f'\n          <p class="panel-note">{note}</p>' if note else ""
    llm_html = render_llm_section(run)

    desc = run["description"] or "Golden trajectory scored against itself."
    return f"""
      <section id="panel-{idx}" class="panel">
        <div class="panel-card">
          <div class="panel-header">
            <div>
              <h3>{run['scenario']}</h3>
              <p class="desc">{desc}</p>
              <div class="chips">{chips}</div>
            </div>
            <div class="atf-headline">
              <div class="label">ATF</div>
              <div class="value" style="color:var(--{atf_band})">{fmt(m['atf'])}{atf_flag}</div>
              <div class="coverage">{pct(m['metric_coverage'])} metric coverage</div>
            </div>
          </div>
          {meters}{note_html}
          {llm_html}
        </div>
      </section>"""


def annotate_notes(runs: list[dict]) -> None:
    """Attach auto-detected caveats so the dashboard explains itself rather
    than silently showing a misleading number."""
    for run in runs:
        m = run["metrics"]
        exp_outcome = m["_exp_outcome"]
        m["_os_flag"] = False  # an unrecorded outcome is N/A now, never a flagged 0
        if exp_outcome is not None and m["_availability"]["outcome"] == "unavailable":
            run["callout"] = (
                f"the observed fixture <code>{run['scenario']}.json</code> never records an "
                f"<code>outcome</code> field, so outcome evidence is declared unavailable and OS is "
                f"N/A rather than a failure against golden's <code>{exp_outcome.id}</code> "
                f"(METRICS.md &sect;6, &sect;11). ATF is renormalized over the other components; see this run's "
                f"metric coverage."
            )
        if run["deviation_type"] == "missing_tool_call" and m["tis_overall"] == 0.0 and (
            (m["nts_overall"] or 0) >= 0.999
        ) and (m["sts_overall"] or 0) >= 0.999:
            m["_note"] = (
                "<strong>This one is a clean, isolated TIS failure.</strong> NTS and STS both stay "
                "perfect &mdash; only the tool call itself is missing."
            )
        elif run["doc_meta"].get("deviation", {}).get("expected_input") == {} and m["tis_overall"] == 1.0:
            m["_note"] = (
                "<strong>Named a tool-input deviation, scores perfect &mdash; and that's correct "
                "behavior.</strong> Golden's own call here expects no arguments (<code>{}</code>), and "
                "input similarity compares the expected arguments (METRICS.md &sect;4), so with none expected there is "
                "nothing to get wrong. The fixture would need non-empty expected arguments to exercise TIS input."
            )


def render_settings_rows(settings: dict) -> str:
    """The settings recorded with the results for auditability (METRICS.md §8, §23)."""
    judge = settings["routing_judge"]
    if judge.get("configured"):
        judge_text = (
            f"{html_escape(str(judge['model']))}, effort {html_escape(str(judge['effort']))}, "
            f"max_tokens {judge['max_tokens']}, temperature {html_escape(judge['temperature'])}, "
            f"prompt <code>{html_escape(judge['system_prompt_id'])}</code>"
        )
    else:
        judge_text = "not configured (RS is N/A)"
    return (
        f"      <div><dt>Routing judge</dt><dd>{judge_text}</dd></div>\n"
        f"      <div><dt>Matching rules</dt><dd>arguments: {html_escape(settings['argument_matching'])}; "
        f"outcome identity: {html_escape(settings['outcome_identity_evaluator'])}</dd></div>\n"
        f"      <div><dt>Weights</dt><dd>ATF "
        + " / ".join(f"{k.upper()} {v:g}" for k, v in settings["weights"].items())
        + "</dd></div>\n"
    )


def render_dashboard(
    runs: list[dict], golden_summary: str, scenarios_dir: Path, run_date: str, settings: dict | None = None
) -> str:
    glance_rows = "".join(
        render_glance_row(run, baseline=(run["deviation_type"] == "none")) for run in runs
    )
    tab_inputs = "".join(
        f'\n      <input type="radio" name="stabs" id="tab-{i}" class="tab-input"{" checked" if i == 1 else ""}>'
        for i in range(1, len(runs) + 1)
    )
    tab_labels = "".join(
        f'\n        <label for="tab-{i}" class="tab-label">{run["scenario"].replace("_", " ").title()}</label>'
        for i, run in enumerate(runs, start=1)
    )
    tab_css = "\n  ".join(
        f'#tab-{i}:checked ~ .tab-bar label[for="tab-{i}"],' for i in range(1, len(runs) + 1)
    ).rstrip(",") + " { background: var(--accent); border-color: var(--accent); color: #fff; }"
    panel_css = "\n  ".join(
        f'#tab-{i}:checked ~ #panel-{i},' for i in range(1, len(runs) + 1)
    ).rstrip(",") + " { display: block; }"
    focus_css = "\n  ".join(
        f'#tab-{i}:focus-visible ~ .tab-bar label[for="tab-{i}"],' for i in range(1, len(runs) + 1)
    ).rstrip(",") + " { outline: 2px solid var(--accent); outline-offset: 2px; }"

    panels = "".join(render_panel(i, run) for i, run in enumerate(runs, start=1))

    any_routing = any(r["metrics"]["rs_overall"] is not None for r in runs)
    notes_parts = [
        f"<p>For <code>{r['scenario']}</code>, {r['callout']}</p>" for r in runs if r.get("callout")
    ]
    if not any_routing:
        judge_off = all(r["metrics"]["_availability"]["routing"] == "not_applicable" for r in runs)
        reason = (
            "no routing judge was configured for this run. The judge score is a required part of RS "
            "(METRICS.md &sect;5), so routing order alone is not reported as RS &mdash; it is shown in each "
            "panel's RS evidence instead."
            if judge_off
            else "no judged routing decisions were available."
        )
        notes_parts.append(f"<p><strong>Routing Similarity (RS)</strong> is N/A across every run: {reason}</p>")
    notes_html = "".join(notes_parts) or "<p>No scoring caveats detected for this run set.</p>"

    html = PAGE_SHELL
    html = html.replace("{{GOLDEN_PATH}}", golden_summary)
    html = html.replace("{{SCENARIOS_PATH}}", str(scenarios_dir))
    html = html.replace("{{RUN_DATE}}", run_date)
    html = html.replace("{{SETTINGS_ROWS}}", render_settings_rows(settings) if settings else "")
    html = html.replace("{{TAB_INPUTS}}", tab_inputs)
    html = html.replace("{{TAB_LABELS}}", tab_labels)
    html = html.replace("{{TAB_ACTIVE_CSS}}", tab_css)
    html = html.replace("{{PANEL_ACTIVE_CSS}}", panel_css)
    html = html.replace("{{TAB_FOCUS_CSS}}", focus_css)
    html = html.replace("{{GLANCE_ROWS}}", glance_rows)
    html = html.replace("{{PANELS}}", panels)
    html = html.replace("{{NOTES}}", notes_html)
    return html


# ---------------------------------------------------------------------------
# LLM / multimodal evaluation (METRICS.md Part B, Groups 1-5)
# ---------------------------------------------------------------------------


def run_llm_evals(runs: list[dict], client, model: str, effort: str | None) -> None:
    """Attach a ConversationLLMReport to each run under run['llm']. Runs are
    processed a few at a time and each conversation fans its own judge calls
    out to a thread pool, so the first full build is I/O-bound rather than
    serial; every subsequent build is served from the on-disk cache."""
    from concurrent.futures import ThreadPoolExecutor

    from atf_eval.llm_evals import build_context, evaluate_conversation

    def _one(run: dict) -> None:
        ctx = build_context(run["scenario"], run["expected"], run["observed"])
        run["llm"] = evaluate_conversation(
            ctx, client, model=model, effort=effort, concurrency=6
        )

    if client is None or len(runs) <= 1:
        for run in runs:
            _one(run)
        return
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(_one, runs))


_LLM_GROUP_TITLES = {
    1: "Group 1 - Response Quality",
    2: "Group 2 - Grounding & Knowledge",
    3: "Group 3 - Safety, Policy & Compliance",
    4: "Group 4 - Conversation Quality",
    5: "Group 5 - Voice / Multimodal",
}


def llm_native_str(scoring_type: str, score) -> str:
    if score is None:
        return "N/A"
    if scoring_type in ("ordinal_1_5", "ordinal_1_5_na"):
        return f"{score}/5"
    if scoring_type == "pass_fail":
        return str(score)
    if scoring_type == "sentiment":
        return f"{score:+d}" if isinstance(score, int) else str(score)
    if scoring_type == "count_severity":
        return f"{score.get('severity','?')} ({score.get('count','?')})" if isinstance(score, dict) else str(score)
    if scoring_type == "count":
        return str(score)
    if scoring_type == "categorical_confidence":
        if isinstance(score, dict):
            conf = score.get("confidence")
            return f"{score.get('category','?')}" + (f" ({conf:.2f})" if isinstance(conf, (int, float)) else "")
        return str(score)
    return str(score)


def _llm_band(normalized: float | None, status: str) -> str:
    if status != "available" or normalized is None:
        return "na"
    if normalized >= 0.95:
        return "good"
    if normalized >= 0.80:
        return "warning"
    if normalized >= 0.50:
        return "serious"
    return "critical"


def write_llm_csv(runs: list[dict], output_dir: Path, timestamp: str) -> Path | None:
    if not any("llm" in r for r in runs):
        return None
    path = output_dir / f"atf_llm_metrics_{timestamp}.csv"
    fields = [
        "golden", "scenario", "deviation_type", "group", "metric_id", "metric_name",
        "level", "scoring_type", "native_score", "normalized_0_1", "status", "reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            report = run.get("llm")
            if report is None:
                continue
            rows = []
            for mid, results in report.turn_results.items():
                rows.extend(results)
            rows.extend(report.overall_results.values())
            for r in rows:
                writer.writerow({
                    "golden": run.get("golden_key", ""),
                    "scenario": run["scenario"],
                    "deviation_type": run["deviation_type"],
                    "group": r.group,
                    "metric_id": r.metric_id + (f"@turn{r.turn_id}" if r.turn_id is not None else ""),
                    "metric_name": r.metric_name,
                    "level": r.level,
                    "scoring_type": r.scoring_type,
                    "native_score": "" if r.score is None else json.dumps(r.score) if isinstance(r.score, dict) else r.score,
                    "normalized_0_1": "" if r.normalized is None else round(r.normalized, 4),
                    "status": r.status,
                    "reason": r.reason,
                })
    return path


def render_llm_metric_row(overall, per_turn: list) -> str:
    native = llm_native_str(overall.scoring_type, overall.score)
    band = _llm_band(overall.normalized, overall.status)
    status_tag = "" if overall.status == "available" else f'<span class="lm-status">{overall.status.replace("_", " ")}</span>'
    turn_rows = ""
    shown_turns = [t for t in per_turn if t.turn_id is not None]
    if shown_turns:
        items = "".join(
            f'<div class="lm-turn"><span class="lm-turn-id">turn {t.turn_id}</span>'
            f'<span class="lm-turn-score {_llm_band(t.normalized, t.status)}">{llm_native_str(t.scoring_type, t.score)}</span>'
            f'<span class="lm-turn-reason">{(t.reason or "")[:400]}</span></div>'
            for t in shown_turns
        )
        turn_rows = f'<div class="lm-turns">{items}</div>'
    return f"""
            <details class="llm-metric">
              <summary>
                <span class="lm-name">{overall.metric_name}<span class="lm-level">{overall.level}</span></span>
                <span class="pill {band}">{native}</span>
                {status_tag}
              </summary>
              <p class="lm-reason">{(overall.reason or "")[:600]}</p>
              {turn_rows}
            </details>"""


def _llm_all_specs():
    from atf_eval.llm_evals import ALL_METRICS
    return ALL_METRICS


def render_llm_section(run: dict) -> str:
    """The 'LLM & Multimodal Evaluation' block embedded inside a run's own
    panel (single dashboard -- no separate tab system for the LLM track)."""
    report = run.get("llm")
    if report is None:
        return ""

    groups_html = ""
    total_scored = total_specs = 0
    for g in (1, 2, 3, 4, 5):
        specs = [s for s in _llm_all_specs() if s.group == g]
        rows = ""
        scored_ct = 0
        for spec in specs:
            overall = report.overall_results.get(spec.metric_id)
            if overall is None:
                continue
            per_turn = report.turn_results.get(spec.metric_id, [])
            rows += render_llm_metric_row(overall, per_turn)
            if overall.status == "available":
                scored_ct += 1
        total_scored += scored_ct
        total_specs += len(specs)
        na_note = ""
        if g == 5 and scored_ct == 0:
            na_note = ('<p class="llm-group-note">All Group 5 metrics are N/A for this run: '
                       'no audio or timing evidence in this trajectory (METRICS.md &sect;18). '
                       'The evaluators are implemented and will score the moment audio/timing '
                       'evidence is supplied.</p>')
        groups_html += f"""
          <details class="llm-group"{' open' if g != 5 else ''}>
            <summary><span class="lg-label">{_LLM_GROUP_TITLES[g]}</span><span class="lg-count">{scored_ct}/{len(specs)} scored</span></summary>
            {na_note}{rows}
          </details>"""

    return f"""
          <div class="llm-section">
            <div class="llm-section-head">
              <h4>LLM &amp; Multimodal Evaluation<span class="count">METRICS.md Part B</span></h4>
              <span class="llm-coverage">{total_scored}/{total_specs} metrics scored</span>
            </div>
            {groups_html}
          </div>"""


# ---------------------------------------------------------------------------
# CLI wiring -- add_arguments()/run() are shared by the `atf-eval dashboard`
# subcommand (cli.py) and this module's own standalone `main()`.
# ---------------------------------------------------------------------------


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--golden-dir",
        default="golden",
        help="Directory of golden *.json files, one per golden scenario (default: ./golden).",
    )
    parser.add_argument(
        "--scenarios",
        default=str(Path("tests") / "fixtures" / "scenarios"),
        help="Directory of observed-fixture *.json files (default: ./tests/fixtures/scenarios).",
    )
    parser.add_argument("--output-dir", default="results", help="Default: ./results")
    parser.add_argument("--judge-model", default="claude-opus-5",
                        help="Anthropic model for RS's routing judge and the Group 1-5 LLM metrics.")
    parser.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], default="low",
                        help="Judge model effort (default: low -- fine for rubric scoring).")
    parser.add_argument("--no-routing-judge", action="store_true",
                        help="Skip RS's LLM routing judge -- RS reports N/A instead.")
    parser.add_argument("--no-llm-evals", action="store_true",
                        help="Skip the Group 1-5 LLM/multimodal metrics (and the LLM section of each panel).")
    parser.add_argument("--weights-file", default=None,
                        help='JSON override of the ATF component weights {"nts", "sts", "tis", "rs", "os"} '
                             "(must sum to 1; default: METRICS.md §7).")


def run(args: argparse.Namespace) -> None:
    golden_dir = Path(args.golden_dir)
    scenarios_dir = Path(args.scenarios)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # One judge client for both RS and the Group 1-5 metrics. None -> those
    # dimensions report N/A (never 0); every deterministic metric is unaffected.
    judge_client = None
    if not (args.no_routing_judge and args.no_llm_evals):
        from atf_eval.llm_evals import get_judge_client
        judge_client = get_judge_client()
        if judge_client is None:
            print("[info] no ANTHROPIC_API_KEY (checked env + a repo-root .env) -- RS and Group 1-5 "
                  "LLM metrics will report N/A", file=sys.stderr)

    rs_client = None if args.no_routing_judge else judge_client

    weights_config = None
    if args.weights_file:
        weights_config = json.loads(Path(args.weights_file).read_text(encoding="utf-8"))
    weights, subweights = resolve_weights(weights_config)

    if not golden_dir.exists():
        raise SystemExit(
            f"Golden directory not found: {golden_dir}\n"
            "Pass --golden-dir, or run from a directory that has a ./golden folder."
        )
    goldens = discover_goldens(golden_dir)
    if not goldens:
        raise SystemExit(f"No golden *.json files found in {golden_dir}")
    if not scenarios_dir.exists():
        raise SystemExit(
            f"Scenarios directory not found: {scenarios_dir}\n"
            "Pass --scenarios, or run from a directory that has a ./tests/fixtures/scenarios folder."
        )
    runs = discover_runs(goldens, scenarios_dir)
    for run_ in runs:
        run_["metrics"] = score_components(
            run_["expected"], run_["observed"], rs_client, args.judge_model, args.effort,
            declared=run_["availability"], weights=weights, subweights=subweights,
        )
    annotate_notes(runs)

    if not args.no_llm_evals:
        print("[info] running Group 1-5 LLM/multimodal metrics "
              f"({'judge=' + args.judge_model if judge_client else 'no client -> all N/A'}) ...",
              file=sys.stderr)
        run_llm_evals(runs, judge_client, args.judge_model, args.effort)

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")

    group_csv, component_csv = write_csvs(runs, output_dir, timestamp)
    golden_summary = ", ".join(f"{g['key']} ({g['path'].name})" for g in goldens)
    # Single dashboard: render_dashboard()/render_panel() embed each run's LLM
    # Groups 1-5 results (if any) directly into that run's own panel, so
    # there is exactly one HTML file to publish, not a deterministic/LLM pair.
    from atf_eval.settings import evaluation_settings

    settings = evaluation_settings(
        args.judge_model if rs_client else None, args.effort, weights=weights, subweights=subweights
    )
    settings_path = output_dir / f"atf_settings_{timestamp}.json"
    settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    dashboard_html = render_dashboard(runs, golden_summary, scenarios_dir, run_date, settings)
    dashboard_path = output_dir / f"atf_dashboard_{timestamp}.html"
    dashboard_path.write_text(dashboard_html, encoding="utf-8")

    print("Wrote:")
    print(f"  {group_csv}")
    print(f"  {component_csv}")
    print(f"  {dashboard_path}")
    print(f"  {settings_path}")

    llm_csv = write_llm_csv(runs, output_dir, timestamp)
    if llm_csv is not None:
        print(f"  {llm_csv}")

    for run_ in runs:
        m = run_["metrics"]
        print(f"[{run_.get('golden_key', '')}] {run_['scenario']}: ATF={fmt(m['atf'])} NTS={fmt(m['nts_overall'])} "
              f"STS={fmt(m['sts_overall'])} TIS={fmt(m['tis_overall'])} "
              f"RS={fmt(m['rs_overall'])} OS={fmt(m['os_overall'])}")
        if "llm" in run_:
            report = run_["llm"]
            avail = sum(1 for r in report.all_results() if r.status == "available")
            print(f"    LLM: {avail} metric-results scored, "
                  f"{sum(1 for r in report.overall_results.values() if r.status == 'available')}/"
                  f"{len(report.overall_results)} overall metrics")


def main() -> None:
    """Standalone entry point (`python -m atf_eval.dashboard.generate ...`).
    The `atf-eval dashboard` CLI subcommand (cli.py) calls add_arguments()/
    run() directly instead of going through this function."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
