"""Behaviour pinned to agent-eval/METRICS.md §2-§8 and the Frozen Rules (§9),
including the spec's own §7 renormalization example."""
from __future__ import annotations

import pytest

from atf_eval.aggregate import atf_score, resolve_weights
from atf_eval.availability import resolve_availability
from atf_eval.metrics.nodes import node_diagnostics, nts_conversation, nts_turn
from atf_eval.metrics.outcome import os_conversation, outcome_completion
from atf_eval.metrics.routing import routing_order_similarity, rs_conversation
from atf_eval.metrics.state import sts_conversation, transition_accuracy, transition_order_similarity
from atf_eval.metrics.tools import (
    tis_conversation,
    tool_coverage,
    tool_diagnostics,
    tool_identity_accuracy,
    tool_input_similarity,
    tool_precision,
)
from atf_eval.normalized import NormalizedNode, NormalizedTurn, Outcome, Routing, StateChange, ToolCall
from atf_eval.runner import score_trajectory


def _nodes(ids, turn_id=1):
    return NormalizedTurn("c", turn_id, nodes=[NormalizedNode(node_id=i) for i in ids])


def _node(node_id, changes=(), tool_ids=()):
    return NormalizedNode(
        node_id=node_id,
        state_changes=[StateChange(key=k, old=o, new=v) for k, o, v in changes],
        tool_calls=[ToolCall(tool_id=t) for t in tool_ids],
    )


def _tools(ids, turn_id=1, args=None):
    return NormalizedTurn(
        "c", turn_id, tool_calls=[ToolCall(tool_id=i, arguments=dict((args or {}).get(i, {}))) for i in ids]
    )


# --- §2 NTS -----------------------------------------------------------------

def test_nts_matched_nodes_come_from_the_position_aware_sequential_alignment():
    # (A,B) vs (B,A): the LCS alignment matches one node -> C = P = O = 0.5
    assert nts_turn(_nodes(["A", "B"]), _nodes(["B", "A"])) == pytest.approx(0.5)


def test_nts_skip_does_not_cascade_into_later_nodes():
    # one skipped node: C = 3/4, P = 1, O = 3/4
    assert nts_turn(_nodes(["a", "b", "c", "d"]), _nodes(["a", "c", "d"])) == pytest.approx(
        0.40 * 0.75 + 0.30 * 1.0 + 0.30 * 0.75
    )


def test_nts_diagnostics_preserve_missing_extra_and_reordered():
    # Frozen Rule 5
    diag = node_diagnostics(["A", "B", "C"], ["B", "A", "X"])
    assert diag["missing"] == ["C"] and diag["extra"] == ["X"] and diag["reordered"] == ["A"]


def test_nts_na_when_node_evidence_unavailable():
    assert nts_conversation([_nodes(["A"])], [_nodes([])], available=False) is None


# --- §3 STS -----------------------------------------------------------------

def _state_turn(changes, turn_id=1, node="n"):
    return NormalizedTurn("c", turn_id, nodes=[_node(node, changes)])


def test_sts_transition_sequence_is_the_sequence_of_full_transitions():
    expected = [_state_turn([("s", "a", "b"), ("s", "b", "c")])]
    observed = [_state_turn([("s", "a", "c"), ("s", "c", "b")])]
    assert transition_order_similarity(expected, observed) == 0.0


def test_sts_compares_state_within_corresponding_aligned_nodes():
    # the LCS node alignment pairs one of the two reordered nodes; the other
    # node's transition is unmatched on both sides
    expected = [NormalizedTurn("c", 1, nodes=[_node("A", [("x", 0, 1)]), _node("B", [("y", 0, 1)])])]
    observed = [NormalizedTurn("c", 1, nodes=[_node("B", [("y", 0, 1)]), _node("A", [("x", 0, 1)])])]
    assert transition_accuracy(expected, observed) == pytest.approx(0.5)


def test_sts_repeated_node_skip_does_not_shift_later_pairings():
    expected = [NormalizedTurn("c", i, nodes=[_node("H", [(f"k{i}", 0, 1)])]) for i in (1, 2, 3)]
    observed = [NormalizedTurn("c", 1, nodes=[])] + [
        NormalizedTurn("c", i, nodes=[_node("H", [(f"k{i}", 0, 1)])]) for i in (2, 3)
    ]
    assert transition_accuracy(expected, observed) == pytest.approx(2 / 3)


def test_sts_is_evaluated_from_node_level_state_changes():
    # Frozen Rule 6: turn-level state on a turn that has nodes is not evaluated
    turn = NormalizedTurn("c", 1, nodes=[_node("n")], state_changes=[StateChange("k", 0, 1)])
    assert sts_conversation([turn], [turn]) is None


def test_sts_unchanged_state_variables_are_not_transitions():
    assert sts_conversation([_state_turn([("a", 1, 1)])], [_state_turn([])]) is None


def test_sts_na_when_state_unavailable_and_loss_when_skipped():
    expected = [_state_turn([("a", 0, 1)])]
    observed = [_state_turn([])]
    assert sts_conversation(expected, observed, available=False) is None
    assert sts_conversation(expected, observed) == 0.0


def test_sts_values_compare_consistently_in_accuracy_and_order():
    def t(old, new):
        return [_state_turn([("amount", old, new)])]

    assert transition_accuracy(t(0, 3000), t(0, 3000.0)) == 1.0
    assert transition_order_similarity(t(0, 3000), t(0, 3000.0)) == 1.0
    assert transition_accuracy(t(False, True), t(0, 1)) == 0.0
    assert transition_order_similarity(t(False, True), t(0, 1)) == 0.0


# --- §4 TIS -----------------------------------------------------------------

def test_tis_matched_tools_and_aligned_pairs():
    expected = [_tools(["verify", "eligibility", "plan"])]
    observed = [_tools(["eligibility", "plan"])]
    assert tool_coverage(expected, observed) == pytest.approx(2 / 3)
    assert tool_precision(expected, observed) == 1.0
    assert tool_identity_accuracy(expected, observed) == 1.0  # a missing call does not misalign later ones
    assert tis_conversation(expected, observed) == pytest.approx(
        0.25 * 2 / 3 + 0.20 * 1 + 0.25 * 1 + 0.20 * 1 + 0.10 * 2 / 3
    )


def test_tis_wrong_tool_lowers_identity():
    expected = [_tools(["a", "b", "c"])]
    observed = [_tools(["a", "x", "c"])]
    assert tool_identity_accuracy(expected, observed) == pytest.approx(2 / 3)
    assert tool_diagnostics(expected, observed)["substituted"] == [{"expected": "b", "observed": "x"}]


def test_tis_input_averages_matched_tool_inputs_and_can_be_partial():
    expected = [_tools(["a"], args={"a": {"amount": 3000, "months": 3}})]
    observed = [_tools(["a"], args={"a": {"amount": 5000, "months": 3}})]
    assert tool_input_similarity(expected, observed) == 0.5


def test_tis_na_only_without_tool_information():
    assert tis_conversation([_tools([])], [_tools([])]) is None
    assert tis_conversation([_tools([])], [_tools(["unexpected"])]) == 0.0  # unexpected tool call


def test_tis_aligns_within_corresponding_nodes_when_association_is_supported():
    def turn(tool):
        return NormalizedTurn("c", 1, nodes=[_node("n", tool_ids=[tool])])

    assert tool_diagnostics([turn("x")], [turn("y")])["scope"] == "node"
    assert tool_identity_accuracy([turn("x")], [turn("y")]) == 0.0


# --- §5 RS ------------------------------------------------------------------

def _routed(path, target, turn_id):
    return NormalizedTurn("c", turn_id, routing=Routing(path=path, target=target))


def test_rs_formula_and_order_over_path_decisions():
    expected = [_routed("hardship", "h", 1), _routed("plan", "p", 2)]
    observed = [_routed("hardship", "h", 1), _routed("pay_now", "p", 2)]
    assert routing_order_similarity(expected, observed) == 0.5
    assert rs_conversation(expected, observed, [1.0, 0.5]) == pytest.approx(0.7 * 0.75 + 0.3 * 0.5)


def test_rs_na_when_routing_cannot_be_judged():
    turns = [_routed("hardship", "h", 1)]
    assert rs_conversation(turns, turns, [None]) is None
    assert resolve_availability(None, judge_configured=False)["routing"] == "not_applicable"


# --- §6 OS ------------------------------------------------------------------

def _outcome_turns(outcome):
    return [NormalizedTurn("c", 1, outcome=outcome)]


def test_os_formula_with_required_outcome():
    expected = Outcome(id="plan", attributes={"amount": 3000, "freq": "monthly"}, required_conditions=["amount"])
    observed = Outcome(id="plan", attributes={"amount": 3000, "freq": "weekly"})
    assert outcome_completion(expected, observed) == 1.0
    assert os_conversation(_outcome_turns(expected), _outcome_turns(observed)) == pytest.approx(
        0.50 * 1 + 0.30 * 0.5 + 0.20 * 1
    )


def test_os_na_when_outcome_information_is_missing():
    expected = _outcome_turns(Outcome(id="plan", attributes={"amount": 1}))
    assert os_conversation(_outcome_turns(None), _outcome_turns(None)) is None
    assert os_conversation(expected, _outcome_turns(None), available=False) is None
    # outcome evidence exists but the agent produced none -> failure (§21)
    assert os_conversation(expected, _outcome_turns(None)) == 0.0


# --- §7 ATF, §8 results -----------------------------------------------------

def test_atf_renormalization_matches_the_spec_example():
    scores = {"nts": 0.90, "sts": 0.80, "tis": None, "rs": 0.70, "os": 1.00}
    atf, coverage = atf_score(scores, resolve_weights(None)[0])
    assert atf == pytest.approx((0.30 * 0.90 + 0.30 * 0.80 + 0.10 * 0.70 + 0.15 * 1.00) / 0.85)
    assert coverage == pytest.approx(0.85)


def test_only_atf_component_weights_are_overridable():
    with pytest.raises(ValueError):
        resolve_weights({"nts": 0.5, "sts": 0.5, "tis": 0.5, "rs": 0.1, "os": 0.1})
    with pytest.raises(ValueError):
        resolve_weights({"subweights": {"nts": {"coverage": 1.0, "precision": 0.0, "order": 0.0}}})


def test_every_metric_result_exposes_the_required_fields():
    turns = [_nodes(["A"])]
    scored = score_trajectory(turns, turns)
    for result in scored.metric_results.values():
        for field in ("metric_id", "score", "status", "applicability", "coverage", "diagnostics"):
            assert hasattr(result, field)
    assert scored.metric_results["sts"].status == "not_applicable"


def test_reports_record_evaluation_settings(tmp_path):
    import json

    from atf_eval.report import write_reports
    from atf_eval.settings import evaluation_settings

    json_path, _ = write_reports([], str(tmp_path), evaluation_settings("claude-opus-5", "low", tolerance=0.5))
    recorded = json.loads(json_path.read_text())["settings"]
    assert recorded["routing_judge"]["model"] == "claude-opus-5"
    assert recorded["routing_judge"]["system_prompt_id"].startswith("sha256:")
    assert "0.5" in recorded["argument_matching"]


def test_rs_order_is_na_when_the_trace_exposes_no_routing_field():
    # §5: RS does not require every source trace to expose a routing field
    expected = [_routed("hardship", "h", 1)]
    observed = [NormalizedTurn("c", 1)]
    assert routing_order_similarity(expected, observed) is None
    assert rs_conversation(expected, observed, [1.0]) == 1.0  # semantic judgment alone


def test_diagnostics_include_matched_elements_and_order_deviations():
    # §8: matched, missing, extra and order deviations are retained
    from atf_eval.metrics.state import state_diagnostics

    expected = [NormalizedTurn("c", 1, nodes=[_node("n", [("a", 0, 1), ("b", 0, 1)])])]
    observed = [NormalizedTurn("c", 1, nodes=[_node("n", [("b", 0, 1), ("a", 0, 1)])])]
    diag = state_diagnostics(expected, observed)
    assert len(diag["matched"]) == 2 and len(diag["order_deviations"]) == 1

    tdiag = tool_diagnostics([_tools(["x", "y"])], [_tools(["y", "x"])])
    assert tdiag["order_deviations"] and "matched" in tdiag


# --- Part B: LLM metric specs transcribed verbatim --------------------------

def _metrics_md():
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "agent-eval" / "METRICS.md"
    if not path.exists():
        pytest.skip("agent-eval/METRICS.md not present locally")
    return path.read_text(encoding="utf-8")


def test_llm_metric_specs_match_metrics_md_verbatim():
    import re

    from atf_eval.llm_evals.base import Level, ScoringType
    from atf_eval.llm_evals.registry import ALL_METRICS

    md = _metrics_md()
    part_b = md[md.index("# Part B"):md.index("# Part C")]
    rows = {
        m.group(1).strip(): [c.strip() for c in m.groups()[1:]]
        for m in re.finditer(r"^\| \*\*([^*]+)\*\* \| ([^|]+) \| ([^|]+) \| ([^|]+) \| ([^|]+) \| ([^|]+) \|", md, re.M)
    }
    bullets = {
        m.group(1).strip(): [line[2:].strip() for line in m.group(2).strip().splitlines()]
        for m in re.finditer(r"\*\*([^*]+)\*\*\n((?:- .*\n)+)", part_b)
    }
    levels = {"Turn": Level.TURN, "Overall": Level.OVERALL, "Turn + Overall": Level.BOTH}
    scoring = {
        "**1–5**": {ScoringType.ORDINAL_1_5}, "**Pass / Fail / N/A**": {ScoringType.PASS_FAIL},
        "**1–5 / N/A**": {ScoringType.ORDINAL_1_5_NA}, "**Count**": {ScoringType.COUNT},
        "**Count + categorical severity**": {ScoringType.COUNT_SEVERITY}, "**-2 to +2**": {ScoringType.SENTIMENT},
        "**Categorical + confidence**": {ScoringType.CATEGORICAL_CONFIDENCE},
    }
    assert {s.name for s in ALL_METRICS} == set(rows) and len(ALL_METRICS) == 32
    for spec in ALL_METRICS:
        definition, level, _evaluator, score_type, na = rows[spec.name]
        assert spec.definition == definition, spec.name
        assert spec.level == levels[level], spec.name
        assert spec.scoring_type in scoring[score_type], spec.name
        assert spec.na_rule == na + ".", spec.name
        if spec.name in bullets:
            want = {}
            for line in bullets[spec.name]:
                key, text = (x.strip() for x in line.split("—", 1))
                want[int(key) if key.isdigit() else key] = text
            assert spec.rubric == want, spec.name
