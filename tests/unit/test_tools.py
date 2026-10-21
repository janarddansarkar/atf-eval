from atf_eval.metrics.tools import argument_match, tis_turn
from atf_eval.normalized import NormalizedTurn, ToolCall


def _turn(calls: list[tuple[str, dict]]) -> NormalizedTurn:
    return NormalizedTurn(
        conversation_id="c",
        turn_id=1,
        tool_calls=[ToolCall(tool_id=tid, arguments=args) for tid, args in calls],
    )


def test_perfect_match_is_one():
    expected = _turn([("eligibility_check", {"amount": 100})])
    observed = _turn([("eligibility_check", {"amount": 100})])
    assert tis_turn(expected, observed) == 1.0


def test_missing_tool_call_lowers_coverage():
    expected = _turn([("eligibility_check", {})])
    observed = _turn([])
    score = tis_turn(expected, observed)
    assert score is not None
    assert score < 1.0


def test_wrong_tool_input_lowers_input_similarity_not_identity():
    expected = _turn([("eligibility_check", {"amount": 100})])
    observed = _turn([("eligibility_check", {"amount": 999})])
    score = tis_turn(expected, observed)
    assert score is not None
    assert 0.0 < score < 1.0


def test_argument_match_no_expected_args_is_full_credit():
    assert argument_match({}, {"anything": "here"}) == 1.0


def test_argument_match_respects_tolerance():
    assert argument_match({"amount": 100.0}, {"amount": 100.4}, tolerance=0.5) == 1.0
    assert argument_match({"amount": 100.0}, {"amount": 100.4}, tolerance=0.0) == 0.0


def test_no_tool_calls_either_side_is_na():
    expected = NormalizedTurn(conversation_id="c", turn_id=1)
    observed = NormalizedTurn(conversation_id="c", turn_id=1)
    assert tis_turn(expected, observed) is None
