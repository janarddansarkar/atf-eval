from atf_eval.metrics.outcome import last_outcome, os_conversation
from atf_eval.normalized import NormalizedTurn, Outcome


def _turns_with_outcome(outcome: Outcome | None) -> list[NormalizedTurn]:
    return [NormalizedTurn(conversation_id="c", turn_id=1, outcome=outcome)]


def test_perfect_match_is_one():
    outcome = Outcome(
        id="installment_arrangement_accepted",
        attributes={"installment_amount": 3000},
        required_conditions=["installment_amount"],
    )
    expected = _turns_with_outcome(outcome)
    observed = _turns_with_outcome(
        Outcome(id="installment_arrangement_accepted", attributes={"installment_amount": 3000})
    )
    assert os_conversation(expected, observed) == 1.0


def test_wrong_outcome_id_is_zero_identity():
    expected = _turns_with_outcome(Outcome(id="a", attributes={}))
    observed = _turns_with_outcome(Outcome(id="b", attributes={}))
    assert os_conversation(expected, observed) == 0.0


def test_missing_observed_outcome_is_zero_not_na():
    expected = _turns_with_outcome(Outcome(id="a", attributes={"x": 1}, required_conditions=["x"]))
    observed = _turns_with_outcome(None)
    assert os_conversation(expected, observed) == 0.0


def test_no_expected_outcome_anywhere_is_na():
    expected = _turns_with_outcome(None)
    observed = _turns_with_outcome(None)
    assert os_conversation(expected, observed) is None


def test_last_outcome_picks_final_non_null():
    turns = [
        NormalizedTurn(conversation_id="c", turn_id=1, outcome=None),
        NormalizedTurn(conversation_id="c", turn_id=2, outcome=Outcome(id="final")),
        NormalizedTurn(conversation_id="c", turn_id=3, outcome=None),
    ]
    assert last_outcome(turns).id == "final"
