from atf_eval.metrics.state import sts_turn
from atf_eval.normalized import NormalizedNode, NormalizedTurn, StateChange


def _turn(node_id: str, changes: list[tuple[str, object, object]]) -> NormalizedTurn:
    return NormalizedTurn(
        conversation_id="c",
        turn_id=1,
        nodes=[
            NormalizedNode(
                node_id=node_id,
                state_changes=[StateChange(key=k, old=o, new=n) for k, o, n in changes],
            )
        ],
    )


def test_perfect_match_is_one():
    expected = _turn("hardship_node", [("hardship_flag", False, True)])
    observed = _turn("hardship_node", [("hardship_flag", False, True)])
    assert sts_turn(expected, observed) == 1.0


def test_wrong_new_value_is_not_a_full_match():
    expected = _turn("hardship_node", [("hardship_flag", False, True)])
    observed = _turn("hardship_node", [("hardship_flag", False, False)])
    score = sts_turn(expected, observed)
    assert score is not None
    assert score < 1.0


def test_missing_transition_denominator_is_max_not_len_expected():
    # METRICS.md: Transition Accuracy denominator = max(expected, observed),
    # not len(expected) -- this is the exact rule the prior implementation
    # got wrong before the METRICS.md alignment rewrite.
    expected = _turn("n", [("a", 0, 1), ("b", 0, 1)])
    observed = _turn("n", [("a", 0, 1)])
    score = sts_turn(expected, observed)
    assert score is not None
    assert score < 1.0


def test_no_state_changes_either_side_is_na():
    expected = NormalizedTurn(conversation_id="c", turn_id=1)
    observed = NormalizedTurn(conversation_id="c", turn_id=1)
    assert sts_turn(expected, observed) is None
