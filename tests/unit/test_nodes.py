from atf_eval.metrics.nodes import node_coverage, nts_conversation, nts_turn
from atf_eval.normalized import NormalizedNode, NormalizedTurn


def _turn(node_ids: list[str]) -> NormalizedTurn:
    return NormalizedTurn(
        conversation_id="c",
        turn_id=1,
        nodes=[NormalizedNode(node_id=n) for n in node_ids],
    )


def test_perfect_match_is_one():
    expected = _turn(["a", "b", "c"])
    observed = _turn(["a", "b", "c"])
    assert nts_turn(expected, observed) == 1.0


def test_missing_node_lowers_coverage_and_order():
    expected = _turn(["a", "b", "c"])
    observed = _turn(["a", "c"])
    score = nts_turn(expected, observed)
    assert score is not None
    assert score < 1.0


def test_extra_node_lowers_precision_not_coverage():
    expected = _turn(["a", "b"])
    observed = _turn(["a", "b", "extra"])
    # coverage=1.0 (both expected nodes matched), precision=2/3, order=2/3
    score = nts_turn(expected, observed)
    assert score is not None
    assert 0.0 < score < 1.0


def test_reordered_nodes_hurts_order_only():
    expected = _turn(["a", "b", "c"])
    observed = _turn(["c", "b", "a"])
    score = nts_turn(expected, observed)
    # matched nodes come from the LCS alignment (METRICS.md §2), so coverage,
    # precision and order all drop
    assert score is not None
    assert score < 1.0


def test_no_expected_nodes_makes_coverage_na_but_not_the_whole_metric():
    # Coverage alone is N/A (nothing was expected, so nothing to "cover"),
    # but precision/order are still meaningful (an unexpected node showed up)
    # and the composite renormalizes around just those -- 0.0, not N/A.
    expected = _turn([])
    observed = _turn(["a"])
    assert node_coverage([], ["a"]) is None
    assert nts_turn(expected, observed) == 0.0


def test_both_sides_empty_is_na():
    expected = _turn([])
    observed = _turn([])
    assert nts_turn(expected, observed) is None


def test_conversation_uses_full_concatenated_trajectory_not_turn_average():
    expected_turns = [_turn(["a"]), _turn(["b"])]
    observed_turns = [_turn(["a"]), _turn([])]
    # per-turn: turn 1 perfect (1.0), turn 2 has no observed nodes (N/A) --
    # a naive turn-average would report 1.0. The concatenated computation
    # sees expected=[a,b], observed=[a] and must show the miss.
    score = nts_conversation(expected_turns, observed_turns)
    assert score is not None
    assert score < 1.0
