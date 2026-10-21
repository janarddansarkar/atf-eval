import json
import math

import pytest

from atf_eval.aggregate import (
    ATF_EPSILON,
    DEFAULT_WEIGHTS,
    atf_score,
    load_weights_file,
    parse_weights_spec,
    resolve_weights,
    weighted_composite,
    weights_from_cli,
)


def test_weighted_composite_excludes_none_not_zero():
    # STS missing entirely (None) must renormalize the remaining weights,
    # not drag the composite down as if STS scored 0.
    components = {"nts": 1.0, "sts": None}
    weights = {"nts": 0.5, "sts": 0.5}
    assert weighted_composite(components, weights) == 1.0


def test_weighted_composite_all_none_is_none():
    assert weighted_composite({"nts": None, "sts": None}, {"nts": 0.5, "sts": 0.5}) is None


# --- ATF weights: RS removed, its weight redistributed proportionally --------

def test_default_weights_drop_rs_and_redistribute_proportionally():
    original = {"nts": 0.30, "sts": 0.30, "tis": 0.15, "rs": 0.10, "os": 0.15}
    remaining = {k: v for k, v in original.items() if k != "rs"}
    assert set(DEFAULT_WEIGHTS) == {"nts", "sts", "tis", "os"}
    assert math.isclose(sum(DEFAULT_WEIGHTS.values()), 1.0, abs_tol=1e-12)
    for key, w in remaining.items():
        # w / 0.90: each keeps its share of the non-RS weight ...
        assert DEFAULT_WEIGHTS[key] == pytest.approx(w / sum(remaining.values()))
        # ... and receives RS's 0.10 in proportion to its original weight
        assert DEFAULT_WEIGHTS[key] - w == pytest.approx(original["rs"] * w / 0.90)
    assert DEFAULT_WEIGHTS == pytest.approx({"nts": 1 / 3, "sts": 1 / 3, "tis": 1 / 6, "os": 1 / 6})
    # relative importance (2:2:1:1) is preserved
    assert DEFAULT_WEIGHTS["nts"] / DEFAULT_WEIGHTS["tis"] == pytest.approx(2.0)
    assert DEFAULT_WEIGHTS["sts"] / DEFAULT_WEIGHTS["os"] == pytest.approx(2.0)


def test_resolve_weights_rejects_rs():
    with pytest.raises(ValueError, match="exactly the keys"):
        resolve_weights({"nts": 0.30, "sts": 0.30, "tis": 0.15, "rs": 0.10, "os": 0.15})
    weights, _ = resolve_weights(None)
    assert "rs" not in weights


# --- ATF formula: exp(Σ wᵢ·log(max(Sᵢ, ε)) / Σ wᵢ) ------------------------------

def test_atf_is_the_weighted_geometric_mean():
    scores = {"nts": 0.9, "sts": 0.8, "tis": 0.6, "os": 0.5}
    w = DEFAULT_WEIGHTS
    atf, coverage = atf_score(scores, w)
    expected = math.exp(sum(w[k] * math.log(s) for k, s in scores.items()) / sum(w.values()))
    assert atf == pytest.approx(expected, rel=1e-12)
    # same thing written as a weighted product: Π Sᵢ^(wᵢ/Σw)
    assert atf == pytest.approx(0.9 ** (1 / 3) * 0.8 ** (1 / 3) * 0.6 ** (1 / 6) * 0.5 ** (1 / 6), rel=1e-12)
    assert coverage == pytest.approx(1.0)


def test_atf_differs_from_the_old_weighted_arithmetic_mean():
    scores = {"nts": 0.9, "sts": 0.8, "tis": 0.6, "os": 0.5}
    atf, _ = atf_score(scores, DEFAULT_WEIGHTS)
    arithmetic = sum(DEFAULT_WEIGHTS[k] * s for k, s in scores.items())
    assert atf < arithmetic  # AM-GM: strictly lower for unequal scores


def test_atf_perfect_and_uniform_scores():
    assert atf_score({k: 1.0 for k in DEFAULT_WEIGHTS}, DEFAULT_WEIGHTS)[0] == 1.0
    assert atf_score({k: 0.7 for k in DEFAULT_WEIGHTS}, DEFAULT_WEIGHTS)[0] == pytest.approx(0.7, rel=1e-12)


# --- zero scores: floored at ε, never log(0) ----------------------------------

def test_atf_zero_component_is_floored_at_epsilon():
    scores = {"nts": 0.0, "sts": 1.0, "tis": 1.0, "os": 1.0}
    atf, coverage = atf_score(scores, DEFAULT_WEIGHTS)
    assert atf == pytest.approx(ATF_EPSILON ** (1 / 3), rel=1e-12)  # 1e-6^(1/3) = 0.01
    assert atf > 0.0 and math.isfinite(atf)
    assert coverage == pytest.approx(1.0)  # a 0 is a scored failure, not N/A


def test_atf_all_zero_scores_equal_epsilon():
    atf, _ = atf_score({k: 0.0 for k in DEFAULT_WEIGHTS}, DEFAULT_WEIGHTS)
    assert atf == pytest.approx(ATF_EPSILON, rel=1e-9)


def test_atf_scores_below_epsilon_are_treated_as_epsilon():
    tiny = {"nts": 1e-12, "sts": 1.0, "tis": 1.0, "os": 1.0}
    zero = {"nts": 0.0, "sts": 1.0, "tis": 1.0, "os": 1.0}
    assert atf_score(tiny, DEFAULT_WEIGHTS)[0] == atf_score(zero, DEFAULT_WEIGHTS)[0]


# --- weight normalization (denominator Σ wᵢ over applicable components) --------

def test_atf_is_invariant_to_weight_scale():
    scores = {"nts": 0.9, "sts": 0.4, "tis": 0.7, "os": 0.2}
    unnormalized = {"nts": 2.0, "sts": 2.0, "tis": 1.0, "os": 1.0}  # same 2:2:1:1 ratios
    assert atf_score(scores, unnormalized)[0] == pytest.approx(atf_score(scores, DEFAULT_WEIGHTS)[0], rel=1e-12)


def test_atf_renormalizes_over_applicable_components():
    scores = {"nts": 0.9, "sts": 0.8, "tis": None, "os": 0.5}
    w = DEFAULT_WEIGHTS
    atf, coverage = atf_score(scores, w)
    applicable_weight = w["nts"] + w["sts"] + w["os"]  # 5/6
    expected = math.exp((w["nts"] * math.log(0.9) + w["sts"] * math.log(0.8) + w["os"] * math.log(0.5)) / applicable_weight)
    assert atf == pytest.approx(expected, rel=1e-12)
    assert coverage == pytest.approx(5 / 6)
    # N/A is excluded, not treated as 0 (which would floor it at ε)
    assert atf > atf_score({**scores, "tis": 0.0}, w)[0]


def test_atf_coverage_renormalizes_around_na_groups():
    group_scores = {"nts": 1.0, "sts": 1.0, "tis": None, "os": 1.0}
    atf, coverage = atf_score(group_scores, DEFAULT_WEIGHTS)
    assert atf == 1.0
    assert coverage == pytest.approx(1 / 3 + 1 / 3 + 1 / 6)


def test_atf_score_all_na_is_none_zero_coverage():
    group_scores = {k: None for k in DEFAULT_WEIGHTS}
    atf, coverage = atf_score(group_scores, DEFAULT_WEIGHTS)
    assert atf is None
    assert coverage == 0.0


def test_atf_zero_applicable_weight_is_none():
    weights = {"nts": 0.0, "sts": 0.5, "tis": 0.5, "os": 0.0}
    atf, coverage = atf_score({"nts": 0.8, "sts": None, "tis": None, "os": 0.9}, weights)
    assert atf is None
    assert coverage == 0.0


# --- configurable weights: shared validation + CLI parsing ---------------------

VALID = {"nts": 0.4, "sts": 0.3, "tis": 0.2, "os": 0.1}


def test_resolve_weights_none_uses_the_shared_defaults():
    weights, _ = resolve_weights(None)
    assert weights == DEFAULT_WEIGHTS and weights is not DEFAULT_WEIGHTS


def test_resolve_weights_accepts_valid_custom_weights_as_floats():
    weights, _ = resolve_weights({"nts": 1, "sts": 0, "tis": 0, "os": 0})
    assert weights == {"nts": 1.0, "sts": 0.0, "tis": 0.0, "os": 0.0}
    assert all(type(v) is float for v in weights.values())
    assert resolve_weights(VALID)[0] == VALID


@pytest.mark.parametrize(
    "config, message",
    [
        ({"nts": 0.5, "sts": 0.5, "tis": 0.0}, r"missing \['os'\]"),
        ({**VALID, "rs": 0.0}, r"unexpected \['rs'\]"),
        ({}, "missing"),  # an empty override is not "omitted"
        ({**VALID, "nts": -0.1, "sts": 0.8}, "'nts' must be non-negative"),
        ({**VALID, "nts": float("nan")}, "'nts' must be finite"),
        ({**VALID, "nts": float("inf")}, "'nts' must be finite"),
        ({**VALID, "nts": True}, "'nts' must be a number"),
        ({**VALID, "nts": "0.4"}, "'nts' must be a number"),
        ({**VALID, "nts": None}, "'nts' must be a number"),
        ({**VALID, "nts": 0.5}, "must sum to 1, got 1.1"),
        ([0.25, 0.25, 0.25, 0.25], "must be an object"),
    ],
)
def test_resolve_weights_rejects_invalid_weights_with_clear_messages(config, message):
    with pytest.raises(ValueError, match=message):
        resolve_weights(config)


def test_sum_tolerance_is_1e6():
    resolve_weights({"nts": 0.4, "sts": 0.3, "tis": 0.2, "os": 0.1 + 5e-7})
    with pytest.raises(ValueError, match="sum to 1"):
        resolve_weights({"nts": 0.4, "sts": 0.3, "tis": 0.2, "os": 0.1 + 1e-5})


def test_parse_weights_spec():
    assert parse_weights_spec("nts=0.4,sts=0.3,tis=0.2,os=0.1") == VALID
    assert parse_weights_spec(" NTS = 0.4 , sts=0.3, tis=0.2 ,os=0.1, ") == VALID
    # syntax only -- the rules are resolve_weights()'s job
    assert parse_weights_spec("nts=0.5") == {"nts": 0.5}


@pytest.mark.parametrize(
    "spec, message",
    [
        ("", "is empty"),
        (" , ", "is empty"),
        ("nts=0.4,sts", "expected component=number"),
        ("nts=", "expected component=number"),
        ("=0.4", "expected component=number"),
        ("nts=abc", "'nts' must be a number, got 'abc'"),
        ("nts=0.4,nts=0.6", "more than once"),
    ],
)
def test_parse_weights_spec_rejects_bad_syntax(spec, message):
    with pytest.raises(ValueError, match=message):
        parse_weights_spec(spec)


def test_parse_then_resolve_rejects_nan_and_inf_strings():
    for bad in ("nan", "inf"):
        with pytest.raises(ValueError, match="'os' must be finite"):
            resolve_weights(parse_weights_spec(f"nts=0.4,sts=0.3,tis=0.2,os={bad}"))


def test_weights_from_cli(tmp_path):
    path = tmp_path / "w.json"
    path.write_text(json.dumps(VALID))
    assert weights_from_cli(None, None) is None
    assert weights_from_cli("nts=0.4,sts=0.3,tis=0.2,os=0.1", None) == VALID
    assert weights_from_cli(None, path) == VALID
    with pytest.raises(ValueError, match="not both"):
        weights_from_cli("nts=1,sts=0,tis=0,os=0", path)


def test_load_weights_file_errors(tmp_path):
    with pytest.raises(ValueError, match="cannot read --weights-file"):
        load_weights_file(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{nts: 0.4")
    with pytest.raises(ValueError, match="is not valid JSON"):
        load_weights_file(bad)


def test_runner_validates_library_weights_too():
    from atf_eval.normalized import NormalizedNode, NormalizedTurn
    from atf_eval.runner import score_trajectory

    turns = [NormalizedTurn("c", 1, nodes=[NormalizedNode(node_id="A")])]
    assert score_trajectory(turns, turns).atf == 1.0  # None -> defaults
    with pytest.raises(ValueError, match="sum to 1"):
        score_trajectory(turns, turns, weights={"nts": 0.5, "sts": 0.5, "tis": 0.5, "os": 0.5})
