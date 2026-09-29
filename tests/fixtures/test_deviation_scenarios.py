"""End-to-end conformance check against agent-eval's own canonical golden
scenario (S001) and its deviation fixtures -- authored by the spec's owners,
not derived from atf_eval's code. See the reference_adapter module for how
the pre-Adapter raw fixture vocabulary is normalized.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from atf_eval.metrics.nodes import nts_conversation
from atf_eval.metrics.state import sts_conversation
from atf_eval.metrics.tools import tis_conversation
from tests.fixtures.reference_adapter import load_fixture_turns, load_golden_turns

_ROOT = Path(__file__).resolve().parents[2]
_LAYOUTS = [
    # (repo folder, golden file, fixture-name prefix)
    ("agent-eval", "S001_motor_insurance_hardship_installment.json", "S001_"),
    ("agent-eval-main", "motor_insurance_hardship_installment.json", ""),
]
_layout = next(
    ((_ROOT / repo, golden, prefix) for repo, golden, prefix in _LAYOUTS if (_ROOT / repo / "golden" / golden).exists()),
    None,
)


def _fixture(name: str) -> Path:
    repo, _, prefix = _layout
    return repo / "tests" / "fixtures" / "scenarios" / f"{prefix}{name}.json"


@pytest.fixture(scope="module")
def golden_turns():
    if _layout is None:
        pytest.skip(
            "agent-eval not found locally (it's a separate repo, not bundled here) -- "
            "clone it as a sibling folder to run this conformance suite"
        )
    repo, golden, _ = _layout
    return load_golden_turns(repo / "golden" / golden)


def test_golden_against_itself_is_a_perfect_baseline(golden_turns):
    assert nts_conversation(golden_turns, golden_turns) == 1.0
    assert sts_conversation(golden_turns, golden_turns) == 1.0
    assert tis_conversation(golden_turns, golden_turns) == 1.0


def test_correct_answer_wrong_trajectory_degrades_nodes_and_state(golden_turns):
    """Deviation: hardship_node is skipped even though the agent's reply is
    contextually appropriate. Both the node sequence and the state
    transition that only hardship_node produces are affected; tool calls
    elsewhere in the conversation are untouched."""
    observed = load_fixture_turns(_fixture("correct_answer_wrong_trajectory"))

    assert nts_conversation(golden_turns, observed) < 1.0
    assert sts_conversation(golden_turns, observed) < 1.0
    assert tis_conversation(golden_turns, observed) == 1.0


def test_missing_tool_call_degrades_only_tis(golden_turns):
    """Deviation: the expected `instalment_eligibility` tool is never
    invoked at turn 5, but the node sequence and the resulting state change
    are otherwise identical to golden -- only TIS should see this."""
    golden_turn_5 = [t for t in golden_turns if t.turn_id == 5]
    observed = load_fixture_turns(_fixture("missing_tool_call"))

    assert nts_conversation(golden_turn_5, observed) == 1.0
    assert sts_conversation(golden_turn_5, observed) == 1.0
    assert tis_conversation(golden_turn_5, observed) == 0.0


def test_wrong_tool_input_stays_perfect_because_nothing_was_required(golden_turns):
    """Deviation: the correct tool is invoked with arguments golden never
    asked for. Golden's expected input for this tool is `{}`, and input
    similarity compares the expected arguments (METRICS.md §4) -- so this
    fixture does not move TIS despite its name. The fixture, not the metric,
    would need non-empty expected arguments to exercise TIS input."""
    golden_turn_5 = [t for t in golden_turns if t.turn_id == 5]
    observed = load_fixture_turns(_fixture("wrong_tool_input"))

    assert nts_conversation(golden_turn_5, observed) == 1.0
    assert sts_conversation(golden_turn_5, observed) == 1.0
    assert tis_conversation(golden_turn_5, observed) == 1.0
