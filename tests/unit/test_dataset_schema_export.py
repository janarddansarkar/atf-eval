"""Proves the Phase 0 rename actually achieves schema conformance: a small
hand-built GoldenTurn list, run through to_normalized() + to_canonical_document(),
must validate against agent-eval-main's normalized_trajectory.schema.json."""
from __future__ import annotations

from atf_eval.dataset import GoldenTurn, to_canonical_document


def _build_turns() -> list[GoldenTurn]:
    turn_1 = GoldenTurn.model_validate(
        {
            "conversation_id": "conv_01",
            "turn_id": 1,
            "user_input": "I can't pay this month.",
            "nodes": [{"node_id": "classify_intent"}, {"node_id": "hardship_node"}],
            "state_changes": [
                {"key": "hardship_flag", "old": False, "new": True, "node_id": "hardship_node"}
            ],
            "tool_calls": [{"tool_id": "eligibility_check", "arguments": {}}],
            "routing": {"path": "hardship", "target": "collection_agent"},
            "outcome": None,
        }
    )
    turn_2 = GoldenTurn.model_validate(
        {
            "conversation_id": "conv_01",
            "turn_id": 2,
            "user_input": "Yes, monthly instalments work.",
            "nodes": [{"node_id": "confirm_outcome"}],
            "state_changes": [],
            "tool_calls": [],
            "routing": None,
            "outcome": {
                "id": "installment_arrangement_accepted",
                "attributes": {"installment_amount": 3000},
                "required_conditions": ["installment_amount"],
            },
        }
    )
    return [turn_1, turn_2]


def test_turn_id_accepts_zero():
    # Regression: turn_id is now an int; `not self.turn_id` would wrongly
    # reject turn_id=0 as "missing".
    turn = GoldenTurn.model_validate({"conversation_id": "c", "turn_id": 0})
    assert turn.turn_id == 0


def test_canonical_document_validates_against_agent_eval_main_schema(canonical_validator):
    golden_turns = _build_turns()
    normalized = [t.to_normalized() for t in golden_turns]
    document = to_canonical_document("conv_01", normalized)

    errors = sorted(canonical_validator.iter_errors(document), key=str)
    assert not errors, "\n".join(f"{e.json_path}: {e.message}" for e in errors)


def test_canonical_document_preserves_outcome():
    golden_turns = _build_turns()
    normalized = [t.to_normalized() for t in golden_turns]
    document = to_canonical_document("conv_01", normalized)

    assert document["outcome"]["id"] == "installment_arrangement_accepted"
    assert document["turns"][0]["nodes"][1]["state_changes"][0] == {
        "key": "hardship_flag",
        "old": False,
        "new": True,
    }
