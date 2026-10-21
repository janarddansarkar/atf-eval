"""Dependency-free TrajectoryAgent for smoke-testing atf_eval itself -- no
external agent, no LLM, no network. Returns a canned trajectory that matches
`mock_golden_dataset.jsonl` turn-for-turn, so a full `atf-eval run` against it is a
runnable, eyeball-able "does the framework still work" check.
"""
from __future__ import annotations

from atf_eval.normalized import NormalizedNode, NormalizedTurn, Outcome, Routing, StateChange, ToolCall

_TURNS = {
    ("demo_conv_01", 1): NormalizedTurn(
        conversation_id="demo_conv_01",
        turn_id=1,
        nodes=[
            NormalizedNode(node_id="classify_intent"),
            NormalizedNode(
                node_id="hardship_node",
                state_changes=[StateChange(key="hardship_flag", old=False, new=True)],
            ),
        ],
        state_changes=[StateChange(key="hardship_flag", old=False, new=True)],
        tool_calls=[ToolCall(tool_id="eligibility_check", status="success")],
        routing=Routing(path="hardship", target="collection_agent"),
        response="I'm sorry to hear that. Let's see if you qualify for a hardship arrangement.",
    ),
    ("demo_conv_01", 2): NormalizedTurn(
        conversation_id="demo_conv_01",
        turn_id=2,
        nodes=[NormalizedNode(node_id="confirm_outcome")],
        routing=Routing(path="confirm", target="customer"),
        outcome=Outcome(id="installment_arrangement_accepted", attributes={"installment_amount": 3000}),
        response="Great, I've set up monthly instalments of 3000 starting next month.",
    ),
    ("demo_conv_02", 1): NormalizedTurn(
        conversation_id="demo_conv_02",
        turn_id=1,
        nodes=[
            NormalizedNode(node_id="classify_intent"),
            NormalizedNode(
                node_id="payment_node",
                state_changes=[StateChange(key="payment_intent", old=False, new=True)],
            ),
        ],
        state_changes=[StateChange(key="payment_intent", old=False, new=True)],
        tool_calls=[ToolCall(tool_id="payment_link_create", status="success")],
        routing=Routing(path="pay_now", target="collection_agent"),
        response="Sure, here's a payment link for the outstanding amount.",
    ),
    ("demo_conv_02", 2): NormalizedTurn(
        conversation_id="demo_conv_02",
        turn_id=2,
        nodes=[NormalizedNode(node_id="confirm_outcome")],
        routing=Routing(path="confirm", target="customer"),
        outcome=Outcome(id="payment_received", attributes={"amount": 5000}),
        response="Thank you, we've received your payment.",
    ),
}


class MockTrajectoryAgent:
    def run_turn(
        self,
        conversation_id: str,
        turn_id: int,
        user_input: str,
        history: list[NormalizedTurn],
    ) -> NormalizedTurn:
        return _TURNS[(conversation_id, turn_id)]
