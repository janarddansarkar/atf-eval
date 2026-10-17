"""A minimal rule-based collections agent wired to atf_eval -- no LLM, no network.

Turn 1: classify the customer's intent by keyword, run that intent's node and
tool, set a state flag and route. Turn 2: confirm the arrangement and report
the outcome. The agent reads its own prior path from `history`, so it holds no
state of its own. Unlike mock_adapter.py it *decides* its trajectory, so it can
genuinely score below 1.0 on a dataset it wasn't hard-coded for.

    atf-eval run --dataset examples/golden_dataset.jsonl \
        --adapter examples.simple_agent:SimpleAgent --no-routing-judge
"""
from __future__ import annotations

from atf_eval.normalized import NormalizedNode, NormalizedTurn, Outcome, Routing, StateChange, ToolCall

# intent -> (keywords, node, state flag, tool, routing target, reply, outcome)
INTENTS = {
    "hardship": (
        ("can't pay", "lost my job", "hardship", "afford"),
        "hardship_node", "hardship_flag", "eligibility_check", "collection_agent",
        "I'm sorry to hear that. Let's see if you qualify for a hardship arrangement.",
        Outcome(id="installment_arrangement_accepted", attributes={"installment_amount": 3000}),
        "Great, I've set up monthly instalments of 3000 starting next month.",
    ),
    "pay_now": (
        ("pay", "forgot", "link"),
        "payment_node", "payment_intent", "payment_link_create", "collection_agent",
        "Sure, here's a payment link for the outstanding amount.",
        Outcome(id="payment_received", attributes={"amount": 5000}),
        "Thank you, we've received your payment.",
    ),
}


class SimpleAgent:
    def run_turn(
        self,
        conversation_id: str,
        turn_id: int,
        user_input: str,
        history: list[NormalizedTurn],
    ) -> NormalizedTurn:
        prior = history[-1].routing.path if history and history[-1].routing else None
        if prior in INTENTS:  # an intent is already open -> this turn confirms it
            outcome, reply = INTENTS[prior][6:]
            return NormalizedTurn(
                conversation_id=conversation_id,
                turn_id=turn_id,
                nodes=[NormalizedNode(node_id="confirm_outcome")],
                routing=Routing(path="confirm", target="customer"),
                outcome=outcome,
                response=reply,
            )

        text = user_input.lower()
        intent = next((name for name, spec in INTENTS.items() if any(k in text for k in spec[0])), None)
        if intent is None:
            return NormalizedTurn(
                conversation_id=conversation_id,
                turn_id=turn_id,
                nodes=[NormalizedNode(node_id="classify_intent")],
                routing=Routing(path="clarify", target="customer"),
                response="Sorry, could you tell me a bit more about what you need?",
            )

        _, node, flag, tool, target, reply, *_ = INTENTS[intent]
        change = StateChange(key=flag, old=False, new=True)
        return NormalizedTurn(
            conversation_id=conversation_id,
            turn_id=turn_id,
            nodes=[NormalizedNode(node_id="classify_intent"), NormalizedNode(node_id=node, state_changes=[change])],
            state_changes=[change],
            tool_calls=[ToolCall(tool_id=tool, status="success")],
            routing=Routing(path=intent, target=target),
            response=reply,
        )
