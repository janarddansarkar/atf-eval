"""Marlow -- a small state-machine collections agent wired to atf_eval.

Where Kestrel infers "what happens next" from the last routing path, Marlow
keeps an explicit conversation stage (idle -> intent_open -> closed) and
records every stage change as a state transition. That makes it a stateful
agent whose trace exposes *more* state than the Golden dataset expects, so the
extra `stage` transitions show up in STS.

    atf-eval run --dataset examples/golden_dataset.jsonl \
        --adapter agents.marlow.agent:Marlow --no-routing-judge
"""
from __future__ import annotations

from atf_eval.normalized import NormalizedNode, NormalizedTurn, Routing, StateChange, ToolCall

from agents.kestrel.agent import INTENTS, Kestrel

# (stage, event) -> next stage
TRANSITIONS = {("idle", "intent"): "intent_open", ("intent_open", "reply"): "closed"}


def _stage(history: list[NormalizedTurn]) -> str:
    """The latest recorded `stage`, rebuilt from the observed history."""
    for turn in reversed(history):
        for change in reversed(turn.state_changes):
            if change.key == "stage":
                return change.new
    return "idle"


class Marlow:
    def __init__(self) -> None:
        self._classify = Kestrel().classify

    def run_turn(
        self,
        conversation_id: str,
        turn_id: int,
        user_input: str,
        history: list[NormalizedTurn],
    ) -> NormalizedTurn:
        stage = _stage(history)

        if stage == "intent_open":
            intent = history[-1].routing.path
            outcome, reply = INTENTS[intent][6:]
            nxt = TRANSITIONS[(stage, "reply")]
            change = StateChange(key="stage", old=stage, new=nxt)
            return NormalizedTurn(
                conversation_id=conversation_id,
                turn_id=turn_id,
                nodes=[NormalizedNode(node_id="confirm_outcome", state_changes=[change])],
                state_changes=[change],
                routing=Routing(path="confirm", target="customer"),
                outcome=outcome,
                response=reply,
            )

        intent = self._classify(user_input)
        if intent is None:  # stay idle and ask again
            return NormalizedTurn(
                conversation_id=conversation_id,
                turn_id=turn_id,
                nodes=[NormalizedNode(node_id="classify_intent")],
                routing=Routing(path="clarify", target="customer"),
                response="Sorry, could you tell me a bit more about what you need?",
            )

        _, node, flag, tool, target, reply, *_ = INTENTS[intent]
        nxt = TRANSITIONS[(stage, "intent")]
        changes = [StateChange(key=flag, old=False, new=True), StateChange(key="stage", old=stage, new=nxt)]
        return NormalizedTurn(
            conversation_id=conversation_id,
            turn_id=turn_id,
            nodes=[NormalizedNode(node_id="classify_intent"), NormalizedNode(node_id=node, state_changes=changes)],
            state_changes=changes,
            tool_calls=[ToolCall(tool_id=tool, status="success")],
            routing=Routing(path=intent, target=target),
            response=reply,
        )
