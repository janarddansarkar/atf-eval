"""Plugs the standalone call-support agent (../../call_support_agent) into atf_eval.

The agent knows nothing about atf_eval; this adapter is the only bridge. It calls
`CallSupportAgent.chat()` and converts the returned `TurnResult` into a
`NormalizedTurn`. The agent saves its own traces as it runs (`trace_dir=`,
$CALL_SUPPORT_TRACE_DIR, or ./traces), so evaluating through the adapter also
leaves trace files behind.
"""
from __future__ import annotations

from atf_eval.normalized import NormalizedNode, NormalizedTurn, Outcome, Routing, StateChange, ToolCall
from call_support import CallSupportAgent


class CallSupportAdapter:
    def __init__(self, agent: CallSupportAgent | None = None, trace_dir=None, save_traces: bool = True) -> None:
        self.agent = agent or CallSupportAgent(trace_dir=trace_dir, save_traces=save_traces)

    @property
    def traces(self):
        return self.agent.traces

    def annotate_trace(self, conversation_id: str, **meta) -> None:
        self.agent.annotate_trace(conversation_id, **meta)

    def run_turn(
        self,
        conversation_id: str,
        turn_id: int,
        user_input: str,
        history: list[NormalizedTurn],
    ) -> NormalizedTurn:
        r = self.agent.chat(conversation_id, user_input, turn_id=turn_id)
        nodes = [
            NormalizedNode(
                node_id=n.node_id,
                state_changes=[StateChange(key=c["key"], old=c["old"], new=c["new"]) for c in n.state_changes],
                tool_calls=[
                    ToolCall(tool_id=c.tool, arguments=c.args, result=c.result,
                             status="success" if c.result else "empty")
                    for c in n.tool_calls
                ],
            )
            for n in r.nodes
        ]
        return NormalizedTurn(
            conversation_id=conversation_id,
            turn_id=turn_id,
            customer_input=user_input,
            nodes=nodes,
            state_changes=[c for n in nodes for c in n.state_changes],
            tool_calls=[t for n in nodes for t in n.tool_calls],
            routing=Routing(path=r.route["path"], target=r.route["target"]) if r.route else None,
            outcome=Outcome(id=r.outcome["id"], attributes=r.outcome["attributes"]) if r.outcome else None,
            response=r.reply,
        )
