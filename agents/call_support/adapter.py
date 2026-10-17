"""Wraps the LangGraph call-support agent as an atf_eval `TrajectoryAgent`.

`run_turn` invokes the graph for one customer turn and normalizes the trace:
  - nodes         <- the graph nodes that ran, in order (stream_mode="updates")
  - state_changes <- diffs of the tracked state keys, attributed to the node that set them
  - tool_calls    <- the per-turn tool log, attached to the node that made each call
  - routing / outcome / response <- set by the nodes themselves
"""
from __future__ import annotations

from atf_eval.normalized import NormalizedNode, NormalizedTurn, Outcome, Routing, StateChange, ToolCall
from agents.call_support.graph import build_graph

TRACKED = {  # state key -> value before the conversation starts
    "authenticated": False,
    "intent": None,
    "ticket_id": None,
    "resolved": False,
}


class CallSupportAdapter:
    def __init__(self, graph=None) -> None:
        self.graph = graph or build_graph()

    def run_turn(
        self,
        conversation_id: str,
        turn_id: int,
        user_input: str,
        history: list[NormalizedTurn],
    ) -> NormalizedTurn:
        cfg = {"configurable": {"thread_id": conversation_id}}
        before = self.graph.get_state(cfg).values
        known = {k: before.get(k, v) for k, v in TRACKED.items()}

        nodes: list[NormalizedNode] = []
        final: dict = {}
        fresh = {"user_input": user_input, "response": None, "route": None, "outcome": None,
                 "outage": False, "calls": None}
        for chunk in self.graph.stream(fresh, cfg, stream_mode="updates"):
            for node_id, update in chunk.items():
                node = NormalizedNode(node_id=node_id)
                for key in TRACKED:
                    if key in update and update[key] != known[key]:
                        node.state_changes.append(StateChange(key=key, old=known[key], new=update[key]))
                        known[key] = update[key]
                for c in update.get("calls", []):
                    node.tool_calls.append(ToolCall(tool_id=c["tool"], arguments=c["args"], result=c["result"],
                                                    status="success" if c["result"] else "empty"))
                nodes.append(node)
                final.update(update)

        route, outcome = final.get("route"), final.get("outcome")
        return NormalizedTurn(
            conversation_id=conversation_id,
            turn_id=turn_id,
            customer_input=user_input,
            nodes=nodes,
            state_changes=[c for n in nodes for c in n.state_changes],
            tool_calls=[t for n in nodes for t in n.tool_calls],
            routing=Routing(path=route["path"], target=route["target"]) if route else None,
            outcome=Outcome(id=outcome["id"], attributes=outcome["attributes"]) if outcome else None,
            response=final.get("response"),
        )
