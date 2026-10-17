"""Wraps the LangGraph call-support agent as an atf_eval `TrajectoryAgent`.

Every turn it runs is also traced (see tracing.py): an ATF-ready trace file per
conversation plus a raw callback event log, saved under the trace directory
(`trace_dir=`, else $CALL_SUPPORT_TRACE_DIR, else agents/call_support/traces).
Pass `save_traces=False` to turn that off. LangSmith picks the runs up from the
LANGSMITH_* env vars with no change here.

`run_turn` invokes the graph for one customer turn and normalizes the trace:
  - nodes         <- the graph nodes that ran, in order (stream_mode="updates")
  - state_changes <- diffs of the tracked state keys, attributed to the node that set them
  - tool_calls    <- the per-turn tool log, attached to the node that made each call
  - routing / outcome / response <- set by the nodes themselves
"""
from __future__ import annotations

import time

from atf_eval.normalized import NormalizedNode, NormalizedTurn, Outcome, Routing, StateChange, ToolCall
from agents.call_support.graph import MODEL, build_graph
from agents.call_support.tracing import TraceStore

TRACKED = {  # state key -> value before the conversation starts
    "authenticated": False,
    "intent": None,
    "ticket_id": None,
    "resolved": False,
}


class CallSupportAdapter:
    def __init__(self, graph=None, trace_dir=None, save_traces: bool = True) -> None:
        self.graph = graph or build_graph()
        self.traces = (
            TraceStore(trace_dir, agent_info={"framework": "langgraph", "model": MODEL}) if save_traces else None
        )

    def annotate_trace(self, conversation_id: str, **meta) -> None:
        """Attach e.g. golden_scenario / deviation_type to a conversation's trace file."""
        if self.traces:
            self.traces.annotate(conversation_id, **meta)

    def run_turn(
        self,
        conversation_id: str,
        turn_id: int,
        user_input: str,
        history: list[NormalizedTurn],
    ) -> NormalizedTurn:
        cfg = {
            "configurable": {"thread_id": conversation_id},
            "run_name": f"call_support/{conversation_id}/turn{turn_id}",
            "tags": ["call_support", f"conversation:{conversation_id}"],
            "metadata": {"conversation_id": conversation_id, "turn_id": turn_id},
        }
        if self.traces:
            cfg["callbacks"] = [self.traces.handler(conversation_id, turn_id)]
        started = time.perf_counter()
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
        turn = NormalizedTurn(
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
        if self.traces:
            self.traces.save_turn(conversation_id, turn, (time.perf_counter() - started) * 1000)
        return turn
