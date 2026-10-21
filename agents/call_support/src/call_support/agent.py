"""The call-support agent as a plain Python object.

    from call_support import CallSupportAgent
    agent = CallSupportAgent()
    turn = agent.chat("conv-1", "My internet is down, I'm ACC-1001")
    print(turn.reply)

`chat()` runs one customer turn through the LangGraph graph (graph.py), returns a
`TurnResult` (reply plus the nodes, tool calls, state changes, routing and outcome
that produced it) and, unless `save_traces=False`, saves the trace (tracing.py).
"""
from __future__ import annotations

import time

from call_support.graph import MODEL, build_graph
from call_support.tracing import TraceStore
from call_support.types import NodeRecord, ToolCallRecord, TurnResult

TRACKED = {  # state key -> value before the conversation starts
    "authenticated": False,
    "intent": None,
    "ticket_id": None,
    "resolved": False,
}


class CallSupportAgent:
    def __init__(self, graph=None, trace_dir=None, save_traces: bool = True) -> None:
        self.graph = graph or build_graph()
        self.traces = (
            TraceStore(trace_dir, agent_info={"framework": "langgraph", "model": MODEL}) if save_traces else None
        )
        self._turns: dict[str, int] = {}

    def annotate_trace(self, conversation_id: str, **meta) -> None:
        """Extra top-level trace fields for a conversation (e.g. a scenario label)."""
        if self.traces:
            self.traces.annotate(conversation_id, **meta)

    def chat(self, conversation_id: str, message: str, turn_id: int | None = None) -> TurnResult:
        """One customer turn. `turn_id` defaults to the next number for this conversation."""
        turn_id = turn_id or self._turns.get(conversation_id, 0) + 1
        self._turns[conversation_id] = turn_id
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

        nodes: list[NodeRecord] = []
        final: dict = {}
        fresh = {"user_input": message, "response": None, "route": None, "outcome": None,
                 "outage": False, "calls": None}
        for chunk in self.graph.stream(fresh, cfg, stream_mode="updates"):
            for node_id, update in chunk.items():
                node = NodeRecord(node_id=node_id)
                for key in TRACKED:
                    if key in update and update[key] != known[key]:
                        node.state_changes.append({"key": key, "old": known[key], "new": update[key]})
                        known[key] = update[key]
                node.tool_calls = [ToolCallRecord(c["tool"], c["args"], c["result"]) for c in update.get("calls", [])]
                nodes.append(node)
                final.update(update)

        turn = TurnResult(
            conversation_id=conversation_id,
            turn_id=turn_id,
            user_input=message,
            reply=final.get("response"),
            nodes=nodes,
            route=final.get("route"),
            outcome=final.get("outcome"),
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        if self.traces:
            self.traces.save_turn(conversation_id, turn)
        return turn
