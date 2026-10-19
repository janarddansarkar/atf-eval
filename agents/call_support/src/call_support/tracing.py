"""Trace capture for the call-support agent -- on whenever the agent runs.

Two local artefacts per conversation, both written as the agent runs:

  <trace_dir>/<conversation_id>.json        the conversation's turns (nodes, tool calls,
                                            state changes, routing, outcome, latency),
                                            rewritten after every turn. A plain JSON file
                                            any evaluation tool can read.
  <trace_dir>/raw/<conversation_id>.jsonl   raw LangChain/LangGraph callback events:
                                            node, tool and LLM calls with timestamps,
                                            prompts, replies and token usage.

LangSmith needs no code here: set LANGSMITH_TRACING=true and LANGSMITH_API_KEY
(optionally LANGSMITH_PROJECT) and LangChain ships every run there; the adapter
tags each run with the conversation and turn so they are easy to find.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from call_support.types import TurnResult


AVAILABILITY = {"nodes": "available", "state": "available", "tools": "available",
                "routing": "available", "outcome": "available"}


def resolve_trace_dir(trace_dir: str | Path | None = None) -> Path:
    """Explicit argument, else $CALL_SUPPORT_TRACE_DIR, else ./traces."""
    return Path(trace_dir or os.environ.get("CALL_SUPPORT_TRACE_DIR") or "traces")


def _clip(value: Any, limit: int = 2000) -> Any:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[:limit] + "…"


class RawEventHandler(BaseCallbackHandler):
    """Appends one JSON line per node / tool / LLM event for a conversation."""

    def __init__(self, path: Path, conversation_id: str, turn_id: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path, self._base = path, {"conversation_id": conversation_id, "turn_id": turn_id}
        self._lock = threading.Lock()
        self._started: dict[Any, float] = {}

    def _emit(self, event: str, run_id, parent_run_id=None, **fields) -> None:
        started = self._started.pop(run_id, None) if event.endswith(("_end", "_error")) else None
        if event.endswith("_start"):
            self._started[run_id] = time.time()
        row = {"ts": time.time(), "event": event, "run_id": str(run_id),
               "parent_run_id": str(parent_run_id) if parent_run_id else None, **self._base, **fields}
        if started is not None:
            row["duration_ms"] = round((time.time() - started) * 1000, 1)
        with self._lock, self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    # LangGraph nodes surface as chains whose metadata names the node
    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, metadata=None, **kw):
        node = (metadata or {}).get("langgraph_node")
        if node and kw.get("name") == node and not node.startswith("__"):
            self._emit("node_start", run_id, parent_run_id, node=node)

    def on_chain_end(self, outputs, *, run_id, parent_run_id=None, **kw):
        if run_id in self._started:
            self._emit("node_end", run_id, parent_run_id)

    def on_chain_error(self, error, *, run_id, parent_run_id=None, **kw):
        if run_id in self._started:
            self._emit("node_error", run_id, parent_run_id, error=repr(error))

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kw):
        self._emit("tool_start", run_id, parent_run_id, tool=(serialized or {}).get("name") or kw.get("name"),
                   input=_clip(input_str))

    def on_tool_end(self, output, *, run_id, parent_run_id=None, **kw):
        self._emit("tool_end", run_id, parent_run_id, output=_clip(getattr(output, "content", output)))

    def on_tool_error(self, error, *, run_id, parent_run_id=None, **kw):
        self._emit("tool_error", run_id, parent_run_id, error=repr(error))

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kw):
        self._emit("llm_start", run_id, parent_run_id,
                   messages=[{"role": m.type, "content": _clip(m.content, 800)} for m in messages[0]])

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kw):
        msg = response.generations[0][0].message if response.generations and response.generations[0] else None
        self._emit("llm_end", run_id, parent_run_id,
                   output=_clip(getattr(msg, "content", "") or getattr(msg, "tool_calls", "")),
                   usage=getattr(msg, "usage_metadata", None))

    def on_llm_error(self, error, *, run_id, parent_run_id=None, **kw):
        self._emit("llm_error", run_id, parent_run_id, error=repr(error))


def to_trace_turn(t: TurnResult) -> dict:
    """One turn in the trace-file shape (tool calls hang off their nodes)."""
    seq, nodes, flat_tools = 0, [], []
    for i, n in enumerate(t.nodes, 1):
        tools = []
        for c in n.tool_calls:
            seq += 1
            call = {"tool_name": c.tool, "sequence_index": seq, "input": c.args, "output": c.result}
            tools.append(call)
            flat_tools.append(call)
        nodes.append({"node_id": n.node_id, "sequence_index": i, "state_changes": n.state_changes,
                      "tool_calls": tools})
    return {
        "turn_id": t.turn_id,
        "conversation": {"customer": t.user_input, "agent": t.reply},
        "nodes": nodes,
        "tool_calls": flat_tools,
        "state_changes": [c for n in t.nodes for c in n.state_changes],
        "routing": t.route,
        "timing": {"response_latency_ms": round(t.latency_ms)},
    }


class TraceStore:
    """Accumulates a conversation's turns and rewrites its trace file after each one."""

    def __init__(self, directory: str | Path | None = None, agent_info: dict | None = None) -> None:
        self.directory = resolve_trace_dir(directory)
        self.agent_info = agent_info or {}
        self._docs: dict[str, dict] = {}
        self._meta: dict[str, dict] = {}
        self._lock = threading.Lock()

    def annotate(self, conversation_id: str, **meta) -> None:
        """Extra top-level trace fields, e.g. golden_scenario / deviation_type."""
        self._meta.setdefault(conversation_id, {}).update(meta)

    def raw_path(self, conversation_id: str) -> Path:
        return self.directory / "raw" / f"{conversation_id}.jsonl"

    def handler(self, conversation_id: str, turn_id: int) -> RawEventHandler:
        if turn_id == 1:  # a new conversation under this id: start its event log afresh
            self.raw_path(conversation_id).unlink(missing_ok=True)
        return RawEventHandler(self.raw_path(conversation_id), conversation_id, turn_id)

    def save_turn(self, conversation_id: str, turn: TurnResult) -> Path:
        with self._lock:
            doc = self._docs.get(conversation_id)
            if doc is None or turn.turn_id == 1:
                doc = self._docs[conversation_id] = {
                    "trace_id": conversation_id,
                    "scenario_id": conversation_id,
                    **self._meta.get(conversation_id, {}),
                    "trace_type": "observed",
                    "agent": self.agent_info,
                    "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "availability": dict(AVAILABILITY),
                    "turns": [],
                }
            doc["turns"] = [t for t in doc["turns"] if t["turn_id"] != turn.turn_id] + [to_trace_turn(turn)]
            if turn.outcome:
                doc["outcome"] = {"id": turn.outcome["id"], "attributes": turn.outcome["attributes"]}
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / f"{conversation_id}.json"
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(doc, indent=2, default=str) + "\n", encoding="utf-8")
            tmp.replace(path)
            return path
