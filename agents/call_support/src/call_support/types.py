"""Plain result types returned by the agent -- no framework or evaluator types."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCallRecord:
    tool: str
    args: dict
    result: Any = None


@dataclass
class NodeRecord:
    node_id: str
    state_changes: list[dict] = field(default_factory=list)  # [{"key", "old", "new"}]
    tool_calls: list[ToolCallRecord] = field(default_factory=list)


@dataclass
class TurnResult:
    conversation_id: str
    turn_id: int
    user_input: str
    reply: str | None
    nodes: list[NodeRecord] = field(default_factory=list)
    route: dict | None = None  # {"path", "target"}
    outcome: dict | None = None  # {"id", "attributes"}
    latency_ms: float = 0.0
