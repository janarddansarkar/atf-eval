"""Golden dataset: pydantic-validated JSONL, one row per turn.

The dataset is authored directly in the normalized trajectory shape (spec §5/§20),
so `GoldenTurn.to_normalized()` is the "Golden Adapter" — an identity mapping. A
pluggable Golden Adapter (for datasets not already in this shape) is a documented
future extension point, not built here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from atf_eval.normalized import (
    NormalizedNode,
    NormalizedTurn,
    Outcome,
    Routing,
    StateChange,
    ToolCall,
)


def _attribute_state_changes_to_nodes(
    golden_changes: list["GoldenStateChange"], nodes: list[NormalizedNode]
) -> None:
    """Populate each NormalizedNode's `state_changes[]` (STS §3, Frozen Rule
    #6: "state is evaluated from node-level state_changes[]"). A change with
    an explicit `node_id` goes to the matching node; one without goes to the
    turn's last node as a documented fallback, since most source systems
    (including this framework's own bundled example dataset) only expose a
    turn-level state diff, not true per-node causal attribution."""
    if not nodes:
        return
    nodes_by_id: dict[str, NormalizedNode] = {}
    for n in nodes:
        nodes_by_id.setdefault(n.node_id, n)

    for change in golden_changes:
        target = nodes_by_id.get(change.node_id) if change.node_id else nodes[-1]
        if target is None:
            target = nodes[-1]
        target.state_changes.append(
            StateChange(key=change.key, old=change.old, new=change.new)
        )


class GoldenNode(BaseModel):
    node_id: str
    node_type: str | None = None


class GoldenStateChange(BaseModel):
    key: str
    old: Any = None
    new: Any = None
    node_id: str | None = None  # which node this transition belongs to, if known (STS §3, Frozen Rule #6)


class GoldenToolCall(BaseModel):
    tool_id: str
    arguments: dict = Field(default_factory=dict)


class GoldenRouting(BaseModel):
    path: str | None = None
    target: str | None = None


class GoldenOutcome(BaseModel):
    id: str | None = None
    attributes: dict = Field(default_factory=dict)
    required_conditions: list[str] = Field(default_factory=list)


class GoldenTurn(BaseModel):
    conversation_id: str
    turn_id: int
    user_input: str = ""
    nodes: list[GoldenNode] = Field(default_factory=list)
    state_changes: list[GoldenStateChange] = Field(default_factory=list)
    tool_calls: list[GoldenToolCall] = Field(default_factory=list)
    routing: GoldenRouting | None = None
    outcome: GoldenOutcome | None = None
    metadata: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def _require_ids(self) -> "GoldenTurn":
        if not self.conversation_id or self.turn_id is None:
            raise ValueError("conversation_id and turn_id must both be present")
        return self

    def to_normalized(self) -> NormalizedTurn:
        flat_state_changes = [
            StateChange(key=s.key, old=s.old, new=s.new)
            for s in self.state_changes
        ]
        normalized_nodes = [
            NormalizedNode(node_id=n.node_id, node_type=n.node_type) for n in self.nodes
        ]
        _attribute_state_changes_to_nodes(self.state_changes, normalized_nodes)

        return NormalizedTurn(
            conversation_id=self.conversation_id,
            turn_id=self.turn_id,
            customer_input=self.user_input or None,
            nodes=normalized_nodes,
            state_changes=flat_state_changes,
            tool_calls=[
                ToolCall(tool_id=t.tool_id, arguments=t.arguments)
                for t in self.tool_calls
            ],
            routing=(
                Routing(path=self.routing.path, target=self.routing.target)
                if self.routing is not None
                else None
            ),
            outcome=(
                Outcome(
                    id=self.outcome.id,
                    attributes=self.outcome.attributes,
                    required_conditions=self.outcome.required_conditions,
                )
                if self.outcome is not None
                else None
            ),
        )


def load_dataset(path: str) -> list[GoldenTurn]:
    turns: list[GoldenTurn] = []
    errors: list[str] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                turns.append(GoldenTurn.model_validate_json(line))
            except Exception as e:  # noqa: BLE001 - collect all row errors, fail loud once
                errors.append(f"line {lineno}: {e}")

    if errors:
        raise ValueError(
            f"{len(errors)} malformed row(s) in {path}:\n" + "\n".join(errors)
        )
    if not turns:
        raise ValueError(f"no valid turns found in {path}")
    return turns


def group_by_conversation(turns: list[GoldenTurn]) -> dict[str, list[GoldenTurn]]:
    grouped: dict[str, list[GoldenTurn]] = {}
    for turn in turns:
        grouped.setdefault(turn.conversation_id, []).append(turn)

    for conversation_id, conv_turns in grouped.items():
        turn_ids = [t.turn_id for t in conv_turns]
        if len(set(turn_ids)) != len(turn_ids):
            raise ValueError(
                f"duplicate turn_id(s) within conversation {conversation_id!r}: {turn_ids}"
            )
        conv_turns.sort(key=lambda t: t.turn_id)

    return grouped


def to_canonical_document(
    conversation_id: str,
    turns: list[NormalizedTurn],
    *,
    trace_id: str | None = None,
    schema_version: str = "1.0",
    agent_id: str | None = None,
    adapter_id: str | None = None,
    availability: dict[str, str] | None = None,
) -> dict:
    """Assemble one canonical-schema-shaped document (agent-eval-main's
    normalized_trajectory.schema.json) from a conversation's ordered
    NormalizedTurns. This is the one place structural (not naming)
    translation lives: turn-incremental runtime objects -> one whole-
    trajectory document, dropping internal-only fields (StateChange.source,
    NormalizedNode.node_type) that have no schema home, and mapping
    turn-level tool_calls to turn.unassociated_tool_calls (never fabricating
    a node/tool association the adapter didn't actually observe)."""

    def _tool_call(tc: ToolCall) -> dict:
        return {
            "tool_id": tc.tool_id,
            "sequence": tc.sequence if tc.sequence is not None else 0,
            "input": tc.arguments,
            "output": tc.result if isinstance(tc.result, dict) else {},
            "status": tc.status,
        }

    def _state_change(sc: StateChange) -> dict:
        return {"key": sc.key, "old": sc.old, "new": sc.new}

    def _node(n: NormalizedNode, index: int) -> dict:
        output = dict(n.output)
        if n.node_type is not None:
            output.setdefault("node_type", n.node_type)
        return {
            "node_id": n.node_id,
            "sequence": n.sequence if n.sequence is not None else index,
            "output": output,
            "state_changes": [_state_change(sc) for sc in n.state_changes],
            "tool_calls": [_tool_call(tc) for tc in n.tool_calls],
        }

    def _turn(t: NormalizedTurn) -> dict:
        turn_doc: dict = {
            "turn_id": t.turn_id,
            "input": {},  # NormalizedTurn carries no raw user_input; golden-side input lives in GoldenTurn.user_input
            "output": {"agent": t.response} if t.response is not None else {},
            "nodes": [_node(n, i + 1) for i, n in enumerate(t.nodes)],
            "unassociated_tool_calls": [_tool_call(tc) for tc in t.tool_calls],
        }
        if t.routing is not None:
            turn_doc["routing"] = {"path": t.routing.path, "target": t.routing.target}
        return turn_doc

    doc: dict = {
        "trace_id": trace_id or conversation_id,
        "schema_version": schema_version,
        "availability": availability
        or {
            "nodes": "available",
            "state": "available",
            "tools": "available",
            "routing": "available",
            "outcome": "available",
        },
        "turns": [_turn(t) for t in turns],
    }
    if agent_id is not None:
        doc["agent_id"] = agent_id
    if adapter_id is not None:
        doc["adapter_id"] = adapter_id

    last_outcome = next((t.outcome for t in reversed(turns) if t.outcome is not None), None)
    if last_outcome is not None:
        doc["outcome"] = {"id": last_outcome.id, "attributes": last_outcome.attributes}

    return doc
