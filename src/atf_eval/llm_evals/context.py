"""Evidence bundles the LLM metrics read.

Built from the same `NormalizedTurn` lists the deterministic metrics use
(expected == golden reference, observed == the trace under evaluation), so
the LLM layer needs no separate trace format. Group 5 splits into a timing
tier (Interruption Count/Recovery/Understanding, Turn-taking Quality,
Perceived Response Latency -- judgeable from structured turn timestamps and
interruption events, no audio needed) and an acoustic tier (Speech
Naturalness, Prosody/Tone, ... -- needs real audio, which the canonical
trajectory does not carry). `has_timing` reflects whether any observed turn
actually carries a `TurnTiming`; `has_audio` stays False until a real
audio-evidence pipeline exists.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from atf_eval.metrics.evidence import turn_state_changes, turn_tool_calls
from atf_eval.normalized import NormalizedTurn, TurnTiming


@dataclass
class ToolEvidence:
    tool_id: str
    arguments: dict[str, Any]
    result: Any = None  # tool call output/return value, when the trace carries one


@dataclass
class TurnContext:
    turn_id: int
    customer_input: str
    agent_response: str
    prior_exchanges: list[tuple[str, str]]        # [(customer, agent), ...] before this turn
    reference_response: str | None                # golden agent response for this turn
    observed_nodes: list[str] = field(default_factory=list)
    reference_nodes: list[str] = field(default_factory=list)
    observed_tools: list[ToolEvidence] = field(default_factory=list)
    reference_tools: list[ToolEvidence] = field(default_factory=list)
    observed_state_changes: list[tuple[str, Any, Any]] = field(default_factory=list)
    reference_state_changes: list[tuple[str, Any, Any]] = field(default_factory=list)
    observed_routing: tuple[str | None, str | None] | None = None
    reference_routing: tuple[str | None, str | None] | None = None
    observed_timing: TurnTiming | None = None

    def asks_question(self) -> bool:
        return "?" in (self.agent_response or "")

    def evidence_block(self) -> str:
        lines = []
        if self.reference_response:
            lines.append(f"reference (golden) agent response: {self.reference_response}")
        if self.reference_nodes:
            lines.append("reference nodes: " + ", ".join(self.reference_nodes))
        if self.observed_nodes:
            lines.append("observed nodes: " + ", ".join(self.observed_nodes))
        if self.reference_tools:
            lines.append(
                "reference tools: "
                + ", ".join(_format_tool_evidence(t) for t in self.reference_tools)
            )
        if self.observed_tools:
            lines.append(
                "observed tools: "
                + ", ".join(_format_tool_evidence(t) for t in self.observed_tools)
            )
        if self.observed_state_changes:
            lines.append(
                "observed state changes: "
                + "; ".join(f"{k}: {o!r}->{n!r}" for k, o, n in self.observed_state_changes)
            )
        if self.observed_timing:
            timing_line = _format_timing(self.observed_timing)
            if timing_line:
                lines.append(timing_line)
        return "\n".join(lines) if lines else "(no additional trajectory evidence)"


@dataclass
class ConversationContext:
    conversation_id: str
    turns: list[TurnContext]
    reference_outcome: dict[str, Any] | None
    observed_outcome: dict[str, Any] | None
    has_audio: bool = False
    has_timing: bool = False

    def transcript(self) -> str:
        out = []
        for t in self.turns:
            out.append(f"[turn {t.turn_id}] customer: {t.customer_input}")
            out.append(f"[turn {t.turn_id}] agent: {t.agent_response}")
        return "\n".join(out)

    def trajectory_evidence(self) -> str:
        """Conversation-wide node/tool/state evidence, turn by turn -- the
        overall-level judge prompt's counterpart to TurnContext.evidence_block().
        Without this, an overall-level grounding metric (Faithfulness,
        Groundedness, Citation/Evidence Accuracy, ...) would only see the raw
        transcript + final outcome, missing the tool results that actually
        ground (or contradict) a claim made mid-conversation."""
        lines: list[str] = []
        for t in self.turns:
            block = t.evidence_block()
            if block and block != "(no additional trajectory evidence)":
                lines.append(f"[turn {t.turn_id}]\n{block}")
        return "\n".join(lines) if lines else "(no additional trajectory evidence available)"


def _format_tool_evidence(t: ToolEvidence) -> str:
    """Include the tool's return value when the trace carries one -- this is
    what lets Faithfulness/Groundedness/Citation-Accuracy check a claim
    against what was actually retrieved, not just against the golden
    reference text."""
    base = f"{t.tool_id}({t.arguments})"
    return f"{base} -> {t.result!r}" if t.result is not None else base


def _format_timing(t: TurnTiming) -> str:
    """Structured timing evidence, in plain English -- what Group 5's
    timing-tier metrics (Interruption Count/Recovery/Understanding,
    Turn-taking Quality, Perceived Response Latency) read. No audio, just
    the timestamps/latency/interruption events a real voice adapter logs."""
    parts = []
    if t.customer_end_ms is not None and t.agent_start_ms is not None:
        parts.append(
            f"customer finished speaking at {t.customer_end_ms}ms, "
            f"agent started responding at {t.agent_start_ms}ms"
        )
    if t.response_latency_ms is not None:
        parts.append(f"response latency ~{t.response_latency_ms}ms")
    for i in t.interruptions:
        note = f" ({i.note})" if i.note else ""
        parts.append(f"interruption: {i.by} interrupted at {i.at_ms}ms{note}")
    return "timing: " + "; ".join(parts) if parts else ""


def _tools(turn: NormalizedTurn) -> list[ToolEvidence]:
    return [
        ToolEvidence(tool_id=tc.tool_id, arguments=dict(tc.arguments), result=tc.result)
        for tc in turn_tool_calls(turn)
    ]


def _state(turn: NormalizedTurn) -> list[tuple[str, Any, Any]]:
    return [(c.key, c.old, c.new) for c in turn_state_changes(turn)]


def _routing(turn: NormalizedTurn) -> tuple[str | None, str | None] | None:
    if turn.routing is None:
        return None
    return (turn.routing.path, turn.routing.target)


def build_context(
    conversation_id: str,
    expected_turns: list[NormalizedTurn],
    observed_turns: list[NormalizedTurn],
) -> ConversationContext:
    ref_by_id = {t.turn_id: t for t in expected_turns}
    prior: list[tuple[str, str]] = []
    turn_ctxs: list[TurnContext] = []

    for obs in observed_turns:
        ref = ref_by_id.get(obs.turn_id)
        customer = _turn_customer(obs) or (_turn_customer(ref) if ref else "") or ""
        agent = obs.response or ""
        tc = TurnContext(
            turn_id=obs.turn_id,
            customer_input=customer,
            agent_response=agent,
            prior_exchanges=list(prior),
            reference_response=ref.response if ref else None,
            observed_nodes=[n.node_id for n in obs.nodes],
            reference_nodes=[n.node_id for n in ref.nodes] if ref else [],
            observed_tools=_tools(obs),
            reference_tools=_tools(ref) if ref else [],
            observed_state_changes=_state(obs),
            reference_state_changes=_state(ref) if ref else [],
            observed_routing=_routing(obs),
            reference_routing=_routing(ref) if ref else None,
            observed_timing=obs.timing,
        )
        turn_ctxs.append(tc)
        prior.append((customer, agent))

    return ConversationContext(
        conversation_id=conversation_id,
        turns=turn_ctxs,
        reference_outcome=_last_outcome_attrs(expected_turns),
        observed_outcome=_last_outcome_attrs(observed_turns),
        has_timing=any(t.timing is not None for t in observed_turns),
    )


def _last_outcome_attrs(turns: list[NormalizedTurn]) -> dict[str, Any] | None:
    for turn in reversed(turns):
        if turn.outcome is not None:
            attrs = dict(turn.outcome.attributes)
            if turn.outcome.id is not None:
                attrs.setdefault("outcome_id", turn.outcome.id)
            return attrs
    return None


def _turn_customer(turn: NormalizedTurn | None) -> str | None:
    """NormalizedTurn doesn't carry the customer utterance directly; the
    dashboard/reference adapters stash it nowhere canonical, so this is a
    best-effort hook. build_context falls back to the reference turn's copy
    and finally to "" -- Relevance/Intent then go N/A if both are empty."""
    if turn is None:
        return None
    return getattr(turn, "customer_input", None)
