from __future__ import annotations

from typing import Protocol, runtime_checkable

from atf_eval.normalized import NormalizedTurn


@runtime_checkable
class TrajectoryAgent(Protocol):
    """Your agent, wrapped to satisfy this one method.

    `run_turn` both invokes your agent for this turn AND normalizes its trace
    into the common `NormalizedTurn` shape. `history` is the normalized
    observed trajectory of every prior turn in this conversation (in order),
    so a stateful/multi-turn agent can rebuild its own context.

    Optionally, the adapter declares which evidence its traces expose via an
    `availability` attribute, e.g. `{"state": "unavailable"}` for an agent
    whose traces never record state. A dimension declared unavailable makes
    its metric N/A; any dimension not declared is treated as available, so
    missing events on it count as agent failures (METRICS.md §11
    "Insufficient evidence", §21).
    """

    def run_turn(
        self,
        conversation_id: str,
        turn_id: int,
        user_input: str,
        history: list[NormalizedTurn],
    ) -> NormalizedTurn: ...
