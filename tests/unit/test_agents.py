"""The example agents (agents/): each produces the Golden dataset's trajectory
for its two-turn conversations. Sage is exercised with a fake provider."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agents.kestrel.agent import Kestrel  # noqa: E402
from agents.marlow.agent import Marlow  # noqa: E402
from agents.sage.agent import IntentChoice, Sage  # noqa: E402


def _converse(agent, first, second="ok"):
    t1 = agent.run_turn("c", 1, first, [])
    t2 = agent.run_turn("c", 2, second, [t1])
    return t1, t2


def test_kestrel_hardship_then_confirm():
    t1, t2 = _converse(Kestrel(), "I can't pay this month, I lost my job.")
    assert [n.node_id for n in t1.nodes] == ["classify_intent", "hardship_node"]
    assert t1.routing.path == "hardship" and t1.tool_calls[0].tool_id == "eligibility_check"
    assert t2.outcome.id == "installment_arrangement_accepted"


def test_kestrel_unclear_input_asks_to_clarify():
    t1 = Kestrel().run_turn("c", 1, "hello", [])
    assert t1.routing.path == "clarify" and not t1.tool_calls


def test_marlow_records_stage_transitions():
    t1, t2 = _converse(Marlow(), "I forgot to pay, can I pay now?")
    assert [(c.key, c.old, c.new) for c in t1.state_changes] == [
        ("payment_intent", False, True),
        ("stage", "idle", "intent_open"),
    ]
    assert [(c.key, c.old, c.new) for c in t2.state_changes] == [("stage", "intent_open", "closed")]
    assert t2.outcome.id == "payment_received"


def _fake_provider(intent):
    def parse_structured(**kwargs):
        assert kwargs["output_model"] is IntentChoice
        if isinstance(intent, Exception):
            raise intent
        return IntentChoice(intent=intent)

    return SimpleNamespace(default_model="m", parse_structured=parse_structured)


def test_sage_uses_the_llm_intent():
    t1 = Sage(provider=_fake_provider("hardship")).run_turn("c", 1, "money is tight", [])
    assert t1.routing.path == "hardship"  # no keyword in the text -- only the LLM could say so


def test_sage_unclear_and_failure_paths():
    assert Sage(provider=_fake_provider("unclear")).run_turn("c", 1, "I can't pay", []).routing.path == "clarify"
    # a failing call falls back to Kestrel's keyword rules
    assert Sage(provider=_fake_provider(RuntimeError("boom"))).run_turn("c", 1, "I can't pay", []).routing.path == "hardship"
