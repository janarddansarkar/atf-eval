"""Run the call-support agent over scripted customer conversations and write one
observed-trace JSON per conversation (the same shape as agent-eval's observed
fixtures), ready for `atf-eval dashboard`.

    python -m agents.call_support.record_traces            # -> agents/call_support/traces/

Each scenario names the golden it is meant to follow (`golden_scenario`) and a
`deviation_type`: "none" for the scripted path, otherwise how the customer
strays from it. The traces hold what the agent really did, LLM replies included,
so re-recording can change them; re-record, then re-score.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from agents.call_support.adapter import CallSupportAdapter
from agents.call_support.graph import MODEL, build_graph
from atf_eval.normalized import NormalizedTurn

HERE = Path(__file__).parent

# (trace name, golden scenario, deviation_type, description, customer messages)
SCENARIOS = [
    ("outage_as_scripted", "support_outage_resolved", "none",
     "Customer follows the golden script.",
     ["Hi, my internet is down. My account is ACC-1001.", "Okay thanks, that's all I needed."]),
    ("outage_paraphrased", "support_outage_resolved", "paraphrased_input",
     "Same journey, different wording (and 'cheers' instead of a thanks).",
     ["Wifi's been dropping all day, acc ACC-1001", "Nope, nothing else. Cheers!"]),
    ("billing_as_scripted", "support_billing_dispute", "none",
     "Customer follows the golden script.",
     ["Hello, I'm ACC-1002. Why is my bill so high this month?",
      "That doesn't look right. I want to dispute it with a human."]),
    ("billing_accepts_explanation", "support_billing_dispute", "customer_accepts_instead_of_disputing",
     "Customer is satisfied with the explanation, so no ticket is raised (outcome differs from the golden).",
     ["Hello, I'm ACC-1002. Why is my bill so high this month?", "Oh I see, the roaming charge. That's all, thanks."]),
    ("slow_as_scripted", "support_slow_no_outage", "none",
     "Customer follows the golden script.",
     ["My internet has been really slow lately.", "Sure, it's ACC-1003."]),
    ("slow_wrong_account_first", "support_slow_no_outage", "extra_turn_wrong_account",
     "Customer first gives an unknown account id, so the agent must ask again (3 turns vs the golden's 2).",
     ["My internet has been really slow lately.", "It's ACC-9999.", "Sorry, ACC-1003."]),
    ("slow_unclear_after_account", "support_slow_no_outage", "off_script_clarification",
     "After verifying, the customer asks something vague instead of repeating the issue.",
     ["Hi, ACC-1003 here.", "Honestly, it's just not great."]),
]


def _raw_turn(t: NormalizedTurn) -> dict:
    seq = 0
    nodes, flat_tools = [], []
    for i, n in enumerate(t.nodes, 1):
        tools = []
        for c in n.tool_calls:
            seq += 1
            call = {"tool_name": c.tool_id, "sequence_index": seq, "input": c.arguments, "output": c.result}
            tools.append(call)
            flat_tools.append(call)
        nodes.append({
            "node_id": n.node_id,
            "sequence_index": i,
            "state_changes": [{"key": c.key, "old": c.old, "new": c.new} for c in n.state_changes],
            "tool_calls": tools,
        })
    return {
        "turn_id": t.turn_id,
        "conversation": {"customer": t.customer_input, "agent": t.response},
        "nodes": nodes,
        "tool_calls": flat_tools,
        "state_changes": [{"key": c.key, "old": c.old, "new": c.new} for c in t.state_changes],
        "routing": {"path": t.routing.path, "target": t.routing.target} if t.routing else None,
    }


def record(name, golden, deviation, description, messages, graph=None) -> dict:
    adapter = CallSupportAdapter(graph or build_graph())
    history: list[NormalizedTurn] = []
    for i, msg in enumerate(messages, 1):
        history.append(adapter.run_turn(name, i, msg, history))
    last = history[-1].outcome
    doc = {
        "trace_id": f"{name}",
        "scenario_id": name,
        "golden_scenario": golden,
        "trace_type": "observed",
        "deviation_type": deviation,
        "deviation": {"description": description},
        "agent": {"framework": "langgraph", "model": MODEL},
        "availability": {"nodes": "available", "state": "available", "tools": "available",
                         "routing": "available", "outcome": "available"},
        "turns": [_raw_turn(t) for t in history],
    }
    if last:
        doc["outcome"] = {"id": last.id, "attributes": last.attributes}
    return doc


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default=str(HERE / "traces"))
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for sc in SCENARIOS:
        t0 = time.time()
        doc = record(*sc)
        (out / f"{sc[0]}.json").write_text(json.dumps(doc, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"recorded {sc[0]:30s} {len(doc['turns'])} turns  {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
