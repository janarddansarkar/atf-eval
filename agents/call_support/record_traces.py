"""Drive the traced call-support agent through scripted customer conversations.

    python -m agents.call_support.record_traces

The agent saves its own traces as it runs (tracing.py); this script only plays
the customer. Each scenario names the golden it is meant to follow
(`golden_scenario`) and a `deviation_type`: "none" for the scripted path,
otherwise how the customer strays from it. Replies come from the live LLM, so
re-running can change them.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from agents.call_support.adapter import CallSupportAdapter
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


def record(name, golden, deviation, description, messages, trace_dir) -> Path:
    """Run one scripted conversation through the traced agent; return its trace file."""
    agent = CallSupportAdapter(trace_dir=trace_dir)
    agent.annotate_trace(name, golden_scenario=golden, deviation_type=deviation,
                         deviation={"description": description})
    history: list[NormalizedTurn] = []
    for i, msg in enumerate(messages, 1):
        history.append(agent.run_turn(name, i, msg, history))
    return agent.traces.directory / f"{name}.json"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default=None, help="trace directory (default: the agent's own)")
    args = p.parse_args()
    for sc in SCENARIOS:
        t0 = time.time()
        path = record(*sc, trace_dir=args.out)
        print(f"ran {sc[0]:30s} {len(sc[4])} turns  {time.time() - t0:.1f}s  -> {path}")


if __name__ == "__main__":
    main()
