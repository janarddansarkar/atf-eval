"""Talk to the call-support agent in the terminal. Every turn is traced.

    python -m agents.call_support.chat [--conversation-id demo_1] [--golden support_outage_resolved]

Traces land in agents/call_support/traces/ (see tracing.py); `--golden` records
which golden scenario the conversation is meant to follow so the dashboard can
score it. Ctrl-D or an empty line to quit.
"""
from __future__ import annotations

import argparse
import time

from agents.call_support.adapter import CallSupportAdapter


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--conversation-id", default=time.strftime("chat_%Y%m%dT%H%M%S"))
    p.add_argument("--golden", help="golden scenario this conversation should follow (for scoring)")
    p.add_argument("--trace-dir", default=None)
    args = p.parse_args()

    agent = CallSupportAdapter(trace_dir=args.trace_dir)
    if args.golden:
        agent.annotate_trace(args.conversation_id, golden_scenario=args.golden)
    history = []
    while True:
        try:
            text = input("you> ").strip()
        except EOFError:
            break
        if not text:
            break
        turn = agent.run_turn(args.conversation_id, len(history) + 1, text, history)
        history.append(turn)
        print(f"agent> {turn.response}")
    if history:
        print(f"trace saved: {agent.traces.directory / (args.conversation_id + '.json')}")


if __name__ == "__main__":
    main()
