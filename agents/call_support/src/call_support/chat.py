"""Talk to the call-support agent in the terminal. Every turn is traced.

    call-support-chat [--conversation-id demo_1] [--meta golden_scenario=x ...]

Traces land in ./traces (or $CALL_SUPPORT_TRACE_DIR / --trace-dir); --meta adds
top-level fields to the trace file. Ctrl-D or an empty line to quit.
"""
from __future__ import annotations

import argparse
import time

from call_support.agent import CallSupportAgent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--conversation-id", default=time.strftime("chat_%Y%m%dT%H%M%S"))
    p.add_argument("--meta", action="append", default=[], metavar="KEY=VALUE", help="extra trace field (repeatable)")
    p.add_argument("--trace-dir", default=None)
    args = p.parse_args()

    agent = CallSupportAgent(trace_dir=args.trace_dir)
    agent.annotate_trace(args.conversation_id, **dict(m.split("=", 1) for m in args.meta))
    turns = 0
    while True:
        try:
            text = input("you> ").strip()
        except EOFError:
            break
        if not text:
            break
        print(f"agent> {agent.chat(args.conversation_id, text).reply}")
        turns += 1
    if turns:
        print(f"trace saved: {agent.traces.directory / (args.conversation_id + '.json')}")


if __name__ == "__main__":
    main()
