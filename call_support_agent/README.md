# Call-support agent

A standalone phone-support agent for a fictional telecom ("Nimbus Telecom"), built with **LangGraph** and **LangChain** (`langchain-anthropic`, `@tool` tools). It has built-in tracing and no dependency on any evaluation framework.

## Install and run

```bash
cd call_support_agent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env            # add ANTHROPIC_API_KEY (optional: see "No API key" below)

call-support-chat               # talk to it; Ctrl-D or an empty line to quit
pytest                          # offline tests, no API key needed
```

Or from Python:

```python
from call_support import CallSupportAgent

agent = CallSupportAgent()
turn = agent.chat("conv-1", "My internet is down. My account is ACC-1001.")
print(turn.reply)          # the agent's answer
print(turn.nodes)          # graph nodes that ran, with tool calls and state changes
print(turn.route, turn.outcome, turn.latency_ms)
```

## How it works

```
START ─▶ identify_customer ─▶ classify_intent ─▶ billing_node ─────────────▶ END
          (if not verified)    (LLM, structured)  outage_node ─(outage)────▶ END
START ─▶ classify_intent        (verified)                   └─(none)─▶ escalate_node ─▶ END
                                                  escalate_node / close_node / clarify_node ─▶ END
```

| File | Role |
|---|---|
| `graph.py` | the LangGraph `StateGraph` (nodes, conditional edges, `MemorySaver`; one conversation = one thread) |
| `tools.py` | LangChain tools over an in-memory CRM: `lookup_account`, `get_bill`, `check_outage`, `create_ticket` |
| `agent.py` | `CallSupportAgent.chat()`: runs a turn, returns a `TurnResult`, saves the trace |
| `tracing.py` | trace capture (below) |
| `chat.py` | the `call-support-chat` terminal client |

- **Verification:** without an account id (`ACC-1234`) the agent asks for one and remembers the complaint; once given, it classifies the original complaint. Test accounts: `ACC-1001` (outage in its area), `ACC-1002` (bill with a roaming charge), `ACC-1003` (no outage).
- **LLM:** Claude (`claude-haiku-5-5`; set `CALL_SUPPORT_MODEL` to change) classifies intent with structured output and writes the replies.
- **No API key:** it falls back to keyword rules and canned replies, so everything still runs offline.
- **Routing targets:** `self_service`, `human_agent`, `customer`. Outcomes: `ticket_created`, `issue_resolved`.

## Tracing

Every `chat()` call saves its trace (turn off with `CallSupportAgent(save_traces=False)`):

| Where | What |
|---|---|
| `traces/<conversation_id>.json` | the conversation's turns: nodes, tool calls, state changes, routing, outcome, latency. Rewritten after every turn. |
| `traces/raw/<conversation_id>.jsonl` | raw LangChain/LangGraph callback events: node, tool and LLM calls with timestamps, durations, prompts, replies, token usage |
| LangSmith (optional) | set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` (and `LANGSMITH_PROJECT`); runs are named `call_support/<conversation>/turn<N>` and tagged with the conversation id. Not tested: no LangSmith key was available. |

Change the folder with `CALL_SUPPORT_TRACE_DIR`, `--trace-dir`, or `CallSupportAgent(trace_dir=...)`. `call-support-chat --meta key=value` adds fields to the trace file.

## Evaluating it

The agent knows nothing about evaluation. An ATF evaluation lives outside this project, in `../agents/call_support/` (adapter, golden scenarios, scoring); it reads this agent through `chat()` and the trace files.
