# How to write an adapter for any agent

An adapter is one class with one method, `run_turn`. It calls your agent for a single turn and returns what the agent did as a `NormalizedTurn`. That is the whole contract (`src/atf_eval/adapter.py:TrajectoryAgent`). A complete working example, about 40 lines: `agents/call_support/adapter.py`.

## Steps

1. Create `adapter.py` in your agent's folder.
2. Write the class (template below).
3. Write `golden_dataset.jsonl`: one expected turn per row, with `conversation_id`, `turn_id`, `user_input`, and the expected `nodes`, `state_changes`, `tool_calls`, `routing` and `outcome`.
4. Run from the agent folder:
   ```bash
   atf-eval submit --server http://127.0.0.1:8000 \
     --adapter adapter:MyAgentAdapter --dataset golden_dataset.jsonl --output-dir results/
   ```

## Template

```python
from atf_eval.normalized import (
    NormalizedTurn, NormalizedNode, StateChange, ToolCall, Routing, Outcome)

class MyAgentAdapter:
    availability = {"state": "unavailable"}     # optional, see Rules

    def __init__(self):
        self.agent = MyAgent()                   # build your agent once

    def run_turn(self, conversation_id, turn_id, user_input, history):
        r = self.agent.chat(conversation_id, user_input)     # 1. run YOUR agent

        nodes = [                                            # 2. map its trace
            NormalizedNode(
                node_id=n.name,
                state_changes=[StateChange(key=k, old=o, new=v) for k, o, v in n.diffs],
                tool_calls=[ToolCall(tool_id=t.name, arguments=t.args, result=t.output)
                            for t in n.tools],
            )
            for n in r.steps
        ]
        return NormalizedTurn(                               # 3. return it
            conversation_id=conversation_id,
            turn_id=turn_id,
            customer_input=user_input,
            nodes=nodes,
            state_changes=[c for n in nodes for c in n.state_changes],   # flat copies
            tool_calls=[t for n in nodes for t in n.tool_calls],
            routing=Routing(path=r.path, target=r.target),
            outcome=Outcome(id=r.outcome_id, attributes=r.attrs) if r.done else None,
            response=r.reply,
        )
```

## What each field is for

| Field | Used by | Notes |
|---|---|---|
| `nodes` (`node_id`, in order) | NTS | the steps, graph nodes or agents that ran |
| `state_changes` (`key`, `old`, `new`) | STS | only real changes count; attach to the node that caused them if known |
| `tool_calls` (`tool_id`, `arguments`) | TIS | attach to nodes if you can, otherwise compared per turn |
| `routing` (`path`, `target`) | — (not scored) | optional trace evidence |
| `outcome` (`id`, `attributes`) | OS | set only on the conversation's final turn |
| `response` | Group 1-5 | the agent's reply text |
| `customer_input` | Group 1-5 | the user message for this turn |

## Rules

- **`history`**: the normalized turns before this one. Use it to rebuild context only if your agent is stateless; if it keeps its own session keyed on `conversation_id`, ignore it.
- **Names must match the golden file**: the `node_id`, `tool_id`, state `key` and outcome `id` you emit must use the same names as the golden dataset.
- **Missing evidence**: if your agent can't expose something, declare it with `availability = {"state": "unavailable"}`. That metric becomes N/A and is skipped instead of scored 0. Undeclared dimensions count as available.
- **No routing or outcome**: leave `routing` or `outcome` as `None` and they are skipped.
- **Errors**: if `run_turn` raises, that conversation stops and the turns collected so far are still submitted. The failed turn is scored 0, not N/A.
- **Imports**: the adapter is imported from the current directory, so run `atf-eval submit` from the folder containing `adapter.py`.

## Reference: a working adapter

`agents/call_support/adapter.py` is the adapter for the call-support agent (LangGraph). It runs and scores end to end (ATF 1.000 on its golden dataset). Run it with `--adapter adapter:CallSupportAdapter` from `agents/call_support/`.

```python
from __future__ import annotations

from atf_eval.normalized import NormalizedNode, NormalizedTurn, Outcome, Routing, StateChange, ToolCall
from call_support import CallSupportAgent


class CallSupportAdapter:
    def __init__(self, agent: CallSupportAgent | None = None, trace_dir=None, save_traces: bool = True) -> None:
        self.agent = agent or CallSupportAgent(trace_dir=trace_dir, save_traces=save_traces)

    def run_turn(
        self,
        conversation_id: str,
        turn_id: int,
        user_input: str,
        history: list[NormalizedTurn],
    ) -> NormalizedTurn:
        r = self.agent.chat(conversation_id, user_input, turn_id=turn_id)       # run the agent
        nodes = [
            NormalizedNode(
                node_id=n.node_id,
                state_changes=[StateChange(key=c["key"], old=c["old"], new=c["new"]) for c in n.state_changes],
                tool_calls=[
                    ToolCall(tool_id=c.tool, arguments=c.args, result=c.result,
                             status="success" if c.result else "empty")
                    for c in n.tool_calls
                ],
            )
            for n in r.nodes
        ]
        return NormalizedTurn(
            conversation_id=conversation_id,
            turn_id=turn_id,
            customer_input=user_input,
            nodes=nodes,
            state_changes=[c for n in nodes for c in n.state_changes],
            tool_calls=[t for n in nodes for t in n.tool_calls],
            routing=Routing(path=r.route["path"], target=r.route["target"]) if r.route else None,
            outcome=Outcome(id=r.outcome["id"], attributes=r.outcome["attributes"]) if r.outcome else None,
            response=r.reply,
        )
```

The matching golden dataset is `agents/call_support/golden_dataset.jsonl`.
