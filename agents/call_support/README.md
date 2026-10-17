# Call-support agent (LangGraph)

A phone-support agent for a fictional telecom ("Nimbus Telecom"), built with **LangGraph** + **LangChain** (`langchain-anthropic`, `@tool` tools), wrapped as an `atf_eval` `TrajectoryAgent` and scored with `atf-eval`.

## The agent

The agent is its own project in [`../../call_support_agent/`](../../call_support_agent/) (LangGraph + LangChain, built-in tracing, no `atf_eval` dependency). This folder is only its evaluation:

| File | Role |
|---|---|
| `adapter.py` | the only bridge: calls `CallSupportAgent.chat()` and converts the result to a `NormalizedTurn` |
| `golden_dataset.jsonl` | 3 conversations / 6 turns of expected behaviour (for live `atf-eval run`) |
| `golden/*.json` | the same 3 goldens in the canonical trace format (for `atf-eval dashboard`) |
| `record_traces.py` | plays 7 scripted customers against the agent; the agent saves the traces itself |
| `traces/` | the traces it saved: `<conversation>.json` and `raw/<conversation>.jsonl` |

## How the trace maps to ATF

| ATF evidence | Source in the agent |
|---|---|
| nodes | LangGraph nodes that ran, in order (`stream_mode="updates"`) |
| state changes | diffs of `authenticated`, `intent`, `ticket_id`, `resolved`, attributed to the node that set them |
| tool calls | the per-turn tool log, attached to the node that made each call |
| routing | each node sets `{path, target}` |
| outcome | `escalate_node` → `ticket_created`; `close_node` → `issue_resolved` (attribute `category`) |
| response | the agent's reply, which RS's LLM judge reads |

## Run it

```bash
pip install -r agents/call_support/requirements.txt   # plus: pip install -e .
python -m atf_eval run \
  --dataset agents/call_support/golden_dataset.jsonl \
  --adapter agents.call_support.adapter:CallSupportAdapter \
  --output-dir results/            # add --no-routing-judge to skip the RS judge
```

## Run the evaluation

```bash
pip install -e .                      # atf-eval, from the repo root
pip install -e call_support_agent     # the agent
python -m agents.call_support.record_traces        # plays the scripted customers; the agent traces itself
atf-eval dashboard --golden-dir agents/call_support/golden \
  --scenarios agents/call_support/traces --output-dir agents/call_support/results \
  --no-llm-evals                                    # drop the flag to add Group 1-5
# or score it live, without saved traces:
atf-eval run --dataset agents/call_support/golden_dataset.jsonl \
  --adapter agents.call_support.adapter:CallSupportAdapter
```

Tracing itself (what is saved, LangSmith) is described in the agent's README. Running through the adapter saves traces too, to `$CALL_SUPPORT_TRACE_DIR` or `./traces`.

A trace is scored against the golden named by its `golden_scenario` (`call-support-chat --meta golden_scenario=<name>`); without one the dashboard can't tell which golden to use. The dashboard also scores each golden against itself as a baseline.

## Results

Scored from the traces the agent saved while running (RS judged by the default Claude judge; Groups 1-5 not run):

| Trace | Golden | Deviation | ATF | NTS | STS | TIS | RS | OS |
|---|---|---|---|---|---|---|---|---|
| outage_as_scripted | outage_resolved | none | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| outage_paraphrased | outage_resolved | paraphrased input | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| billing_as_scripted | billing_dispute | none | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| billing_accepts_explanation | billing_dispute | customer doesn't dispute | 0.682 | 0.800 | 0.500 | 0.883 | 0.850 | 0.500 |
| slow_as_scripted | slow_no_outage | none | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| slow_wrong_account_first | slow_no_outage | extra turn, unknown account | 0.909 | 0.900 | 1.000 | 0.925 | 0.500 | 1.000 |
| slow_unclear_after_account | slow_no_outage | vague follow-up | 0.370 | 0.400 | 0.333 | 0.767 | 0.350 | 0.000 |

The agent reproduces the golden path when the customer follows it (even paraphrased). The three lower scores are the customer going off-script, not agent bugs: the "deviations" test how ATF scores a *different but reasonable* journey against a fixed golden. Note that adapter errors are scored 0, not N/A.
