# Evaluating the call-support agent

How the standalone [call-support agent](../../call_support_agent/README.md) (LangGraph + LangChain, fictional telecom "Nimbus Telecom") is evaluated with atf-eval. This folder is everything the **agent side** needs: an adapter, golden data and a scenario runner. The evaluation itself is done by the atf-eval API server ([docs/API.md](../../docs/API.md)).

## Files

The agent is its own project in [`../../call_support_agent/`](../../call_support_agent/) (built-in tracing, no `atf_eval` dependency). Here:

| File | Role |
|---|---|
| `adapter.py` | the only bridge: calls `CallSupportAgent.chat()` and converts the result to a `NormalizedTurn` |
| `golden_dataset.jsonl` | 3 conversations / 6 turns: the customer inputs to drive the agent with and the expected trajectory (for `atf-eval submit` / `run`) |
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

## Run the evaluation

**Through the API** (the intended setup):

```bash
# terminal 1: the atf-eval server (needs ANTHROPIC_API_KEY in its env or .env for RS / Group 1-5)
pip install -e ".[server]" && atf-eval serve --port 8000

# terminal 2: the agent side
pip install -e ".[client]" -e call_support_agent
atf-eval submit --server http://localhost:8000 \
  --adapter agents.call_support.adapter:CallSupportAdapter \
  --dataset agents/call_support/golden_dataset.jsonl --output-dir results/
```

That runs the agent live over the dataset's customer inputs, submits what it did, and downloads the
dashboard, CSVs and `result.json`. Add `--no-llm-evals` to skip the Group 1-5 metrics, or
`--judge-model claude-haiku-5-5` for a cheaper judge. **Without a server**, `atf-eval run` scores in-process (`atf-eval run --dataset agents/call_support/golden_dataset.jsonl --adapter agents.call_support.adapter:CallSupportAdapter`), and `atf-eval dashboard` scores traces the agent already saved:

```bash
python -m agents.call_support.record_traces        # plays the scripted customers; the agent traces itself
atf-eval dashboard --golden-dir agents/call_support/golden \
  --scenarios agents/call_support/traces --output-dir agents/call_support/results --no-llm-evals
```

Tracing itself (what is saved, LangSmith) is described in the agent's README. Running through the adapter saves traces too, to `$CALL_SUPPORT_TRACE_DIR` or `./traces`.

A trace is scored against the golden named by its `golden_scenario` (`call-support-chat --meta golden_scenario=<name>`); without one the dashboard can't tell which golden to use. The dashboard also scores each golden against itself as a baseline.

## Results

Scored with `atf-eval dashboard` from the traces the agent saved while running (RS judged by the default Claude judge; Groups 1-5 not run). Through the API, the three golden conversations also scored ATF 1.000 with a Haiku judge, and the Group 1-5 metrics added a signal the deterministic ones miss (e.g. `correctness: FAIL` on the no-outage conversation):

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
