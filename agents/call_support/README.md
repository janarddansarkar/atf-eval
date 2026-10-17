# Call-support agent (LangGraph)

A phone-support agent for a fictional telecom ("Nimbus Telecom"), built with **LangGraph** + **LangChain** (`langchain-anthropic`, `@tool` tools), wrapped as an `atf_eval` `TrajectoryAgent` and scored with `atf-eval`.

## The agent

```
START ─▶ identify_customer ─▶ classify_intent ─▶ billing_node ─────────────▶ END
          (if not verified)    (LLM, structured)  outage_node ─(outage)────▶ END
START ─▶ classify_intent        (verified)                   └─(none)─▶ escalate_node ─▶ END
                                                  escalate_node / close_node / clarify_node ─▶ END
```

| File | Role |
|---|---|
| `graph.py` | The LangGraph `StateGraph` (nodes, conditional edges, `MemorySaver` checkpointer; one conversation = one thread) |
| `tools.py` | LangChain tools over an in-memory CRM: `lookup_account`, `get_bill`, `check_outage`, `create_ticket` |
| `adapter.py` | `CallSupportAdapter`: runs one turn and normalizes the trace for atf-eval |
| `golden_dataset.jsonl` | 3 conversations / 6 turns of expected behaviour (for live `atf-eval run`) |
| `golden/*.json` | the same 3 goldens in the canonical trace format (for `atf-eval dashboard`) |
| `record_traces.py` | runs the agent over 7 scripted conversations and writes the traces |
| `traces/*.json` | the recorded observed traces |

- **Verification:** if the caller hasn't given an account id (`ACC-1234`), the agent asks for it and remembers what they said; once given, it classifies the *original* complaint.
- **LLM use:** Claude (`claude-haiku-5-5` by default, `CALL_SUPPORT_MODEL` to change) classifies intent (structured output) and writes replies. With no `ANTHROPIC_API_KEY` it falls back to keyword rules and canned replies, so it still runs offline.
- **Routing targets:** `self_service`, `human_agent`, `customer`.

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

## Record traces, then score them

```bash
python -m agents.call_support.record_traces        # runs the live agent -> traces/*.json
atf-eval dashboard --golden-dir agents/call_support/golden \
  --scenarios agents/call_support/traces --output-dir agents/call_support/results \
  --no-llm-evals                                    # drop this flag to add the Group 1-5 LLM metrics
```

Each trace names the golden it follows (`golden_scenario`) and a `deviation_type`. Re-recording calls the LLM again, so replies (and RS) can differ slightly; the committed traces are the ones scored below. The dashboard also scores each golden against itself as a baseline.

## Results

Scored from the recorded traces (RS judged by the default Claude judge; Groups 1-5 not run):

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
