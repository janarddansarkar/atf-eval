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
| `golden_dataset.jsonl` | 3 conversations / 6 turns of expected behaviour |

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

## Results

Live run (Claude Haiku for the agent, the default Claude judge for RS): **ATF 1.000**, NTS / STS / TIS / RS / OS all 1.000 on 3 conversations / 6 turns. The offline keyword mode also scores 1.000.

That is a first baseline, not a hard test: the golden dataset describes the flow the graph was designed for, and the utterances are clear. To stress it, add paraphrased, ambiguous or off-script conversations to the dataset and re-run.

Note that when the adapter raises (e.g. an API error), atf-eval scores that turn 0 rather than N/A, so a failed run shows up as a very low score.
