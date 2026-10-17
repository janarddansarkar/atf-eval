# atf-eval

[![GitHub](https://img.shields.io/badge/GitHub-2597688js%2Fatf--eval-blue?logo=github)](https://github.com/2597688js/atf-eval)

Standalone evaluation framework for **Agent Trajectory Fidelity (ATF)** — scores how
closely an agent's observed execution trajectory (nodes visited, state transitions, tool
calls, routing decisions, final outcome) matches an expected trajectory defined by a
golden dataset. See `METRICS.md` for the frozen scoring specification this package
implements (it lives in the separate `agent-eval` repo, checked out here as `agent-eval/`).

## Ways to run it

| | Command | Use when |
|---|---|---|
| **Local run** | `atf-eval run --dataset … --adapter module:Class` | you want scores for an agent in one process (no server) |
| **Dashboard on saved traces** | `atf-eval dashboard --golden-dir … --scenarios …` | you already have golden + observed trace JSON files |
| **API** | `atf-eval serve`, then `atf-eval submit --adapter …` | the agent lives elsewhere; the agent owner writes only an adapter ([docs/API.md](docs/API.md)) |

## Quickstart

```bash
pip install git+https://github.com/2597688js/atf-eval.git
mkdir my-atf-project && cd my-atf-project
atf-eval init
atf-eval run --dataset golden_dataset.jsonl --adapter adapter:MyAgentAdapter --no-routing-judge
```

`atf-eval init` generates a starter adapter + a matching golden dataset that runs with
`ATF score: 1.000` immediately — confirms the framework works before you touch anything.
Then edit `adapter.py` and `golden_dataset.jsonl` to point at your real agent.

Optional extras: `[server]` (FastAPI + uvicorn, for `atf-eval serve`), `[client]` (httpx, for
`atf-eval submit`), `[nvidia]` (the NVIDIA NIM judge provider), `[dev]` (tests + all of the above).

## Repository layout

| Path | What |
|---|---|
| `src/atf_eval/` | the framework: metrics (`metrics/`), LLM metrics (`llm_evals/`), judge providers (`llm_provider.py`), CLI, dashboard generator, API server (`api/`) and client (`client.py`) |
| `agent-eval/` | the frozen spec (`METRICS.md`), golden scenarios and observed-scenario fixtures for the motor-insurance example |
| `examples/` | a dependency-free mock agent + tiny golden dataset (framework smoke test) |
| `call_support_agent/` | a **standalone** LangGraph + LangChain call-support agent with built-in tracing; no `atf_eval` dependency |
| `agents/call_support/` | how that agent is evaluated: adapter, goldens, scripted scenarios, recorded traces |
| `docs/API.md` | the HTTP API reference |
| `tests/` | unit tests (`pytest`) |

**For step-by-step setup and run instructions (install, API keys, the `atf-eval
dashboard` command), see `HOW_TO_USE.md`.** This README is the API/conceptual
reference: the adapter contract, golden dataset schema, metric group definitions, and


Four of five metric groups (NTS, STS, TIS, OS) are fully deterministic — closed-form
formulas, no LLM calls. **RS is the one exception**: it's an LLM-based semantic judge
(METRICS.md §5) of whether the agent's routing choice was appropriate given the
conversation, not a field comparison. Without the selected provider's API key set (or with
`--no-routing-judge`), routing cannot be evaluated, so RS is N/A and excluded from ATF
(routing order is still reported as a diagnostic) — the rest of the framework needs no
API key at all.

The LLM judges (RS and the Group 1-5 metrics) support two providers, chosen with
`--provider` or the `ATF_LLM_PROVIDER` env var:

| Provider | Key env var | Default `--judge-model` |
|---|---|---|
| `anthropic` (default) | `ANTHROPIC_API_KEY` | `claude-opus-5` |
| `nvidia` (NIM, OpenAI-compatible; `NVIDIA_BASE_URL` overrides the endpoint) | `NVIDIA_API_KEY` | `openai/gpt-oss-20b` |

The NVIDIA provider needs the optional `openai` package: `pip install -e ".[nvidia]"`.
`--effort` applies only to Anthropic. All provider-specific code lives in
`src/atf_eval/llm_provider.py`; to add a provider, implement `parse_structured()` there
and register the class in `PROVIDERS`. Scores from different judge models are not
directly comparable, so the provider and model are recorded in each run's settings.

## atf-eval as an API

atf-eval can run as a FastAPI service so any agent, in any repo or framework, is evaluated by writing only an
adapter. The agent side runs the agent and submits what it did; the server scores it (all five deterministic
metrics, the LLM routing judge, the Group 1-5 LLM metrics) and serves the dashboard and CSVs. The server never
calls your agent, and judge API keys stay on the server.

```bash
pip install -e ".[server]" && atf-eval serve --port 8000          # server (ANTHROPIC_API_KEY in its env or .env)

pip install -e ".[client]"                                          # agent side
atf-eval submit --server http://localhost:8000 \
  --adapter my_adapter:MyAgentAdapter --dataset golden_dataset.jsonl --output-dir results/
```

Full reference — architecture, every flag and endpoint, request/response schemas, `AtfClient` — is in
[`docs/API.md`](docs/API.md). A complete worked example (standalone agent + adapter) is in
[`agents/call_support/`](agents/call_support/README.md).

## Integrate your own agent

Implement the `TrajectoryAgent` protocol from `atf_eval.adapter`:

```python
from atf_eval.normalized import NormalizedTurn, NormalizedNode, ToolCall, Routing, Outcome

class MyAgentAdapter:
    # Declare evidence your traces never expose, so those metrics are N/A instead of
    # scored as failures (METRICS.md §11 "Insufficient evidence"). Undeclared
    # dimensions count as available. Optional.
    availability = {"state": "unavailable"}

    def run_turn(self, conversation_id, turn_id, user_input, history):
        raw_trace = my_agent.invoke(user_input, history=history)
        nodes = [NormalizedNode(node_id=n) for n in raw_trace.nodes_visited]
        # attach state changes to the node that caused them if you know it;
        # otherwise the golden side's own fallback (last node) is a reasonable match
        if nodes:
            nodes[-1].state_changes = [...]
        return NormalizedTurn(
            conversation_id=conversation_id,
            turn_id=turn_id,
            nodes=nodes,
            state_changes=[...],  # flat copy, used as a fallback when nodes[] is empty
            tool_calls=[ToolCall(tool_id=t.name, arguments=t.args) for t in raw_trace.tool_calls],
            routing=Routing(path=raw_trace.path, target=raw_trace.target),
            outcome=Outcome(id=..., attributes={...}) if raw_trace.is_final_turn else None,
            response=raw_trace.response_text,  # feeds RS's LLM judge; omit if you don't want RS
        )
```

`run_turn` both invokes your agent for this turn *and* normalizes its trace — same pattern
as `rag_eval`'s `RAGPipeline.query()`. `history` is the normalized trajectory of every
prior turn in the conversation, so a stateful agent can rebuild its own context. Point the
CLI at it with `--adapter my_project.adapters:MyAgentAdapter`.

## Golden dataset format

JSONL, one turn per row, authored directly in the normalized trajectory schema (no
separate Golden Adapter in v1 — this dataset format *is* the normalized contract):

```json
{"conversation_id": "conv_01", "turn_id": 1, "user_input": "...",
 "nodes": [{"node_id": "path_selector", "node_type": "routing"}],
 "state_changes": [{"key": "discount_stage", "old": "none", "new": "offered", "node_id": "path_selector"}],
 "tool_calls": [{"tool_id": "create_payment_plan", "arguments": {"amount": 2500}}],
 "routing": {"path": "escalation", "target": "discount_planning_agent"},
 "outcome": {"id": "instalment_arrangement_confirmed",
             "attributes": {"amount": 2500}, "required_conditions": ["amount"]}}
```

Field names match `agent-eval-main/schema/normalized_trajectory.schema.json` (`tool_id`, `old`/`new`,
`id` for outcome, integer `turn_id`) — the canonical vocabulary every adapter and golden dataset in
this project now targets.

`state_changes[].node_id` is optional — if you know which node caused a transition, tag it
so STS's node-level alignment (METRICS.md §3) uses your real attribution; if omitted, it's
attributed to the turn's last node as a fallback.

Turns sharing a `conversation_id` are grouped and ordered by `turn_id`. `routing` and
`outcome` are optional per turn — `outcome` is typically only set on a conversation's
final turn, since Outcome Similarity is evaluated conversation-wide.

## Metric groups

| Group | Weight | What it measures |
|---|---:|---|
| NTS — Node Traversal Similarity | 30% | node coverage (40%) / precision (30%) / order (30%) over the exact, position-aware (LCS) node alignment; recall is a diagnostic (== coverage); missing / extra / reordered nodes in diagnostics |
| STS — State Transition Similarity | 30% | transition accuracy (70%) / order over full (key, old, new) transitions (30%); key/old/new-value accuracy are diagnostics. Compared within LCS-aligned nodes via `NormalizedNode.state_changes[]`, not turn-level; unchanged values are not transitions |
| TIS — Tool Invocation Similarity | 15% | tool coverage / precision / identity / input args / order; tools aligned within LCS-aligned nodes when the trace attaches them to nodes, else per turn |
| RS — Routing Similarity | 10% | **LLM judge** semantic correctness (70%) + order of path decisions (30%) — see above |
| OS — Outcome Similarity | 15% | outcome identity / attributes / completion (conversation-level) |

Overall NTS/STS/TIS/RS are computed from the **complete concatenated trajectory**, not by
averaging each turn's score (METRICS.md Frozen Rule #10) — per-turn scores in the report
are diagnostics explaining *where* a conversation deviated, not inputs to a turn-average.

Weights are overridable via `--weights-file weights.json` (`{"nts": ..., "sts": ..., "tis": ..., "rs": ..., "os": ...}`).

A metric that's **not applicable** for a turn/conversation (e.g. no expected tool calls)
is reported as `null`/N/A and excluded from both the numerator and denominator of every
weighted average above it — never coalesced to a score of `0`. The report always includes
`metric_coverage` alongside the ATF score so you can see how much of the score is backed
by applicable evidence.

## CLI flags and worked examples

Full flag reference and copy-pasteable commands (`atf-eval run`, `atf-eval dashboard`, `atf-eval serve`,
`atf-eval submit`) live in `HOW_TO_USE.md` and `docs/API.md` — not duplicated here to avoid the two docs drifting out of sync.

## Design notes

- **Fully separate from `rag_eval`.** Different domain (agent trajectories vs. RAG
  retrieval/answer quality), no shared code — mirrors its conventions (pydantic for
  validated input, plain dataclasses for in-process value objects, argparse CLI, no
  plugin/registry abstraction) without depending on it.
- **The adapter invokes your agent live.** `run_turn()` is both "call the agent" and
  "normalize its trace," same shape as `RAGPipeline.query()`.
- **A handful of formula details the spec leaves open are documented as explicit
  assumptions in the source** (tool-call/state-transition pairing strategy, how
  `outcome_completion`'s "required conditions" are sourced) — see the docstrings in `metrics/`.
- **State is evaluated per-node**, not per-turn (METRICS.md Frozen Rule #6):
  `NormalizedNode.state_changes[]`, not just `NormalizedTurn.state_changes[]`. Golden
  dataset rows can tag a `GoldenStateChange` with an explicit `node_id`; if omitted, it's
  attributed to the turn's last node as a documented fallback (`dataset.py`), since most
  source systems only expose a turn-level state diff, not true per-node causal attribution.
- **`NormalizedTurn.response`** (the agent's reply text) exists solely to feed RS's LLM
  judge — it's optional and unused by every other metric group.
