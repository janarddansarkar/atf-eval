# atf-eval

[![GitHub](https://img.shields.io/badge/GitHub-2597688js%2Fatf--eval-blue?logo=github)](https://github.com/2597688js/atf-eval)

Standalone evaluation framework for **Agent Trajectory Fidelity (ATF)** — scores how
closely an agent's observed execution trajectory (nodes visited, state transitions, tool
calls, routing decisions, final outcome) matches an expected trajectory defined by a
golden dataset. See `METRICS.md` for the frozen scoring specification this package
implements (supersedes `ATF_Metric_Definition_Evaluation_Specification.md`, the
original draft spec).

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

**For step-by-step setup and run instructions (install, API keys, the `atf-eval
dashboard` command), see `HOW_TO_USE.md`.** This README is the API/conceptual
reference: the adapter contract, golden dataset schema, metric group definitions, and
the Python API for Policy Compliance and the scorecard.

Four of five metric groups (NTS, STS, TIS, OS) are fully deterministic — closed-form
formulas, no LLM calls. **RS is the one exception**: it's an LLM-based semantic judge
(METRICS.md §5) of whether the agent's routing choice was appropriate given the
conversation, not a field comparison. Without `ANTHROPIC_API_KEY` set (or with
`--no-routing-judge`), routing cannot be evaluated, so RS is N/A and excluded from ATF
(routing order is still reported as a diagnostic) — the rest of the framework needs no
API key at all.

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

## Policy Compliance (separate from ATF)

ATF answers "did the agent follow the expected trajectory?" — it says nothing about
whether an action the agent took was actually *allowed*. A conversation can score well on
ATF while still granting a waiver, discount, or exception the domain's rules never
permitted. Policy Compliance is a deliberately separate, parallel check for exactly that
case, reusing the same `NormalizedTurn` objects your adapter already produces (no second
agent invocation needed — see "Reusing the captured trace" below).

Write `PolicyRule`s as plain Python predicates (mirrors the adapter pattern: domain
knowledge belongs in code you write, not a config format the engine has to interpret):

```python
from atf_eval.policy.rules import PolicyRule
from atf_eval.policy.helpers import tool_calls_named, latest_state_value

def waiver_requires_capacity_check(turn, history, context):
    for tc in tool_calls_named(turn, "waive_premium"):
        capacity = latest_state_value(history, turn, "customer_payment_capacity")
        if capacity is None:
            return "Premium waived without ever checking payment capacity."
    return None

RULES = [
    PolicyRule(
        rule_id="waiver_requires_capacity_check",
        description="A premium waiver must not be granted before payment capacity is assessed.",
        severity="critical",   # "critical" fails the PASS/FAIL gate; "warning" is advisory only
        check=waiver_requires_capacity_check,
    ),
]
```

```python
from atf_eval.policy.checker import evaluate_conversation

result = evaluate_conversation(conv_result.observed_turns, RULES, context={"policy": {...}})
# result.status: "pass" | "fail"
# result.violations: list[PolicyViolation] (rule_id, severity, turn_id, detail)
```

`context` is whatever domain data your rules need (policy limits, eligibility rules,
etc.) — supplied by you, not derived from the trajectory.

### Reusing the captured trace

`run_evaluation()` returns each `ConversationResult` with `.expected_turns` /
`.observed_turns` already attached, so Policy Compliance (and the scorecard's deviation
detection) run against the *same* captured trace ATF already scored — no need to
re-invoke a live agent a second time.

## Scorecard (diagnostic report)

`render_scorecard()` turns one conversation's `ConversationResult` (+ optional
`PolicyResult`) into the actionable text report the spec's §17 asks for ("never collapse
the evaluation into ATF alone") — ATF score and group breakdown, the policy PASS/FAIL gate
and any violations, plus the single most significant state deviation and tool deviation:

```python
from atf_eval.scorecard import render_scorecard

print(render_scorecard("Motor Insurance — Payment Difficulty", conv_result, policy_result))
```

```
Scenario
Motor Insurance — Payment Difficulty

────────────────────────────────────────

ATF SCORE                          91.4%

Node Traversal                     95.0%
State Transition                   88.0%
Tool Invocation                    92.0%
Routing                           100.0%
Outcome                           100.0%

────────────────────────────────────────

POLICY COMPLIANCE                   PASS

────────────────────────────────────────

Key Deviation

STS ↓

Expected:
cannot_pay → partial_pay

Observed:
(not updated)

Reason:
Agent did not update 'customer_payment_posture'.

────────────────────────────────────────

Tool Deviation

Expected:
check_installment_eligibility

Observed:
Not invoked
```

## CLI flags and worked examples

Full flag reference and copy-pasteable commands (`atf-eval run`, `atf-eval dashboard`)
live in `HOW_TO_USE.md` — not duplicated here to avoid the two docs drifting out of sync.

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
