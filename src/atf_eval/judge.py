"""RS's LLM-based semantic routing judge — METRICS.md §5.

RS is the one metric group that is not a closed-form formula: it asks
whether the agent chose the appropriate conversational direction given the
customer input and its own response, which is a judgment call a plain field
comparison can't make (an agent can route to the textually "wrong" target
and still be doing the right thing, or vice versa). This is the only place
in atf_eval that makes an LLM call — every other metric group stays pure
Python. Mirrors `rag_eval.metrics.judge`'s structured-output pattern.
"""
from __future__ import annotations

from typing import Literal

import anthropic
from pydantic import BaseModel


ROUTING_JUDGE_MAX_TOKENS = 1024


class RoutingJudgeScore(BaseModel):
    verdict: Literal["correct", "partial", "incorrect"]
    reasoning: str


ROUTING_JUDGE_SYSTEM_PROMPT = """You are a strict, impartial evaluator of an agent's routing decision for one \
turn of a conversation.

You will be given the customer's message, the agent's response, the expected path/target for \
this turn (from a golden reference trajectory), and whatever else is known about what the \
agent actually did (nodes visited, state changes, tools called).

Judge whether the agent chose the appropriate conversational direction — NOT whether the \
response text is well-written, and NOT whether every internal node matches the golden \
trajectory exactly. Routing is about the *destination*: did the agent send this turn where it \
needed to go (e.g. handle it directly vs. hand off to a specialist vs. escalate to a human), \
given what the customer actually said and what the agent's own response indicates it decided.

Score exactly one of:
- correct: the agent's actual routing destination matches the expected path/target, or is a \
  reasonable equivalent given the conversational context.
- partial: the agent's routing is defensible but diverges from the expected path/target in a \
  meaningful way (e.g. handled directly when a specialist handoff was expected, but did so \
  competently), or the evidence is ambiguous.
- incorrect: the agent's routing destination clearly does not match what the conversational \
  context called for.

Be strict and consistent. Do not give benefit of the doubt beyond what the evidence supports. \
Provide a one-to-two sentence reasoning citing the specific evidence for your verdict."""


def _build_routing_judge_user_prompt(
    customer_input: str,
    agent_response: str | None,
    expected_path: str | None,
    expected_target: str | None,
    trajectory_evidence: str,
) -> str:
    return f"""<customer_input>
{customer_input}
</customer_input>

<agent_response>
{agent_response or "(no response text available)"}
</agent_response>

<expected_path_and_target>
path: {expected_path or "(not specified)"}
target: {expected_target or "(not specified)"}
</expected_path_and_target>

<available_trajectory_evidence>
{trajectory_evidence}
</available_trajectory_evidence>

Judge the agent's routing decision for this turn against the expected path/target above."""


def run_routing_judge(
    client: anthropic.Anthropic,
    model: str,
    customer_input: str,
    agent_response: str | None,
    expected_path: str | None,
    expected_target: str | None,
    trajectory_evidence: str,
    max_retries: int = 1,
    effort: str | None = None,
) -> RoutingJudgeScore:
    user_prompt = _build_routing_judge_user_prompt(
        customer_input, agent_response, expected_path, expected_target, trajectory_evidence
    )
    parse_kwargs = dict(
        model=model,
        max_tokens=ROUTING_JUDGE_MAX_TOKENS,
        system=ROUTING_JUDGE_SYSTEM_PROMPT,
        output_format=RoutingJudgeScore,
        messages=[{"role": "user", "content": user_prompt}],
    )
    if effort:
        parse_kwargs["output_config"] = {"effort": effort}

    last_err: Exception | None = None
    for _ in range(max_retries + 1):
        try:
            response = client.messages.parse(**parse_kwargs)
            if response.parsed_output is None:
                raise ValueError("model did not return parseable structured output")
            return response.parsed_output
        except Exception as e:  # noqa: BLE001 - isolate any judge-call failure
            last_err = e
    raise RuntimeError(
        f"Routing judge failed after {max_retries + 1} attempt(s): {last_err}"
    ) from last_err


def verdict_to_score(verdict: str) -> float:
    return {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}[verdict]
