# Bundled example

A small, dependency-free smoke test for the framework itself — no live agent,
no LLM, no network calls. `golden_dataset.jsonl` is a 2-conversation, 4-turn
golden dataset (hand-written, schema-conformant with
`agent-eval-main/schema/normalized_trajectory.schema.json` field names, not
copied from `agent-eval-main`'s own insurance-domain golden scenario — this
one is collection-agent-adjacent, used only as a shape reference). `mock_adapter.py`
returns a canned trajectory that matches it turn-for-turn.

## Run it

```bash
cd "/Users/janarddan/1.jana files/3.MyMacProjects/atf_eval"
source .venv/bin/activate   # pip install -e ".[dev]" first if you haven't
python -m atf_eval run \
  --dataset examples/golden_dataset.jsonl \
  --adapter examples.mock_adapter:MockTrajectoryAgent \
  --output-dir results/ \
  --no-routing-judge
```

## Expected output

Since the mock adapter's trajectory matches the golden dataset exactly, every
metric should come back a perfect (or near-perfect) score:

```
=== ATF Evaluation Summary ===
Conversations:   2
Turns:           4

ATF score:       1.000
Metric coverage: 100.0%

Metric groups (mean):
  NTS     1.000  (available)
  STS     1.000  (available)
  TIS     1.000  (available)
  RS      1.000  (available)
  OS      1.000  (available)
```

RS scores 1.000 even with `--no-routing-judge` because both turns carry a
`routing` block that matches exactly — with no judge configured, RS's
semantic component reports N/A and the formula renormalizes to just the
order-similarity component, which is a perfect match here. Drop
`--no-routing-judge` (with `ANTHROPIC_API_KEY` set, or `NVIDIA_API_KEY` plus
`--provider nvidia`) to see the real LLM judge run instead.

If this doesn't reproduce, the framework itself is broken — start
troubleshooting here before touching a real integration.

## Next: a real agent

This mock is only a framework smoke test. For a real agent (LangGraph + LangChain, with tracing, evaluated through
the atf-eval API) see [`../agents/call_support/`](../agents/call_support/README.md) and the standalone agent in
[`../call_support_agent/`](../call_support_agent/README.md).

