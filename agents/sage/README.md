# Sage — LLM-classified agent

Kestrel's trajectory with an LLM choosing the intent instead of keyword rules.

**What it does**
- Asks the configured LLM (through `atf_eval.llm_provider`, so Anthropic or NVIDIA) to label the customer's message `hardship`, `pay_now` or `unclear`, using structured output.
- Everything after the intent — nodes, tool, state change, routing, confirm turn — is inherited from Kestrel.
- **Falls back to Kestrel's keyword rules** when no API key is set or a call fails (a warning is printed), so a run never breaks.

**Use it for:** an agent whose behaviour depends on a model, e.g. to see how paraphrased messages ("money is tight") change the trajectory and the score.

**Run it**
```bash
ANTHROPIC_API_KEY=... python -m atf_eval run --dataset examples/golden_dataset.jsonl \
  --adapter agents.sage.agent:Sage --output-dir results/ --no-routing-judge
# or: ATF_LLM_PROVIDER=nvidia NVIDIA_API_KEY=... (needs pip install -e ".[nvidia]")
```
The agent's own provider and the judge's provider are selected by the same `ATF_LLM_PROVIDER`.

**Expected on the bundled dataset:** ATF 1.000 — with or without a key, since the dataset's messages also match the keyword fallback. The unit tests exercise the LLM path with a fake provider; it has not been run against a live API.
