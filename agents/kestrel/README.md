# Kestrel — rule-based agent

The simplest agent: no LLM, no network, no API key.

**What it does**
- **Turn 1:** matches keywords in the customer's message to an intent (`hardship` or `pay_now`), then runs that intent's node (`hardship_node` / `payment_node`), calls its tool (`eligibility_check` / `payment_link_create`), flips its state flag, and routes to `collection_agent`. An unrecognised message routes to `clarify` and asks the customer to say more.
- **Turn 2:** sees from `history` that an intent is open, runs `confirm_outcome`, and reports the outcome (`installment_arrangement_accepted` or `payment_received`).
- It keeps no state of its own; everything is rebuilt from `history`.

**Use it for:** the smallest working template of a `TrajectoryAgent`, and a perfect-score baseline.

**Run it**
```bash
python -m atf_eval run --dataset examples/golden_dataset.jsonl \
  --adapter agents.kestrel.agent:Kestrel --output-dir results/ --no-routing-judge
```
**Expected on the bundled dataset:** ATF 1.000 (RS is N/A with `--no-routing-judge`).
