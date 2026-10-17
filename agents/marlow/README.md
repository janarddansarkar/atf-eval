# Marlow — state-machine agent

A stateful agent with an explicit conversation stage: `idle → intent_open → closed`. It reuses Kestrel's intents, nodes and tools.

**What it does**
- **Turn 1:** classifies the intent, runs the same node and tool as Kestrel, and moves the stage `idle → intent_open`.
- **Turn 2:** reads the stage back from its own `history`, confirms the arrangement, and moves the stage `intent_open → closed`.
- Every stage change is recorded as a `StateChange(key="stage", …)`.

**Use it for:** seeing how STS reacts to an agent whose trace carries *more* state than the Golden dataset expects. The Golden dataset has no `stage` transitions, so they count as extra transitions.

**Run it**
```bash
python -m atf_eval run --dataset examples/golden_dataset.jsonl \
  --adapter agents.marlow.agent:Marlow --output-dir results/ --no-routing-judge
```
**Expected on the bundled dataset:** NTS, TIS and OS 1.000; **STS ≈ 0.333**; ATF ≈ 0.778. The drop is entirely the extra `stage` transitions (transition accuracy divides by `max(expected, observed)`).
