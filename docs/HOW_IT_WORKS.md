# How the atf-eval API and aggregation work

The server never touches the agent. The agent side runs the agent and POSTs the resulting trajectories over HTTP; the server scores them. Paths below are under `src/atf_eval/`.

```
agent folder                                     atf-eval server (atf-eval serve)
adapter.py + golden_dataset.jsonl
  atf-eval submit
   1. collect()        run agent locally
   2. POST /v1/evaluations  ──────────────────▶  api/app.py -> background job
   3. poll GET /v1/evaluations/{id}  ◀─────────  api/service.py:evaluate -> metrics/* -> aggregate.py
   4. download dashboard, CSVs  ◀──────────────  atf_server_data/evaluations/<id>/
```

## Agent side: what `atf-eval submit` does

`cmd_submit` in `cli.py:201`:

```python
golden  = load_dataset(args.dataset)            # golden_dataset.jsonl
adapter = load_adapter(args.adapter)            # adapter:CallSupportAdapter, resolved from the current folder
client  = AtfClient(args.server); client.health()          # GET /health
collected = collect(adapter, golden, ...)                  # 1. run the agent locally
eval_id = client.submit(collected, golden=golden, ...)     # 2. POST /v1/evaluations
result  = client.wait(eval_id)                             # 3. poll GET /v1/evaluations/{id}
client.download(result, args.output_dir)                   # 4. fetch dashboard and CSVs
```

**`collect()`** (`client.py:43`) runs in the agent's process and calls the adapter once per turn:

```python
history.append(adapter.run_turn(cid, gt.turn_id, gt.user_input, list(history)))
```

Customer inputs come from the golden file. Each agent trace becomes a `NormalizedTurn`.

**`AtfClient.submit()`** (`client.py:103`) POSTs plain JSON:

```python
body = {"observed": {cid: _payload(turns) ...}, "availability": ..., "options": ..., "label": ...,
        "golden_turns": [...]}            # or "golden_id"
self._http.post("/v1/evaluations", json=body)
```

Any language that can POST this JSON can use the API.

## Server side

`api/app.py:80` receives the POST and returns `202` immediately. Scoring runs in a background thread pool (`workers=2`); `wait()` polls until the status is `done`:

```python
result = evaluate(golden, body.observed, body.availability, body.options, eval_dir, body.label)
```

Judge API keys live on this side.

## Where the aggregation happens

1. **`api/service.py:build_runs`**: pairs each observed conversation with the golden conversation of the same `conversation_id`, sliced to the turns the agent produced.
2. **`api/service.py:evaluate`**: passes the runs to `dashboard.generate.build_outputs`, the same pipeline as `atf-eval dashboard`, so CLI and API agree.
3. **`metrics/*.py`**: each group is computed over the whole concatenated conversation, not per turn:
   - `nodes.py` NTS
   - `state.py` STS
   - `tools.py` TIS
   - `routing.py` RS (LLM judge)
   - `outcome.py` OS

   Each group is a weighted mix of its sub-scores via `weighted_composite(components, subweights)`.
4. **`aggregate.py`**: the rules.

   ```python
   DEFAULT_WEIGHTS = {"nts": 0.30, "sts": 0.30, "tis": 0.15, "rs": 0.10, "os": 0.15}

   def weighted_composite(components, weights):   # sum(w*score) / sum(w), skipping None
   def atf_score(group_scores, weights):          # ATF = weighted mean of NTS/STS/TIS/RS/OS
   ```

   N/A (`None`) components are dropped from both numerator and denominator, so the remaining weights are renormalized instead of counting as 0. `metric_coverage` is the applicable share of the total weight (e.g. RS N/A gives 90% coverage).
5. **`api/service.py:_run_summary` / `_mean`**: build the result JSON. `summary.atf_mean` is the plain mean of the per-conversation ATF scores.

Outputs (`result.json`, dashboard HTML, CSVs) are written to `atf_server_data/evaluations/<id>/`; `client.download()` copies them to `--output-dir`.
