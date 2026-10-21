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

`api/app.py` receives the POST. The request body is validated first — including `options.weights`, which goes through the same `aggregate.resolve_weights()` as the CLI — so invalid weights are a `422` and nothing is queued. A valid request returns `202` immediately with status `queued`. Scoring runs in a background thread pool of `--workers` threads (`$ATF_SERVER_WORKERS`, default 2), so at most that many evaluations run at once; later ones stay `queued`, and their status carries `queue_position`, `running` and `workers` until a worker frees up. Every step of a job, status writes included, is guarded, so a crash always ends as `failed` with an `error` (and a traceback in the server log) rather than leaving the evaluation `queued`/`running`. On start-up the server marks evaluations a previous server process left `queued`/`running` as `failed`, since nothing will pick them up again. `status.json` is written atomically (temp file + `os.replace`). On Windows, a file can't be replaced while another handle has it open (Windows Defender or the search indexer scanning it, for example), so the swap and the status read are retried for up to 5 s on `PermissionError`, and the server's own status reads take the same lock as its writes. If a status still can't be saved, the server keeps the outcome in memory and reports it, so a client is never left polling a stale `queued`. `wait()` polls until the status is `done`, printing each status change during `atf-eval submit`:

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
   - `outcome.py` OS

   Each group is a weighted mix of its sub-scores via `weighted_composite(components, subweights)`.
4. **`aggregate.py`**: the rules.

   ```python
   # original NTS .30 / STS .30 / TIS .15 / OS .15, each / 0.90 (RS's .10 redistributed proportionally)
   DEFAULT_WEIGHTS = {"nts": 1/3, "sts": 1/3, "tis": 1/6, "os": 1/6}
   ATF_EPSILON = 1e-6

   def weighted_composite(components, weights):   # sum(w*score) / sum(w), skipping None
   def atf_score(group_scores, weights):          # ATF = exp(sum(w*log(max(S, eps))) / sum(w)), skipping None
   ```

   `resolve_weights(config)` is the single entry point for weights: `None` → `DEFAULT_WEIGHTS`; anything else must have all four keys, finite non-negative numbers, summing to 1. The CLI (`--weights` / `--weights-file`, parsed by `parse_weights_spec` / `load_weights_file`), the API (`EvalOptions.weights`) and `runner.score_trajectory` all call it.

   ATF is the weighted geometric mean of NTS/STS/TIS/OS. N/A (`None`) components are dropped from both numerator and denominator, so the remaining weights are renormalized instead of counting as 0; a score of 0 is floored at ε before the log. `metric_coverage` is the applicable share of the total weight (e.g. TIS N/A gives 5/6 ≈ 83.3% coverage).
5. **`api/service.py:_run_summary` / `_mean`**: build the result JSON. `summary.atf_mean` is the plain mean of the per-conversation ATF scores.

Outputs (`result.json`, dashboard HTML, CSVs) are written to `atf_server_data/evaluations/<id>/`; `client.download()` copies them to `--output-dir`.
