# atf-eval API

atf-eval can run as an HTTP service (FastAPI). An agent owner writes only an **adapter** and runs one
command; the server does the whole evaluation — the five deterministic ATF metrics, the LLM routing
judge, the Group 1-5 LLM metrics, the dashboard and the CSVs.

## Architecture

```
 agent side (any repo, any framework)                 atf-eval server
┌────────────────────────────────────────┐          ┌──────────────────────────────────────┐
│ your agent                             │          │ POST /v1/goldens                     │
│   ▲                                    │          │ POST /v1/evaluations ──▶ background  │
│   │ run_turn()                         │ observed │      job: score_components()         │
│ adapter.py ── NormalizedTurn ──────────┼─────────▶│      + routing judge (RS)            │
│   ▲                                    │          │      + Group 1-5 LLM metrics         │
│   │ runs the agent over the dataset's  │  scores, │      + dashboard / CSVs              │
│ atf-eval submit  (or AtfClient)  ◀─────┼──────────┤ GET  /v1/evaluations/{id}[/...]      │
└────────────────────────────────────────┘ dashboard└──────────────────────────────────────┘
```

Design choices:

- **The server never calls or imports the agent.** The agent side runs the agent (so private code, data and
  credentials stay there) and submits the trajectory it produced. Any language that can POST JSON can use the
  API; `atf-eval submit` / `AtfClient` are the Python convenience layer.
- **Judge API keys live on the server** (`ANTHROPIC_API_KEY` / `NVIDIA_API_KEY` in its environment or in a `.env` in the directory it is started from; see `.env.example`).
  They are never part of a request.
- **Same scoring code as the CLI.** The server calls the pipeline behind `atf-eval dashboard`
  (`dashboard.generate.build_outputs`), so API and CLI results agree.
- **Evaluations are asynchronous.** The LLM judges take a while, so `POST /v1/evaluations` returns `202`
  immediately and you poll `GET /v1/evaluations/{id}`.
- **No authentication.** Run it on localhost or behind your own gateway.

## Run it

```bash
# server
pip install -e ".[server]"
atf-eval serve --host 127.0.0.1 --port 8000 [--data-dir ./atf_server_data]

# agent side
pip install -e ".[client]"
atf-eval submit --server http://127.0.0.1:8000 \
  --adapter my_adapter:MyAgentAdapter \
  --dataset golden_dataset.jsonl \
  --output-dir results/
```

`--server` defaults to `$ATF_SERVER`, then `http://127.0.0.1:8000`. The server stores data under `--data-dir`
(default `$ATF_SERVER_DATA`, else `./atf_server_data`); interactive docs are at `/docs`.

### What `atf-eval submit` does

1. Loads the golden dataset and your adapter (`module:Class`, no constructor args).
2. Runs the adapter over every golden conversation's `user_input`s, turn by turn, passing `history`.
   A turn that raises ends that conversation (a stateful agent's context may be corrupt); the turns up to
   then are still submitted and a warning is printed.
3. `POST`s the observed trajectories (plus the golden, inline or by id) and polls until the job is done.
4. Prints the per-conversation scores and saves the dashboard, CSVs and `result.json` to `--output-dir`.

| Flag | Meaning |
|---|---|
| `--dataset` | golden JSONL: the customer inputs the agent is driven with and the expected trajectory |
| `--upload-golden` / `--golden-id` | store the golden on the server once, then reference it by id |
| `--no-routing-judge` | skip RS's LLM judge (RS is N/A) |
| `--no-llm-evals` | skip the Group 1-5 LLM metrics (they make many judge calls; results are cached on the server) |
| `--provider`, `--judge-model`, `--effort` | judge selection (`anthropic` / `nvidia`) |
| `--weights-file` | JSON override of the ATF component weights |
| `--concurrency`, `--limit` | conversations run in parallel / only the first N |
| `--timeout` | how long to wait for the evaluation (seconds, default 3600) |

### From Python

```python
from atf_eval.client import AtfClient

result = AtfClient("http://127.0.0.1:8000").evaluate(
    MyAgentAdapter(), "golden_dataset.jsonl",
    options={"llm_evals": False}, out_dir="results",
)
print(result["summary"]["atf_mean"], result["agent_errors"])
```

Lower-level pieces: `collect(adapter, golden)`, `client.upload_golden()`, `client.submit()`, `client.wait()`,
`client.download()`.

## Endpoints

### `GET /health`
`{"status": "ok"}`

### `POST /v1/goldens` → `201`
Store a golden dataset. Body: `{"name": "my-golden", "turns": [<GoldenTurn>, ...]}`, where each turn is a row of
the golden JSONL format (see the README). Returns `{golden_id, name, created_at, conversations}`.
`GET /v1/goldens` lists them; `GET /v1/goldens/{golden_id}` returns one including its turns.

### `POST /v1/evaluations` → `202`
```jsonc
{
  "golden_id": "my-golden-1a2b3c",      // or "golden_turns": [<GoldenTurn>, ...] to send the golden inline
  "observed": {                          // conversation_id -> the agent's normalized turns
    "support_outage_resolved": [
      {"conversation_id": "...", "turn_id": 1, "nodes": [{"node_id": "classify_intent", "tool_calls": [],
        "state_changes": [{"key": "intent", "old": null, "new": "outage"}]}],
       "tool_calls": [], "state_changes": [], "routing": {"path": "outage", "target": "self_service"},
       "outcome": null, "response": "...", "customer_input": "..."}
    ]
  },
  "availability": {"state": "unavailable"},   // evidence your traces never expose -> that metric is N/A
  "options": {
    "routing_judge": true,        // RS's LLM judge; false -> RS is N/A
    "llm_evals": true,            // Group 1-5 LLM metrics
    "provider": null,             // "anthropic" | "nvidia"; default: server's ATF_LLM_PROVIDER / anthropic
    "judge_model": null,          // default: the provider's default model
    "effort": "low",
    "weights": null               // {"nts": .3, "sts": .3, "tis": .15, "rs": .1, "os": .15}
  },
  "label": "free text shown on the dashboard"
}
```
`observed` turns have the shape of `atf_eval.normalized.NormalizedTurn` (the same objects your adapter returns).
An observed conversation is scored against the golden conversation with the **same id**, sliced to the turns the
agent produced; conversations with no golden counterpart are reported under `skipped`, not scored.

Returns `{"evaluation_id": "...", "status": "queued", "url": "/v1/evaluations/..."}`. Errors: `404` unknown
`golden_id`, `422` invalid body / no golden / empty `observed`.

### `GET /v1/evaluations/{id}`
`status` is `queued`, `running`, `done` or `failed` (with `error`). Once `done`:

```jsonc
{
  "evaluation_id": "...", "status": "done",
  "summary": {"conversations": 3, "nts_mean": 1.0, "sts_mean": 1.0, "tis_mean": 1.0,
              "rs_mean": 1.0, "os_mean": 1.0, "atf_mean": 1.0, "judge": "anthropic:claude-haiku-5-5"},
  "runs": [{
    "conversation_id": "...", "turns_compared": "2 (full conversation)",
    "nts": 1.0, "sts": 1.0, "tis": 1.0, "rs": 1.0, "os": 1.0, "atf": 1.0, "metric_coverage": 1.0,
    "availability": {"nodes": "available", "state": "available", "...": "..."},
    "components": {"nts_coverage": 1.0, "nts_precision": 1.0, "sts_transition_accuracy": 1.0, "...": "..."},
    "diagnostics": {"nts": {"matched": [], "missing": [], "extra": [], "reordered": []}, "sts": {}, "tis": {}, "rs": {}, "os": {}},
    "llm": {"correctness": {"score": "PASS", "normalized": 1.0, "status": "available", "reason": "..."}}
  }],
  "skipped": [], "settings": {"...": "judge, weights, matching rules"},
  "files": {"dashboard": "atf_dashboard_<ts>.html", "metrics_csv": "...", "components_csv": "...", "settings": "...", "llm_csv": "..."},
  "links": {"dashboard": "/v1/evaluations/{id}/dashboard", "metrics_csv": "/v1/evaluations/{id}/files/..."}
}
```
A score of `null` means **N/A** (not evaluable), never 0. `llm` is present only when `llm_evals` was on;
metrics that could not be judged report `status: "not_applicable"`.

### `GET /v1/evaluations/{id}/dashboard`
The HTML dashboard (one tab per conversation, deterministic and LLM tracks together).

### `GET /v1/evaluations/{id}/files/{name}`
Any file listed under `files` (CSVs, settings JSON).

## curl example

```bash
curl -s localhost:8000/v1/goldens -H 'content-type: application/json' \
  -d "$(jq -Rs '{name:"demo", turns: [split("\n")[] | select(length>0) | fromjson]}' golden_dataset.jsonl)"
# -> {"golden_id": "demo-ab12cd", ...}

curl -s localhost:8000/v1/evaluations -H 'content-type: application/json' \
  -d '{"golden_id":"demo-ab12cd","observed":{...},"options":{"llm_evals":false}}'
# -> {"evaluation_id": "20261017T135717-4b00b1", "status": "queued", ...}

curl -s localhost:8000/v1/evaluations/20261017T135717-4b00b1 | jq .summary
curl -s localhost:8000/v1/evaluations/20261017T135717-4b00b1/dashboard > dashboard.html
```

## Writing the agent-side adapter

An adapter is any class with `run_turn(conversation_id, turn_id, user_input, history) -> NormalizedTurn`
(the `TrajectoryAgent` protocol — see the README, "Integrate your own agent"). For a minimal complete example,
see [`agents/call_support/adapter.py`](../agents/call_support/adapter.py): about 40 lines that call a standalone
LangGraph agent and map its result to a `NormalizedTurn`.
