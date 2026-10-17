# How to use atf-eval

Project root dir: agent-eval (has golden/ + tests/fixtures/scenarios/)

```bash
git clone https://github.com/2597688js/atf-eval.git
python3 -m venv .venv && source .venv/bin/activate    # create + activate a venv first
pip install -e atf-eval/                              # install dependencies + the atf-eval command
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env            # optional, at project root — for RS + LLM metrics

atf-eval dashboard                                     # zero flags — uses ./golden + ./tests/fixtures/scenarios
```

To use NVIDIA NIM as the judge instead of Anthropic, put this in `.env`
(or pass `--provider nvidia`; pick another model with `--judge-model`):
```bash
ATF_LLM_PROVIDER=nvidia
NVIDIA_API_KEY=nvapi-...
```

Custom paths (golden/tests not at the default locations, or a different output dir):
```bash
atf-eval dashboard \
  --golden-dir path/to/golden \
  --scenarios path/to/tests \
  --output-dir path/to/output
```

Output: `results/atf_dashboard_<timestamp>.html` — open it in a browser.

## Evaluate an agent through the API

If the agent lives in a different repo, run atf-eval as a server and write only an adapter on the agent side:

```bash
pip install -e "atf-eval/[server]"           # on the server machine
atf-eval serve --port 8000                    # reads ANTHROPIC_API_KEY from its env or a .env in its cwd

pip install -e "atf-eval/[client]"           # on the agent side
atf-eval submit --server http://localhost:8000 \
  --adapter my_adapter:MyAgentAdapter \
  --dataset golden_dataset.jsonl \
  --output-dir results/                       # saves the dashboard, CSVs and result.json
```

Add `--no-llm-evals` to skip the (many-call) Group 1-5 LLM metrics, `--no-routing-judge` to leave RS as N/A, or
`--judge-model claude-haiku-5-5` for a cheaper judge. See `docs/API.md` for the full reference and
`agents/call_support/` for a worked example.

