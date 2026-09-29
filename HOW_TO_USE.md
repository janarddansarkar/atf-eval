# How to use atf-eval

Project root dir: agent-eval (has golden/ + tests/fixtures/scenarios/)

```bash
git clone https://github.com/2597688js/atf-eval.git
python3 -m venv .venv && source .venv/bin/activate    # create + activate a venv first
pip install -e atf-eval/                              # install dependencies + the atf-eval command
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env            # optional, at project root — for RS + LLM metrics

atf-eval dashboard                                     # zero flags — uses ./golden + ./tests/fixtures/scenarios
```

Custom paths (golden/tests not at the default locations, or a different output dir):
```bash
atf-eval dashboard \
  --golden-dir path/to/golden \
  --scenarios path/to/tests \
  --output-dir path/to/output
```

Output: `results/atf_dashboard_<timestamp>.html` — open it in a browser.
