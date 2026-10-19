# Call-support agent + atf-eval

```bash
cd agents/call_support
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install "atf-eval[server,client] @ git+https://github.com/janarddansarkar/atf-eval.git@atf_with_api"
cp .env.example .env    # set ANTHROPIC_API_KEY
```

Terminal 1:
```bash
cd agents/call_support && source .venv/bin/activate
atf-eval serve --port 8000
```

Terminal 2:
```bash
cd agents/call_support && source .venv/bin/activate
atf-eval submit --server http://127.0.0.1:8000 \
  --adapter adapter:CallSupportAdapter \
  --dataset golden_dataset.jsonl \
  --output-dir results/
```

Stop:
```bash
pkill -f "atf-eval serve"
```
