# Commands

## 1. Install
```bash
cd agents/call_support
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install "atf-eval[server,client] @ git+https://github.com/2597688js/atf-eval.git"
cp .env.example .env    # set ANTHROPIC_API_KEY
```

## 2. Start atf-eval server (Terminal 1)
```bash
cd agents/call_support && source .venv/bin/activate
atf-eval serve --port 8000
```

## 3. Run atf-eval on call_support (Terminal 2)
```bash
cd agents/call_support && source .venv/bin/activate
atf-eval submit --server http://127.0.0.1:8000 \
  --adapter adapter:CallSupportAdapter \
  --dataset golden_dataset.jsonl \
  --output-dir results/
open "$(ls -t results/atf_dashboard_*.html | head -1)"
```

## 4. Stop the server
```bash
pkill -f "atf-eval serve"
```
