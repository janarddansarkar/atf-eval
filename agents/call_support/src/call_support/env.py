"""Tiny .env loader: fills ANTHROPIC_API_KEY (and CALL_SUPPORT_*) from ./.env without overriding exports."""
from __future__ import annotations

import os
from pathlib import Path

NAMES = ("ANTHROPIC_API_KEY", "CALL_SUPPORT_MODEL", "CALL_SUPPORT_TRACE_DIR")


def load_dotenv(path: str | Path = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name, value = name.strip(), value.strip().strip('"').strip("'")
        if name in NAMES and value and not os.environ.get(name):
            os.environ[name] = value
