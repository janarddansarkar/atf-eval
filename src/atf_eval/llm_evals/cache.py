"""Content-addressed on-disk cache for LLM judge responses.

Keyed by a hash of (model, effort, system prompt, user prompt, output-schema
name), so an identical judge call is made exactly once ever. Makes a full
dashboard rebuild after the first one free and fully deterministic, and lets
tests replay recorded responses with no network.

Default location: <repo>/.llm_cache/ (git-ignored). Override with
ATF_LLM_CACHE_DIR, or disable entirely with ATF_LLM_CACHE=off.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

_DEFAULT_DIR = Path(__file__).resolve().parents[3] / ".llm_cache"


def cache_dir() -> Path:
    return Path(os.environ.get("ATF_LLM_CACHE_DIR", str(_DEFAULT_DIR)))


def enabled() -> bool:
    return os.environ.get("ATF_LLM_CACHE", "on").lower() not in ("off", "0", "false")


def key(*, model: str, effort: str | None, system: str, user: str, schema: str) -> str:
    blob = json.dumps(
        {"model": model, "effort": effort, "system": system, "user": user, "schema": schema},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load(k: str) -> dict[str, Any] | None:
    if not enabled():
        return None
    path = cache_dir() / f"{k}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def store(k: str, value: dict[str, Any]) -> None:
    if not enabled():
        return
    d = cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{k}.json").write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
