"""Shared fixtures: paths into agent-eval-main (the canonical spec/schema
repo -- a separate, standalone repo that lives as a local sibling folder,
NOT part of this repo) and a loaded JSON Schema validator for the normalized
trajectory contract.

Tests that depend on agent-eval-main's canonical schema/fixtures skip
gracefully (not fail) when that folder isn't present locally -- e.g. a fresh
clone of this repo alone, or CI without the sibling checkout -- since it is
external spec infrastructure, not bundled here.
"""
from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
# the spec repo may be checked out as either folder name
AGENT_EVAL_MAIN = next(
    (p for p in (REPO_ROOT / "agent-eval-main", REPO_ROOT / "agent-eval") if p.exists()),
    REPO_ROOT / "agent-eval-main",
)


@pytest.fixture(scope="session")
def canonical_schema() -> dict:
    schema_path = AGENT_EVAL_MAIN / "schema" / "normalized_trajectory.schema.json"
    if not schema_path.exists():
        pytest.skip(
            "agent-eval-main not found locally (it's a separate repo, not bundled here) -- "
            "clone it as a sibling folder to run schema-conformance tests"
        )
    return json.loads(schema_path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def canonical_validator(canonical_schema: dict) -> jsonschema.Validator:
    return jsonschema.Draft202012Validator(canonical_schema)
