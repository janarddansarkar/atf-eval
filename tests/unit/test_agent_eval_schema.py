"""Schema/structural conformance checks for agent-eval's golden dataset(s)
and observed-scenario fixtures (S001, S002, ... -- however many there are on
disk).

Complements test_dataset_schema_export.py (which checks atf_eval's own
`to_canonical_document()` export against the canonical schema using a
hand-built fixture) by checking the actual files on disk: the same golden
and scenario JSON that generate_atf_dashboard.py and reference_adapter.py
consume.

Two different conformance bars, matching METRICS.md §12's documented
Adapter/fixture split:
  - agent-eval/golden/*.json is already canonical -- it must validate
    against normalized_trajectory.schema.json exactly.
  - agent-eval/tests/fixtures/scenarios/*.json is pre-Adapter, raw-ish
    observed traces (`sequence_index`/`tool_name` instead of
    `sequence`/`tool_id`, `turns`/`turn` instead of a fixed shape) -- it is
    NOT expected to validate against that schema, so instead this checks
    the structural contract the dashboard/reference adapter actually rely
    on: the required top-level keys are present, `golden_scenario` names a
    golden file that actually exists (a typo here silently orphans or
    misroutes a fixture), and the file normalizes into at least one turn
    without raising.

Skips gracefully (not fails) when agent-eval isn't present locally -- it's a
separate repo that lives as a local sibling folder, not bundled here.
"""
from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from tests.fixtures.reference_adapter import load_fixture_turns, load_golden_turns

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_EVAL = REPO_ROOT / "agent-eval"
GOLDEN_DIR = AGENT_EVAL / "golden"
SCENARIOS_DIR = AGENT_EVAL / "tests" / "fixtures" / "scenarios"
SCHEMA_PATH = AGENT_EVAL / "schema" / "normalized_trajectory.schema.json"

pytestmark = pytest.mark.skipif(
    not AGENT_EVAL.exists(),
    reason="agent-eval not found locally (it's a separate repo, not bundled here) -- "
    "clone it as a sibling folder to run this conformance suite",
)


def _golden_files() -> list[Path]:
    return sorted(GOLDEN_DIR.glob("*.json")) if GOLDEN_DIR.exists() else []


def _scenario_files() -> list[Path]:
    return sorted(SCENARIOS_DIR.glob("*.json")) if SCENARIOS_DIR.exists() else []


def _golden_key(doc: dict) -> str:
    """The identifier a scenario fixture's `golden_scenario` field is
    expected to name -- metadata.scenario_id, falling back to trace_id,
    same precedence generate_atf_dashboard.py's discover_goldens() uses."""
    return doc.get("metadata", {}).get("scenario_id") or doc["trace_id"]


@pytest.fixture(scope="module")
def validator() -> jsonschema.protocols.Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return jsonschema.Draft202012Validator(schema)


@pytest.fixture(scope="module")
def golden_keys() -> set[str]:
    keys = set()
    for path in _golden_files():
        doc = json.loads(path.read_text(encoding="utf-8"))
        keys.add(_golden_key(doc))
    return keys


# ---------------------------------------------------------------------------
# Golden files: must validate against the canonical schema exactly.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", _golden_files(), ids=lambda p: p.name)
def test_golden_file_matches_canonical_schema(
    path: Path, validator: jsonschema.protocols.Validator
) -> None:
    doc = json.loads(path.read_text(encoding="utf-8"))
    errors = sorted(validator.iter_errors(doc), key=str)
    assert not errors, "\n".join(f"{e.json_path}: {e.message}" for e in errors)


@pytest.mark.parametrize("path", _golden_files(), ids=lambda p: p.name)
def test_golden_file_loads_via_reference_adapter(path: Path) -> None:
    """Schema-valid doesn't automatically mean usable -- confirm the same
    loader generate_atf_dashboard.py and the metric functions actually
    consume produces at least one turn."""
    turns = load_golden_turns(path)
    assert turns, f"{path.name} produced zero turns"


# ---------------------------------------------------------------------------
# Scenario fixtures: pre-Adapter raw shape, not schema-conformant by design
# -- checked structurally instead.
# ---------------------------------------------------------------------------

REQUIRED_FIXTURE_KEYS = {
    "trace_id", "scenario_id", "golden_scenario", "trace_type", "deviation_type", "deviation",
}


@pytest.mark.parametrize("path", _scenario_files(), ids=lambda p: p.name)
def test_scenario_fixture_has_required_keys(path: Path) -> None:
    doc = json.loads(path.read_text(encoding="utf-8"))
    missing = REQUIRED_FIXTURE_KEYS - doc.keys()
    assert not missing, f"{path.name} is missing required keys: {sorted(missing)}"
    assert "turns" in doc or "turn" in doc, f"{path.name} has neither 'turns' nor 'turn'"


@pytest.mark.parametrize("path", _scenario_files(), ids=lambda p: p.name)
def test_scenario_fixture_references_a_real_golden(path: Path, golden_keys: set[str]) -> None:
    """golden_scenario must name a golden file that actually exists in
    agent-eval/golden/ -- a typo here would silently orphan the fixture
    (skipped with a [WARN]) or, with only one golden on disk, silently
    misroute it to the wrong one."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    golden_scenario = doc.get("golden_scenario")
    assert golden_scenario, f"{path.name} has no golden_scenario field"
    assert golden_scenario in golden_keys, (
        f"{path.name}: golden_scenario={golden_scenario!r} matches none of {sorted(golden_keys)}"
    )


@pytest.mark.parametrize("path", _scenario_files(), ids=lambda p: p.name)
def test_scenario_fixture_loads_via_reference_adapter(path: Path) -> None:
    """Every scenario fixture must actually normalize into at least one
    turn via the same loader the metric functions consume -- catches a
    malformed node/tool_call (e.g. a missing node_id or tool_name) as a
    clear assertion here instead of a confusing KeyError deep inside a
    metric run."""
    turns = load_fixture_turns(path)
    assert turns, f"{path.name} produced zero turns"
    for turn in turns:
        assert turn.turn_id is not None, f"{path.name} has a turn with no turn_id"


@pytest.mark.parametrize("path", _scenario_files(), ids=lambda p: p.name)
def test_scenario_fixture_shares_at_least_one_turn_with_its_golden(
    path: Path, golden_keys: set[str]
) -> None:
    """An observed fixture may legitimately run *past* its golden's last
    turn_id (e.g. a drift/disconnect scenario where the agent drags the
    conversation on before the customer hangs up -- turn_id there is part
    of the deviation itself, not an error), so this doesn't require every
    observed turn_id to exist in golden. It does require at least one turn
    of overlap: with zero overlap, generate_atf_dashboard.py's
    `[t for t in golden_turns if t.turn_id in observed_ids] or golden_turns`
    would silently fall back to comparing against the *entire* golden
    conversation instead of the intended slice."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    golden_scenario = doc.get("golden_scenario")
    if golden_scenario not in golden_keys:
        pytest.skip("covered by test_scenario_fixture_references_a_real_golden")

    golden_path = next(
        p for p in _golden_files()
        if _golden_key(json.loads(p.read_text(encoding="utf-8"))) == golden_scenario
    )
    golden_turn_ids = {t.turn_id for t in load_golden_turns(golden_path)}
    observed_turn_ids = {t.turn_id for t in load_fixture_turns(path)}

    assert golden_turn_ids & observed_turn_ids, (
        f"{path.name} shares no turn_id with its golden {golden_path.name} "
        f"(observed {sorted(observed_turn_ids)} vs. golden {sorted(golden_turn_ids)}) -- "
        "the dashboard would silently fall back to comparing against the whole conversation"
    )
