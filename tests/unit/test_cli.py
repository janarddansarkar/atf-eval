"""`atf-eval run` / `submit` / `dashboard` weight options: --weights and
--weights-file share one parser, one validator (aggregate.resolve_weights) and
the same defaults as the API -- all offline."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from atf_eval import cli  # noqa: E402
from atf_eval.aggregate import DEFAULT_WEIGHTS, atf_score  # noqa: E402
from atf_eval.normalized import NormalizedNode, NormalizedTurn  # noqa: E402
from tests.fixtures.mock_adapter import MockTrajectoryAgent  # noqa: E402

DATASET = str(ROOT / "tests" / "fixtures" / "mock_golden_dataset.jsonl")
CUSTOM = {"nts": 0.4, "sts": 0.3, "tis": 0.2, "os": 0.1}
CUSTOM_SPEC = "nts=0.4,sts=0.3,tis=0.2,os=0.1"


class SkipsANode(MockTrajectoryAgent):
    """NTS < 1, every other group perfect, so the ATF depends on the weights."""

    def run_turn(self, conversation_id, turn_id, user_input, history):
        t = super().run_turn(conversation_id, turn_id, user_input, history)
        return NormalizedTurn(**{**t.__dict__, "nodes": [NormalizedNode(node_id="wrong_node")]})


@pytest.fixture
def agent(monkeypatch):
    calls = []

    def load(_spec):
        calls.append(_spec)
        return SkipsANode()

    monkeypatch.setattr(cli, "load_adapter", load)
    return calls


def _run(tmp_path, *extra) -> tuple[int, dict | None]:
    args = cli._build_parser().parse_args(
        ["run", "--dataset", DATASET, "--adapter", "x:Y", "--output-dir", str(tmp_path / "out"), *extra]
    )
    code = cli.cmd_run(args)
    reports = sorted((tmp_path / "out").glob("report_*.json")) if (tmp_path / "out").exists() else []
    return code, (json.loads(reports[-1].read_text()) if reports else None)


def _check_atf(report, weights):
    assert report["settings"]["weights"] == pytest.approx(weights)
    for c in report["conversations"]:
        expected = atf_score({k: c[k] for k in ("nts", "sts", "tis", "os")}, weights)[0]
        assert c["atf"] == pytest.approx(expected)


# --- atf-eval run ----------------------------------------------------------------

def test_run_without_weights_uses_the_defaults(tmp_path, agent):
    code, report = _run(tmp_path)
    assert code == 0
    _check_atf(report, DEFAULT_WEIGHTS)


def test_run_with_weights_spec(tmp_path, agent):
    code, report = _run(tmp_path, "--weights", CUSTOM_SPEC)
    assert code == 0
    _check_atf(report, CUSTOM)
    _, default_report = _run(tmp_path / "d")
    assert report["conversations"][0]["atf"] != pytest.approx(default_report["conversations"][0]["atf"])


def test_run_with_weights_file(tmp_path, agent):
    path = tmp_path / "w.json"
    path.write_text(json.dumps(CUSTOM))
    code, report = _run(tmp_path, "--weights-file", str(path))
    assert code == 0
    _check_atf(report, CUSTOM)


def test_run_rejects_both_weight_options(tmp_path, agent, capsys):
    path = tmp_path / "w.json"
    path.write_text(json.dumps(CUSTOM))
    with pytest.raises(SystemExit) as exc:
        _run(tmp_path, "--weights", CUSTOM_SPEC, "--weights-file", str(path))
    assert exc.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err
    assert agent == []


@pytest.mark.parametrize(
    "spec, message",
    [
        ("nts=0.5,sts=0.5,tis=0.0", "missing ['os']"),
        ("nts=0.4,sts=0.3,tis=0.2,os=0.1,rs=0", "unexpected ['rs']"),
        ("nts=0.5,sts=0.3,tis=0.2,os=0.1", "must sum to 1, got 1.1"),
        ("nts=-0.1,sts=0.8,tis=0.2,os=0.1", "'nts' must be non-negative"),
        ("nts=nan,sts=0.3,tis=0.2,os=0.1", "'nts' must be finite"),
        ("nts=abc,sts=0.3,tis=0.2,os=0.1", "'nts' must be a number"),
        ("nts=0.4;sts=0.6", "'nts' must be a number, got '0.4;sts=0.6'"),
        ("nts=0.4,sts", "expected component=number"),
    ],
)
def test_run_rejects_invalid_weights_before_running_the_agent(tmp_path, agent, capsys, spec, message):
    code, report = _run(tmp_path, "--weights", spec)
    assert code == 2 and report is None
    err = capsys.readouterr().err
    assert "invalid ATF weights" in err and message in err
    assert agent == []  # the adapter was never even loaded


@pytest.mark.parametrize(
    "content, message",
    [
        ('{"nts": 0.4, "sts": 0.3, "tis": 0.2, "os": "0.1"}', "'os' must be a number"),
        ('{"nts": 0.4, "sts": 0.3, "tis": 0.2, "os": true}', "'os' must be a number"),
        ('[0.25, 0.25, 0.25, 0.25]', "must be an object"),
        ("{nts: 0.4", "is not valid JSON"),
    ],
)
def test_run_rejects_invalid_weights_file(tmp_path, agent, capsys, content, message):
    path = tmp_path / "w.json"
    path.write_text(content)
    code, _ = _run(tmp_path, "--weights-file", str(path))
    assert code == 2 and message in capsys.readouterr().err
    assert agent == []


def test_run_rejects_missing_weights_file(tmp_path, agent, capsys):
    code, _ = _run(tmp_path, "--weights-file", str(tmp_path / "nope.json"))
    assert code == 2 and "cannot read --weights-file" in capsys.readouterr().err


# --- atf-eval submit -------------------------------------------------------------

class FakeClient:
    """Stands in for AtfClient: records what `submit` would send to the server."""
    instances: list = []

    def __init__(self, server=None):
        self.options = None
        FakeClient.instances.append(self)

    def health(self):
        return {"status": "ok"}

    def submit(self, collected, *, golden_id=None, golden=None, options=None, label=""):
        self.options = options
        return "eval-1"

    def wait(self, eval_id, timeout=None, on_status=None):
        return {"summary": {"conversations": 0, "judge": None, "atf_mean": None}, "runs": [], "skipped": []}

    def download(self, result, out_dir):
        return {}


@pytest.fixture
def fake_server(monkeypatch):
    import atf_eval.client

    FakeClient.instances = []
    monkeypatch.setattr(atf_eval.client, "AtfClient", FakeClient)
    return FakeClient.instances


def _submit(tmp_path, *extra) -> int:
    args = cli._build_parser().parse_args(
        ["submit", "--dataset", DATASET, "--adapter", "x:Y", "--no-llm-evals",
         "--output-dir", str(tmp_path / "out"), *extra]
    )
    return cli.cmd_submit(args)


def test_submit_without_weights_sends_none_so_the_server_uses_the_defaults(tmp_path, agent, fake_server):
    assert _submit(tmp_path) == 0
    assert fake_server[0].options["weights"] is None


def test_submit_sends_validated_weights(tmp_path, agent, fake_server):
    assert _submit(tmp_path, "--weights", CUSTOM_SPEC) == 0
    assert fake_server[0].options["weights"] == CUSTOM
    path = tmp_path / "w.json"
    path.write_text(json.dumps({"nts": 1, "sts": 0, "tis": 0, "os": 0}))
    assert _submit(tmp_path, "--weights-file", str(path)) == 0
    assert fake_server[1].options["weights"] == {"nts": 1.0, "sts": 0.0, "tis": 0.0, "os": 0.0}


def test_submit_rejects_invalid_weights_before_running_the_agent(tmp_path, agent, fake_server, capsys):
    assert _submit(tmp_path, "--weights", "nts=0.5,sts=0.5,tis=0.5,os=0.5") == 2
    assert "must sum to 1" in capsys.readouterr().err
    assert agent == [] and fake_server == []  # no agent run, no server contact


def test_submit_rejects_both_weight_options(tmp_path, agent, fake_server, capsys):
    with pytest.raises(SystemExit) as exc:
        _submit(tmp_path, "--weights", CUSTOM_SPEC, "--weights-file", "w.json")
    assert exc.value.code == 2 and "not allowed with argument" in capsys.readouterr().err


# --- atf-eval dashboard ------------------------------------------------------------

CALL_SUPPORT = ROOT / "agents" / "call_support"


def _dashboard(tmp_path, *extra):
    from atf_eval.dashboard.generate import run as run_dashboard

    args = cli._build_parser().parse_args(
        ["dashboard", "--golden-dir", str(CALL_SUPPORT / "golden"), "--scenarios", str(CALL_SUPPORT / "traces"),
         "--output-dir", str(tmp_path / "dash"), "--no-llm-evals", *extra]
    )
    run_dashboard(args)
    html = next((tmp_path / "dash").glob("atf_dashboard_*.html")).read_text()
    settings = json.loads(next((tmp_path / "dash").glob("atf_settings_*.json")).read_text())
    return html, settings


def test_dashboard_shows_the_default_weights_when_none_given(tmp_path):
    html, settings = _dashboard(tmp_path)
    assert settings["weights"] == pytest.approx(DEFAULT_WEIGHTS)
    assert "ATF weights used: NTS 0.3333 &middot; STS 0.3333 &middot; TIS 0.1667 &middot; OS 0.1667" in html
    assert '<span class="weight">33.3%</span>' in html and "{{" not in html


def test_dashboard_shows_the_custom_weights_actually_used(tmp_path):
    html, settings = _dashboard(tmp_path, "--weights", CUSTOM_SPEC)
    assert settings["weights"] == CUSTOM
    assert "ATF weights used: NTS 0.4 &middot; STS 0.3 &middot; TIS 0.2 &middot; OS 0.1" in html
    for pct in ("40.0%", "30.0%", "20.0%", "10.0%"):
        assert f'<span class="weight">{pct}</span>' in html and f"{pct} of ATF" in html
    assert '<span class="weight">33.3%</span>' not in html and "33.3% of ATF" not in html
    assert "{{" not in html


def test_dashboard_rejects_invalid_weights(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        _dashboard(tmp_path, "--weights", "nts=1")
    assert exc.value.code == 2 and "missing" in capsys.readouterr().err
    assert not (tmp_path / "dash").exists()
