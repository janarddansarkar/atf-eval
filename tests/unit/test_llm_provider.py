"""The LLM provider seam (atf_eval.llm_provider): provider selection, API-key
discovery, and that the judge goes through `parse_structured()` -- all
network-free, using fake SDK clients."""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Literal

import pytest
from pydantic import BaseModel

from atf_eval import llm_provider
from atf_eval.llm_evals.judge import JudgeUnavailable, judge_structured
from atf_eval.llm_provider import (
    AnthropicProvider,
    NvidiaProvider,
    _extract_json,
    get_provider,
    load_dotenv,
    provider_name,
)



class JudgeScore(BaseModel):
    """A small structured-output model for exercising the provider seam."""
    verdict: Literal["correct", "partial", "incorrect"]
    reasoning: str


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    """No real keys, no real .env, no on-disk judge cache."""
    for name in ("ATF_LLM_PROVIDER", "ANTHROPIC_API_KEY", "NVIDIA_API_KEY", "NVIDIA_BASE_URL", "ATF_LLM_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(llm_provider, "_dotenv_candidates", lambda: [tmp_path / ".env"])
    monkeypatch.setenv("ATF_LLM_CACHE", "off")


class FakeProvider:
    name = "fake"
    api_key_env = "FAKE_API_KEY"
    default_model = "fake-model"
    supports_effort = False

    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, []

    def parse_structured(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.result


def test_provider_name_precedence(monkeypatch):
    assert provider_name() == "anthropic"
    monkeypatch.setenv("ATF_LLM_PROVIDER", "NVIDIA")
    assert provider_name() == "nvidia"
    assert provider_name("anthropic") == "anthropic"


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="unknown LLM provider"):
        provider_name("nope")


def test_no_api_key_means_no_provider():
    assert get_provider("anthropic") is None
    assert get_provider("nvidia") is None


def test_each_provider_is_built_from_its_own_key(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    assert isinstance(get_provider("anthropic"), AnthropicProvider)
    nvidia = get_provider("nvidia")
    assert isinstance(nvidia, NvidiaProvider)
    assert str(nvidia._client.base_url).startswith(NvidiaProvider.default_base_url)


def test_dotenv_fills_provider_and_key_but_never_overrides_exported(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text(
        "# comment\nATF_LLM_PROVIDER=nvidia\nNVIDIA_API_KEY='nvapi-from-file'\n"
        "ANTHROPIC_API_KEY=from-file\nUNRELATED=x\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "exported")
    monkeypatch.delenv("UNRELATED", raising=False)
    load_dotenv()
    assert os.environ["ATF_LLM_PROVIDER"] == "nvidia"
    assert os.environ["NVIDIA_API_KEY"] == "nvapi-from-file"
    assert os.environ["ANTHROPIC_API_KEY"] == "exported"
    assert "UNRELATED" not in os.environ


def test_dotenv_also_fills_server_and_endpoint_settings(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("ATF_SERVER=http://example:9000\nNVIDIA_BASE_URL=http://nim:8000/v1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    for name in ("ATF_SERVER", "NVIDIA_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    load_dotenv()
    assert os.environ["ATF_SERVER"] == "http://example:9000"
    assert os.environ["NVIDIA_BASE_URL"] == "http://nim:8000/v1"


def test_extract_json_strips_reasoning_and_fences():
    text = '<think>{"not": "this"}</think>Sure:\n```json\n{"verdict": "correct", "reasoning": "ok"}\n```'
    assert _extract_json(text) == '{"verdict": "correct", "reasoning": "ok"}'
    with pytest.raises(ValueError):
        _extract_json("no json here")


class _FakeBadRequestError(Exception):
    pass


def _nvidia_with_fake_client(responses):
    """An NvidiaProvider whose chat.completions.create replays `responses`
    (an Exception instance is raised instead of returned)."""
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=r))])

    provider = NvidiaProvider.__new__(NvidiaProvider)
    provider._openai = SimpleNamespace(BadRequestError=_FakeBadRequestError)
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return provider, calls


def test_nvidia_parses_structured_output_with_guided_json():
    provider, calls = _nvidia_with_fake_client(['{"verdict": "partial", "reasoning": "close"}'])
    out = provider.parse_structured(
        model="m", system="sys", user="u", output_model=JudgeScore, max_tokens=50
    )
    assert out == JudgeScore(verdict="partial", reasoning="close")
    assert calls[0]["extra_body"]["nvext"]["guided_json"] == JudgeScore.model_json_schema()
    assert "JSON schema" in calls[0]["messages"][0]["content"]


def test_nvidia_falls_back_to_prompt_only_json_when_guided_json_is_rejected():
    provider, calls = _nvidia_with_fake_client(
        [_FakeBadRequestError("guided_json unsupported"), '{"verdict": "correct", "reasoning": "r"}']
    )
    out = provider.parse_structured(
        model="m", system="sys", user="u", output_model=JudgeScore, max_tokens=50
    )
    assert out.verdict == "correct"
    assert "extra_body" not in calls[1]


def test_judge_structured_uses_the_providers_default_model():
    verdict = JudgeScore(verdict="correct", reasoning="r")
    provider = FakeProvider(result=verdict)
    assert judge_structured(provider, system="s", user="u", output_model=JudgeScore) == verdict
    assert provider.calls[0]["model"] == "fake-model"


def test_judge_structured_turns_provider_errors_into_judge_unavailable():
    with pytest.raises(JudgeUnavailable, match="judge call failed"):
        judge_structured(FakeProvider(error=RuntimeError("boom")), system="s", user="u",
                         output_model=JudgeScore)
    with pytest.raises(JudgeUnavailable, match="no judge client"):
        judge_structured(None, system="s", user="u", output_model=JudgeScore)



# --- per-request timeout (keeps one stalled judge call from pinning a server worker) ---

def test_request_timeout_default_and_override(monkeypatch):
    assert llm_provider.request_timeout() == llm_provider.DEFAULT_LLM_TIMEOUT == 120.0
    monkeypatch.setenv("ATF_LLM_TIMEOUT", "45")
    assert llm_provider.request_timeout() == 45.0


@pytest.mark.parametrize("raw", ["abc", "0", "-5", "nan", "inf"])
def test_request_timeout_rejects_bad_values(monkeypatch, raw):
    monkeypatch.setenv("ATF_LLM_TIMEOUT", raw)
    with pytest.raises(ValueError, match="ATF_LLM_TIMEOUT"):
        llm_provider.request_timeout()


def test_providers_are_built_with_the_timeout(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.setenv("ATF_LLM_TIMEOUT", "30")
    monkeypatch.setenv("NVIDIA_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    nvidia = NvidiaProvider()
    assert nvidia._client.timeout == 30.0 and nvidia._client.max_retries == 2
    anthropic_provider = AnthropicProvider()
    assert anthropic_provider._client.timeout == 30.0 and anthropic_provider._client.max_retries == 2
