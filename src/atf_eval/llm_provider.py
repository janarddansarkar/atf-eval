"""The one place that knows which LLM API the judges call.

Both LLM judges (RS's routing judge, judge.py, and the Group 1-5 metrics,
llm_evals/judge.py) need exactly one thing from a provider: send a system +
user prompt and get back an instance of a Pydantic model. `LLMProvider` is
that interface; everything else in atf_eval takes a provider object (or None
-> the LLM-judged dimensions report N/A) and never imports a vendor SDK.

Selecting a provider: `--provider` on the CLI, else the ATF_LLM_PROVIDER env
var, else "anthropic". Each provider reads its own API key env var, which may
also come from a `.env` file (see `load_dotenv()`).

Adding a provider: write a class with the `LLMProvider` attributes and
`parse_structured()`, and register it in `PROVIDERS`. Import its SDK inside
`__init__` so users of other providers don't need it installed.
"""
from __future__ import annotations

import json
import os
import re
import ssl
import sys
from pathlib import Path
from typing import ClassVar, Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

PROVIDER_ENV = "ATF_LLM_PROVIDER"
DEFAULT_PROVIDER = "anthropic"


class LLMProvider(Protocol):
    name: ClassVar[str]
    api_key_env: ClassVar[str]
    default_model: ClassVar[str]
    # Whether the `effort` setting means anything to this provider. When it
    # doesn't, callers pass effort=None so it isn't recorded as if applied.
    supports_effort: ClassVar[bool]

    def parse_structured(
        self,
        *,
        model: str,
        system: str,
        user: str,
        output_model: type[T],
        max_tokens: int,
        effort: str | None = None,
    ) -> T:
        """One call; returns a validated `output_model`, or raises."""
        ...


class AnthropicProvider:
    name = "anthropic"
    api_key_env = "ANTHROPIC_API_KEY"
    default_model = "claude-opus-5"
    supports_effort = True

    def __init__(self) -> None:
        import anthropic

        self._client = anthropic.Anthropic()

    def parse_structured(self, *, model, system, user, output_model, max_tokens, effort=None):
        parse_kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            system=system,
            output_format=output_model,
            messages=[{"role": "user", "content": user}],
        )
        if effort:
            parse_kwargs["output_config"] = {"effort": effort}
        response = self._client.messages.parse(**parse_kwargs)
        if response.parsed_output is None:
            raise ValueError("model did not return parseable structured output")
        return response.parsed_output


_JSON_INSTRUCTIONS = """

Respond with only a single JSON object (no prose, no code fences) that conforms to this JSON schema:
{schema}"""


def _extract_json(text: str) -> str:
    """The JSON object in a chat completion: drops a reasoning model's
    <think> block and any code fence or prose around the object."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise ValueError("model response contained no JSON object")
    return text[start : end + 1]


class NvidiaProvider:
    """NVIDIA NIM's hosted, OpenAI-compatible API (build.nvidia.com).
    Override the endpoint with NVIDIA_BASE_URL for a self-hosted NIM."""

    name = "nvidia"
    api_key_env = "NVIDIA_API_KEY"
    default_model = "openai/gpt-oss-20b"
    supports_effort = False
    default_base_url = "https://integrate.api.nvidia.com/v1"

    def __init__(self) -> None:
        import openai

        self._openai = openai
        self._client = openai.OpenAI(
            base_url=os.environ.get("NVIDIA_BASE_URL", self.default_base_url),
            api_key=os.environ[self.api_key_env],
            # The SDK's own default trusts *only* SSL_CERT_FILE when it is set,
            # so a corporate bundle holding just an internal root CA fails every
            # call ("Connection error" / CERTIFICATE_VERIFY_FAILED). The stdlib
            # default context trusts the OS store plus SSL_CERT_FILE.
            http_client=openai.DefaultHttpxClient(verify=ssl.create_default_context()),
        )

    def parse_structured(self, *, model, system, user, output_model, max_tokens, effort=None):
        schema = output_model.model_json_schema()
        create_kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system + _JSON_INSTRUCTIONS.format(schema=json.dumps(schema))},
                {"role": "user", "content": user},
            ],
        )
        try:
            # NIM's guided decoding constrains the output to the schema.
            response = self._client.chat.completions.create(
                **create_kwargs, extra_body={"nvext": {"guided_json": schema}}
            )
        except self._openai.BadRequestError:
            # Not every hosted model accepts guided decoding; the schema in the
            # system prompt plus validation below still applies.
            response = self._client.chat.completions.create(**create_kwargs)
        text = response.choices[0].message.content or ""
        return output_model.model_validate_json(_extract_json(text))


PROVIDERS: dict[str, type] = {p.name: p for p in (AnthropicProvider, NvidiaProvider)}


# Checked in order, first match wins. Current working directory first: when
# atf_eval is consumed as a dependency (pip install -e ./atf-eval from inside
# your own project), `.env` belongs at *your* project root, where you run
# `atf-eval dashboard` from -- not inside the atf-eval package checkout,
# which is wherever __file__ happens to resolve to and may not even be
# writable (an installed, non-editable package). The package-root path is
# kept as a fallback for this repo's own local dev workflow.
def _dotenv_candidates() -> list[Path]:
    candidates = [Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"]
    seen: list[Path] = []
    for p in candidates:
        if p not in seen:
            seen.append(p)
    return seen


def load_dotenv() -> None:
    """Fill the provider selector and every provider's API key from a `.env`
    when not already exported (dev convenience -- the deterministic pipeline
    never needs a key). An exported variable always wins."""
    names = {PROVIDER_ENV} | {p.api_key_env for p in PROVIDERS.values()}
    for env_path in _dotenv_candidates():
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            name, value = name.strip(), value.strip().strip('"').strip("'")
            if name in names and value and not os.environ.get(name):
                os.environ[name] = value


def provider_name(explicit: str | None = None) -> str:
    """`explicit` (the --provider flag), else ATF_LLM_PROVIDER, else the default."""
    load_dotenv()
    name = (explicit or os.environ.get(PROVIDER_ENV) or DEFAULT_PROVIDER).strip().lower()
    if name not in PROVIDERS:
        raise ValueError(f"unknown LLM provider {name!r}; choose one of: {', '.join(PROVIDERS)}")
    return name


def get_provider(name: str | None = None) -> LLMProvider | None:
    """The selected provider if its API key is reachable, else None (the
    LLM-judged dimensions then report N/A)."""
    cls = PROVIDERS[provider_name(name)]
    if not os.environ.get(cls.api_key_env):
        return None
    try:
        return cls()
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] could not construct {cls.name} client: {e}", file=sys.stderr)
        return None
