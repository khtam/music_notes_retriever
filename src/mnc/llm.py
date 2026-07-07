"""Provider-agnostic LLM access for musical analysis (song-structure labeling).

The pipeline only ever talks to the small `LLMClient` interface below, so
providers are interchangeable. An API key can be passed per request (the web
form) and beats the environment; otherwise configuration is env-driven:

    MNC_LLM_PROVIDER   a PROVIDERS id (below) or "none"  (default: auto-detect by API key)
    MNC_LLM_MODEL      model override for the chosen provider
    MNC_LLM_BASE_URL   endpoint override (for "local"/"custom", or any compat server)
    ANTHROPIC_API_KEY  enables the Anthropic provider
    OPENAI_API_KEY     enables the OpenAI provider
    OPENAI_BASE_URL    point the OpenAI client at any OpenAI-compatible server
                       (Ollama, vLLM, LM Studio, ...)

Almost every non-Anthropic provider (Google, DeepSeek, Qwen, Moonshot, Zhipu,
xAI, Groq, OpenRouter, local servers, ...) speaks the OpenAI chat-completions
API, so they all reuse `OpenAIClient` with a different `base_url` — see
`PROVIDERS`. Only Anthropic gets a native client.

Everything degrades gracefully: with no provider configured the pipeline
falls back to pure-heuristic analysis, never an error.
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-8"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


class LLMError(Exception):
    pass


class LLMClient(ABC):
    """Minimal interface the pipeline depends on. Implement this to add a provider."""

    name: str = "llm"

    @abstractmethod
    def generate_json(self, system: str, prompt: str, schema: dict) -> dict:
        """Return a JSON object conforming to `schema` for the given prompt."""


@dataclass(frozen=True)
class ProviderSpec:
    """One entry in the provider picker (backend registry + frontend dropdown)."""

    id: str
    label: str
    group: str  # "major" | "regional" | "local"
    client: str  # "anthropic" | "openai"
    default_model: str
    base_url: Optional[str] = None  # None => SDK default (api.openai.com) for client="openai"
    key_prefix: Optional[str] = None  # placeholder hint, e.g. "sk-ant-..."
    key_env: tuple[str, ...] = ()  # env vars that also supply the key, checked in order
    docs_url: Optional[str] = None
    needs_key: bool = True
    editable_base_url: bool = False

    def public(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "group": self.group,
            "default_model": self.default_model,
            "base_url": self.base_url,
            "key_prefix": self.key_prefix,
            "docs_url": self.docs_url,
            "needs_key": self.needs_key,
            "editable_base_url": self.editable_base_url,
        }


PROVIDERS: dict[str, ProviderSpec] = {
    spec.id: spec
    for spec in [
        # -- major players --------------------------------------------------
        ProviderSpec(
            id="anthropic",
            label="Anthropic (Claude)",
            group="major",
            client="anthropic",
            default_model=DEFAULT_ANTHROPIC_MODEL,
            key_prefix="sk-ant-...",
            key_env=("ANTHROPIC_API_KEY",),
            docs_url="https://console.anthropic.com/settings/keys",
        ),
        ProviderSpec(
            id="openai",
            label="OpenAI",
            group="major",
            client="openai",
            default_model=DEFAULT_OPENAI_MODEL,
            key_prefix="sk-...",
            key_env=("OPENAI_API_KEY",),
            docs_url="https://platform.openai.com/api-keys",
        ),
        ProviderSpec(
            id="google",
            label="Google (Gemini)",
            group="major",
            client="openai",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            default_model="gemini-2.0-flash",
            key_prefix="AIza...",
            key_env=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
            docs_url="https://aistudio.google.com/apikey",
        ),
        ProviderSpec(
            id="xai",
            label="xAI (Grok)",
            group="major",
            client="openai",
            base_url="https://api.x.ai/v1",
            default_model="grok-2-latest",
            key_prefix="xai-...",
            key_env=("XAI_API_KEY",),
            docs_url="https://console.x.ai",
        ),
        # -- low-cost / regional ---------------------------------------------
        ProviderSpec(
            id="deepseek",
            label="DeepSeek",
            group="regional",
            client="openai",
            base_url="https://api.deepseek.com",
            default_model="deepseek-chat",
            key_prefix="sk-...",
            key_env=("DEEPSEEK_API_KEY",),
            docs_url="https://platform.deepseek.com/api_keys",
        ),
        ProviderSpec(
            id="qwen",
            label="Alibaba Qwen (DashScope)",
            group="regional",
            client="openai",
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            default_model="qwen-plus",
            key_prefix="sk-...",
            key_env=("DASHSCOPE_API_KEY", "QWEN_API_KEY"),
            docs_url="https://dashscope.console.aliyun.com/apiKey",
        ),
        ProviderSpec(
            id="moonshot",
            label="Moonshot (Kimi)",
            group="regional",
            client="openai",
            base_url="https://api.moonshot.ai/v1",
            default_model="moonshot-v1-8k",
            key_prefix="sk-...",
            key_env=("MOONSHOT_API_KEY",),
            docs_url="https://platform.moonshot.ai/console/api-keys",
        ),
        ProviderSpec(
            id="zhipu",
            label="Zhipu (GLM)",
            group="regional",
            client="openai",
            base_url="https://open.bigmodel.cn/api/paas/v4",
            default_model="glm-4-flash",
            key_prefix="...",
            key_env=("ZHIPU_API_KEY",),
            docs_url="https://open.bigmodel.cn/usercenter/apikeys",
        ),
        ProviderSpec(
            id="groq",
            label="Groq",
            group="regional",
            client="openai",
            base_url="https://api.groq.com/openai/v1",
            default_model="llama-3.3-70b-versatile",
            key_prefix="gsk_...",
            key_env=("GROQ_API_KEY",),
            docs_url="https://console.groq.com/keys",
        ),
        ProviderSpec(
            id="openrouter",
            label="OpenRouter",
            group="regional",
            client="openai",
            base_url="https://openrouter.ai/api/v1",
            default_model="deepseek/deepseek-chat",
            key_prefix="sk-or-...",
            key_env=("OPENROUTER_API_KEY",),
            docs_url="https://openrouter.ai/keys",
        ),
        # -- local / custom ---------------------------------------------------
        ProviderSpec(
            id="local",
            label="Local (Ollama / LM Studio)",
            group="local",
            client="openai",
            base_url="http://localhost:11434/v1",
            default_model="llama3.1",
            needs_key=False,
            editable_base_url=True,
        ),
        ProviderSpec(
            id="custom",
            label="Custom (OpenAI-compatible)",
            group="local",
            client="openai",
            base_url=None,
            default_model="",
            key_prefix=None,
            editable_base_url=True,
        ),
    ]
}


def _extract_json(text: str) -> dict:
    """Tolerant JSON extraction for providers without native structured output."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end <= start:
        raise LLMError(f"No JSON object found in LLM response: {text[:200]!r}")
    return json.loads(candidate[start : end + 1])


class AnthropicClient(LLMClient):
    name = "anthropic"

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        try:
            import anthropic
        except ImportError as exc:
            raise LLMError(
                "The 'anthropic' package is not installed. Run: pip install -e '.[llm]'"
            ) from exc
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMError("No Anthropic API key (paste one in the form or set ANTHROPIC_API_KEY)")
        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model or DEFAULT_ANTHROPIC_MODEL

    def generate_json(self, system: str, prompt: str, schema: dict) -> dict:
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=16000,
                thinking={"type": "adaptive"},
                system=system,
                messages=[{"role": "user", "content": prompt}],
                output_config={"format": {"type": "json_schema", "schema": schema}},
            )
        except self._anthropic.APIError as exc:
            raise LLMError(f"Anthropic request failed: {exc}") from exc
        if response.stop_reason == "refusal":
            raise LLMError("Anthropic declined the request")
        text = next((b.text for b in response.content if b.type == "text"), "")
        return json.loads(text)


class OpenAIClient(LLMClient):
    """OpenAI, or any OpenAI-compatible endpoint (Google, DeepSeek, Qwen, Moonshot,
    Zhipu, xAI, Groq, OpenRouter, local servers, ...) via `base_url`."""

    name = "openai"

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        try:
            import openai
        except ImportError as exc:
            raise LLMError(
                "The 'openai' package is not installed. Run: pip install -e '.[llm]'"
            ) from exc
        self._openai = openai
        try:
            self.client = openai.OpenAI(
                # A local/unauthenticated server still needs a non-empty string here.
                api_key=api_key or os.getenv("OPENAI_API_KEY") or "not-needed",
                base_url=base_url or os.getenv("OPENAI_BASE_URL"),
            )
        except openai.OpenAIError as exc:  # e.g. no key anywhere
            raise LLMError(f"OpenAI client unavailable: {exc}") from exc
        self.model = model or DEFAULT_OPENAI_MODEL

    def generate_json(self, system: str, prompt: str, schema: dict) -> dict:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt + "\n\nRespond with a single JSON object."},
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
            )
        except self._openai.BadRequestError:
            # Some OpenAI-compatible servers reject `response_format`; retry
            # in plain-text mode and lean on the tolerant extractor below.
            response = self.client.chat.completions.create(model=self.model, messages=messages)
        except self._openai.OpenAIError as exc:
            raise LLMError(f"OpenAI request failed: {exc}") from exc
        content = response.choices[0].message.content or ""
        return _extract_json(content)


def resolve_provider(provider: Optional[str] = None, api_key: Optional[str] = None) -> Optional[str]:
    """Normalize the provider choice; None means heuristic-only analysis.

    With no explicit provider, an ad-hoc key is sniffed by prefix (Anthropic
    keys start with 'sk-ant-'), then the environment decides.
    """
    provider = (provider or os.getenv("MNC_LLM_PROVIDER") or "").strip().lower()
    if provider in ("none", "off", "disabled"):
        return None
    if provider:
        if provider in PROVIDERS:
            return provider
        known = ", ".join(sorted(PROVIDERS)) + ", none"
        raise LLMError(f"Unknown LLM provider {provider!r}; use one of: {known}")
    if api_key:
        return "anthropic" if api_key.startswith("sk-ant-") else "openai"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_BASE_URL"):
        return "openai"
    for pid, spec in PROVIDERS.items():
        if pid in ("openai", "anthropic"):
            continue
        if any(os.getenv(var) for var in spec.key_env):
            return pid
    return None


def get_llm_client(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Optional[LLMClient]:
    """Build the configured LLM client, or None when analysis should be heuristic-only."""
    api_key = (api_key or "").strip() or None
    provider = resolve_provider(provider, api_key)
    model = model or os.getenv("MNC_LLM_MODEL")
    base_url = base_url or os.getenv("MNC_LLM_BASE_URL")

    if provider is None:
        return None

    if provider == "anthropic":
        return AnthropicClient(model=model, api_key=api_key)
    spec = PROVIDERS[provider]

    base_url = base_url or spec.base_url
    if spec.editable_base_url and not spec.base_url and not base_url:
        raise LLMError(f"{spec.label} requires a base URL (paste one in the form)")

    if not api_key:
        api_key = next((os.getenv(var) for var in spec.key_env if os.getenv(var)), None)
    if spec.needs_key and not api_key and provider != "openai":
        raise LLMError(
            f"No API key for {spec.label} (paste one in the form or set {spec.key_env[0]!r})"
            if spec.key_env
            else f"No API key for {spec.label} (paste one in the form)"
        )

    return OpenAIClient(
        model=model or spec.default_model or None,
        base_url=base_url,
        api_key=api_key,
    )
