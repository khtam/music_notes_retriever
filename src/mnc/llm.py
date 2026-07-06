"""Provider-agnostic LLM access for musical analysis (song-structure labeling).

The pipeline only ever talks to the small `LLMClient` interface below, so
providers are interchangeable. Configuration is env-driven:

    MNC_LLM_PROVIDER   anthropic | openai | none   (default: auto-detect by API key)
    MNC_LLM_MODEL      model override for the chosen provider
    ANTHROPIC_API_KEY  enables the Anthropic provider
    OPENAI_API_KEY     enables the OpenAI provider
    OPENAI_BASE_URL    point the OpenAI client at any OpenAI-compatible server
                       (Ollama, vLLM, LM Studio, ...)

Everything degrades gracefully: with no provider configured the pipeline
falls back to pure-heuristic analysis, never an error.
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
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

    def __init__(self, model: Optional[str] = None):
        try:
            import anthropic
        except ImportError as exc:
            raise LLMError(
                "The 'anthropic' package is not installed. Run: pip install -e '.[llm]'"
            ) from exc
        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
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
    """OpenAI, or any OpenAI-compatible endpoint via OPENAI_BASE_URL."""

    name = "openai"

    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None):
        try:
            import openai
        except ImportError as exc:
            raise LLMError(
                "The 'openai' package is not installed. Run: pip install -e '.[llm]'"
            ) from exc
        self._openai = openai
        self.client = openai.OpenAI(base_url=base_url or os.getenv("OPENAI_BASE_URL"))
        self.model = model or DEFAULT_OPENAI_MODEL

    def generate_json(self, system: str, prompt: str, schema: dict) -> dict:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt + "\n\nRespond with a single JSON object."},
                ],
                response_format={"type": "json_object"},
            )
        except self._openai.OpenAIError as exc:
            raise LLMError(f"OpenAI request failed: {exc}") from exc
        content = response.choices[0].message.content or ""
        return _extract_json(content)


def get_llm_client(
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[LLMClient]:
    """Build the configured LLM client, or None when analysis should be heuristic-only."""
    provider = (provider or os.getenv("MNC_LLM_PROVIDER") or "").strip().lower()
    model = model or os.getenv("MNC_LLM_MODEL")

    if not provider:  # auto-detect from available credentials
        if os.getenv("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_BASE_URL"):
            provider = "openai"
        else:
            return None

    if provider in ("none", "off", "disabled"):
        return None
    if provider == "anthropic":
        return AnthropicClient(model=model)
    if provider == "openai":
        return OpenAIClient(model=model)
    raise LLMError(f"Unknown LLM provider {provider!r}; use 'anthropic', 'openai', or 'none'")
