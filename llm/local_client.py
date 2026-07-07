"""Local provider: Ollama's OpenAI-compatible /v1 endpoint, reusing the OpenAI adapter.

Ollama ignores the API key but the SDK requires one — "ollama" by convention.
A down Ollama server surfaces as the SDK's connection error naming the base_url;
no special handling (same fail-loud policy as the cloud clients).
"""

from openai import OpenAI
from pydantic import BaseModel

from config import settings
from llm.base import LLMResponse
from llm.openai_client import generate_openai

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=settings.local_base_url, api_key="ollama",
                         max_retries=settings.llm_max_retries)
    return _client


def generate_local(
    messages: list[dict],
    *,
    system: str | list[dict] | None = None,
    tools: list[dict] | None = None,
    structured_schema: type[BaseModel] | None = None,
    max_tokens: int = 4096,
) -> LLMResponse:
    return generate_openai(
        messages, system=system, tools=tools, structured_schema=structured_schema,
        max_tokens=max_tokens, client=_get_client(), model=settings.local_model,
    )
