from typing import Any

from pydantic import BaseModel

from config import settings


class ToolCall(BaseModel):
    id: str
    name: str
    input: dict


class LLMResponse(BaseModel):
    text: str = ""
    tool_calls: list[ToolCall] = []
    parsed: Any | None = None  # populated when structured_schema was given
    stop_reason: str | None = None
    usage: dict = {}


def generate(
    messages: list[dict],
    *,
    system: str | list[dict] | None = None,
    tools: list[dict] | None = None,
    structured_schema: type[BaseModel] | None = None,
    provider: str | None = None,
    max_tokens: int | None = None,
) -> LLMResponse:
    """Provider-neutral chat entrypoint.

    `messages` and `tools` use the Anthropic shape everywhere in this codebase;
    the OpenAI client adapts them. `structured_schema` is a pydantic model class;
    the validated instance comes back on `LLMResponse.parsed`.
    """
    provider = provider or settings.llm_provider
    max_tokens = max_tokens or settings.llm_max_tokens
    if provider == "anthropic":
        from llm.anthropic_client import generate_anthropic

        return generate_anthropic(
            messages, system=system, tools=tools,
            structured_schema=structured_schema, max_tokens=max_tokens,
        )
    if provider == "openai":
        from llm.openai_client import generate_openai

        return generate_openai(
            messages, system=system, tools=tools,
            structured_schema=structured_schema, max_tokens=max_tokens,
        )
    raise ValueError(f"Unknown provider: {provider}")
