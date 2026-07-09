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
    """Provider-neutral chat entrypoint (anthropic | openai | local/Ollama).

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
    if provider == "local":
        from llm.local_client import generate_local

        return generate_local(
            messages, system=system, tools=tools,
            structured_schema=structured_schema, max_tokens=max_tokens,
        )
    raise ValueError(f"Unknown provider: {provider}")


def generate_stream(
    messages: list[dict],
    *,
    system: str | list[dict] | None = None,
    tools: list[dict] | None = None,
    on_delta,
    provider: str | None = None,
    max_tokens: int | None = None,
) -> LLMResponse:
    """Streaming variant of generate(): on_delta(str) fires per text chunk,
    the complete LLMResponse (with tool_calls) is returned at the end.
    No structured_schema — structured calls stay on generate()."""
    provider = provider or settings.llm_provider
    max_tokens = max_tokens or settings.llm_max_tokens
    if provider == "anthropic":
        import llm.anthropic_client as anthropic_client

        return anthropic_client.generate_anthropic_stream(
            messages, system=system, tools=tools, max_tokens=max_tokens,
            on_delta=on_delta,
        )
    if provider == "openai":
        import llm.openai_client as openai_client

        return openai_client.generate_openai_stream(
            messages, system=system, tools=tools, max_tokens=max_tokens,
            on_delta=on_delta,
        )
    if provider == "local":
        import llm.local_client as local_client

        return local_client.generate_local_stream(
            messages, system=system, tools=tools, max_tokens=max_tokens,
            on_delta=on_delta,
        )
    raise ValueError(f"Unknown provider: {provider}")
