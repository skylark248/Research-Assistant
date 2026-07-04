import anthropic
from pydantic import BaseModel

from config import settings
from llm.base import LLMResponse, ToolCall

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        # max_retries: SDK retries 429/5xx with exponential backoff.
        _client = anthropic.Anthropic(max_retries=settings.llm_max_retries)
    return _client


def generate_anthropic(
    messages: list[dict],
    *,
    system: str | list[dict] | None = None,
    tools: list[dict] | None = None,
    structured_schema: type[BaseModel] | None = None,
    max_tokens: int = 4096,
) -> LLMResponse:
    client = _get_client()
    kwargs: dict = {
        "model": settings.anthropic_model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system is not None:
        kwargs["system"] = system
    if tools:
        kwargs["tools"] = tools
    if structured_schema is not None:
        response = client.messages.parse(output_format=structured_schema, **kwargs)
    else:
        response = client.messages.create(**kwargs)
    return _to_llm_response(response, structured_schema)


def _to_llm_response(response, structured_schema=None) -> LLMResponse:
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(ToolCall(id=block.id, name=block.name, input=dict(block.input)))
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
    }
    parsed = getattr(response, "parsed_output", None) if structured_schema else None
    return LLMResponse(
        text="\n".join(text_parts),
        tool_calls=tool_calls,
        parsed=parsed,
        stop_reason=response.stop_reason,
        usage=usage,
    )
