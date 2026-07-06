import json

from openai import OpenAI
from pydantic import BaseModel

from config import settings
from llm.base import LLMResponse, ToolCall

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key or None, max_retries=settings.llm_max_retries)
    return _client


def _system_to_text(system: str | list[dict]) -> str:
    if isinstance(system, str):
        return system
    # Anthropic system blocks; cache_control has no OpenAI equivalent, drop it.
    return "\n\n".join(b["text"] for b in system if b.get("type") == "text")


def convert_messages(messages: list[dict], system: str | list[dict] | None = None) -> list[dict]:
    """Anthropic-shaped messages -> OpenAI chat.completions messages."""
    out: list[dict] = []
    if system is not None:
        out.append({"role": "system", "content": _system_to_text(system)})
    for msg in messages:
        content = msg["content"]
        if isinstance(content, str):
            out.append({"role": msg["role"], "content": content})
            continue
        if msg["role"] == "assistant":
            text = "".join(b["text"] for b in content if b["type"] == "text")
            tool_calls = [
                {"id": b["id"], "type": "function",
                 "function": {"name": b["name"], "arguments": json.dumps(b["input"])}}
                for b in content if b["type"] == "tool_use"
            ]
            entry: dict = {"role": "assistant", "content": text or None}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            out.append(entry)
        else:  # user message with content blocks
            for b in content:
                if b["type"] == "tool_result":
                    out.append({"role": "tool", "tool_call_id": b["tool_use_id"],
                                "content": b["content"]})
                elif b["type"] == "text":
                    out.append({"role": "user", "content": b["text"]})
    return out


def convert_tools(tools: list[dict]) -> list[dict]:
    """Anthropic tool spec -> OpenAI function-calling spec."""
    return [
        {"type": "function",
         "function": {"name": t["name"], "description": t.get("description", ""),
                      "parameters": t["input_schema"]}}
        for t in tools
    ]


def _usage(completion) -> dict:
    u = completion.usage
    return {"input_tokens": u.prompt_tokens, "output_tokens": u.completion_tokens}


def generate_openai(
    messages: list[dict],
    *,
    system: str | list[dict] | None = None,
    tools: list[dict] | None = None,
    structured_schema: type[BaseModel] | None = None,
    max_tokens: int = 4096,
) -> LLMResponse:
    client = _get_client()
    kwargs: dict = {
        "model": settings.openai_model,
        "messages": convert_messages(messages, system),
        "max_completion_tokens": max_tokens,
    }
    if tools:
        kwargs["tools"] = convert_tools(tools)
    if structured_schema is not None:
        completion = client.chat.completions.parse(response_format=structured_schema, **kwargs)
        choice = completion.choices[0]
        return LLMResponse(text=choice.message.content or "", parsed=choice.message.parsed,
                           stop_reason=choice.finish_reason, usage=_usage(completion))
    completion = client.chat.completions.create(**kwargs)
    choice = completion.choices[0]
    tool_calls = [
        ToolCall(id=tc.id, name=tc.function.name, input=json.loads(tc.function.arguments))
        for tc in (choice.message.tool_calls or [])
    ]
    return LLMResponse(text=choice.message.content or "", tool_calls=tool_calls,
                       stop_reason=choice.finish_reason, usage=_usage(completion))
