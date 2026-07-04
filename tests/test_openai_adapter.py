import json
from types import SimpleNamespace

from pydantic import BaseModel


def _fake_completion(message, finish="stop"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish)],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


def test_convert_messages_full_tool_round_trip():
    from llm.openai_client import convert_messages

    anthropic_msgs = [
        {"role": "user", "content": "find the paper"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Searching."},
            {"type": "tool_use", "id": "tu_1", "name": "arxiv_search",
             "input": {"query": "attention"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu_1",
             "content": "found 1706.03762", "is_error": False},
        ]},
    ]
    out = convert_messages(anthropic_msgs, system="be terse")

    assert out[0] == {"role": "system", "content": "be terse"}
    assert out[1] == {"role": "user", "content": "find the paper"}
    assert out[2]["role"] == "assistant"
    assert out[2]["content"] == "Searching."
    assert out[2]["tool_calls"][0]["id"] == "tu_1"
    assert out[2]["tool_calls"][0]["function"]["name"] == "arxiv_search"
    assert json.loads(out[2]["tool_calls"][0]["function"]["arguments"]) == {"query": "attention"}
    assert out[3] == {"role": "tool", "tool_call_id": "tu_1", "content": "found 1706.03762"}


def test_convert_messages_joins_system_blocks_and_drops_cache_control():
    from llm.openai_client import convert_messages

    system = [
        {"type": "text", "text": "instructions"},
        {"type": "text", "text": "context", "cache_control": {"type": "ephemeral"}},
    ]
    out = convert_messages([{"role": "user", "content": "q"}], system=system)
    assert out[0] == {"role": "system", "content": "instructions\n\ncontext"}


def test_convert_tools():
    from llm.openai_client import convert_tools

    schema = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
    out = convert_tools([{"name": "rag_query", "description": "d", "input_schema": schema}])
    assert out == [{"type": "function",
                    "function": {"name": "rag_query", "description": "d", "parameters": schema}}]


def test_generate_openai_tool_call(monkeypatch):
    import llm.openai_client as oc
    from llm.base import generate

    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="rag_query", arguments='{"question": "q"}'),
    )
    message = SimpleNamespace(content=None, tool_calls=[tool_call])

    class FakeCompletions:
        def create(self, **kwargs):
            return _fake_completion(message, finish="tool_calls")

    fake = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr(oc, "_get_client", lambda: fake)

    tools = [{"name": "rag_query", "description": "d",
              "input_schema": {"type": "object", "properties": {}}}]
    resp = generate([{"role": "user", "content": "hi"}], tools=tools, provider="openai")

    assert resp.tool_calls[0].id == "call_1"
    assert resp.tool_calls[0].input == {"question": "q"}


def test_generate_openai_structured(monkeypatch):
    import llm.openai_client as oc
    from llm.base import generate

    class Scores(BaseModel):
        relevance: int

    message = SimpleNamespace(content='{"relevance": 4}', tool_calls=None,
                              parsed=Scores(relevance=4))

    class FakeCompletions:
        def parse(self, **kwargs):
            return _fake_completion(message)

    fake = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr(oc, "_get_client", lambda: fake)

    resp = generate([{"role": "user", "content": "judge"}],
                    structured_schema=Scores, provider="openai")
    assert resp.parsed == Scores(relevance=4)
