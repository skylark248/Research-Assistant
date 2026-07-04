from types import SimpleNamespace

from pydantic import BaseModel


def _fake_response(blocks, stop="end_turn", parsed=None):
    resp = SimpleNamespace(
        content=blocks,
        stop_reason=stop,
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )
    if parsed is not None:
        resp.parsed_output = parsed
    return resp


def test_anthropic_tool_call(monkeypatch):
    import llm.anthropic_client as ac
    from llm.base import generate

    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _fake_response(
                [
                    SimpleNamespace(type="text", text="Looking that up."),
                    SimpleNamespace(type="tool_use", id="tu_1", name="rag_query",
                                    input={"question": "what is attention?"}),
                ],
                stop="tool_use",
            )

    monkeypatch.setattr(ac, "_get_client", lambda: SimpleNamespace(messages=FakeMessages()))

    tools = [{"name": "rag_query", "description": "d",
              "input_schema": {"type": "object", "properties": {"question": {"type": "string"}},
                               "required": ["question"]}}]
    resp = generate([{"role": "user", "content": "hi"}], tools=tools, provider="anthropic")

    assert captured["tools"] == tools
    assert resp.stop_reason == "tool_use"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "rag_query"
    assert resp.tool_calls[0].input == {"question": "what is attention?"}


def test_anthropic_structured_output(monkeypatch):
    import llm.anthropic_client as ac
    from llm.base import generate

    class Scores(BaseModel):
        faithfulness: int

    captured = {}

    class FakeMessages:
        def parse(self, **kwargs):
            captured.update(kwargs)
            return _fake_response(
                [SimpleNamespace(type="text", text='{"faithfulness": 5}')],
                parsed=Scores(faithfulness=5),
            )

    monkeypatch.setattr(ac, "_get_client", lambda: SimpleNamespace(messages=FakeMessages()))

    resp = generate([{"role": "user", "content": "judge"}],
                    structured_schema=Scores, provider="anthropic")

    assert captured["output_format"] is Scores
    assert resp.parsed == Scores(faithfulness=5)
