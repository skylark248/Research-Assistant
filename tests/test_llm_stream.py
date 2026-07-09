from types import SimpleNamespace

from llm.base import LLMResponse


def _chunk(content=None, tool_calls=None, finish_reason=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta,
                                                    finish_reason=finish_reason)])


def _tc_delta(index, id=None, name=None, arguments=None):
    fn = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=id, function=fn)


class FakeOpenAIClient:
    def __init__(self, chunks):
        completions = SimpleNamespace(create=self._create)
        self.chat = SimpleNamespace(completions=completions)
        self._chunks = chunks
        self.kwargs = None

    def _create(self, **kwargs):
        self.kwargs = kwargs
        return iter(self._chunks)


def test_openai_stream_accumulates_text_and_calls_on_delta():
    from llm.openai_client import generate_openai_stream

    client = FakeOpenAIClient([
        _chunk(content="Hel"),
        _chunk(content="lo"),
        _chunk(finish_reason="stop"),
    ])
    deltas = []
    resp = generate_openai_stream([{"role": "user", "content": "hi"}],
                                  on_delta=deltas.append, client=client, model="m")
    assert deltas == ["Hel", "lo"]
    assert resp.text == "Hello"
    assert resp.stop_reason == "stop"
    assert client.kwargs["stream"] is True


def test_openai_stream_accumulates_tool_calls():
    from llm.openai_client import generate_openai_stream

    client = FakeOpenAIClient([
        _chunk(tool_calls=[_tc_delta(0, id="tc_1", name="rag_query", arguments='{"que')]),
        _chunk(tool_calls=[_tc_delta(0, arguments='stion": "x"}')]),
        _chunk(finish_reason="tool_calls"),
    ])
    resp = generate_openai_stream([{"role": "user", "content": "hi"}],
                                  on_delta=lambda t: None, client=client, model="m")
    assert resp.text == ""
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].id == "tc_1"
    assert resp.tool_calls[0].name == "rag_query"
    assert resp.tool_calls[0].input == {"question": "x"}


def test_anthropic_stream(monkeypatch):
    import llm.anthropic_client as ac

    final_message = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="Hello")],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=1, output_tokens=2),
    )

    class FakeStream:
        text_stream = iter(["Hel", "lo"])

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get_final_message(self):
            return final_message

    class FakeClient:
        messages = SimpleNamespace(stream=lambda **kwargs: FakeStream())

    monkeypatch.setattr(ac, "_get_client", lambda: FakeClient())
    deltas = []
    resp = ac.generate_anthropic_stream([{"role": "user", "content": "hi"}],
                                        on_delta=deltas.append)
    assert deltas == ["Hel", "lo"]
    assert resp.text == "Hello"
    assert resp.stop_reason == "end_turn"


def test_base_dispatch(monkeypatch):
    import llm.anthropic_client as ac
    import llm.base as base

    def fake_stream(messages, **kwargs):
        kwargs["on_delta"]("x")
        return LLMResponse(text="x")

    monkeypatch.setattr(ac, "generate_anthropic_stream", fake_stream)
    deltas = []
    resp = base.generate_stream([{"role": "user", "content": "hi"}],
                                on_delta=deltas.append, provider="anthropic")
    assert resp.text == "x" and deltas == ["x"]


def test_base_dispatch_unknown_provider():
    import pytest

    import llm.base as base

    with pytest.raises(ValueError, match="Unknown provider"):
        base.generate_stream([{"role": "user", "content": "hi"}],
                             on_delta=lambda t: None, provider="gemini")
