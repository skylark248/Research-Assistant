from types import SimpleNamespace

import pytest


def _fake_response(blocks, stop="end_turn"):
    return SimpleNamespace(
        content=blocks,
        stop_reason=stop,
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )


def test_generate_text(monkeypatch):
    import llm.anthropic_client as ac
    from llm.base import generate

    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _fake_response([SimpleNamespace(type="text", text="hello")])

    monkeypatch.setattr(ac, "_get_client", lambda: SimpleNamespace(messages=FakeMessages()))

    resp = generate([{"role": "user", "content": "hi"}], provider="anthropic")

    assert resp.text == "hello"
    assert resp.tool_calls == []
    assert resp.stop_reason == "end_turn"
    assert resp.usage["input_tokens"] == 10
    assert captured["model"] == "claude-opus-4-8"
    assert captured["max_tokens"] > 0
    assert "system" not in captured  # omitted when None


def test_system_passed_through(monkeypatch):
    import llm.anthropic_client as ac
    from llm.base import generate

    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _fake_response([SimpleNamespace(type="text", text="ok")])

    monkeypatch.setattr(ac, "_get_client", lambda: SimpleNamespace(messages=FakeMessages()))

    generate([{"role": "user", "content": "hi"}], system="be terse", provider="anthropic")
    assert captured["system"] == "be terse"


def test_unknown_provider_raises():
    from llm.base import generate

    with pytest.raises(ValueError, match="Unknown provider"):
        generate([{"role": "user", "content": "hi"}], provider="grok")
