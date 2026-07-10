from llm.base import LLMResponse


def test_retry_rewrite_returns_alternative_query(monkeypatch):
    import rag.rewrite as rewrite_mod
    from llm.base import LLMResponse
    from rag.rewrite import RewrittenQuery

    def fake_generate(messages, **kwargs):
        assert kwargs["system"] == rewrite_mod.RETRY_REWRITE_SYSTEM_PROMPT
        assert kwargs["provider"] == "local"
        return LLMResponse(parsed=RewrittenQuery(query="transformer self-attention architecture"))

    monkeypatch.setattr(rewrite_mod, "generate", fake_generate)
    result = rewrite_mod.retry_rewrite_query("how do transformers work?", provider="local")
    assert result == "transformer self-attention architecture"


def test_retry_rewrite_fails_open_to_original(monkeypatch):
    import rag.rewrite as rewrite_mod

    monkeypatch.setattr(rewrite_mod, "generate",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    assert rewrite_mod.retry_rewrite_query("original q") == "original q"


def test_retry_rewrite_empty_result_falls_back(monkeypatch):
    import rag.rewrite as rewrite_mod
    from llm.base import LLMResponse
    from rag.rewrite import RewrittenQuery

    monkeypatch.setattr(rewrite_mod, "generate",
                        lambda *a, **k: LLMResponse(parsed=RewrittenQuery(query="  ")))
    assert rewrite_mod.retry_rewrite_query("original q") == "original q"


def test_rewrite_returns_parsed_query(monkeypatch):
    import rag.rewrite as rewrite_mod

    def fake_generate(messages, **kwargs):
        assert kwargs["structured_schema"] is rewrite_mod.RewrittenQuery
        assert messages[-1]["content"] == "what's that attention thing?"
        return LLMResponse(parsed=rewrite_mod.RewrittenQuery(query="transformer self-attention mechanism"))

    monkeypatch.setattr(rewrite_mod, "generate", fake_generate)
    assert rewrite_mod.rewrite_query("what's that attention thing?") == \
        "transformer self-attention mechanism"


def test_rewrite_falls_back_on_llm_error(monkeypatch):
    import rag.rewrite as rewrite_mod

    def boom(*a, **k):
        raise RuntimeError("no api key")

    monkeypatch.setattr(rewrite_mod, "generate", boom)
    assert rewrite_mod.rewrite_query("original question") == "original question"


def test_rewrite_falls_back_on_empty_result(monkeypatch):
    import rag.rewrite as rewrite_mod

    monkeypatch.setattr(rewrite_mod, "generate",
                        lambda *a, **k: LLMResponse(parsed=rewrite_mod.RewrittenQuery(query="  ")))
    assert rewrite_mod.rewrite_query("original question") == "original question"
