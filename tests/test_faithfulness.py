from llm.base import LLMResponse

CONTEXTS = [{"paper_id": "1706.03762", "title": "Attention", "text": "self-attention"}]


def _patch_generate(monkeypatch, text=None, exc=None):
    import rag.faithfulness as faith_mod

    calls = []

    def fake_generate(messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        if exc is not None:
            raise exc
        return LLMResponse(text=text)

    monkeypatch.setattr(faith_mod, "generate", fake_generate)
    return calls


def test_yes_means_supported(monkeypatch):
    from rag.faithfulness import check_faithfulness

    _patch_generate(monkeypatch, text="Yes")
    assert check_faithfulness("q", "a [1706.03762]", CONTEXTS) is True


def test_no_means_unsupported_even_with_trailing_prose(monkeypatch):
    from rag.faithfulness import check_faithfulness

    _patch_generate(monkeypatch, text="no. The answer cites a paper not in the excerpts.")
    assert check_faithfulness("q", "a [9999.00001]", CONTEXTS) is False


def test_garbage_verdict_is_none(monkeypatch):
    from rag.faithfulness import check_faithfulness

    _patch_generate(monkeypatch, text="Maybe? Hard to say.")
    assert check_faithfulness("q", "a", CONTEXTS) is None


def test_llm_error_is_none(monkeypatch):
    from rag.faithfulness import check_faithfulness

    _patch_generate(monkeypatch, exc=RuntimeError("api down"))
    assert check_faithfulness("q", "a", CONTEXTS) is None


def test_prompt_carries_excerpts_question_answer_and_provider(monkeypatch):
    from rag.faithfulness import check_faithfulness

    calls = _patch_generate(monkeypatch, text="yes")
    check_faithfulness("what is attention?", "It is attention [1706.03762].",
                       CONTEXTS, provider="local")
    prompt = calls[0]["messages"][0]["content"]
    assert "self-attention" in prompt              # excerpt text
    assert "what is attention?" in prompt          # question
    assert "It is attention [1706.03762]." in prompt  # answer under test
    assert calls[0]["provider"] == "local"
