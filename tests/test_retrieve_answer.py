from types import SimpleNamespace


def _chunk(pid="1706.03762", title="Attention", text="self-attention", score=0.9):
    from rag.store import ScoredChunk

    return ScoredChunk(paper_id=pid, title=title, text=text, score=score)


def test_retrieve_embeds_and_searches(monkeypatch):
    import rag.retrieve as retrieve_mod

    monkeypatch.setattr(retrieve_mod, "embed_query", lambda q: [0.5])
    fake_store = SimpleNamespace(search=lambda vector, top_k: [_chunk()] if vector == [0.5] else [])

    chunks = retrieve_mod.retrieve("what is attention?", top_k=3, store=fake_store)
    assert len(chunks) == 1
    assert chunks[0].paper_id == "1706.03762"


def test_answer_question_builds_grounded_prompt(monkeypatch):
    import rag.answer as answer_mod
    from llm.base import LLMResponse

    monkeypatch.setattr(answer_mod, "retrieve",
                        lambda q, store=None: [_chunk(), _chunk(pid="1810.04805", title="BERT")])
    captured = {}

    def fake_generate(messages, **kwargs):
        captured["messages"] = messages
        captured.update(kwargs)
        return LLMResponse(text="Self-attention is key [1706.03762].",
                           usage={"cache_read_input_tokens": 0})

    monkeypatch.setattr(answer_mod, "generate", fake_generate)

    result = answer_mod.answer_question("what is attention?")

    assert result.text == "Self-attention is key [1706.03762]."
    assert result.sources == ["1706.03762", "1810.04805"]
    # Grounded prompt: context in system with a cache breakpoint, question last.
    assert captured["system"][1]["cache_control"] == {"type": "ephemeral"}
    assert "self-attention" in captured["system"][1]["text"]
    assert "what is attention?" in captured["messages"][-1]["content"]


def test_answer_question_empty_store(monkeypatch):
    import rag.answer as answer_mod

    monkeypatch.setattr(answer_mod, "retrieve", lambda q, store=None: [])
    monkeypatch.setattr(answer_mod, "generate",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no LLM call")))

    result = answer_mod.answer_question("anything")
    assert result.sources == []
    assert "ingest" in result.text.lower()
