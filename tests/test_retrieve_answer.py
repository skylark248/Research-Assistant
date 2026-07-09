from types import SimpleNamespace

from config import settings
from rag.sparse import SparseVector


def _chunk(pid="1706.03762", title="Attention", text="self-attention", score=0.9):
    from rag.store import ScoredChunk

    return ScoredChunk(paper_id=pid, title=title, text=text, score=score)


class CapturingStore:
    def __init__(self, hits=None):
        self.hits = hits if hits is not None else [_chunk()]
        self.calls = []

    def search(self, dense=None, sparse=None, top_k=None):
        self.calls.append({"dense": dense, "sparse": sparse, "top_k": top_k})
        return self.hits[:top_k]


def _patch_pipeline(monkeypatch, mode="hybrid", rerank_on=False, rewrite_on=False):
    import rag.retrieve as retrieve_mod

    monkeypatch.setattr(settings, "retrieval_mode", mode)
    monkeypatch.setattr(settings, "rerank_enabled", rerank_on)
    monkeypatch.setattr(settings, "rewrite_enabled", rewrite_on)
    monkeypatch.setattr(retrieve_mod, "embed_query", lambda q: [0.5])
    monkeypatch.setattr(retrieve_mod, "sparse_embed_query",
                        lambda q: SparseVector(indices=[1], values=[1.0]))
    return retrieve_mod


def test_retrieve_dense_mode(monkeypatch):
    retrieve_mod = _patch_pipeline(monkeypatch, mode="dense")
    store = CapturingStore()
    chunks = retrieve_mod.retrieve("q", top_k=3, store=store)
    assert len(chunks) == 1
    assert store.calls[0]["dense"] == [0.5]
    assert store.calls[0]["sparse"] is None
    assert store.calls[0]["top_k"] == 3


def test_retrieve_sparse_mode(monkeypatch):
    retrieve_mod = _patch_pipeline(monkeypatch, mode="sparse")
    store = CapturingStore()
    retrieve_mod.retrieve("q", top_k=3, store=store)
    assert store.calls[0]["dense"] is None
    assert store.calls[0]["sparse"].indices == [1]


def test_retrieve_hybrid_mode(monkeypatch):
    retrieve_mod = _patch_pipeline(monkeypatch, mode="hybrid")
    store = CapturingStore()
    retrieve_mod.retrieve("q", top_k=3, store=store)
    assert store.calls[0]["dense"] == [0.5]
    assert store.calls[0]["sparse"].indices == [1]


def test_retrieve_rerank_overfetches_then_truncates(monkeypatch):
    retrieve_mod = _patch_pipeline(monkeypatch, mode="dense", rerank_on=True)
    monkeypatch.setattr(settings, "rerank_candidates", 20)
    hits = [_chunk(pid=f"p{i}") for i in range(20)]
    store = CapturingStore(hits=hits)
    captured = {}

    def fake_rerank(question, chunks, top_k):
        captured["n_in"] = len(chunks)
        captured["question"] = question
        return list(reversed(chunks))[:top_k]

    monkeypatch.setattr(retrieve_mod, "rerank", fake_rerank)
    result = retrieve_mod.retrieve("original q", top_k=5, store=store)
    assert store.calls[0]["top_k"] == 20  # over-fetch
    assert captured["n_in"] == 20
    assert len(result) == 5
    assert result[0].paper_id == "p19"  # rerank order won


def test_retrieve_rewrite_shapes_search_not_rerank(monkeypatch):
    retrieve_mod = _patch_pipeline(monkeypatch, mode="dense", rerank_on=True, rewrite_on=True)
    monkeypatch.setattr(retrieve_mod, "rewrite_query", lambda q: "rewritten query")
    embedded = []
    monkeypatch.setattr(retrieve_mod, "embed_query", lambda q: embedded.append(q) or [0.5])
    captured = {}

    def fake_rerank(question, chunks, top_k):
        captured["question"] = question
        return chunks[:top_k]

    monkeypatch.setattr(retrieve_mod, "rerank", fake_rerank)
    retrieve_mod.retrieve("original q", top_k=3, store=CapturingStore())
    assert embedded == ["rewritten query"]      # search sees the rewrite
    assert captured["question"] == "original q"  # relevance judged on the original


def test_retrieve_rewrite_off_by_default(monkeypatch):
    retrieve_mod = _patch_pipeline(monkeypatch, mode="dense")

    def boom(q):
        raise AssertionError("rewrite_query must not be called when disabled")

    monkeypatch.setattr(retrieve_mod, "rewrite_query", boom)
    retrieve_mod.retrieve("q", top_k=3, store=CapturingStore())


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
    assert captured["system"][1]["cache_control"] == {"type": "ephemeral"}
    assert "self-attention" in captured["system"][1]["text"]
    assert "what is attention?" in captured["messages"][-1]["content"]


def test_answer_question_threads_provider_to_generate(monkeypatch):
    import rag.answer as answer_mod
    from llm.base import LLMResponse

    monkeypatch.setattr(answer_mod, "retrieve", lambda q, store=None: [_chunk()])
    captured = {}

    def fake_generate(messages, **kwargs):
        captured.update(kwargs)
        return LLMResponse(text="Self-attention is key [1706.03762].",
                           usage={"cache_read_input_tokens": 0})

    monkeypatch.setattr(answer_mod, "generate", fake_generate)

    answer_mod.answer_question("what is attention?", provider="local")

    assert captured["provider"] == "local"


def test_answer_question_empty_store(monkeypatch):
    import rag.answer as answer_mod

    monkeypatch.setattr(answer_mod, "retrieve", lambda q, store=None: [])
    monkeypatch.setattr(answer_mod, "generate",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no LLM call")))

    result = answer_mod.answer_question("anything")
    assert result.sources == []
    assert "ingest" in result.text.lower()
