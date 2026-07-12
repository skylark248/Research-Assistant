from types import SimpleNamespace

import pytest

from config import settings
from rag.sparse import SparseVector


@pytest.fixture
def guardrails_off(monkeypatch):
    """Phase-5 guardrails default on; these tests exercise the pre-existing path."""
    monkeypatch.setattr(settings, "grading_enabled", False)
    monkeypatch.setattr(settings, "faithfulness_enabled", False)


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


def test_answer_question_builds_grounded_prompt(monkeypatch, guardrails_off):
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


def test_answer_question_threads_provider_to_generate(monkeypatch, guardrails_off):
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


def test_answer_question_empty_store(monkeypatch, guardrails_off):
    import rag.answer as answer_mod

    monkeypatch.setattr(answer_mod, "retrieve", lambda q, store=None: [])
    monkeypatch.setattr(answer_mod, "generate",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no LLM call")))

    result = answer_mod.answer_question("anything")
    assert result.sources == []
    assert "ingest" in result.text.lower()


def _fake_llm(text):
    from llm.base import LLMResponse

    return LLMResponse(text=text, usage={"cache_read_input_tokens": 0})


def test_grading_filters_chunks_before_prompt(monkeypatch):
    import rag.answer as answer_mod

    monkeypatch.setattr(settings, "grading_enabled", True)
    monkeypatch.setattr(settings, "faithfulness_enabled", False)
    monkeypatch.setattr(answer_mod, "retrieve", lambda q, store=None: [
        _chunk(), _chunk(pid="1810.04805", title="BERT", text="bert stuff")])
    monkeypatch.setattr(answer_mod, "grade_chunks",
                        lambda q, chunks, provider=None: chunks[:1])
    captured = {}

    def fake_generate(messages, **kwargs):
        captured.update(kwargs)
        return _fake_llm("A [1706.03762].")

    monkeypatch.setattr(answer_mod, "generate", fake_generate)
    result = answer_mod.answer_question("q")
    assert result.sources == ["1706.03762"]              # BERT graded out
    assert "bert stuff" not in captured["system"][1]["text"]


def test_retry_fires_once_on_zero_survivors(monkeypatch):
    import rag.answer as answer_mod

    monkeypatch.setattr(settings, "grading_enabled", True)
    monkeypatch.setattr(settings, "faithfulness_enabled", False)
    retrievals = []
    monkeypatch.setattr(answer_mod, "retrieve",
                        lambda q, store=None, rewrite=None: retrievals.append(q) or [_chunk()])
    grades = iter([[], [_chunk()]])  # first grade: nothing; retry grade: survivor
    monkeypatch.setattr(answer_mod, "grade_chunks",
                        lambda q, chunks, provider=None: next(grades))
    monkeypatch.setattr(answer_mod, "retry_rewrite_query",
                        lambda q, provider=None: "alternative query")
    monkeypatch.setattr(answer_mod, "generate",
                        lambda *a, **k: _fake_llm("A [1706.03762]."))
    result = answer_mod.answer_question("original q")
    assert retrievals == ["original q", "alternative query"]
    assert result.sources == ["1706.03762"]


def test_honest_degradation_skips_generate(monkeypatch):
    import rag.answer as answer_mod

    monkeypatch.setattr(settings, "grading_enabled", True)
    monkeypatch.setattr(settings, "faithfulness_enabled", False)
    monkeypatch.setattr(answer_mod, "retrieve", lambda q, store=None, rewrite=None: [_chunk()])
    monkeypatch.setattr(answer_mod, "grade_chunks",
                        lambda q, chunks, provider=None: [])
    monkeypatch.setattr(answer_mod, "retry_rewrite_query",
                        lambda q, provider=None: "alt")
    monkeypatch.setattr(answer_mod, "generate",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no LLM call")))
    result = answer_mod.answer_question("q")
    assert result.sources == []
    assert result.faithful is None
    assert "ingest" in result.text.lower()


def test_retry_skipped_when_rewrite_fails_open(monkeypatch):
    import rag.answer as answer_mod

    monkeypatch.setattr(settings, "grading_enabled", True)
    monkeypatch.setattr(settings, "faithfulness_enabled", False)
    retrievals = []
    monkeypatch.setattr(answer_mod, "retrieve",
                        lambda q, store=None: retrievals.append(q) or [_chunk()])
    monkeypatch.setattr(answer_mod, "grade_chunks",
                        lambda q, chunks, provider=None: [])
    monkeypatch.setattr(answer_mod, "retry_rewrite_query",
                        lambda q, provider=None: q)  # failed open → identical query
    result = answer_mod.answer_question("q")
    assert retrievals == ["q"]  # no pointless second retrieval
    assert result.sources == []


def test_grading_disabled_never_calls_grader(monkeypatch):
    import rag.answer as answer_mod

    monkeypatch.setattr(settings, "grading_enabled", False)
    monkeypatch.setattr(settings, "faithfulness_enabled", False)
    monkeypatch.setattr(answer_mod, "retrieve", lambda q, store=None: [_chunk()])
    monkeypatch.setattr(answer_mod, "grade_chunks",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("grader must not run")))
    monkeypatch.setattr(answer_mod, "generate",
                        lambda *a, **k: _fake_llm("A [1706.03762]."))
    result = answer_mod.answer_question("q")
    assert result.sources == ["1706.03762"]


def test_faithfulness_verdict_attached(monkeypatch):
    import rag.answer as answer_mod

    monkeypatch.setattr(settings, "grading_enabled", False)
    monkeypatch.setattr(settings, "faithfulness_enabled", True)
    monkeypatch.setattr(answer_mod, "retrieve", lambda q, store=None: [_chunk()])
    monkeypatch.setattr(answer_mod, "generate",
                        lambda *a, **k: _fake_llm("A [1706.03762]."))
    captured = {}

    def fake_check(question, answer, contexts, provider=None):
        captured.update(question=question, answer=answer,
                        n=len(contexts), provider=provider)
        return False

    monkeypatch.setattr(answer_mod, "check_faithfulness", fake_check)
    result = answer_mod.answer_question("q", provider="local")
    assert result.faithful is False
    assert captured["n"] == 1
    assert captured["provider"] == "local"


def test_on_status_receives_pipeline_events(monkeypatch):
    import rag.answer as answer_mod

    monkeypatch.setattr(settings, "grading_enabled", True)
    monkeypatch.setattr(settings, "faithfulness_enabled", True)
    monkeypatch.setattr(answer_mod, "retrieve",
                        lambda q, store=None: [_chunk(), _chunk(pid="p2")])
    monkeypatch.setattr(answer_mod, "grade_chunks",
                        lambda q, chunks, provider=None: chunks[:1])
    monkeypatch.setattr(answer_mod, "check_faithfulness", lambda *a, **k: True)
    monkeypatch.setattr(answer_mod, "generate", lambda *a, **k: _fake_llm("A."))
    statuses = []
    answer_mod.answer_question("q", on_status=statuses.append)
    assert statuses == ["grading 2 chunks…", "1 of 2 chunks relevant",
                        "verifying citations…"]


def test_retrieve_rewrite_param_overrides_setting(monkeypatch):
    retrieve_mod = _patch_pipeline(monkeypatch, mode="dense", rewrite_on=True)

    def boom(q):
        raise AssertionError("rewrite_query must not run when rewrite=False")

    monkeypatch.setattr(retrieve_mod, "rewrite_query", boom)
    retrieve_mod.retrieve("q", top_k=3, store=CapturingStore(), rewrite=False)


def test_retry_retrieval_never_rewrites_again(monkeypatch):
    import rag.answer as answer_mod

    monkeypatch.setattr(settings, "grading_enabled", True)
    monkeypatch.setattr(settings, "faithfulness_enabled", False)
    monkeypatch.setattr(settings, "rewrite_enabled", True)
    calls = []

    def fake_retrieve(q, store=None, rewrite=None):
        calls.append({"q": q, "rewrite": rewrite})
        return [_chunk()]

    monkeypatch.setattr(answer_mod, "retrieve", fake_retrieve)
    grades = iter([[], [_chunk()]])
    monkeypatch.setattr(answer_mod, "grade_chunks",
                        lambda q, chunks, provider=None: next(grades))
    monkeypatch.setattr(answer_mod, "retry_rewrite_query",
                        lambda q, provider=None: "alt query")
    monkeypatch.setattr(answer_mod, "generate",
                        lambda *a, **k: _fake_llm("A [1706.03762]."))

    answer_mod.answer_question("orig")
    assert calls[0]["rewrite"] is None            # first retrieval: global setting
    assert calls[1] == {"q": "alt query", "rewrite": False}  # retry: never re-rewritten
