from llm.base import LLMResponse
from rag.store import ScoredChunk


def _chunk(pid="1706.03762", title="Attention", text="self-attention", score=0.9):
    return ScoredChunk(paper_id=pid, title=title, text=text, score=score)


def _patch_generate(monkeypatch, text=None, exc=None):
    import rag.grade as grade_mod

    calls = []

    def fake_generate(messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        if exc is not None:
            raise exc
        return LLMResponse(text=text)

    monkeypatch.setattr(grade_mod, "generate", fake_generate)
    return calls


def test_grade_keeps_relevant_chunks_in_order(monkeypatch):
    from rag.grade import grade_chunks

    calls = _patch_generate(monkeypatch, text="1: yes\n2: no\n3: yes")
    chunks = [_chunk(pid="p1"), _chunk(pid="p2"), _chunk(pid="p3")]
    kept = grade_chunks("q", chunks)
    assert [c.paper_id for c in kept] == ["p1", "p3"]
    assert len(calls) == 1  # one batched call, not one per chunk


def test_grade_prompt_carries_question_and_all_chunks(monkeypatch):
    from rag.grade import grade_chunks

    calls = _patch_generate(monkeypatch, text="1: yes\n2: yes")
    grade_chunks("what is attention?",
                 [_chunk(pid="p1", text="alpha"), _chunk(pid="p2", text="beta")])
    prompt = calls[0]["messages"][0]["content"]
    assert "alpha" in prompt and "beta" in prompt
    assert "what is attention?" in prompt


def test_grade_threads_provider(monkeypatch):
    from rag.grade import grade_chunks

    calls = _patch_generate(monkeypatch, text="1: yes")
    grade_chunks("q", [_chunk()], provider="local")
    assert calls[0]["provider"] == "local"


def test_grade_missing_verdict_fails_open_per_chunk(monkeypatch):
    from rag.grade import grade_chunks

    _patch_generate(monkeypatch, text="1: no")  # chunk 2 never mentioned
    kept = grade_chunks("q", [_chunk(pid="p1"), _chunk(pid="p2")])
    assert [c.paper_id for c in kept] == ["p2"]  # unmentioned chunk passes


def test_grade_garbage_output_returns_all_chunks(monkeypatch):
    from rag.grade import grade_chunks

    _patch_generate(monkeypatch, text="I think they all look great!")
    chunks = [_chunk(pid="p1"), _chunk(pid="p2")]
    assert grade_chunks("q", chunks) == chunks


def test_grade_llm_error_returns_all_chunks(monkeypatch):
    from rag.grade import grade_chunks

    _patch_generate(monkeypatch, exc=RuntimeError("api down"))
    chunks = [_chunk(pid="p1")]
    assert grade_chunks("q", chunks) == chunks


def test_grade_empty_input_makes_no_llm_call(monkeypatch):
    from rag.grade import grade_chunks

    calls = _patch_generate(monkeypatch, text="unused")
    assert grade_chunks("q", []) == []
    assert calls == []


def test_grade_tolerates_format_variants(monkeypatch):
    from rag.grade import grade_chunks

    _patch_generate(monkeypatch, text="1. YES\n 2) no\n3 - yes")
    kept = grade_chunks("q", [_chunk(pid="p1"), _chunk(pid="p2"), _chunk(pid="p3")])
    assert [c.paper_id for c in kept] == ["p1", "p3"]
