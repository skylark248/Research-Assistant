def _chunk(pid, text, score=0.5):
    from rag.store import ScoredChunk

    return ScoredChunk(paper_id=pid, title="T", text=text, score=score)


class FakeCrossEncoder:
    """Scores by document length — deterministic and obviously different from input order."""

    def rerank(self, query, documents):
        return [float(len(d)) for d in documents]


def _patch_model(monkeypatch):
    import rag.rerank as rerank_mod

    monkeypatch.setattr(rerank_mod, "_model", FakeCrossEncoder())


def test_rerank_reorders_and_truncates(monkeypatch):
    from rag.rerank import rerank

    _patch_model(monkeypatch)
    chunks = [_chunk("a", "short"), _chunk("b", "the longest text here"), _chunk("c", "medium txt")]
    result = rerank("q", chunks, top_k=2)
    assert [c.paper_id for c in result] == ["b", "c"]
    assert result[0].score == float(len("the longest text here"))


def test_rerank_empty_list(monkeypatch):
    from rag.rerank import rerank

    _patch_model(monkeypatch)
    assert rerank("q", [], top_k=5) == []
