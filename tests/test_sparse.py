from types import SimpleNamespace

import numpy as np


class FakeBM25:
    """Mimics fastembed.SparseTextEmbedding: embed() is TF-weighted, query_embed() is binary."""

    def embed(self, texts):
        for _ in texts:
            yield SimpleNamespace(indices=np.array([3, 7]), values=np.array([2.0, 1.0]))

    def query_embed(self, text):
        yield SimpleNamespace(indices=np.array([3]), values=np.array([1.0]))


def _patch_model(monkeypatch):
    import rag.sparse as sparse_mod

    monkeypatch.setattr(sparse_mod, "_model", FakeBM25())


def test_sparse_embed_texts_converts_arrays(monkeypatch):
    from rag.sparse import sparse_embed_texts

    _patch_model(monkeypatch)
    vecs = sparse_embed_texts(["chunk a", "chunk b"])
    assert len(vecs) == 2
    assert vecs[0].indices == [3, 7]
    assert vecs[0].values == [2.0, 1.0]


def test_sparse_embed_texts_empty():
    from rag.sparse import sparse_embed_texts

    assert sparse_embed_texts([]) == []


def test_sparse_embed_query_uses_query_embed(monkeypatch):
    from rag.sparse import sparse_embed_query

    _patch_model(monkeypatch)
    vec = sparse_embed_query("what is attention?")
    assert vec.indices == [3]
    assert vec.values == [1.0]
