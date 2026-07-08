from types import SimpleNamespace

import pytest

from rag.sparse import SparseVector


class FakeQdrant:
    def __init__(self, exists=False, hits=None, fail=False, legacy=False, dense_size=1536):
        self._exists = exists
        self._hits = hits or []
        self._fail = fail
        self._legacy = legacy
        self._dense_size = dense_size
        self.created = []
        self.upserted = []
        self.deleted = []
        self.queries = []

    def get_collections(self):
        if self._fail:
            raise ConnectionError("refused")
        return []

    def collection_exists(self, name):
        return self._exists

    def get_collection(self, name):
        if self._legacy:
            # phase-1 schema: single unnamed dense vector, no sparse vectors
            params = SimpleNamespace(vectors=SimpleNamespace(size=1536), sparse_vectors=None)
        else:
            params = SimpleNamespace(vectors={"dense": SimpleNamespace(size=self._dense_size)},
                                     sparse_vectors={"bm25": SimpleNamespace()})
        return SimpleNamespace(config=SimpleNamespace(params=params))

    def create_collection(self, collection_name, vectors_config, sparse_vectors_config):
        self.created.append((collection_name, vectors_config, sparse_vectors_config))

    def delete_collection(self, collection_name):
        self.deleted.append(collection_name)

    def upsert(self, collection_name, points):
        self.upserted.append((collection_name, points))

    def query_points(self, collection_name, query=None, prefetch=None, using=None, limit=None):
        self.queries.append({"query": query, "prefetch": prefetch,
                             "using": using, "limit": limit})
        return SimpleNamespace(points=self._hits[:limit])

    def scroll(self, collection_name, scroll_filter, limit):
        return (self._hits[:limit], None)


def _store(fake):
    from rag.store import VectorStore

    return VectorStore(client=fake)


def _record():
    from rag.store import ChunkRecord

    return ChunkRecord(paper_id="1706.03762", title="Attention", chunk_index=0,
                       text="chunk", vector=[0.1, 0.2],
                       sparse=SparseVector(indices=[3], values=[2.0]))


def test_ping_fail_fast():
    store = _store(FakeQdrant(fail=True))
    with pytest.raises(RuntimeError, match="docker compose up"):
        store.ping()


def test_ping_ok():
    _store(FakeQdrant()).ping()  # no raise


def test_ensure_collection_creates_named_vectors(monkeypatch):
    from config import settings
    from rag.store import DENSE_VECTOR, SPARSE_VECTOR

    monkeypatch.setattr(settings, "embedding_dim", 1536)
    fake = FakeQdrant(exists=False)
    _store(fake).ensure_collection()
    assert len(fake.created) == 1
    _, vectors_config, sparse_config = fake.created[0]
    assert DENSE_VECTOR in vectors_config
    assert SPARSE_VECTOR in sparse_config

    fake2 = FakeQdrant(exists=True)
    _store(fake2).ensure_collection()
    assert fake2.created == []


def test_ensure_collection_rejects_legacy_schema():
    fake = FakeQdrant(exists=True, legacy=True)
    with pytest.raises(RuntimeError, match="rag.migrate"):
        _store(fake).ensure_collection()


def test_check_schema_passes_when_collection_missing():
    _store(FakeQdrant(exists=False)).check_schema()  # no raise


def test_check_schema_rejects_legacy():
    fake = FakeQdrant(exists=True, legacy=True)
    with pytest.raises(RuntimeError, match="rag.migrate"):
        _store(fake).check_schema()


def test_upsert_builds_deterministic_named_points():
    fake = FakeQdrant()
    store = _store(fake)
    rec = _record()
    store.upsert_chunks([rec])
    store.upsert_chunks([rec])

    (_, points1), (_, points2) = fake.upserted
    assert points1[0].id == points2[0].id  # uuid5 → idempotent re-ingest
    assert points1[0].payload == {"paper_id": "1706.03762", "title": "Attention",
                                  "chunk_index": 0, "chunk_text": "chunk", "section": ""}
    assert points1[0].vector["dense"] == [0.1, 0.2]
    assert points1[0].vector["bm25"].indices == [3]
    assert points1[0].vector["bm25"].values == [2.0]


def _hit():
    return SimpleNamespace(score=0.87, payload={"paper_id": "1706.03762", "title": "Attention",
                                                "chunk_index": 0, "chunk_text": "self-attention",
                                                "section": ""})


def test_search_dense_only():
    fake = FakeQdrant(hits=[_hit()])
    results = _store(fake).search(dense=[0.1, 0.2], top_k=3)
    assert len(results) == 1
    assert results[0].text == "self-attention"
    assert fake.queries[0]["using"] == "dense"
    assert fake.queries[0]["prefetch"] is None


def test_search_sparse_only():
    fake = FakeQdrant(hits=[_hit()])
    results = _store(fake).search(sparse=SparseVector(indices=[3], values=[1.0]), top_k=3)
    assert len(results) == 1
    assert fake.queries[0]["using"] == "bm25"
    assert fake.queries[0]["query"].indices == [3]


def test_search_hybrid_builds_rrf_fusion():
    from qdrant_client import models

    fake = FakeQdrant(hits=[_hit()])
    results = _store(fake).search(dense=[0.1, 0.2],
                                  sparse=SparseVector(indices=[3], values=[1.0]), top_k=5)
    assert len(results) == 1
    q = fake.queries[0]
    assert isinstance(q["query"], models.FusionQuery)
    assert q["query"].fusion == models.Fusion.RRF
    assert len(q["prefetch"]) == 2
    assert {p.using for p in q["prefetch"]} == {"dense", "bm25"}
    # each side over-fetches so fusion has candidates to merge
    assert all(p.limit == 10 for p in q["prefetch"])
    assert q["limit"] == 5


def test_search_requires_a_vector():
    with pytest.raises(ValueError, match="dense and/or sparse"):
        _store(FakeQdrant()).search(top_k=3)


def test_has_paper():
    hit = SimpleNamespace(score=1.0, payload={})
    assert _store(FakeQdrant(hits=[hit])).has_paper("1706.03762") is True
    assert _store(FakeQdrant(hits=[])).has_paper("1706.03762") is False


def test_check_schema_rejects_dimension_mismatch(monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "embedding_dim", 384)  # e.g. switched to local embeddings
    fake = FakeQdrant(exists=True, dense_size=1536)      # collection built with openai dims
    with pytest.raises(RuntimeError, match="1536.*384|384.*1536"):
        _store(fake).check_schema()


def test_check_schema_accepts_matching_dimension(monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "embedding_dim", 384)
    fake = FakeQdrant(exists=True, dense_size=384)
    _store(fake).check_schema()  # no raise
