from types import SimpleNamespace

import pytest


class FakeQdrant:
    def __init__(self, exists=False, hits=None, fail=False):
        self._exists = exists
        self._hits = hits or []
        self._fail = fail
        self.created = []
        self.upserted = []

    def get_collections(self):
        if self._fail:
            raise ConnectionError("refused")
        return []

    def collection_exists(self, name):
        return self._exists

    def create_collection(self, collection_name, vectors_config):
        self.created.append((collection_name, vectors_config))

    def upsert(self, collection_name, points):
        self.upserted.append((collection_name, points))

    def query_points(self, collection_name, query, limit):
        return SimpleNamespace(points=self._hits[:limit])

    def scroll(self, collection_name, scroll_filter, limit):
        return (self._hits[:limit], None)


def _store(fake):
    from rag.store import VectorStore

    return VectorStore(client=fake)


def test_ping_fail_fast():
    store = _store(FakeQdrant(fail=True))
    with pytest.raises(RuntimeError, match="docker compose up"):
        store.ping()


def test_ping_ok():
    _store(FakeQdrant()).ping()  # no raise


def test_ensure_collection_creates_once():
    fake = FakeQdrant(exists=False)
    _store(fake).ensure_collection()
    assert len(fake.created) == 1

    fake2 = FakeQdrant(exists=True)
    _store(fake2).ensure_collection()
    assert fake2.created == []


def test_upsert_builds_deterministic_points():
    from rag.store import ChunkRecord

    fake = FakeQdrant()
    store = _store(fake)
    rec = ChunkRecord(paper_id="1706.03762", title="Attention", chunk_index=0,
                      text="chunk", vector=[0.1, 0.2])
    store.upsert_chunks([rec])
    store.upsert_chunks([rec])

    (_, points1), (_, points2) = fake.upserted
    assert points1[0].id == points2[0].id  # uuid5 → idempotent re-ingest
    assert points1[0].payload == {"paper_id": "1706.03762", "title": "Attention",
                                  "chunk_index": 0, "chunk_text": "chunk", "section": ""}


def test_search_maps_hits():
    hit = SimpleNamespace(score=0.87, payload={"paper_id": "1706.03762", "title": "Attention",
                                               "chunk_index": 0, "chunk_text": "self-attention",
                                               "section": ""})
    store = _store(FakeQdrant(hits=[hit]))
    results = store.search([0.1, 0.2], top_k=3)
    assert len(results) == 1
    assert results[0].paper_id == "1706.03762"
    assert results[0].text == "self-attention"
    assert results[0].score == 0.87


def test_has_paper():
    hit = SimpleNamespace(score=1.0, payload={})
    assert _store(FakeQdrant(hits=[hit])).has_paper("1706.03762") is True
    assert _store(FakeQdrant(hits=[])).has_paper("1706.03762") is False
