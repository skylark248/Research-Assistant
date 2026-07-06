# Phase 2: Retrieval Quality + Agent Depth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add hybrid search + reranking + query rewriting (config-flagged, eval-ablatable), multi-turn agent memory (SQLite checkpointing + summarization), and a multi-agent supervisor mode to the paper-research-assistant.

**Architecture:** Retrieval becomes a staged pipeline (`[rewrite] → embed → search dense|sparse|hybrid → [rerank]`) where every stage is a `config.py` flag, so `eval.run --ablation` can isolate each technique's effect. The Qdrant collection moves to named dense+sparse vectors (breaking schema — `rag/migrate.py` recreates it). The LangGraph agent gains an `AsyncSqliteSaver` checkpointer keyed by `thread_id` plus a summarize node; a separate `agents/multi.py` supervisor (planner → researcher → synthesizer) sits behind `agent_mode=multi`.

**Tech Stack:** fastembed (BM25 sparse + cross-encoder rerank, local ONNX, keyless), qdrant-client Query API (RRF fusion, IDF modifier), langgraph-checkpoint-sqlite (AsyncSqliteSaver), existing `llm/base.generate` structured output.

**Spec:** `docs/superpowers/specs/2026-07-06-phase-2-retrieval-and-agents-design.md`

## Global Constraints

- Python >=3.11, `uv` for everything: `uv sync`, `uv run pytest`, `uv add`.
- Unit tests are keyless and offline: mock fastembed models, LLM calls, and Qdrant exactly like the existing suite (monkeypatch module attributes, `FakeQdrant`-style fakes). Real APIs/models only behind `@pytest.mark.integration`.
- `messages` / `tools` use the Anthropic shape everywhere (see `llm/base.py`).
- Blocking `generate(...)` calls from async code go through `asyncio.to_thread` (see `agents/graph.py`).
- Import direction: `api → agents → rag/llm`; `eval → rag/agents/llm`. Never backwards.
- No silent fallback for reranker load failures; query-rewrite failures DO fall back to the original question with a logged warning.
- Run the full unit suite (`uv run pytest`) before every commit; all tests must pass — with one planned exception: Tasks 3–6 carry known failures in `tests/test_ingest.py` and `tests/test_retrieve_answer.py` (schema/signature change lands before the pipeline that consumes it); Task 7 restores a fully green suite. No OTHER failures are acceptable at any commit.
- Commit after every task with the repo's style: `type: lowercase summary` (`feat:`, `fix:`, `docs:`, `test:`).

---

### Task 1: Dependencies + config flags

**Files:**
- Modify: `pyproject.toml`
- Modify: `config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: settings used by all later tasks — `retrieval_mode` ("dense"|"sparse"|"hybrid", default "hybrid"), `rerank_enabled` (True), `rerank_candidates` (20), `rerank_model` ("Xenova/ms-marco-MiniLM-L-6-v2"), `sparse_model` ("Qdrant/bm25"), `rewrite_enabled` (False), `agent_mode` ("single"|"multi", default "single"), `checkpoint_db` ("data/checkpoints.db"), `memory_max_messages` (20), `memory_keep_messages` (8).

- [ ] **Step 1: Add dependencies**

```bash
uv add "fastembed>=0.6" "langgraph-checkpoint-sqlite>=2.0"
```

Expected: `uv.lock` updated, both packages resolve. (`langgraph-checkpoint-sqlite` pulls `aiosqlite`.)

- [ ] **Step 2: Write the failing test**

Append to `tests/test_config.py`:

```python
def test_phase2_defaults():
    from config import Settings

    s = Settings(_env_file=None)
    assert s.retrieval_mode == "hybrid"
    assert s.rerank_enabled is True
    assert s.rerank_candidates == 20
    assert s.rerank_model == "Xenova/ms-marco-MiniLM-L-6-v2"
    assert s.sparse_model == "Qdrant/bm25"
    assert s.rewrite_enabled is False
    assert s.agent_mode == "single"
    assert s.checkpoint_db == "data/checkpoints.db"
    assert s.memory_max_messages == 20
    assert s.memory_keep_messages == 8
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::test_phase2_defaults -v`
Expected: FAIL with `AttributeError: 'Settings' object has no attribute 'retrieval_mode'` (or pydantic validation error).

- [ ] **Step 4: Add the settings**

In `config.py`, replace the `# RAG` block and add new blocks so the class body contains:

```python
    # RAG
    chunk_max_tokens: int = 500
    chunk_overlap_tokens: int = 50
    retrieval_top_k: int = 5
    pdf_dir: str = "data/pdfs"

    # Retrieval pipeline (phase 2) — every stage is a flag so eval ablation
    # can isolate each technique's effect.
    retrieval_mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    rerank_enabled: bool = True
    rerank_candidates: int = 20  # over-fetch size fed to the reranker
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    sparse_model: str = "Qdrant/bm25"
    rewrite_enabled: bool = False  # needs an LLM key; off until keys exist

    # Agent
    agent_max_steps: int = 8
    agent_mode: Literal["single", "multi"] = "single"

    # Memory (phase 2)
    checkpoint_db: str = "data/checkpoints.db"
    memory_max_messages: int = 20  # summarize when history exceeds this
    memory_keep_messages: int = 8  # recent messages kept verbatim in the prompt
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (all).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock config.py tests/test_config.py
git commit -m "feat: add phase-2 deps and retrieval/agent config flags"
```

---

### Task 2: BM25 sparse embeddings (`rag/sparse.py`)

**Files:**
- Create: `rag/sparse.py`
- Test: `tests/test_sparse.py`

**Interfaces:**
- Consumes: `settings.sparse_model` (Task 1).
- Produces: `SparseVector(indices: list[int], values: list[float])` pydantic model; `sparse_embed_texts(texts: list[str]) -> list[SparseVector]`; `sparse_embed_query(text: str) -> SparseVector`. Tasks 3 and 7 import these.

- [ ] **Step 1: Write the failing test**

Create `tests/test_sparse.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sparse.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag.sparse'`.

- [ ] **Step 3: Implement `rag/sparse.py`**

```python
"""BM25 sparse embeddings via fastembed — local ONNX, no API keys.

Passages use embed() (term-frequency weighted); queries use query_embed()
(binary term presence) — the asymmetry is part of the BM25 formulation.
IDF weighting happens server-side in Qdrant (Modifier.IDF on the collection).
"""

from fastembed import SparseTextEmbedding
from pydantic import BaseModel

from config import settings

_model: SparseTextEmbedding | None = None


class SparseVector(BaseModel):
    indices: list[int]
    values: list[float]


def _get_model() -> SparseTextEmbedding:
    global _model
    if _model is None:
        _model = SparseTextEmbedding(model_name=settings.sparse_model)
    return _model


def sparse_embed_texts(texts: list[str]) -> list[SparseVector]:
    """Embed passages in order (TF side of BM25)."""
    if not texts:
        return []
    return [
        SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
        for e in _get_model().embed(texts)
    ]


def sparse_embed_query(text: str) -> SparseVector:
    emb = next(iter(_get_model().query_embed(text)))
    return SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())
```

Note for the tests: they patch `rag.sparse._model` directly, which bypasses `_get_model()`'s `None` check — that works because `_get_model` returns the patched instance only when `_model is None` is False. Trace it: `_model` patched to `FakeBM25()` → `_get_model()` skips construction and returns it. No fastembed model download in unit tests.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_sparse.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add rag/sparse.py tests/test_sparse.py
git commit -m "feat: add BM25 sparse embeddings via fastembed"
```

---

### Task 3: Named dense+sparse vectors and hybrid search in the store

**Files:**
- Modify: `rag/store.py`
- Modify: `tests/test_store.py`

**Interfaces:**
- Consumes: `SparseVector` from `rag.sparse` (Task 2), `settings.embedding_dim`, `settings.retrieval_top_k`.
- Produces: `ChunkRecord` gains required field `sparse: SparseVector`; `VectorStore.search(dense: list[float] | None = None, sparse: SparseVector | None = None, top_k: int | None = None) -> list[ScoredChunk]`; `VectorStore.check_schema() -> None` (raises `RuntimeError` naming `rag.migrate` on legacy schema); vector names are module constants `DENSE_VECTOR = "dense"`, `SPARSE_VECTOR = "bm25"`. Tasks 4, 7, and the API/eval wiring rely on these exact names.

- [ ] **Step 1: Update the fake and write failing tests**

Replace `tests/test_store.py` entirely with:

```python
from types import SimpleNamespace

import pytest

from rag.sparse import SparseVector


class FakeQdrant:
    def __init__(self, exists=False, hits=None, fail=False, legacy=False):
        self._exists = exists
        self._hits = hits or []
        self._fail = fail
        self._legacy = legacy
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
            params = SimpleNamespace(vectors={"dense": SimpleNamespace(size=1536)},
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


def test_ensure_collection_creates_named_vectors():
    from rag.store import DENSE_VECTOR, SPARSE_VECTOR

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_store.py -v`
Expected: FAIL — `ImportError` (no `DENSE_VECTOR`), `ValidationError` (no `sparse` field), `TypeError` on `search` kwargs.

- [ ] **Step 3: Rewrite `rag/store.py`**

```python
import uuid

from pydantic import BaseModel
from qdrant_client import QdrantClient, models

from config import settings
from rag.sparse import SparseVector

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"

LEGACY_SCHEMA_MESSAGE = (
    "Collection '{collection}' uses the legacy phase-1 schema (single unnamed "
    "vector). Recreate it with: uv run python -m rag.migrate --yes "
    "(drops ingested chunks; re-ingest papers afterwards)"
)


class ChunkRecord(BaseModel):
    paper_id: str
    title: str
    chunk_index: int
    text: str
    section: str = ""  # kept for schema parity; pypdf gives no reliable sections
    vector: list[float]
    sparse: SparseVector


class ScoredChunk(BaseModel):
    paper_id: str
    title: str
    text: str
    score: float


class VectorStore:
    """Thin wrapper around Qdrant for paper chunks (named dense+sparse vectors)."""

    def __init__(self, url: str | None = None, collection: str | None = None, client=None):
        self.collection = collection or settings.qdrant_collection
        self.client = client or QdrantClient(url=url or settings.qdrant_url)

    def ping(self) -> None:
        """Fail fast with a clear message when Qdrant is down."""
        try:
            self.client.get_collections()
        except Exception as exc:
            raise RuntimeError(
                f"Qdrant is not reachable at {settings.qdrant_url}. "
                "Start it with: docker compose up -d"
            ) from exc

    def check_schema(self) -> None:
        """Raise when the collection predates the named dense+sparse schema."""
        if not self.client.collection_exists(self.collection):
            return
        params = self.client.get_collection(self.collection).config.params
        if not isinstance(params.vectors, dict) or DENSE_VECTOR not in params.vectors \
                or not params.sparse_vectors:
            raise RuntimeError(LEGACY_SCHEMA_MESSAGE.format(collection=self.collection))

    def ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    DENSE_VECTOR: models.VectorParams(
                        size=settings.embedding_dim, distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    # IDF computed server-side from the stored corpus — the
                    # client only ships term frequencies (rag/sparse.py).
                    SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
            )
        else:
            self.check_schema()

    def upsert_chunks(self, records: list[ChunkRecord]) -> None:
        points = [
            models.PointStruct(
                # uuid5 of paper_id:chunk_index → re-ingesting a paper overwrites
                # its old points instead of duplicating them.
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{r.paper_id}:{r.chunk_index}")),
                vector={
                    DENSE_VECTOR: r.vector,
                    SPARSE_VECTOR: models.SparseVector(
                        indices=r.sparse.indices, values=r.sparse.values
                    ),
                },
                payload={"paper_id": r.paper_id, "title": r.title,
                         "chunk_index": r.chunk_index, "chunk_text": r.text,
                         "section": r.section},
            )
            for r in records
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, dense: list[float] | None = None,
               sparse: SparseVector | None = None,
               top_k: int | None = None) -> list[ScoredChunk]:
        top_k = top_k or settings.retrieval_top_k
        if dense is not None and sparse is not None:
            # RRF fusion over both vector types; each side over-fetches 2x so
            # fusion has candidates that only one side ranked highly.
            resp = self.client.query_points(
                collection_name=self.collection,
                prefetch=[
                    models.Prefetch(query=dense, using=DENSE_VECTOR, limit=top_k * 2),
                    models.Prefetch(
                        query=models.SparseVector(indices=sparse.indices, values=sparse.values),
                        using=SPARSE_VECTOR, limit=top_k * 2,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
            )
        elif dense is not None:
            resp = self.client.query_points(
                collection_name=self.collection, query=dense,
                using=DENSE_VECTOR, limit=top_k,
            )
        elif sparse is not None:
            resp = self.client.query_points(
                collection_name=self.collection,
                query=models.SparseVector(indices=sparse.indices, values=sparse.values),
                using=SPARSE_VECTOR, limit=top_k,
            )
        else:
            raise ValueError("search needs a dense and/or sparse vector")
        return [
            ScoredChunk(paper_id=h.payload["paper_id"], title=h.payload["title"],
                        text=h.payload["chunk_text"], score=h.score)
            for h in resp.points
        ]

    def has_paper(self, paper_id: str) -> bool:
        hits, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="paper_id",
                                            match=models.MatchValue(value=paper_id))]
            ),
            limit=1,
        )
        return len(hits) > 0
```

- [ ] **Step 4: Run tests — store passes, collateral failures expected**

Run: `uv run pytest tests/test_store.py -v`
Expected: PASS (all store tests).

Run: `uv run pytest`
Expected: FAILURES in `tests/test_ingest.py` and `tests/test_retrieve_answer.py` (`ChunkRecord` now requires `sparse`; `search` signature changed). Those are fixed in Task 7 — do NOT fix them here. Everything else passes.

- [ ] **Step 5: Commit**

```bash
git add rag/store.py tests/test_store.py
git commit -m "feat: named dense+sparse vectors with RRF hybrid search in the store"
```

(Committing with two known-failing test files is a deliberate exception here; Task 7 restores a green suite. If your workflow forbids it, squash Tasks 3–7 into one commit at the end of Task 7 instead.)

---

### Task 4: Migration script + schema checks at entry points

**Files:**
- Create: `rag/migrate.py`
- Modify: `api/main.py` (lifespan only)
- Modify: `eval/run.py` (ping block only)
- Test: `tests/test_migrate.py`
- Modify: `tests/test_api.py` (FakeStore gains `check_schema`)

**Interfaces:**
- Consumes: `VectorStore.check_schema()`, `VectorStore.ensure_collection()` (Task 3).
- Produces: `rag.migrate.migrate(store: VectorStore | None = None) -> None`; CLI `uv run python -m rag.migrate --yes`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_migrate.py`:

```python
class FakeStore:
    def __init__(self, exists=True):
        self.collection = "papers"
        self.client = self
        self._exists = exists
        self.deleted = []
        self.ensured = False
        self.pinged = False

    # VectorStore surface
    def ping(self):
        self.pinged = True

    def ensure_collection(self):
        self.ensured = True

    # client surface used by migrate
    def collection_exists(self, name):
        return self._exists

    def delete_collection(self, name):
        self.deleted.append(name)


def test_migrate_drops_and_recreates():
    from rag.migrate import migrate

    store = FakeStore(exists=True)
    migrate(store=store)
    assert store.pinged
    assert store.deleted == ["papers"]
    assert store.ensured


def test_migrate_fresh_collection_no_delete():
    from rag.migrate import migrate

    store = FakeStore(exists=False)
    migrate(store=store)
    assert store.deleted == []
    assert store.ensured
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_migrate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag.migrate'`.

- [ ] **Step 3: Implement `rag/migrate.py`**

```python
"""Recreate the Qdrant collection with the phase-2 named dense+sparse schema.

Destructive: drops all ingested chunks. PDFs stay cached in data/pdfs, so
re-ingesting re-parses and re-embeds them (dense re-embedding costs OpenAI
tokens).

Usage: uv run python -m rag.migrate --yes
"""

import argparse

from rag.store import VectorStore


def migrate(store: VectorStore | None = None) -> None:
    store = store or VectorStore()
    store.ping()
    if store.client.collection_exists(store.collection):
        store.client.delete_collection(store.collection)
    store.ensure_collection()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true",
                        help="confirm dropping the existing collection")
    args = parser.parse_args()
    if not args.yes:
        parser.error("refusing to drop the collection without --yes")
    migrate()
    print("Collection recreated with the dense+sparse schema. Re-ingest papers now.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Wire `check_schema` into the API and eval entry points**

In `api/main.py`, replace the lifespan body:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    store = VectorStore()
    store.ping()  # fail fast at startup if Qdrant is down (docker compose up -d)
    store.check_schema()  # fail fast on a phase-1 collection (rag.migrate)
    yield
```

In `eval/run.py`, replace the first line of `run_eval`'s body:

```python
    store = VectorStore()
    store.ping()  # fail fast with a clear message when Qdrant is down
    store.check_schema()
```

In `tests/test_api.py`, the `_client` helper's `FakeStore` gains the new method — replace the class with:

```python
    class FakeStore:
        def ping(self):
            pass

        def check_schema(self):
            pass
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_migrate.py tests/test_api.py -v`
Expected: PASS. (`tests/test_ingest.py` / `tests/test_retrieve_answer.py` still fail — Task 7.)

- [ ] **Step 6: Commit**

```bash
git add rag/migrate.py tests/test_migrate.py api/main.py eval/run.py tests/test_api.py
git commit -m "feat: add rag.migrate and legacy-schema checks at startup/eval"
```

---

### Task 5: Cross-encoder reranking (`rag/rerank.py`)

**Files:**
- Create: `rag/rerank.py`
- Test: `tests/test_rerank.py`

**Interfaces:**
- Consumes: `settings.rerank_model` (Task 1), `ScoredChunk` (Task 3).
- Produces: `rerank(question: str, chunks: list[ScoredChunk], top_k: int) -> list[ScoredChunk]` — sorted best-first, truncated to `top_k`, `score` replaced by the cross-encoder score. Task 7 imports this.

- [ ] **Step 1: Write the failing test**

Create `tests/test_rerank.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rerank.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag.rerank'`.

- [ ] **Step 3: Implement `rag/rerank.py`**

```python
"""Cross-encoder reranking via fastembed — local ONNX, no API keys.

First use downloads the model (~80MB) into the local HF cache. A load failure
raises: no silent fallback to un-reranked results, a degraded pipeline should
be loud (the query-rewrite stage is the one that fails open, not this one).
"""

from fastembed.rerank.cross_encoder import TextCrossEncoder

from config import settings
from rag.store import ScoredChunk

_model: TextCrossEncoder | None = None


def _get_model() -> TextCrossEncoder:
    global _model
    if _model is None:
        _model = TextCrossEncoder(model_name=settings.rerank_model)
    return _model


def rerank(question: str, chunks: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
    """Re-score chunks against the question with a cross-encoder; best first, keep top_k."""
    if not chunks:
        return []
    scores = list(_get_model().rerank(question, [c.text for c in chunks]))
    rescored = [c.model_copy(update={"score": float(s)}) for c, s in zip(chunks, scores)]
    return sorted(rescored, key=lambda c: c.score, reverse=True)[:top_k]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_rerank.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add rag/rerank.py tests/test_rerank.py
git commit -m "feat: add local cross-encoder reranking"
```

---

### Task 6: LLM query rewriting (`rag/rewrite.py`)

**Files:**
- Create: `rag/rewrite.py`
- Modify: `llm/prompts.py` (append one constant)
- Test: `tests/test_rewrite.py`

**Interfaces:**
- Consumes: `llm.base.generate` with `structured_schema`.
- Produces: `rewrite_query(question: str) -> str` — rewritten search query, or the original question on any failure. Task 7 imports this. `llm.prompts.REWRITE_SYSTEM_PROMPT`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_rewrite.py`:

```python
from llm.base import LLMResponse


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rewrite.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag.rewrite'`.

- [ ] **Step 3: Append the prompt to `llm/prompts.py`**

```python
REWRITE_SYSTEM_PROMPT = """You rewrite user questions into search queries for a research-paper vector database.

Return one focused query: expand acronyms, drop conversational filler, keep every technical term. Do not answer the question."""
```

- [ ] **Step 4: Implement `rag/rewrite.py`**

```python
"""LLM query rewriting: question → focused search query (used for retrieval only;
answer generation still sees the original question).

Fails open: any LLM error logs a warning and returns the original question —
retrieval must never break because rewriting broke.
"""

import logging

from pydantic import BaseModel

from llm.base import generate
from llm.prompts import REWRITE_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class RewrittenQuery(BaseModel):
    query: str


def rewrite_query(question: str) -> str:
    try:
        resp = generate(
            [{"role": "user", "content": question}],
            system=REWRITE_SYSTEM_PROMPT,
            structured_schema=RewrittenQuery,
        )
        rewritten = resp.parsed.query.strip()
        return rewritten or question
    except Exception:
        logger.warning("Query rewrite failed; using the original question", exc_info=True)
        return question
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_rewrite.py tests/test_prompts.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add rag/rewrite.py llm/prompts.py tests/test_rewrite.py
git commit -m "feat: add fail-open LLM query rewriting"
```

---

### Task 7: Staged retrieval pipeline + ingest writes both vectors

**Files:**
- Modify: `rag/retrieve.py`
- Modify: `rag/ingest.py`
- Modify: `tests/test_retrieve_answer.py`
- Modify: `tests/test_ingest.py`

**Interfaces:**
- Consumes: `sparse_embed_texts`/`sparse_embed_query` (Task 2), `store.search(dense=, sparse=, top_k=)` (Task 3), `rerank` (Task 5), `rewrite_query` (Task 6), all settings flags (Task 1).
- Produces: `retrieve(question, top_k=None, store=None) -> list[ScoredChunk]` — same public signature as phase 1, now honoring the flags. `ingest_paper` unchanged signature, records carry `sparse`.

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_retrieve_answer.py` entirely with:

```python
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


def test_answer_question_empty_store(monkeypatch):
    import rag.answer as answer_mod

    monkeypatch.setattr(answer_mod, "retrieve", lambda q, store=None: [])
    monkeypatch.setattr(answer_mod, "generate",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no LLM call")))

    result = answer_mod.answer_question("anything")
    assert result.sources == []
    assert "ingest" in result.text.lower()
```

In `tests/test_ingest.py`, replace `_patch_pipeline` and `test_ingest_paper_happy_path` with:

```python
def _patch_pipeline(monkeypatch, text="some paper text"):
    import rag.ingest as ingest
    from rag.sparse import SparseVector

    monkeypatch.setattr(ingest, "download_pdf", lambda pid: f"/tmp/{pid}.pdf")
    monkeypatch.setattr(ingest, "extract_text", lambda path: text)
    monkeypatch.setattr(ingest, "chunk_text", lambda t: ["chunk a", "chunk b"])
    monkeypatch.setattr(ingest, "embed_texts", lambda chunks: [[0.1], [0.2]])
    monkeypatch.setattr(ingest, "sparse_embed_texts",
                        lambda chunks: [SparseVector(indices=[1], values=[1.0]),
                                        SparseVector(indices=[2], values=[1.0])])


def test_ingest_paper_happy_path(monkeypatch):
    import rag.ingest as ingest

    _patch_pipeline(monkeypatch)
    store = FakeStore()
    n = ingest.ingest_paper(_meta(), store)

    assert n == 2
    records = store.upserts[0]
    assert [r.chunk_index for r in records] == [0, 1]
    assert records[0].paper_id == "1706.03762"
    assert records[0].vector == [0.1]
    assert records[0].sparse.indices == [1]
    assert records[1].sparse.indices == [2]
```

(The other ingest tests keep working — they never build `ChunkRecord`s past the patched pipeline.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_retrieve_answer.py tests/test_ingest.py -v`
Expected: FAIL — `retrieve` doesn't honor flags yet, `ingest` doesn't produce `sparse`.

- [ ] **Step 3: Rewrite `rag/retrieve.py`**

```python
"""Staged retrieval pipeline: [rewrite] → embed → search (dense|sparse|hybrid) → [rerank].

Every stage is a config.py flag so eval ablation can isolate each technique's
effect. Reranking scores against the ORIGINAL question — the rewrite only
shapes the search, never the relevance judgment.
"""

from config import settings
from rag.embed import embed_query
from rag.rerank import rerank
from rag.rewrite import rewrite_query
from rag.sparse import sparse_embed_query
from rag.store import ScoredChunk, VectorStore


def retrieve(question: str, top_k: int | None = None,
             store: VectorStore | None = None) -> list[ScoredChunk]:
    store = store or VectorStore()
    top_k = top_k or settings.retrieval_top_k

    search_text = rewrite_query(question) if settings.rewrite_enabled else question
    fetch_k = settings.rerank_candidates if settings.rerank_enabled else top_k

    dense = embed_query(search_text) if settings.retrieval_mode in ("dense", "hybrid") else None
    sparse = sparse_embed_query(search_text) if settings.retrieval_mode in ("sparse", "hybrid") else None
    chunks = store.search(dense=dense, sparse=sparse, top_k=fetch_k)

    if settings.rerank_enabled:
        chunks = rerank(question, chunks, top_k=top_k)
    return chunks[:top_k]
```

- [ ] **Step 4: Update `rag/ingest.py`**

Add the import:

```python
from rag.sparse import sparse_embed_texts
```

In `ingest_paper`, replace the chunk/embed/records block with:

```python
    chunks = chunk_text(text)
    vectors = embed_texts(chunks)
    sparse_vectors = sparse_embed_texts(chunks)
    records = [
        ChunkRecord(paper_id=meta.paper_id, title=meta.title,
                    chunk_index=i, text=chunk, vector=vector, sparse=sparse)
        for i, (chunk, vector, sparse) in enumerate(zip(chunks, vectors, sparse_vectors))
    ]
```

- [ ] **Step 5: Run the FULL suite to verify green**

Run: `uv run pytest`
Expected: PASS — this task ends the Task-3 collateral breakage. If anything else fails, fix it before committing.

- [ ] **Step 6: Commit**

```bash
git add rag/retrieve.py rag/ingest.py tests/test_retrieve_answer.py tests/test_ingest.py
git commit -m "feat: staged retrieval pipeline with flags; ingest writes sparse vectors"
```

---

### Task 8: Eval ablation mode

**Files:**
- Modify: `eval/run.py`
- Test: `tests/test_eval_run.py` (append new tests)

**Interfaces:**
- Consumes: `run_eval(dataset_path, report_path)` (existing), settings flags (Task 1).
- Produces: `eval.run.PRESETS: dict[str, dict]`, `run_ablation(dataset_path="eval/golden.json", report_path="report-ablation.json") -> dict` (shape: `{"presets": {name: summary}}`), CLI `uv run python -m eval.run --ablation`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_eval_run.py`:

```python
def test_ablation_sweeps_presets_and_restores_settings(monkeypatch, tmp_path):
    import eval.run as run_mod
    from config import settings

    monkeypatch.setattr(settings, "retrieval_mode", "hybrid")
    monkeypatch.setattr(settings, "rerank_enabled", True)
    monkeypatch.setattr(settings, "rewrite_enabled", False)

    seen = []

    def fake_run_eval(dataset_path, report_path):
        seen.append({"preset_report": report_path,
                     "mode": settings.retrieval_mode,
                     "rerank": settings.rerank_enabled,
                     "rewrite": settings.rewrite_enabled})
        return {"summary": {"n": 1, "avg_precision": 1.0, "avg_recall": 1.0,
                            "avg_faithfulness": 5.0, "avg_relevance": 5.0,
                            "avg_citation_accuracy": 5.0}}

    monkeypatch.setattr(run_mod, "run_eval", fake_run_eval)
    report = run_mod.run_ablation(report_path=str(tmp_path / "ablation.json"))

    assert list(report["presets"]) == ["baseline-dense", "sparse", "hybrid",
                                       "hybrid+rerank", "full"]
    assert [s["mode"] for s in seen] == ["dense", "sparse", "hybrid", "hybrid", "hybrid"]
    assert [s["rerank"] for s in seen] == [False, False, False, True, True]
    assert [s["rewrite"] for s in seen] == [False, False, False, False, True]
    assert seen[0]["preset_report"] == str(tmp_path / "report-baseline-dense.json")
    # settings restored after the sweep
    assert settings.retrieval_mode == "hybrid"
    assert settings.rerank_enabled is True
    assert settings.rewrite_enabled is False


def test_ablation_restores_settings_on_failure(monkeypatch, tmp_path):
    import pytest

    import eval.run as run_mod
    from config import settings

    monkeypatch.setattr(settings, "retrieval_mode", "hybrid")

    def boom(dataset_path, report_path):
        raise RuntimeError("qdrant down")

    monkeypatch.setattr(run_mod, "run_eval", boom)
    with pytest.raises(RuntimeError):
        run_mod.run_ablation(report_path=str(tmp_path / "ablation.json"))
    assert settings.retrieval_mode == "hybrid"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_eval_run.py -v`
Expected: New tests FAIL with `AttributeError: module 'eval.run' has no attribute 'run_ablation'`; existing tests still pass.

- [ ] **Step 3: Implement ablation in `eval/run.py`**

Add imports at the top (`argparse`, `settings`):

```python
import argparse
import json
from pathlib import Path
from statistics import mean

from config import settings
from eval.judge import judge_answer
from eval.metrics import precision_recall
from rag.answer import answer_question
from rag.retrieve import retrieve
from rag.store import VectorStore
```

Append after `run_eval`:

```python
# Ablation presets: each row of the comparison table. Order matters — the
# printed table reads as "what did each added technique buy us".
PRESETS: dict[str, dict] = {
    "baseline-dense": {"retrieval_mode": "dense", "rerank_enabled": False, "rewrite_enabled": False},
    "sparse": {"retrieval_mode": "sparse", "rerank_enabled": False, "rewrite_enabled": False},
    "hybrid": {"retrieval_mode": "hybrid", "rerank_enabled": False, "rewrite_enabled": False},
    "hybrid+rerank": {"retrieval_mode": "hybrid", "rerank_enabled": True, "rewrite_enabled": False},
    "full": {"retrieval_mode": "hybrid", "rerank_enabled": True, "rewrite_enabled": True},
}


def run_ablation(dataset_path: str = "eval/golden.json",
                 report_path: str = "report-ablation.json") -> dict:
    """Run the golden dataset once per preset; collect summaries side by side.

    Mutates settings per preset and restores the originals afterwards —
    fine for this offline, single-threaded harness.
    """
    fields = ["retrieval_mode", "rerank_enabled", "rewrite_enabled"]
    original = {f: getattr(settings, f) for f in fields}
    summaries: dict[str, dict] = {}
    try:
        for name, overrides in PRESETS.items():
            for field, value in overrides.items():
                setattr(settings, field, value)
            per_preset_path = str(Path(report_path).with_name(f"report-{name}.json"))
            summaries[name] = run_eval(dataset_path=dataset_path,
                                       report_path=per_preset_path)["summary"]
    finally:
        for field, value in original.items():
            setattr(settings, field, value)
    report = {"presets": summaries}
    Path(report_path).write_text(json.dumps(report, indent=2))
    return report


def _print_ablation(report: dict) -> None:
    cols = ["avg_precision", "avg_recall", "avg_faithfulness",
            "avg_relevance", "avg_citation_accuracy"]
    print(f"\n{'preset':<16}" + "".join(f"{c.removeprefix('avg_'):>19}" for c in cols))
    for name, s in report["presets"].items():
        print(f"{name:<16}" + "".join(f"{s[c]:>19.2f}" for c in cols))
```

Replace `main`:

```python
def main() -> None:
    parser = argparse.ArgumentParser(description="Offline eval harness")
    parser.add_argument("--ablation", action="store_true",
                        help="sweep retrieval presets and print a comparison table")
    args = parser.parse_args()
    if args.ablation:
        _print_ablation(run_ablation())
        return
    report = run_eval()
    s = report["summary"]
    print(f"\nEvaluated {s['n']} questions -> report.json")
    print(f"  retrieval precision : {s['avg_precision']:.2f}")
    print(f"  retrieval recall    : {s['avg_recall']:.2f}")
    print(f"  faithfulness        : {s['avg_faithfulness']:.2f} / 5")
    print(f"  relevance           : {s['avg_relevance']:.2f} / 5")
    print(f"  citation accuracy   : {s['avg_citation_accuracy']:.2f} / 5")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_eval_run.py -v`
Expected: PASS (all, old and new).

- [ ] **Step 5: Commit**

```bash
git add eval/run.py tests/test_eval_run.py
git commit -m "feat: eval ablation mode sweeping retrieval presets"
```

---

### Task 9: Agent memory — checkpointing, message reducer, summarization

**Files:**
- Modify: `agents/graph.py`
- Modify: `llm/prompts.py` (append one constant)
- Modify: `tests/test_graph.py`

**Interfaces:**
- Consumes: `settings.checkpoint_db`, `settings.memory_max_messages`, `settings.memory_keep_messages` (Task 1); `AsyncSqliteSaver` from `langgraph.checkpoint.sqlite.aio`.
- Produces: `build_graph(toolbox, checkpointer=None)`; `run_agent(question: str, thread_id: str | None = None) -> str` (fresh UUID when omitted — direct callers stay single-shot); `AgentState` with `messages: Annotated[list[dict], operator.add]`, `steps: int`, `summary: str`; nodes return DELTAS (only new messages), not the full list. Tasks 10–11 call `run_agent(question, thread_id)`.

**Key behavior change to understand before editing:** with `operator.add` as the messages reducer, LangGraph appends whatever a node returns to the channel. Phase-1 nodes returned the full list (`state["messages"] + [new]`) — under the reducer that would duplicate history. Every node must now return only its new messages. This is what makes multi-turn work: invoking a checkpointed thread with one new user message appends it to the restored history.

- [ ] **Step 1: Update tests**

In `tests/test_graph.py`, apply these changes:

1. `test_run_agent_falls_back_when_step_limit_hit` — `run_agent` now opens a SQLite checkpointer; point it at a tmp file. Add `tmp_path` to the signature and a monkeypatch line:

```python
async def test_run_agent_falls_back_when_step_limit_hit(monkeypatch, tmp_path):
    import agents.graph as graph_mod
    from config import settings
    from rag.answer import RagAnswer

    monkeypatch.setattr(settings, "checkpoint_db", str(tmp_path / "checkpoints.db"))
    monkeypatch.setattr(settings, "agent_max_steps", 1)
    endless = [
        LLMResponse(tool_calls=[ToolCall(id=f"tu_{i}", name="rag_query",
                                         input={"question": "q"})])
        for i in range(5)
    ]
    _scripted_generate(monkeypatch, endless)
    monkeypatch.setattr(graph_mod, "answer_question",
                        lambda q: RagAnswer(text="partial", sources=[]))

    class FakeToolboxCM:
        async def __aenter__(self):
            return FakeToolbox()

        async def __aexit__(self, *exc):
            return None

    monkeypatch.setattr(graph_mod, "MCPToolbox", FakeToolboxCM)

    reply = await graph_mod.run_agent("q")

    assert reply == graph_mod.STEP_LIMIT_MESSAGE
```

2. Append three new tests:

```python
async def test_multi_turn_thread_restores_history(monkeypatch, tmp_path):
    """Second invoke on the same thread sees the first turn's messages."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    import agents.graph as graph_mod

    seen = _scripted_generate(monkeypatch, [
        LLMResponse(text="Paris."),
        LLMResponse(text="About 2 million."),
    ])
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "cp.db")) as saver:
        graph = graph_mod.build_graph(FakeToolbox(), checkpointer=saver)
        config = {"configurable": {"thread_id": "t1"}}
        await graph.ainvoke({"messages": [{"role": "user", "content": "Capital of France?"}],
                             "steps": 0}, config=config)
        state = await graph.ainvoke({"messages": [{"role": "user", "content": "Its population?"}],
                                     "steps": 0}, config=config)

    assert graph_mod.final_text(state) == "About 2 million."
    second_call_messages = seen[1]["messages"]
    contents = [m["content"] for m in second_call_messages if isinstance(m["content"], str)]
    assert "Capital of France?" in contents  # history restored from the checkpoint
    assert "Its population?" in contents


async def test_summarize_compresses_long_history(monkeypatch, tmp_path):
    """Past memory_max_messages, older turns get summarized into the system prompt."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    import agents.graph as graph_mod
    from config import settings

    monkeypatch.setattr(settings, "memory_max_messages", 3)
    monkeypatch.setattr(settings, "memory_keep_messages", 2)

    # Turn 1 and 2: direct answers. Turn 3: history (5 messages) exceeds the
    # threshold → summarize node calls generate first, then the agent node.
    seen = _scripted_generate(monkeypatch, [
        LLMResponse(text="answer one"),
        LLMResponse(text="answer two"),
        LLMResponse(text="THE SUMMARY"),
        LLMResponse(text="answer three"),
    ])
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "cp.db")) as saver:
        graph = graph_mod.build_graph(FakeToolbox(), checkpointer=saver)
        config = {"configurable": {"thread_id": "t1"}}
        for q in ["q1", "q2", "q3"]:
            state = await graph.ainvoke(
                {"messages": [{"role": "user", "content": q}], "steps": 0}, config=config)

    assert graph_mod.final_text(state) == "answer three"
    assert len(seen) == 4
    # the agent call after summarization carries the summary in its system prompt
    assert "THE SUMMARY" in seen[3]["system"]
    # and only the keep-window of messages (trimmed history)
    assert len(seen[3]["messages"]) <= 3


async def test_trimmed_history_starts_at_plain_user_turn():
    from agents.graph import _trimmed_history

    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1",
                                           "name": "rag_query", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1",
                                      "content": "result"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "a1"}]},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": [{"type": "text", "text": "a2"}]},
    ]
    # keep=3 starts the window at "assistant a1"; the walk-back must pass the
    # tool_result (list content, not plain) and land on plain user "q1"
    trimmed = _trimmed_history(messages, keep=3)
    assert trimmed[0] == {"role": "user", "content": "q1"}
    # keep=2 starts at "q2" — already a clean boundary, no walk
    trimmed2 = _trimmed_history(messages, keep=2)
    assert trimmed2[0] == {"role": "user", "content": "q2"}
    # short history returned whole
    assert _trimmed_history(messages[:2], keep=8) == messages[:2]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_graph.py -v`
Expected: FAIL — `build_graph` takes no `checkpointer`, `_trimmed_history` doesn't exist, multi-turn duplicates/replaces state.

- [ ] **Step 3: Append the summarization prompt to `llm/prompts.py`**

```python
SUMMARIZE_SYSTEM_PROMPT = """Compress this research-assistant conversation into a factual running summary.

Keep: questions asked, papers found or ingested (with arXiv ids), key findings, open follow-ups.
Drop: pleasantries and tool-call mechanics. Under 300 words."""
```

- [ ] **Step 4: Rewrite `agents/graph.py`**

```python
"""LangGraph agent with multi-turn memory.

Flow per invoke: summarize (no-op below the threshold) → agent ⇄ tools → END.
State is checkpointed per thread_id (AsyncSqliteSaver); `messages` uses an
operator.add reducer, so nodes return ONLY their new messages and invoking a
checkpointed thread with one new user message appends to restored history.

Long threads: the full history lives in the checkpoint, but the LLM sees a
trimmed window (_trimmed_history) plus a running summary in the system prompt.
The summarize node re-summarizes everything outside the window each time it
fires — costs some tokens, keeps the bookkeeping trivial.
"""

import asyncio
import logging
import operator
import uuid
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph

from agents.mcp_client import MCPToolbox
from config import settings
from llm.base import generate
from llm.prompts import AGENT_SYSTEM_PROMPT, SUMMARIZE_SYSTEM_PROMPT
from rag.answer import answer_question

logger = logging.getLogger(__name__)

STEP_LIMIT_MESSAGE = (
    "I hit my tool-step limit before reaching a final answer. "
    "Any papers fetched so far are ingested - try asking again or narrowing the question."
)

RAG_QUERY_TOOL = {
    "name": "rag_query",
    "description": (
        "Answer a question from the already-ingested arXiv papers, with [paper_id] "
        "citations. Tells you when it doesn't have enough information."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"question": {"type": "string"}},
        "required": ["question"],
    },
}


class AgentState(TypedDict):
    messages: Annotated[list[dict], operator.add]
    steps: int
    summary: str


def _trimmed_history(messages: list[dict], keep: int) -> list[dict]:
    """Window of recent messages that starts at a plain user turn.

    Walking back to a plain (string-content) user message keeps tool_use /
    tool_result pairs intact — an orphaned tool_result at the front of the
    window would be an API error. Index 0 is always a plain user turn, so the
    walk terminates.
    """
    if len(messages) <= keep:
        return messages
    start = len(messages) - keep
    while start > 0 and not (messages[start]["role"] == "user"
                             and isinstance(messages[start]["content"], str)):
        start -= 1
    return messages[start:]


def _render_for_summary(messages: list[dict]) -> str:
    lines: list[str] = []
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            lines.append(f"{m['role']}: {content}")
            continue
        for block in content:
            if block["type"] == "text":
                lines.append(f"{m['role']}: {block['text']}")
            elif block["type"] == "tool_use":
                lines.append(f"{m['role']} called {block['name']}({block['input']})")
            elif block["type"] == "tool_result":
                lines.append(f"tool result: {str(block['content'])[:300]}")
    return "\n".join(lines)


def build_graph(toolbox, checkpointer=None):
    tools = [RAG_QUERY_TOOL] + toolbox.list_tools()

    async def summarize_node(state: AgentState) -> dict:
        messages = state["messages"]
        if len(messages) <= settings.memory_max_messages:
            return {}
        window = _trimmed_history(messages, settings.memory_keep_messages)
        older = messages[: len(messages) - len(window)]
        if not older:
            return {}
        prior = state.get("summary", "")
        prompt = (f"Previous summary:\n{prior}\n\n" if prior else "") + \
            "Conversation to compress:\n" + _render_for_summary(older)
        resp = await asyncio.to_thread(
            generate, [{"role": "user", "content": prompt}], system=SUMMARIZE_SYSTEM_PROMPT
        )
        return {"summary": resp.text}

    async def agent_node(state: AgentState) -> dict:
        history = _trimmed_history(state["messages"], settings.memory_keep_messages)
        system = AGENT_SYSTEM_PROMPT
        if state.get("summary"):
            system = (f"{AGENT_SYSTEM_PROMPT}\n\n"
                      f"Conversation so far (summarized):\n{state['summary']}")
        resp = await asyncio.to_thread(generate, history, system=system, tools=tools)
        content: list[dict] = []
        if resp.text:
            content.append({"type": "text", "text": resp.text})
        for tc in resp.tool_calls:
            content.append({"type": "tool_use", "id": tc.id, "name": tc.name,
                            "input": tc.input})
        return {"messages": [{"role": "assistant", "content": content}]}

    async def tools_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        results: list[dict] = []
        for block in last["content"]:
            if block["type"] != "tool_use":
                continue
            name, args = block["name"], block["input"]
            logger.info("Tool call: %s(%s)", name, args)
            if name == "rag_query":
                try:
                    ans = await asyncio.to_thread(answer_question, args["question"])
                    content = f"{ans.text}\n\nSources: {', '.join(ans.sources) or 'none'}"
                    is_error = False
                except Exception as exc:  # e.g. Qdrant down — agent decides what to do
                    content, is_error = f"rag_query failed: {exc}", True
            else:
                content, is_error = await toolbox.call_tool(name, args)
            results.append({"type": "tool_result", "tool_use_id": block["id"],
                            "content": content, "is_error": is_error})
        return {
            "messages": [{"role": "user", "content": results}],
            "steps": state["steps"] + 1,
        }

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        has_tool_use = isinstance(last["content"], list) and any(
            b["type"] == "tool_use" for b in last["content"]
        )
        if has_tool_use and state["steps"] < settings.agent_max_steps:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("summarize", summarize_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.set_entry_point("summarize")
    graph.add_edge("summarize", "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)


def final_text(state: dict) -> str:
    """Text of the last assistant message that has any text."""
    for message in reversed(state["messages"]):
        if message["role"] != "assistant":
            continue
        content = message["content"]
        if isinstance(content, list):
            texts = [b["text"] for b in content if b["type"] == "text"]
            if texts:
                return "\n".join(texts)
        elif content:
            return content
    return ""


async def run_agent(question: str, thread_id: str | None = None) -> str:
    """One agent turn. Same thread_id continues a conversation; omitted → fresh
    single-shot thread (direct callers like eval stay stateless)."""
    thread_id = thread_id or str(uuid.uuid4())
    Path(settings.checkpoint_db).parent.mkdir(parents=True, exist_ok=True)
    async with MCPToolbox() as toolbox, \
            AsyncSqliteSaver.from_conn_string(settings.checkpoint_db) as saver:
        graph = build_graph(toolbox, checkpointer=saver)
        state = await graph.ainvoke(
            {"messages": [{"role": "user", "content": question}], "steps": 0},
            config={"recursion_limit": settings.agent_max_steps * 2 + 6,
                    "configurable": {"thread_id": thread_id}},
        )
        text = final_text(state)
        return text or STEP_LIMIT_MESSAGE
```

Differences worth noting against phase 1: nodes return deltas; `summarize` is the entry point (no-op below threshold); `recursion_limit` grew by 2 for the summarize hop; `run_agent` gained `thread_id` and the checkpointer context manager.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_graph.py -v`
Expected: PASS (all — the four phase-1 tests exercise single invokes where delta-vs-full-list behavior is equivalent, plus the three new memory tests).

Run: `uv run pytest`
Expected: PASS (full suite).

- [ ] **Step 6: Commit**

```bash
git add agents/graph.py llm/prompts.py tests/test_graph.py
git commit -m "feat: agent memory — sqlite checkpointing, message reducer, summarization"
```

---

### Task 10: thread_id through API + frontend

**Files:**
- Modify: `api/main.py`
- Modify: `api/static/app.js`
- Modify: `api/static/index.html`
- Modify: `tests/test_api.py`

**Interfaces:**
- Consumes: `run_agent(question, thread_id)` (Task 9).
- Produces: `POST /api/chat` accepts `{"message": str, "thread_id": str | null}`, responds `{"reply": str, "thread_id": str}` (server-generated UUID when absent). Task 11 swaps the `run_agent` call for `run_chat`.

- [ ] **Step 1: Update the tests**

In `tests/test_api.py`, replace `test_chat_calls_agent` with:

```python
def test_chat_generates_thread_id(monkeypatch):
    import api.main as api_main

    async def fake_run_agent(question, thread_id=None):
        return f"echo: {question} [{thread_id}]"

    monkeypatch.setattr(api_main, "run_agent", fake_run_agent)
    with _client(monkeypatch) as client:
        resp = client.post("/api/chat", json={"message": "what is attention?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["thread_id"]  # server minted one
    assert body["reply"] == f"echo: what is attention? [{body['thread_id']}]"


def test_chat_reuses_given_thread_id(monkeypatch):
    import api.main as api_main

    async def fake_run_agent(question, thread_id=None):
        return f"echo: {question} [{thread_id}]"

    monkeypatch.setattr(api_main, "run_agent", fake_run_agent)
    with _client(monkeypatch) as client:
        resp = client.post("/api/chat", json={"message": "follow-up", "thread_id": "t-42"})
    assert resp.json() == {"reply": "echo: follow-up [t-42]", "thread_id": "t-42"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_api.py -v`
Expected: FAIL — response has no `thread_id`, `run_agent` called without one.

- [ ] **Step 3: Update `api/main.py`**

Add `import uuid` at the top. Replace the chat models and route:

```python
class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None  # omit to start a new conversation


class ChatResponse(BaseModel):
    reply: str
    thread_id: str


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    thread_id = req.thread_id or str(uuid.uuid4())
    reply = await run_agent(req.message, thread_id)
    return ChatResponse(reply=reply, thread_id=thread_id)
```

- [ ] **Step 4: Update the frontend**

In `api/static/index.html`, add a button next to Send in the chat fieldset:

```html
  <fieldset>
    <legend>Chat</legend>
    <input id="chat-input" type="text" placeholder="Ask about the ingested papers…">
    <button id="chat-btn">Send</button>
    <button id="new-conv-btn" title="Forget this conversation and start fresh">New conversation</button>
  </fieldset>
```

In `api/static/app.js`, add at the top (after the `log` line):

```js
let threadId = null; // set from the first reply; sent back to continue the thread
```

Replace the chat click handler and add the new-conversation handler:

```js
document.getElementById("chat-btn").addEventListener("click", async () => {
  const input = document.getElementById("chat-input");
  const message = input.value.trim();
  if (!message) return;
  input.value = "";
  append("user", `You: ${message}`);
  append("status", "thinking…");
  try {
    const body = threadId ? { message, thread_id: threadId } : { message };
    const result = await post("/api/chat", body);
    threadId = result.thread_id;
    log.lastChild.remove();
    append("bot", `Assistant: ${result.reply}`);
  } catch (err) {
    log.lastChild.remove();
    append("status", `Chat failed: ${err.message}`);
  }
});

document.getElementById("new-conv-btn").addEventListener("click", () => {
  threadId = null;
  log.replaceChildren();
  append("status", "New conversation started.");
});
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_api.py -v`
Expected: PASS (all, including `test_index_served`).

- [ ] **Step 6: Commit**

```bash
git add api/main.py api/static/app.js api/static/index.html tests/test_api.py
git commit -m "feat: thread_id through chat API and frontend for multi-turn memory"
```

---

### Task 11: Multi-agent supervisor (`agents/multi.py`)

**Files:**
- Create: `agents/multi.py`
- Modify: `llm/prompts.py` (append two constants)
- Modify: `api/main.py` (dispatch via `run_chat`)
- Test: `tests/test_multi.py`
- Modify: `tests/test_api.py` (patch target becomes `run_chat`)

**Interfaces:**
- Consumes: `run_agent(question, thread_id)` (Task 9), `generate` with `structured_schema`, `settings.agent_mode` (Task 1).
- Produces: `agents.multi.Plan(simple: bool, sub_questions: list[str])`; `run_multi_agent(question, thread_id=None) -> str`; `run_chat(message, thread_id=None) -> str` — the dispatcher the API calls from now on.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_multi.py`:

```python
from llm.base import LLMResponse


def _patch_plan(monkeypatch, plan, synth_text="synthesized"):
    """Patch agents.multi.generate: first call returns the plan, later calls synthesize."""
    import agents.multi as multi_mod

    calls = []

    def fake_generate(messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        if kwargs.get("structured_schema") is multi_mod.Plan:
            return LLMResponse(parsed=plan)
        return LLMResponse(text=synth_text)

    monkeypatch.setattr(multi_mod, "generate", fake_generate)
    return calls


async def test_simple_question_falls_through_to_single_agent(monkeypatch):
    import agents.multi as multi_mod

    _patch_plan(monkeypatch, multi_mod.Plan(simple=True))

    async def fake_run_agent(question, thread_id=None):
        return f"single: {question} [{thread_id}]"

    monkeypatch.setattr(multi_mod, "run_agent", fake_run_agent)
    reply = await multi_mod.run_multi_agent("what is attention?", thread_id="t-1")
    assert reply == "single: what is attention? [t-1]"


async def test_decomposed_question_researches_and_synthesizes(monkeypatch):
    import agents.multi as multi_mod

    plan = multi_mod.Plan(simple=False,
                          sub_questions=["what is BERT?", "what is GPT?"])
    calls = _patch_plan(monkeypatch, plan, synth_text="combined answer [1810.04805]")
    researched = []

    async def fake_run_agent(question, thread_id=None):
        researched.append((question, thread_id))
        return f"finding about {question}"

    monkeypatch.setattr(multi_mod, "run_agent", fake_run_agent)
    reply = await multi_mod.run_multi_agent("compare BERT and GPT")

    assert reply == "combined answer [1810.04805]"
    # researchers run per sub-question, single-shot (no thread)
    assert researched == [("what is BERT?", None), ("what is GPT?", None)]
    # synthesizer saw the question and both findings
    synth_input = calls[-1]["messages"][-1]["content"]
    assert "compare BERT and GPT" in synth_input
    assert "finding about what is BERT?" in synth_input


async def test_failed_researcher_reported_to_synthesizer(monkeypatch):
    import agents.multi as multi_mod

    plan = multi_mod.Plan(simple=False, sub_questions=["good q", "bad q"])
    calls = _patch_plan(monkeypatch, plan)

    async def flaky_run_agent(question, thread_id=None):
        if question == "bad q":
            raise RuntimeError("mcp exploded")
        return "a finding"

    monkeypatch.setattr(multi_mod, "run_agent", flaky_run_agent)
    reply = await multi_mod.run_multi_agent("q")

    assert reply == "synthesized"
    synth_input = calls[-1]["messages"][-1]["content"]
    assert "FAILED: mcp exploded" in synth_input


async def test_run_chat_dispatches_on_agent_mode(monkeypatch):
    import agents.multi as multi_mod
    from config import settings

    async def fake_single(question, thread_id=None):
        return "single"

    async def fake_multi(question, thread_id=None):
        return "multi"

    monkeypatch.setattr(multi_mod, "run_agent", fake_single)
    monkeypatch.setattr(multi_mod, "run_multi_agent", fake_multi)

    monkeypatch.setattr(settings, "agent_mode", "single")
    assert await multi_mod.run_chat("q") == "single"
    monkeypatch.setattr(settings, "agent_mode", "multi")
    assert await multi_mod.run_chat("q") == "multi"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_multi.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agents.multi'`.

- [ ] **Step 3: Append the prompts to `llm/prompts.py`**

```python
PLANNER_SYSTEM_PROMPT = """You decompose research questions for a paper-research assistant.

If the question needs only one lookup, mark it simple. Otherwise split it into
1-4 self-contained sub-questions, each answerable independently against arXiv
papers. Sub-questions must not reference each other."""

SYNTHESIZER_SYSTEM_PROMPT = """You combine research findings into one answer.

You get a question and findings for its sub-questions, each carrying [paper_id]
citations. Compose a single coherent answer, keep every citation next to the
claim it supports, and never invent paper ids. If a finding says FAILED, work
with the remaining findings and state what is missing."""
```

- [ ] **Step 4: Implement `agents/multi.py`**

```python
"""Multi-agent supervisor: planner → researcher (per sub-question) → synthesizer.

Enabled with agent_mode=multi (config.py); run_chat is the dispatcher the API
calls. The planner may return simple=True and fall through to the single-agent
loop (which keeps thread memory). Researchers run the existing agent loop
sequentially, each on a fresh single-shot thread — multi mode itself keeps no
conversation memory. A failed sub-question is reported to the synthesizer,
which answers from what remains.
"""

import asyncio
import logging

from pydantic import BaseModel, Field

from agents.graph import run_agent
from config import settings
from llm.base import generate
from llm.prompts import PLANNER_SYSTEM_PROMPT, SYNTHESIZER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class Plan(BaseModel):
    simple: bool = Field(description="True when the question needs no decomposition")
    sub_questions: list[str] = Field(default_factory=list, max_length=4)


def _plan(question: str) -> Plan:
    resp = generate([{"role": "user", "content": question}],
                    system=PLANNER_SYSTEM_PROMPT, structured_schema=Plan)
    return resp.parsed


def _synthesize(question: str, findings: list[tuple[str, str]]) -> str:
    parts = [f"Sub-question: {sq}\nFinding: {answer}" for sq, answer in findings]
    content = f"Question: {question}\n\n" + "\n\n---\n\n".join(parts)
    resp = generate([{"role": "user", "content": content}],
                    system=SYNTHESIZER_SYSTEM_PROMPT)
    return resp.text


async def run_multi_agent(question: str, thread_id: str | None = None) -> str:
    plan = await asyncio.to_thread(_plan, question)
    if plan.simple or not plan.sub_questions:
        return await run_agent(question, thread_id)
    findings: list[tuple[str, str]] = []
    for sub_question in plan.sub_questions[:4]:
        try:
            findings.append((sub_question, await run_agent(sub_question)))
        except Exception as exc:
            logger.exception("Researcher failed for %r", sub_question)
            findings.append((sub_question, f"FAILED: {exc}"))
    return await asyncio.to_thread(_synthesize, question, findings)


async def run_chat(message: str, thread_id: str | None = None) -> str:
    """Dispatch on agent_mode: the single loop (default) or the supervisor."""
    if settings.agent_mode == "multi":
        return await run_multi_agent(message, thread_id)
    return await run_agent(message, thread_id)
```

- [ ] **Step 5: Point the API at the dispatcher**

In `api/main.py`, replace the import and the call:

```python
from agents.multi import run_chat
```

(drop the now-unused `from agents.graph import run_agent` import), and in the chat route:

```python
    reply = await run_chat(req.message, thread_id)
```

In `tests/test_api.py`, both chat tests patch the new symbol — replace `monkeypatch.setattr(api_main, "run_agent", fake_run_agent)` with `monkeypatch.setattr(api_main, "run_chat", fake_run_agent)` in `test_chat_generates_thread_id` and `test_chat_reuses_given_thread_id` (the fake's signature already matches).

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_multi.py tests/test_api.py -v`
Expected: PASS.

Run: `uv run pytest`
Expected: PASS (full suite).

- [ ] **Step 7: Commit**

```bash
git add agents/multi.py llm/prompts.py api/main.py tests/test_multi.py tests/test_api.py
git commit -m "feat: multi-agent supervisor mode behind agent_mode flag"
```

---

### Task 12: Integration tests + README

**Files:**
- Create: `tests/test_integration_phase2.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: everything above. Integration tests need real keys / Qdrant / network — they document the keys-arrive-later validation pass and run with `uv run pytest -m integration`.

- [ ] **Step 1: Write the integration tests**

Create `tests/test_integration_phase2.py`:

```python
"""Phase-2 integration tests: real Qdrant, real models, real API keys.

Run: uv run pytest -m integration tests/test_integration_phase2.py
test_rerank_real_model needs only network (model download), no keys.
"""

import pytest

pytestmark = pytest.mark.integration


def test_rerank_real_model_orders_by_relevance():
    from rag.rerank import rerank
    from rag.store import ScoredChunk

    chunks = [
        ScoredChunk(paper_id="a", title="T", score=0.5,
                    text="The capital of France is Paris."),
        ScoredChunk(paper_id="b", title="T", score=0.5,
                    text="Self-attention relates positions of a sequence to compute representations."),
        ScoredChunk(paper_id="c", title="T", score=0.5,
                    text="Bananas are rich in potassium."),
    ]
    result = rerank("How does self-attention work in transformers?", chunks, top_k=2)
    assert result[0].paper_id == "b"


def test_hybrid_round_trip():
    """Migrate → ingest → retrieve in all three modes against live Qdrant."""
    from config import settings
    from rag.ingest import ingest_query
    from rag.migrate import migrate
    from rag.retrieve import retrieve

    migrate()
    result = ingest_query("attention is all you need", max_results=1)
    assert result.ingested

    for mode in ("dense", "sparse", "hybrid"):
        settings.retrieval_mode = mode
        chunks = retrieve("what is multi-head attention?", top_k=3)
        assert chunks, f"no results in {mode} mode"


async def test_memory_two_turns():
    from agents.graph import run_agent

    reply1 = await run_agent("Fetch and summarize 'Attention is All You Need'.",
                             thread_id="it-memory")
    assert reply1
    reply2 = await run_agent("What did I just ask you about?", thread_id="it-memory")
    assert "attention" in reply2.lower()


async def test_multi_agent_e2e():
    from agents.multi import run_multi_agent

    reply = await run_multi_agent(
        "Compare the transformer architecture with BERT's pretraining objectives.")
    assert len(reply) > 50


def test_ablation_smoke():
    """One full ablation sweep; scores land in [0,1] / [0,5] ranges."""
    from eval.run import run_ablation

    report = run_ablation()
    assert set(report["presets"]) == {"baseline-dense", "sparse", "hybrid",
                                      "hybrid+rerank", "full"}
    for summary in report["presets"].values():
        assert 0.0 <= summary["avg_precision"] <= 1.0
        assert 0.0 <= summary["avg_faithfulness"] <= 5.0
```

- [ ] **Step 2: Verify collection behavior**

Run: `uv run pytest tests/test_integration_phase2.py`
Expected: `5 deselected` (integration marker excluded by default) — proves the file doesn't leak into unit runs.

- [ ] **Step 3: Update `README.md`**

Replace the `## Use` and `## Layout` sections with:

```markdown
## Use

```bash
# Web UI (chat + ingest) at http://localhost:8000
uv run uvicorn api.main:app --reload

# Ingest from the CLI
uv run python -c "from rag.ingest import ingest_query; print(ingest_query('attention is all you need'))"

# Offline eval -> report.json + printed summary
uv run python -m eval.run

# Retrieval ablation: golden dataset across dense/sparse/hybrid/rerank/rewrite presets
uv run python -m eval.run --ablation

# Upgrading from phase 1? The collection schema changed (named dense+sparse
# vectors) — recreate it and re-ingest:
uv run python -m rag.migrate --yes
```

Retrieval is a staged pipeline — `[rewrite] → embed → search (dense|sparse|hybrid) → [rerank]` —
controlled by `.env` flags (`RETRIEVAL_MODE`, `RERANK_ENABLED`, `REWRITE_ENABLED`; see `config.py`).
BM25 sparse search and reranking run on local ONNX models — no API keys needed.
Chat is multi-turn: the UI carries a `thread_id`, history is checkpointed to
`data/checkpoints.db`, long conversations get summarized. `AGENT_MODE=multi`
switches to a planner → researcher → synthesizer supervisor.

## Layout

- `llm/` — provider abstraction (Anthropic + OpenAI), prompts, structured output, prompt caching
- `rag/` — arXiv fetch, PDF parse, chunk, embed (dense + BM25 sparse), Qdrant store (hybrid RRF),
  rerank, query rewrite, retrieve, answer, migrate
- `agents/` — LangGraph agent with SQLite-checkpointed memory; multi-agent supervisor (`agents/multi.py`);
  custom MCP server (`python -m agents.mcp_server`); MCP client (also consumes `mcp-server-fetch`)
- `eval/` — golden dataset, LLM judge, retrieval metrics, report generator, ablation mode
- `api/` — FastAPI routes + static frontend

Imports flow one way: `api → agents → rag/llm`; `eval → rag/agents/llm`.
```

- [ ] **Step 4: Full suite + commit**

Run: `uv run pytest`
Expected: PASS, integration tests deselected.

```bash
git add tests/test_integration_phase2.py README.md
git commit -m "test: phase-2 integration suite; docs: phase-2 README"
```

---

## Post-plan validation (when API keys arrive)

Not tasks — the recorded keys-later debt, in order:

1. `cp .env.example .env` + real `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`.
2. `docker compose up -d`, `uv run python -m rag.migrate --yes`, re-ingest the golden-dataset papers.
3. `uv run pytest -m integration` — including the pre-existing phase-1 integration tests.
4. `uv run python -m eval.run --ablation` — the ablation table is the phase-2 proof: each preset should move retrieval precision/recall (and downstream judge scores) vs `baseline-dense`.
5. Flip `rewrite_enabled` default to `True` if the `full` preset wins.
