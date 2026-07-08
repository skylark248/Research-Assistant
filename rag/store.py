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

DIM_MISMATCH_MESSAGE = (
    "Collection '{collection}' stores {found}-dim dense vectors but the configured "
    "embedding provider expects {expected}-dim. Recreate it with: "
    "uv run python -m rag.migrate --yes (then re-ingest papers)"
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
        dense = params.vectors[DENSE_VECTOR]
        if dense.size != settings.embedding_dim:
            raise RuntimeError(DIM_MISMATCH_MESSAGE.format(
                collection=self.collection, found=dense.size,
                expected=settings.embedding_dim,
            ))

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
