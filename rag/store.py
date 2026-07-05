import uuid

from pydantic import BaseModel
from qdrant_client import QdrantClient, models

from config import settings


class ChunkRecord(BaseModel):
    paper_id: str
    title: str
    chunk_index: int
    text: str
    section: str = ""  # kept for schema parity; pypdf gives no reliable sections
    vector: list[float]


class ScoredChunk(BaseModel):
    paper_id: str
    title: str
    text: str
    score: float


class VectorStore:
    """Thin wrapper around Qdrant for paper chunks."""

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

    def ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=settings.embedding_dim, distance=models.Distance.COSINE
                ),
            )

    def upsert_chunks(self, records: list[ChunkRecord]) -> None:
        points = [
            models.PointStruct(
                # uuid5 of paper_id:chunk_index → re-ingesting a paper overwrites
                # its old points instead of duplicating them.
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{r.paper_id}:{r.chunk_index}")),
                vector=r.vector,
                payload={"paper_id": r.paper_id, "title": r.title,
                         "chunk_index": r.chunk_index, "chunk_text": r.text,
                         "section": r.section},
            )
            for r in records
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, vector: list[float], top_k: int | None = None) -> list[ScoredChunk]:
        top_k = top_k or settings.retrieval_top_k
        hits = self.client.query_points(
            collection_name=self.collection, query=vector, limit=top_k
        ).points
        return [
            ScoredChunk(paper_id=h.payload["paper_id"], title=h.payload["title"],
                        text=h.payload["chunk_text"], score=h.score)
            for h in hits
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
