from config import settings
from rag.embed import embed_query
from rag.store import ScoredChunk, VectorStore


def retrieve(question: str, top_k: int | None = None,
             store: VectorStore | None = None) -> list[ScoredChunk]:
    """Embed the question and return the top-k chunks from Qdrant."""
    store = store or VectorStore()
    return store.search(embed_query(question), top_k=top_k or settings.retrieval_top_k)
