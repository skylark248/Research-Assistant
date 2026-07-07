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
