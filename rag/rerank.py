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
