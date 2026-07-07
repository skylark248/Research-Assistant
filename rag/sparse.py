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
