"""Dense embeddings: OpenAI (default) or local fastembed, per settings.embedding_provider.

Switching providers changes the vector dimension (1536 vs 384) — the Qdrant
collection must be recreated (python -m rag.migrate --yes) and papers re-ingested.
"""

from fastembed import TextEmbedding
from openai import OpenAI

from config import settings

_client: OpenAI | None = None
_local_model: TextEmbedding | None = None

BATCH_SIZE = 100  # texts per embeddings API request


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key or None, max_retries=settings.llm_max_retries)
    return _client


def _get_local_model() -> TextEmbedding:
    global _local_model
    if _local_model is None:
        _local_model = TextEmbedding(model_name=settings.local_embedding_model)
    return _local_model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts in order; provider chosen by settings.embedding_provider."""
    if not texts:
        return []
    if settings.embedding_provider == "local":
        return [vector.tolist() for vector in _get_local_model().embed(texts)]
    client = _get_client()
    vectors: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        resp = client.embeddings.create(model=settings.embedding_model, input=batch)
        vectors.extend(d.embedding for d in resp.data)
    return vectors


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
