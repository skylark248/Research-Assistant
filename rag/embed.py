from openai import OpenAI

from config import settings

_client: OpenAI | None = None

BATCH_SIZE = 100  # texts per embeddings API request


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key or None, max_retries=settings.llm_max_retries)
    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts in order, batching requests."""
    if not texts:
        return []
    client = _get_client()
    vectors: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        resp = client.embeddings.create(model=settings.embedding_model, input=batch)
        vectors.extend(d.embedding for d in resp.data)
    return vectors


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
