"""Full round trip against real APIs. Requires: docker compose up -d, real keys in .env.

Run: uv run pytest tests/test_integration_rag.py -m integration -v
"""

import pytest

pytestmark = pytest.mark.integration


def test_ingest_then_query_round_trip():
    from rag.answer import answer_question
    from rag.arxiv_client import get_paper
    from rag.ingest import ingest_paper
    from rag.store import VectorStore

    store = VectorStore()
    store.ping()
    store.ensure_collection()

    meta = get_paper("1706.03762")
    assert meta is not None
    n = ingest_paper(meta, store)
    assert n is not None  # ingested now or already present

    result = answer_question("What attention mechanism does the Transformer use?", store=store)
    assert "1706.03762" in result.sources
    assert "[1706.03762]" in result.text  # inline citation present
