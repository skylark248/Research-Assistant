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
