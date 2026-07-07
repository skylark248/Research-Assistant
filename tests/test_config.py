def test_defaults():
    from config import Settings

    s = Settings(_env_file=None)
    assert s.llm_provider == "anthropic"
    assert s.anthropic_model == "claude-opus-4-8"
    assert s.embedding_model == "text-embedding-3-small"
    assert s.embedding_dim == 1536
    assert s.qdrant_url == "http://localhost:6333"
    assert s.qdrant_collection == "papers"
    assert s.chunk_max_tokens == 500
    assert s.chunk_overlap_tokens == 50
    assert s.retrieval_top_k == 5


def test_env_override(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("RETRIEVAL_TOP_K", "9")
    from config import Settings

    s = Settings(_env_file=None)
    assert s.llm_provider == "openai"
    assert s.retrieval_top_k == 9


def test_phase2_defaults():
    from config import Settings

    s = Settings(_env_file=None)
    assert s.retrieval_mode == "hybrid"
    assert s.rerank_enabled is True
    assert s.rerank_candidates == 20
    assert s.rerank_model == "Xenova/ms-marco-MiniLM-L-6-v2"
    assert s.sparse_model == "Qdrant/bm25"
    assert s.rewrite_enabled is False
    assert s.agent_mode == "single"
    assert s.checkpoint_db == "data/checkpoints.db"
    assert s.memory_max_messages == 20
    assert s.memory_keep_messages == 8
