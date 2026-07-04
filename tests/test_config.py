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
