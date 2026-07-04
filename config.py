from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration. Values come from env vars / .env (12-factor style)."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # API keys (empty default so unit tests never need real keys)
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # LLM
    llm_provider: Literal["anthropic", "openai"] = "anthropic"
    anthropic_model: str = "claude-opus-4-8"
    openai_model: str = "gpt-5"
    llm_max_tokens: int = 4096
    llm_max_retries: int = 4  # SDK retries 429/5xx with exponential backoff

    # Embeddings
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # Vector store
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "papers"

    # RAG
    chunk_max_tokens: int = 500
    chunk_overlap_tokens: int = 50
    retrieval_top_k: int = 5
    pdf_dir: str = "data/pdfs"

    # Agent
    agent_max_steps: int = 8


settings = Settings()
