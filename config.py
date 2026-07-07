from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration. Values come from env vars / .env (12-factor style)."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # API keys (empty default so unit tests never need real keys)
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # LLM
    llm_provider: Literal["anthropic", "openai", "local"] = "anthropic"
    anthropic_model: str = "claude-opus-4-8"
    openai_model: str = "gpt-5"
    llm_max_tokens: int = 4096
    llm_max_retries: int = 4  # SDK retries 429/5xx with exponential backoff

    # Local LLM (Ollama's OpenAI-compatible endpoint; qwen2.5:3b fits an 8GB M1)
    local_base_url: str = "http://localhost:11434/v1"
    local_model: str = "qwen2.5:3b"

    # Embeddings
    embedding_provider: Literal["openai", "local"] = "openai"
    embedding_model: str = "text-embedding-3-small"
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"  # fastembed ONNX, 384-dim
    embedding_dim: int | None = None  # derived from provider below; explicit value wins

    @model_validator(mode="after")
    def _derive_embedding_dim(self) -> "Settings":
        if self.embedding_dim is None:
            self.embedding_dim = 1536 if self.embedding_provider == "openai" else 384
        return self

    # Vector store
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "papers"

    # RAG
    chunk_max_tokens: int = 500
    chunk_overlap_tokens: int = 50
    retrieval_top_k: int = 5
    pdf_dir: str = "data/pdfs"

    # Retrieval pipeline (phase 2) — every stage is a flag so eval ablation
    # can isolate each technique's effect.
    retrieval_mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    rerank_enabled: bool = True
    rerank_candidates: int = 20  # over-fetch size fed to the reranker
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    sparse_model: str = "Qdrant/bm25"
    rewrite_enabled: bool = False  # needs an LLM key; off until keys exist

    # Agent
    agent_max_steps: int = 8
    agent_mode: Literal["single", "multi"] = "single"

    # Memory (phase 2)
    checkpoint_db: str = "data/checkpoints.db"
    memory_max_messages: int = 20  # summarize when history exceeds this
    memory_keep_messages: int = 8  # recent messages kept verbatim in the prompt


settings = Settings()
