# Paper Research Assistant — Phase 3 Design: Local LLM Provider

## Purpose

Add a fully local execution path: a third LLM provider ("local", Qwen 2.5 via Ollama) and a local dense-embedding provider (fastembed), so the entire system — ingest, chat, agent memory, multi-agent, eval ablation — runs with zero API keys on the development machine (M1 MacBook Air, 8 GB RAM).

Cloud providers stay untouched and remain the defaults; local is opt-in via `.env`. No automatic fallback between providers (phase-1 decision stands).

## Constraints

- **Hardware:** M1 Air, 8 GB unified memory, with macOS (~2–3 GB), Qdrant in Docker, and uvicorn already resident. Default model must fit comfortably: `qwen2.5:3b` (~1.9 GB Q4). `qwen2.5:7b` (~4.7 GB) documented as opt-in only.
- **Runtime:** Ollama (`brew install ollama`), consumed through its OpenAI-compatible `/v1` endpoint — no new Python dependency for the LLM path.
- **Context window:** Ollama's default context is too small for grounded prompts (top_k=5 × ~500-token chunks + system prompts ≈ 4k tokens). README documents `OLLAMA_CONTEXT_LENGTH=8192` for the Ollama server.

## Track 1: LLM provider "local"

Reuse the existing OpenAI adapter — Ollama's `/v1` endpoint speaks the same protocol, including function calling and `response_format` structured output.

- `config.py`:
  - `llm_provider: Literal["anthropic", "openai", "local"]` (default unchanged: "anthropic")
  - `local_base_url: str = "http://localhost:11434/v1"`
  - `local_model: str = "qwen2.5:3b"`
- `llm/openai_client.py`: `generate_openai(...)` gains optional `client` and `model` parameters; omitted → exactly current behavior. No call-site changes for the cloud path.
- `llm/local_client.py` (new, thin): lazy-singleton OpenAI SDK client with `base_url=settings.local_base_url`, `api_key="ollama"` (Ollama ignores the key but the SDK requires one); `generate_local(...)` delegates to `generate_openai(client=..., model=settings.local_model, ...)`.
- `llm/base.py`: `generate(...)` dispatches `"local"` to `generate_local`.

Every downstream consumer (agent loop, judge, query rewrite, summarization, planner, synthesizer) flows through `generate()` and needs zero changes.

**Known quality risks (accepted, stated):** a 3B model adheres to JSON schemas and citation formats less reliably than cloud models. Failure behavior follows existing per-call policy: rewrite fails open to the original question; judge/planner validation errors fail loud. No new retry machinery.

## Track 2: Local dense embeddings

- `config.py`:
  - `embedding_provider: Literal["openai", "local"] = "openai"`
  - `local_embedding_model: str = "BAAI/bge-small-en-v1.5"` (fastembed ONNX, ~130 MB, 384-dim)
  - `embedding_dim` becomes provider-derived (1536 for openai, 384 for local) with explicit `.env` override still honored.
- `rag/embed.py`: provider switch inside `embed_texts`/`embed_query`; local path uses a lazy-singleton fastembed `TextEmbedding` — same pattern as `rag/sparse.py` and `rag/rerank.py`. fastembed is already a dependency; no new packages.
- Switching embedding providers changes vector dimensions → collection schema mismatch → existing `python -m rag.migrate --yes` + re-ingest flow applies. `VectorStore.check_schema` is extended to also compare the collection's dense-vector size against `settings.embedding_dim`, so a provider switch fails fast at startup with the migrate message instead of a raw Qdrant error at upsert/search time. README documents the switch procedure.

## Keyless end-to-end story

`.env` with `LLM_PROVIDER=local` + `EMBEDDING_PROVIDER=local` gives, with no API keys:

- ingest (arXiv fetch → parse → chunk → local dense + BM25 sparse → Qdrant)
- chat with citations, multi-turn memory, summarization
- multi-agent mode
- `python -m eval.run` and `--ablation` (judge runs on the local model)

Cross-provider eval comparison (cloud judge vs local judge) is explicitly out of scope for phase 3 — the ablation compares retrieval presets under whatever provider is configured.

## Testing

- **Unit (keyless, offline, default run):** provider dispatch (`"local"` → `generate_local`), local client construction (base_url/model wiring, mocked SDK), `generate_openai` client/model parameterization (cloud defaults preserved), embed provider switch + dim derivation (mocked fastembed), config defaults.
- **New pytest marker `local`:** tests needing a running Ollama but no keys — one real `generate()` round trip, one structured-output parse, one tool-call loop against qwen2.5:3b, one fastembed real-embedding shape check. Deselected by default alongside `integration` (`addopts = "-m 'not integration and not local'"`); run via `pytest -m local`. This makes real-model validation possible on this machine today, unlike the keys-blocked `integration` suite.
- Existing `integration` marker semantics unchanged (real cloud APIs + keys).

## README

New "Fully local (no API keys)" section: `brew install ollama`, `ollama pull qwen2.5:3b`, `OLLAMA_CONTEXT_LENGTH=8192 ollama serve`, `.env` flags, migrate + re-ingest note for the embedding switch, RAM guidance (3b default; 7b only with Docker/browser closed), `pytest -m local`.

## Out of Scope

- Automatic provider fallback/routing (unchanged phase-1 decision)
- MLX / llama.cpp / LM Studio runtimes
- Cross-provider eval comparison harness
- Local reranker/sparse changes (already local since phase 2)
- Streaming, observability (still phase-4+ candidates)
