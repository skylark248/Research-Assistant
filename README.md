# Paper Research Assistant

Learning project covering LLM APIs + prompting, RAG, evaluation, and agents + MCP.
Ingests arXiv papers, answers questions grounded in them with [paper_id] citations,
and autonomously fetches papers it doesn't have yet.

## Status: complete (all 5 phases shipped)

| Phase | Delivered |
|-------|-----------|
| 1 | Core RAG pipeline (ingest → chunk → embed → Qdrant → cited answers), LangGraph agent + MCP tools, eval harness, web UI |
| 2 | Hybrid retrieval (BM25 + dense + RRF), cross-encoder reranking, query rewriting, agent memory (summarization), multi-agent supervisor, retrieval ablation |
| 3 | Fully local, no-keys operation via Ollama (`qwen2.5:3b`) + local embeddings; live-validated end to end |
| 4 | Per-request provider toggle in the UI, provider availability checks, SSE streaming (activity + tokens), citation chips, persistent thread sidebar |
| 5 | Corrective RAG (LLM chunk grading + one rewritten-query retry + honest degradation) and a citation-faithfulness guardrail with an "unverified citations" badge in the UI |

Design specs and implementation plans for each phase live in
`docs/superpowers/specs/` and `docs/superpowers/plans/`. Known limitation:
the 3B local model occasionally fumbles the agent tool-call loop — use
`LOCAL_MODEL=qwen2.5:7b` if you have the RAM (see notes below).

## Setup

```bash
uv sync
cp .env.example .env   # add ANTHROPIC_API_KEY and OPENAI_API_KEY
docker compose up -d   # Qdrant on localhost:6333
```

## Fully local (no API keys)

Runs the whole system — ingest, cited chat, memory, multi-agent, eval — on a
local model. Fits an 8GB M1 MacBook Air.

```bash
brew install ollama
ollama pull qwen2.5:3b                      # ~1.9GB
OLLAMA_CONTEXT_LENGTH=8192 ollama serve     # grounded prompts need >4k context
```

Set in `.env`: `LLM_PROVIDER=local` and `EMBEDDING_PROVIDER=local`.

Switching the embedding provider changes vector dimensions (1536 → 384), so
recreate the collection and re-ingest: `uv run python -m rag.migrate --yes`.

Notes: `qwen2.5:3b` is the safe default — `qwen2.5:7b` (~4.7GB) answers better
but only fits with Docker/browser mostly closed. A 3B model follows citation
and JSON-schema instructions less reliably than the cloud models; the eval
harness quantifies the gap. Real-model tests: `uv run pytest -m local`.

## Use

```bash
# Web UI (chat + ingest) at http://localhost:8000
uv run uvicorn api.main:app --reload

# Ingest from the CLI
uv run python -c "from rag.ingest import ingest_query; print(ingest_query('attention is all you need'))"

# Offline eval -> report.json + printed summary
uv run python -m eval.run

# Retrieval ablation: golden dataset across dense/sparse/hybrid/rerank/rewrite presets
uv run python -m eval.run --ablation

# Upgrading from phase 1? The collection schema changed (named dense+sparse
# vectors) — recreate it and re-ingest:
uv run python -m rag.migrate --yes
```

Retrieval is a staged pipeline — `[rewrite] → embed → search (dense|sparse|hybrid) → [rerank] → [grade → retry once]` —
controlled by `.env` flags (`RETRIEVAL_MODE`, `RERANK_ENABLED`, `REWRITE_ENABLED`, `GRADING_ENABLED`; see `config.py`).
After answering, `FAITHFULNESS_ENABLED` runs a citation-faithfulness check; an
unsupported answer gets an "⚠ citations unverified" badge in the UI (live
responses only — verdicts aren't persisted, so restored threads never show it).
Both guardrails fail open and add 1–3 LLM calls per request — turn them off on
slow local models if latency hurts.
Chat is multi-turn: the UI carries a `thread_id`, history is checkpointed to
`data/checkpoints.db`, long conversations get summarized. `AGENT_MODE=multi`
switches to a planner → researcher → synthesizer supervisor.

The web UI (phase 4):
- **Per-request provider toggle** — dropdown switches reasoning between
  Anthropic / OpenAI / local Ollama per message; `GET /api/providers` grays out
  providers with no key or no reachable Ollama.
- **Streaming** — replies arrive over SSE (`POST /api/chat/stream`): live agent
  activity ("calling rag_query…") plus token-by-token text.
- **Citations** — chips under each reply link the arXiv papers the answer drew on.
- **Thread sidebar** — past conversations persist (list / restore / delete);
  markdown rendering via vendored `marked` + `DOMPurify` (works offline, no build step).

Keyless demo: set `RETRIEVAL_MODE=sparse` — BM25-only retrieval, no OpenAI key.
First reranked query downloads the ~80MB cross-encoder to the local cache.

## Tests

```bash
uv run pytest                  # unit tests (mocked, no keys needed)
uv run pytest -m integration   # real cloud APIs; needs keys, Qdrant, network, uvx
uv run pytest -m local         # real local model; needs Ollama running, no keys
```

## Layout

- `llm/` — provider abstraction (Anthropic + OpenAI + local/Ollama), streaming, prompts,
  structured output, prompt caching
- `rag/` — arXiv fetch, PDF parse, chunk, embed (dense + BM25 sparse), Qdrant store (hybrid RRF),
  rerank, query rewrite, retrieve, answer, migrate
- `agents/` — LangGraph agent with SQLite-checkpointed memory; multi-agent supervisor (`agents/multi.py`);
  custom MCP server (`python -m agents.mcp_server`); MCP client (also consumes `mcp-server-fetch`)
- `eval/` — golden dataset, LLM judge, retrieval metrics, report generator, ablation mode
- `api/` — FastAPI routes (chat, SSE stream, ingest, providers, threads) + static frontend

Imports flow one way: `api → agents → rag/llm`; `eval → rag/agents/llm`.
