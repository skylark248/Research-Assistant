# Paper Research Assistant

Learning project covering LLM APIs + prompting, RAG, evaluation, and agents + MCP.
Ingests arXiv papers, answers questions grounded in them with [paper_id] citations,
and autonomously fetches papers it doesn't have yet.

## Setup

```bash
uv sync
cp .env.example .env   # add ANTHROPIC_API_KEY and OPENAI_API_KEY
docker compose up -d   # Qdrant on localhost:6333
```

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

Retrieval is a staged pipeline — `[rewrite] → embed → search (dense|sparse|hybrid) → [rerank]` —
controlled by `.env` flags (`RETRIEVAL_MODE`, `RERANK_ENABLED`, `REWRITE_ENABLED`; see `config.py`).
BM25 sparse search and reranking run on local ONNX models — no API keys needed.
Chat is multi-turn: the UI carries a `thread_id`, history is checkpointed to
`data/checkpoints.db`, long conversations get summarized. `AGENT_MODE=multi`
switches to a planner → researcher → synthesizer supervisor.

## Tests

```bash
uv run pytest                  # unit tests (mocked, no keys needed)
uv run pytest -m integration   # real APIs; needs keys, Qdrant, network, uvx
```

## Layout

- `llm/` — provider abstraction (Anthropic + OpenAI), prompts, structured output, prompt caching
- `rag/` — arXiv fetch, PDF parse, chunk, embed (dense + BM25 sparse), Qdrant store (hybrid RRF),
  rerank, query rewrite, retrieve, answer, migrate
- `agents/` — LangGraph agent with SQLite-checkpointed memory; multi-agent supervisor (`agents/multi.py`);
  custom MCP server (`python -m agents.mcp_server`); MCP client (also consumes `mcp-server-fetch`)
- `eval/` — golden dataset, LLM judge, retrieval metrics, report generator, ablation mode
- `api/` — FastAPI routes + static frontend

Imports flow one way: `api → agents → rag/llm`; `eval → rag/agents/llm`.
