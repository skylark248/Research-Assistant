# Paper Research Assistant — Design

## Purpose

Learning project covering four topics in one codebase: LLM APIs + prompting, RAG, evaluation, and agents + MCP. Real-world shape: a research paper assistant that ingests arXiv papers, answers questions grounded in them with citations, and can autonomously fetch new papers it doesn't have yet.

Built via Claude Code; user reads the generated code to learn. Depth target: "solid learning project" — realistic depth on all four topics, not a toy, not portfolio-polished.

## Module Breakdown

Top-level Python packages, each with one job and a thin interface:

- `llm/` — provider abstraction over Anthropic + OpenAI, prompt templates, structured output, prompt caching demo
- `rag/` — arXiv fetch, PDF parsing, chunking, embedding, Qdrant storage, retrieval
- `agents/` — LangGraph agent graph; custom MCP server exposing arXiv tools; MCP client consuming one external server (`mcp-server-fetch`)
- `eval/` — golden Q&A dataset, LLM-as-judge scorer, retrieval metrics, report generator
- `api/` — FastAPI routes (chat, ingest) + minimal static HTML/JS frontend (no build step)
- `tests/` — pytest, mirrors package structure

Modules import from each other in one direction only: `api` → `agents` → `rag`/`llm`; `eval` → `rag`/`agents`/`llm`. No package reaches back into a consumer.

## Data Flow

**Ingest:**
arXiv API search → download PDF → parse text (pypdf) → chunk (recursive splitter, ~500 tokens, overlap) → embed (OpenAI `text-embedding-3-small`) → upsert into Qdrant (id, vector, metadata: paper_id, title, chunk_text, section).

**RAG query:**
user question → embed query → Qdrant top-k retrieve → build grounded prompt (context + citation instructions) → LLM generate → answer with inline `[paper_id]` citations.

**Agent (LangGraph):**
user message → agent node decides: (a) answer from existing RAG store, (b) call custom MCP tool (`arxiv_search` / `arxiv_fetch_paper`) to pull a paper not yet ingested, then re-ingest and retry RAG, or (c) call the external `fetch` MCP tool for e.g. a URL the user pasted → loop until final answer → respond.

**Eval (offline, not runtime):**
golden dataset (question, expected_paper_ids, expected_answer_gist) → run through RAG/agent pipeline → LLM-judge scores faithfulness / relevance / citation-accuracy → retrieval precision/recall computed against expected_paper_ids → `report.json` + printed summary.

## Tech Stack

- Package management: `uv`
- LLM SDKs: `anthropic`, `openai`, behind `llm/base.py` (`generate(messages, tools=None, structured_schema=None)`)
- Prompting techniques demonstrated: system prompt design, few-shot examples (citation format), structured output via Pydantic schema (judge scores, agent decisions), Anthropic prompt caching (`cache_control` on long paper context)
- Embeddings: OpenAI `text-embedding-3-small`
- Vector store: Qdrant, run via `docker-compose.yml`
- Agent orchestration: LangGraph `StateGraph` with a tool-calling node
- MCP: custom server via the `mcp` Python SDK exposing `arxiv_search` and `arxiv_fetch_paper`; consumes the official `mcp-server-fetch` as an external server
- API: FastAPI + minimal vanilla HTML/JS static page
- Eval: no framework — custom LLM-as-judge scorer with structured output, wrapped by pytest for pass/fail thresholds
- Config: `.env` + `pydantic-settings`
- Testing: `pytest`; unit tests mock LLM calls; real-API integration tests behind `@pytest.mark.integration`

## Error Handling

- LLM calls: retry with backoff on rate limits; no automatic provider fallback (explicit user choice only)
- PDF parse failures: skip paper, log, continue batch ingest
- MCP tool call failures: returned to the agent as a tool error result — agent decides retry vs. give up, no silent crash
- Qdrant unavailable: fail fast at startup with a clear message (Docker not running)

## Testing

- Unit: chunking logic, prompt template rendering, MCP tool schemas — pure functions, mocked LLM
- Integration (`@pytest.mark.integration`, real API calls, run manually or opt-in in CI): full ingest→query round trip, agent tool-call loop, MCP server tool invocation
- Eval harness runs standalone (`python -m eval.run`) producing a report; one pytest smoke test asserts scores stay above a floor threshold to catch regressions

## Out of Scope

- Automatic multi-provider fallback/routing
- Auth/multi-user support
- Production deployment (Docker Compose for local dev only)
- Frontend build tooling / framework (React etc.) — plain HTML/JS is enough
