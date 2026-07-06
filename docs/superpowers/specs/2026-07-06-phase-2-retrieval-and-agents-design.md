# Paper Research Assistant — Phase 2 Design: Retrieval Quality + Agent Depth

## Purpose

Phase 2 deepens the learning project along two tracks, in order:

1. **Retrieval quality** — hybrid search, reranking, query rewriting; each config-flagged and measurable against the phase-1 baseline via an eval ablation mode.
2. **Agent depth** — multi-turn conversation memory (LangGraph checkpointing + summarization) and a multi-agent supervisor mode (planner → researcher → synthesizer).

Retrieval lands first because agents consume retrieval; the agent track builds on the improved pipeline.

**Key constraint:** real API keys arrive later. Everything is built now with mocked unit tests (phase-1 style). Hybrid search (BM25 sparse) and reranking (local cross-encoder) run fully keyless; query rewriting, dense embeddings, summarization, and the eval ablation run need keys and get one real validation pass when keys arrive.

## Track 1: Retrieval Quality

New dependency: `fastembed` (ONNX runtime — local models, no torch, no API keys).

### Hybrid search

- Collection schema moves to **named vectors**: `dense` (OpenAI `text-embedding-3-small`, existing) + `bm25` sparse (`Qdrant/bm25` model via fastembed, `Modifier.IDF` so Qdrant applies IDF server-side).
- Ingest writes both vectors per chunk.
- Query path: Qdrant Query API — prefetch on dense and sparse, RRF fusion server-side.
- **Breaking schema change.** The collection must be recreated and papers re-ingested (PDFs are cached in `data/pdfs`; re-ingest re-parses and re-embeds — dense re-embedding costs OpenAI tokens). A small `python -m rag.migrate` script recreates the collection; `ensure_collection`/startup detects the legacy unnamed-vector schema and raises a clear error naming the migrate command.

### Reranking

- fastembed `TextCrossEncoder`, model `Xenova/ms-marco-MiniLM-L-6-v2` (~80MB, downloaded on first use, local inference).
- Over-fetch `rerank_candidates` (default 20) from the store, rerank against the user question, keep `retrieval_top_k` (default 5).

### Query rewriting

- LLM structured output via existing `llm/base.generate`: user question → one rewritten search query, used for embedding/retrieval only (the original question still drives answer generation).
- Off by default (needs keys). On LLM failure: log a warning, fall back to the original question — retrieval never breaks because rewriting broke.

### Config flags (config.py, existing pydantic-settings style)

```python
retrieval_mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
rerank_enabled: bool = True
rerank_candidates: int = 20
rewrite_enabled: bool = False
```

`sparse` mode is a free bonus: BM25-only retrieval works fully keyless and is demo-able immediately.

### Pipeline

question → [rewrite] → embed (dense and/or sparse per mode) → Qdrant query (RRF fusion if hybrid) → [rerank] → top_k chunks → grounded prompt (unchanged from phase 1).

### Modules

- `rag/sparse.py` — new: BM25 sparse embedding via fastembed
- `rag/rerank.py` — new: cross-encoder reranking
- `rag/migrate.py` — new: collection recreation script
- `rag/store.py` — changed: named dense+sparse vectors, hybrid query with RRF prefetch, legacy-schema detection
- `rag/retrieve.py` — changed: staged pipeline honoring the config flags
- `rag/ingest.py` — changed: compute and upsert both vectors

## Track 1b: Eval Ablation Mode

`python -m eval.run --ablation` sweeps the golden dataset across config presets:

| preset | retrieval_mode | rerank | rewrite |
|---|---|---|---|
| baseline-dense | dense | off | off |
| sparse | sparse | off | off |
| hybrid | hybrid | off | off |
| hybrid+rerank | hybrid | on | off |
| full | hybrid | on | on |

Report (`report.json` + printed table) shows retrieval precision/recall and judge scores side by side per preset. This is the retrieval track's proof of value. Existing single-config mode stays the default. Requires keys (dense embeddings + LLM judge); until then, unit tests mock the pipeline and verify preset wiring.

## Track 2: Agent Memory + Checkpointing

New dependency: `langgraph-checkpoint-sqlite`.

- **Checkpointing:** `AsyncSqliteSaver` at `data/checkpoints.db`, passed into `build_graph(toolbox, checkpointer)`. `run_agent(question, thread_id=None)` — a fresh UUID is generated when omitted, so direct callers (eval, tests) keep working single-shot — invokes the graph with `config={"configurable": {"thread_id": ...}}`. Same thread → LangGraph restores prior state; the conversation continues.
- **API:** `/api/chat` accepts optional `thread_id`; the server generates a UUID when absent and returns it in the response.
- **Frontend:** keeps `thread_id` in a JS variable, sends it with subsequent messages, and gets a "New conversation" button that clears it. No conversation-list UI.
- **Summarization:** `AgentState` gains `summary: str`. A `summarize` node runs when the message count exceeds `memory_max_messages` (new setting, default 20): the LLM compresses older turns into a running summary; recent messages are kept verbatim; the summary is injected into the system prompt. Needs keys live; mocked in unit tests.

## Track 3: Multi-Agent Mode

New setting: `agent_mode: Literal["single", "multi"] = "single"`. Single is the untouched phase-1 loop and stays the default.

Multi = supervisor pattern, three stages:

1. **Planner** — LLM structured output: question → 1–4 sub-questions, or a "simple question" verdict that falls through to the single-agent loop.
2. **Researcher** — the existing agent loop (RAG + MCP tools) executed per sub-question, sequentially (rate-limit friendly, easy to trace).
3. **Synthesizer** — LLM composes the final cited answer from researcher outputs. It is told about any failed sub-question and answers from what it has.

`agent_mode` becomes another eval ablation row later (single vs multi on the golden dataset).

## Error Handling

- Query rewrite LLM failure → log warning, use the original question.
- Reranker model load failure → explicit error, no silent fallback (learning project: explicit beats magic).
- Researcher sub-question failure → synthesizer is informed and continues with the remaining results.
- Legacy collection schema → clear startup error naming `python -m rag.migrate`.
- Existing phase-1 policies (retry with backoff, Qdrant fail-fast, MCP tool errors returned to the agent) unchanged.

## Testing

- **Unit (keyless, mocked LLM/Qdrant):** sparse embed output shape, hybrid query construction, rerank ordering with a mocked cross-encoder, rewrite fallback path, planner/synthesizer prompt rendering and structured-output parsing, migrate legacy-schema detection, eval preset wiring.
- **Checkpointer:** tested against a real SQLite file in a pytest tmp path — local, no mocking needed.
- **Integration (`@pytest.mark.integration`, real APIs/models):** hybrid ingest→query round trip, real cross-encoder rerank, multi-turn memory across two `/api/chat` calls, multi-agent E2E, full ablation smoke run.

## Out of Scope (Phase 2)

- Streaming (SSE) and async pipeline rework
- Observability/tracing (Langfuse etc.), token/cost tracking
- Conversation-list UI, auth, deployment
- Parallel researcher execution
- Automatic provider fallback (unchanged from phase 1)
