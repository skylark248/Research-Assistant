# Phase 4: Per-Request Provider Toggle, Streaming, Citations, Threads

**Date:** 2026-07-08
**Status:** Approved

## Goal

Final phase. Make the reasoning LLM switchable from the UI per request
(Anthropic / OpenAI / local Ollama), and close the remaining UX gaps:
provider availability checks, streaming responses, source citations, and
persistent conversation threads with a polished frontend.

## Context

`llm/base.py::generate()` already accepts a `provider` override, and all three
clients (`anthropic_client`, `openai_client`, `local_client`) exist. What is
missing is plumbing: no call site passes `provider`, the API has no way to
receive it, and the frontend has no control for it.

## 1. Per-request provider

- `ChatRequest` (and the new stream endpoint) gain
  `provider: Literal["anthropic", "openai", "local"] | None = None`.
  `None` falls back to `settings.llm_provider`, so existing callers and eval
  are unaffected.
- Threading path:
  `api/main.py` → `run_chat(message, thread_id, provider)` →
  - `run_agent(question, thread_id, provider)` →
    `build_graph(toolbox, checkpointer, provider)`; node closures pass
    `provider=` to every `generate()` call (agent node, summarize node).
  - `run_multi_agent(question, thread_id, provider)` → `_plan(question,
    provider)` and `_synthesize(question, findings, provider)`; researchers
    reuse `run_agent(..., provider)`.
- `rag_query`'s answer generation receives the per-request provider, threaded
  through `answer_question(question, store, provider)` into its `generate()`
  call. Only `rag/rewrite.py` stays on the global setting: `rewrite_enabled`
  is off by default and the rewrite runs inside the RAG pipeline, several
  layers below the agent — not worth threading a parameter through.
- Invalid provider strings are rejected with 422 by the `Literal` type.

## 2. Provider status — `GET /api/providers`

Response: list of `{provider, available, detail, model}`.

| Provider  | Availability check                                | `model` field              |
|-----------|---------------------------------------------------|----------------------------|
| anthropic | `settings.anthropic_api_key` non-empty            | `settings.anthropic_model` |
| openai    | `settings.openai_api_key` non-empty               | `settings.openai_model`    |
| local     | `GET {local_base_url}/models`, 1.5 s timeout      | `settings.local_model`     |

- `detail` carries the human-readable reason when unavailable
  ("no API key set", "Ollama unreachable at http://localhost:11434/v1").
- The Ollama probe uses `requests` (already a runtime dependency), executed
  via `run_in_threadpool`; `httpx` stays dev-only.
- Frontend calls this on load: dropdown lists all three providers, disables
  unavailable ones, selects `settings.llm_provider` if alive, else the first
  alive one.
- Choosing an unavailable provider via raw API returns 400 with the `detail`
  message before any agent work starts.

## 3. Streaming

One mechanism serves both "activity" and "token" streaming.

### `generate_stream()` in all three clients

- Same signature as `generate()` plus it takes an `on_delta(str)` callback invoked per text chunk and returns the final `LLMResponse` (callback-style survives `asyncio.to_thread`; a generator would not).
- Anthropic: `client.messages.stream(...)`. OpenAI + local: shared
  chat-completions path with `stream=True` (local client already speaks the
  OpenAI protocol).
- Structured-output calls (planner) keep non-streaming `generate()`.

### `POST /api/chat/stream` — SSE

Events, in order of appearance during a request:

| Event      | Payload                          | Meaning                                    |
|------------|----------------------------------|--------------------------------------------|
| `status`   | `{text}`                         | Node/tool activity ("calling rag_query…")  |
| `delta`    | `{text}`                         | Text tokens from the current agent turn    |
| `turn_end` | `{has_tools}`                    | If `has_tools`: client collapses streamed text into the activity feed (it was tool-reasoning). Else: streamed text is the final answer. |
| `done`     | `{reply, thread_id, citations}`  | Terminal. `reply` authoritative (client may already have it via deltas). |
| `error`    | `{message}`                      | Terminal on failure.                        |

- Plumbing: an `asyncio.Queue` is created per request and passed into
  `build_graph`; `agent_node` runs `generate_stream` in a worker thread and
  pushes deltas with `loop.call_soon_threadsafe`. `tools_node` pushes `status`
  events per tool call. The SSE handler drains the queue.
- Multi-agent mode: planner and researcher phases emit `status` events
  ("planning…", "researching: <sub-question>"); the synthesizer token-streams
  as the final answer.
- `/api/chat` (non-streaming) remains unchanged for eval and tests.

## 4. Citations

- `AgentState` gains `citations: Annotated[list[str], operator.add]`.
- `tools_node` appends `ans.sources` (paper ids) after each `rag_query` call.
- `run_agent` returns `(text, citations)`; `run_multi_agent` unions researcher
  citations (order-preserving dedupe).
- `ChatResponse` gains `citations: list[str]`; `done` SSE event carries the
  same list.
- UI renders citation chips under each assistant reply, linking to
  `https://arxiv.org/abs/{paper_id}`.

## 5. Threads

- New table in the existing `data/checkpoints.db` (one DB file to manage;
  LangGraph's tables are untouched):

  ```sql
  CREATE TABLE IF NOT EXISTS threads (
    thread_id  TEXT PRIMARY KEY,
    title      TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
  );
  ```

- Upsert on every chat request: insert with `title = first message truncated
  to 80 chars` on first sight, bump `updated_at` afterwards.
- Endpoints:
  - `GET /api/threads` — list ordered by `updated_at` desc.
  - `GET /api/threads/{id}` — transcript restored via `graph.aget_state`;
    only plain-text user/assistant turns are returned (tool traffic omitted).
    404 if unknown.
  - `DELETE /api/threads/{id}` — removes the row and the thread's LangGraph
    checkpoint rows.

## 6. Frontend

Stays vanilla JS, no build step. `marked` + `DOMPurify` vendored under
`api/static/vendor/` (no build step, works offline).

- Sidebar: thread list (title + relative time), click to load transcript,
  delete button per thread, "New conversation" on top.
- Provider dropdown next to the chat input, populated from `/api/providers`.
- Chat over SSE: activity lines render as dim status text; deltas render
  live into the pending assistant bubble; `turn_end {has_tools:true}`
  collapses the bubble text into the activity feed.
- Assistant replies render as sanitized markdown; citation chips underneath.

## Error handling

- Invalid provider value → 422 (pydantic `Literal`).
- Unavailable provider → 400 with the availability `detail`, checked before
  the agent starts.
- Mid-stream failure → `error` SSE event, stream closes; UI shows the message
  in place of the pending bubble.
- Ollama down is surfaced twice: grayed out in the dropdown (status check)
  and as a clean 400/`error` if forced via raw API.

## Testing

- Unit (no keys, existing pattern): fake streaming clients for
  `generate_stream`; provider threading asserted by monkeypatching `generate`
  and capturing the `provider` kwarg; threads table CRUD; `/api/providers`
  with monkeypatched settings and a stubbed Ollama probe; SSE event sequence
  with a stubbed agent.
- Integration (existing `local` marker): real Ollama end-to-end stream.
- Frontend: manual smoke via the browse tool (dropdown state, streaming
  render, thread restore, citation links).

## Order of work

Each step lands green and independently useful:

1. `GET /api/providers` + httpx runtime dep
2. Per-request provider threading (API → agents → generate)
3. Citations (state → response → UI-ready payload)
4. Threads table + endpoints
5. `generate_stream` in three clients + SSE endpoint
6. Frontend rebuild (sidebar, dropdown, markdown, streaming, chips)

## Out of scope

- Embedding-provider toggle in UI (retrieval must match the collection's
  embedding dim; switching requires re-ingest — stays a config/migration
  concern).
- Auth/multi-user, token cost display, mobile layout.
