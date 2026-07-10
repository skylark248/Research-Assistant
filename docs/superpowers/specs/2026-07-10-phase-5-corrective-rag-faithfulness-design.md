# Phase 5 Design: Corrective RAG + Citation Faithfulness

Date: 2026-07-10
Status: approved

## Goal

Two small features sharing one pattern ("LLM grades LLM"), one on each side of
the pipeline:

1. **Corrective RAG** — grade retrieved chunks for relevance; if nothing
   relevant survives, retry once with a rewritten query; if still nothing,
   degrade honestly instead of hallucinating.
2. **Citation faithfulness guardrail** — after answering, check whether the
   cited excerpts actually support the answer's claims; surface an
   "unverified citations" warning in the UI when they don't.

Learning targets: reflection loops (self-/corrective RAG), guardrail
engineering (fail-open design, honest degradation), and grader-prompt design
that survives a 3B local model.

## Non-goals

- Prompt-injection scanning of ingested PDFs (deferred).
- Regenerating answers on faithfulness failure (flag only — honest degradation).
- Persisting faithfulness verdicts to the threads DB (live-only badge;
  verdict is lost on thread restore — accepted limitation, noted in README).
- LangGraph-level reflection nodes (grading lives in the RAG pipeline so all
  callers — agent, multi-agent, eval, API — benefit).

## Behavior decisions (locked)

- **Fail behavior: honest degradation.**
  (a) Zero relevant chunks after retry → answer states the corpus lacks the
  info and suggests ingesting; no generate call is made.
  (b) Unfaithful citations → answer still shown, UI marks it with an
  "unverified citations" badge. No silent failures, no infinite retries.
- **Defaults: on.** `GRADING_ENABLED=true`, `FAITHFULNESS_ENABLED=true`;
  local-model users can disable to save 1–3 extra LLM calls per request.
- **Fail-open guardrails.** A grader error can never make results worse than
  no grader: parse/LLM errors keep all chunks (grading) or yield verdict
  `None` (faithfulness). Guardrails never fail the request.

## Architecture

Pipeline grows two stages:

```
[rewrite] → embed → search (dense|sparse|hybrid) → [rerank]
          → [grade → retry once] → generate → [faithfulness]
```

### New module: `rag/grade.py`

`grade_chunks(question, chunks, provider=None) -> list[ScoredChunk]`

- ONE batched LLM call. Prompt lists numbered chunk excerpts; model returns
  one line per chunk: `1: yes` / `2: no`. Line format, not JSON — a 3B model
  follows it far more reliably.
- Returns only the chunks graded relevant, original order preserved.
- Fail-open: unparseable line → that chunk passes; LLM exception or fully
  unparseable output → log warning, return all chunks unchanged.

### New module: `rag/faithfulness.py`

`check_faithfulness(question, answer, contexts, provider=None) -> bool | None`

- ONE LLM call: given excerpts + answer, does every cited claim have support?
  Model answers `yes` / `no` (first token wins on parsing).
- `True` = supported, `False` = unsupported, `None` = check errored or output
  unparseable (UI shows nothing for `None`).

### Corrective loop: `rag/answer.py`

`answer_question` orchestrates (it already owns per-request provider
threading; `retrieve` stays as-is):

1. `chunks = retrieve(question)`
2. If `settings.grading_enabled`: `chunks = grade_chunks(question, chunks, provider)`
3. If zero chunks survive: rewrite the question with
   `RETRY_REWRITE_SYSTEM_PROMPT` ("previous search found nothing relevant —
   write one alternative query"), then `retrieve` + `grade_chunks` once more.
4. Still zero → return the existing "don't have enough information" honest
   answer with empty sources; **no generate call**.
5. Otherwise build the grounded prompt and generate as today.
6. If `settings.faithfulness_enabled`: `faithful = check_faithfulness(...)`.

`RagAnswer` gains `faithful: bool | None = None`.

Cost envelope: +2 LLM calls typical (grade + faithfulness), +3 worst case
(grade, retry-grade, faithfulness). Retry capped at exactly one.

### Config (`config.py`)

```python
# Guardrails (phase 5)
grading_enabled: bool = True
faithfulness_enabled: bool = True
```

Matches the existing `rerank_enabled` flag pattern so eval ablation can
isolate each stage.

### Prompts (`llm/prompts.py`)

`GRADE_SYSTEM_PROMPT`, `FAITHFULNESS_SYSTEM_PROMPT`,
`RETRY_REWRITE_SYSTEM_PROMPT`. Contexts remain plain dicts
(`{paper_id, title, text}`) — `llm/` never imports `rag/`.

## Event + verdict plumbing

- `answer_question` gains optional `on_status: Callable[[str], None] = None`.
  Emitted statuses: `"grading N chunks…"`, `"M of N chunks relevant"`,
  `"retrying with rewritten query…"`, `"verifying citations…"`. Plain
  callable — `rag/` gains no new dependencies.
- `agents/graph.py` `tools_node` passes
  `on_status=lambda t: on_event({"event": "status", "text": t})` when
  `on_event` is set, so grading activity appears in the existing SSE
  activity stream.
- Verdict propagation: `AgentState` gains a `faithful` channel collecting
  per-`rag_query` verdicts; `AgentResult` gains `faithful: bool | None`
  computed as an AND: any `False` → `False`; else any `None` → `None`;
  else `True` (no rag_query calls → `None`).
- `agents/multi.py` propagates the same way across sub-question findings.
- `ChatResponse` gains `faithful: bool | None = None`; the SSE `done` event
  carries it alongside citations.

## UI (`api/static`)

- Amber badge `⚠ citations unverified` rendered next to the citation chips
  when `faithful === false`. Nothing rendered for `true`/`null` — quiet when
  healthy.
- Live responses only (both `/api/chat` and `/api/chat/stream`); restored
  transcripts never show the badge.

## Eval

- New ablation preset `hybrid+rerank+grade` in `eval/` — measures grading's
  effect on retrieval metrics exactly as rerank was measured.
- Eval report additionally logs the faithfulness rate (fraction `True` of
  non-`None`) across the golden dataset.

## Error handling summary

| Failure | Behavior |
|---|---|
| Grader LLM error / unparseable output | Keep all chunks, log warning |
| Zero relevant after retry | Honest "not enough information" answer, no generate |
| Faithfulness LLM error / unparseable | Verdict `None`, UI silent |
| Any guardrail failure | Request still succeeds |

## Testing

Unit tests (mocked `generate`, no keys — existing pattern):

- `grade_chunks`: filters by grades; preserves order; malformed lines
  fail-open per-chunk; full parse failure returns all chunks; LLM exception
  returns all chunks.
- Corrective loop: retry fires only on zero survivors; retry runs at most
  once; honest-degradation path makes no generate call; grading disabled →
  behavior identical to today.
- `check_faithfulness`: yes/no/garbage → `True`/`False`/`None`; exception →
  `None`.
- Agent: verdict AND-ing across multiple `rag_query` calls; `faithful`
  channel through checkpointed state; `on_status` events forwarded to
  `on_event`.
- API: `ChatResponse.faithful` present; SSE `done` event carries `faithful`;
  status events for grading appear in the stream.
- One `-m local` marker test: real-Ollama batch grading returns parseable
  verdicts.

## File touch list

| File | Change |
|---|---|
| `rag/grade.py` | new |
| `rag/faithfulness.py` | new |
| `rag/answer.py` | corrective loop, `on_status`, `faithful` field |
| `config.py` | 2 flags |
| `llm/prompts.py` | 3 prompts |
| `agents/graph.py` | status forwarding, `faithful` channel + AgentResult field |
| `agents/multi.py` | verdict propagation |
| `api/main.py` | `ChatResponse.faithful`, SSE done payload |
| `api/static/*` | warning badge |
| `eval/*` | ablation preset + faithfulness rate |
| `tests/*` | new unit tests + 1 local test |
| `README.md` | phase 5 row, flags, badge limitation |
