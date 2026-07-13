# Paper Research Assistant

Learning project covering LLM APIs + prompting, RAG, evaluation, and agents + MCP.
Ingests arXiv papers, answers questions grounded in them with [paper_id] citations,
and autonomously fetches papers it doesn't have yet.

## Status: complete (all 8 phases shipped)

| Phase | Delivered |
|-------|-----------|
| 1 | Core RAG pipeline (ingest → chunk → embed → Qdrant → cited answers), LangGraph agent + MCP tools, eval harness, web UI |
| 2 | Hybrid retrieval (BM25 + dense + RRF), cross-encoder reranking, query rewriting, agent memory (summarization), multi-agent supervisor, retrieval ablation |
| 3 | Fully local, no-keys operation via Ollama (`qwen2.5:3b`) + local embeddings; live-validated end to end |
| 4 | Per-request provider toggle in the UI, provider availability checks, SSE streaming (activity + tokens), citation chips, persistent thread sidebar |
| 5 | Corrective RAG (LLM chunk grading + one rewritten-query retry + honest degradation) and a citation-faithfulness guardrail with an "unverified citations" badge in the UI |
| 6 | Synthetic eval generation (LLM question/gist from ingested chunks, fail-closed self-check, grows golden set 3 → 50+) and 95% bootstrap CIs on every eval metric and ablation cell |
| 7 | Judge calibration: blind human-labeling CLI, judge-vs-human agreement (quadratic-weighted kappa + MAE with bootstrap CIs), test-retest consistency mode |
| 8 | UI/UX refresh: token design system with auto/manual dark mode, per-reply agent-activity accordion, welcome/empty states, error toasts, scroll pinning, mobile drawer layout |

Design specs and implementation plans for each phase live in
`docs/superpowers/specs/` and `docs/superpowers/plans/`. Known limitation:
the 3B local model occasionally fumbles the agent tool-call loop — use
`LOCAL_MODEL=qwen2.5:7b` if you have the RAM (see notes below).

## Current eval snapshot (2026-07-13, qwen2.5:3b, 53 questions)

From the committed `report.json` (3 hand-written + 50 synthetic items, 95% bootstrap CIs):

```
retrieval precision : 0.83 [0.74, 0.92]
retrieval recall    : 0.89 [0.79, 0.96]
faithfulness        : 3.51 [3.23, 3.81] / 5
relevance           : 4.55 [4.38, 4.70] / 5
citation accuracy   : 2.43 [2.19, 2.70] / 5   <- the 3B model's known weak spot
verified answers    : 67% [55%, 80%]           (phase-5 faithfulness guardrail)

[hand]      n=3   precision 0.83  recall 1.00  faith 4.67  rel 4.00  cite 2.67
[synthetic] n=50  precision 0.83  recall 0.88  faith 3.44  rel 4.58  cite 2.42
```

`report.json` is normally gitignored (regenerable); this snapshot is committed
deliberately as the pinned baseline for the pending calibration session.

### Pending work

- **Human labeling session (phase 7's payoff, ~15 min, not yet done):**
  `uv run python -m eval.calibrate label` → blind-score 20 sampled answers →
  commit `eval/human-labels.json` → `uv run python -m eval.calibrate report`.
  Until then the judge behind every number above is uncalibrated.
- **Cloud API keys never configured:** `.env` has placeholder
  `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` values. Everything above ran on local
  `qwen2.5:3b`; the cloud providers, `pytest -m integration`, and a
  3B-vs-cloud comparison (the obvious fix for citation accuracy) all await
  real keys.
- **Ablation on the 53-item set:** `--ablation` sweeps 6 presets × 53 items —
  hours on the 3B model. Run it chunked or wait for a cloud key.

## Setup

```bash
uv sync
cp .env.example .env   # cloud keys OPTIONAL — this repo currently runs fully
                       # local; add ANTHROPIC_API_KEY / OPENAI_API_KEY to
                       # unlock the provider toggle + integration tests
docker compose up -d   # Qdrant on localhost:6333
```

## Fully local (no API keys)

Runs the whole system — ingest, cited chat, memory, multi-agent, eval — on a
local model. Fits an 8GB M1 MacBook Air.

```bash
brew install ollama
ollama pull qwen2.5:3b                      # ~1.9GB
OLLAMA_CONTEXT_LENGTH=8192 OLLAMA_NUM_PARALLEL=1 ollama serve
# >4k context for grounded prompts; a single parallel slot keeps the KV cache
# from multiplying — without it, long eval runs can OOM an 8GB machine
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

# Offline eval -> report.json + printed summary (CIs + hand/synthetic split).
# ~1h for 53 items on qwen2.5:3b. On 8GB RAM use the serve flags above AND
# split the run: pass --dataset with slices of the item list in separate
# processes, then merge rows (ONNX models + KV cache in one long process
# is what OOMed an M1 Air)
uv run python -m eval.run

# Retrieval ablation: dataset across dense/sparse/hybrid/rerank/grade/rewrite presets
uv run python -m eval.run --ablation

# Generate synthetic eval items from ingested chunks (grows the golden set;
# eval.run picks the file up automatically)
uv run python -m eval.generate --count 50

# Run eval on ONLY the hand-written set
uv run python -m eval.run --dataset eval/golden.json

# Calibrate the judge: hand-label ~20 sampled answers (blind), then measure
# judge-vs-human agreement; --consistency re-judges each item (live LLM)
uv run python -m eval.calibrate label
uv run python -m eval.calibrate report
uv run python -m eval.calibrate report --consistency

# Upgrading from phase 1? The collection schema changed (named dense+sparse
# vectors) — recreate it and re-ingest:
uv run python -m rag.migrate --yes
```

Eval metrics print with 95% bootstrap confidence intervals — `0.67 [0.51, 0.82]`
in the summary, `0.67 ±0.08` per ablation cell — so preset differences can be
read against their noise floor.
Caveat: synthetic items share a generator with the LLM judge and are sampled
from the ingested corpus, so absolute scores on the mixed set skew friendly —
trust the ablation deltas, not the absolute numbers.
`eval.calibrate` quantifies the judge itself — with one annotator and small n,
treat its kappa as a sanity check, not a certification.

Retrieval is a staged pipeline — `[rewrite] → embed → search (dense|sparse|hybrid) → [rerank] → [grade → retry once]` —
controlled by `.env` flags (`RETRIEVAL_MODE`, `RERANK_ENABLED`, `REWRITE_ENABLED`, `GRADING_ENABLED`; see `config.py`).
BM25 sparse search and reranking run on local ONNX models — no API keys needed.
After answering, `FAITHFULNESS_ENABLED` runs a citation-faithfulness check; an
unsupported answer gets an "⚠ citations unverified" badge in the UI (live
responses only — verdicts aren't persisted, so restored threads never show it).
The verdict is per-message: each reply's badge reflects only the rag_query
calls behind that reply, not earlier turns.
Both guardrails fail open and add 1–3 LLM calls per request — turn them off on
slow local models if latency hurts.
Chat is multi-turn: the UI carries a `thread_id`, history is checkpointed to
`data/checkpoints.db`, long conversations get summarized. `AGENT_MODE=multi`
switches to a planner → researcher → synthesizer supervisor.

The web UI (phases 4 + 8):
- **Per-request provider toggle** — header dropdown switches reasoning between
  Anthropic / OpenAI / local Ollama per message; a banner + disabled input
  appear when no provider is available.
- **Streaming with an activity accordion** — replies stream over SSE; each
  reply's agent trace ("calling rag_query…", "grading 8 chunks…") lives in a
  collapsible block attached to that reply, folding to "⚙ N steps" when done.
- **Citations + faithfulness** — [paper_id] chips link to arXiv; unverified
  answers get a warning chip; a copy button grabs the reply's markdown.
- **Dark mode** — follows the OS by default; header toggle cycles
  auto → light → dark (persisted).
- **Thread sidebar** — persistent conversations (list / restore / delete);
  collapses to a drawer on small screens; ingest lives at the sidebar bottom.
- Vanilla JS + vendored `marked`/`DOMPurify` — no build step, works offline.

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
  rerank, query rewrite, retrieve, relevance grading, answer, faithfulness check, migrate
- `agents/` — LangGraph agent with SQLite-checkpointed memory; multi-agent supervisor (`agents/multi.py`);
  custom MCP server (`python -m agents.mcp_server`); MCP client (also consumes `mcp-server-fetch`)
- `eval/` — golden + synthetic datasets, synthetic-item generator, LLM judge, retrieval metrics,
  bootstrap CIs, report generator, ablation mode
- `api/` — FastAPI routes (chat, SSE stream, ingest, providers, threads) + static frontend

Imports flow one way: `api → agents → rag/llm`; `eval → rag/agents/llm`.
