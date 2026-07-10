# Phase 5: Corrective RAG + Citation Faithfulness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Grade retrieved chunks for relevance (retry once on zero survivors, degrade honestly), and verify answers against their cited excerpts, surfacing an "unverified citations" badge in the UI.

**Architecture:** Two new fail-open pipeline stages in `rag/` (`grade.py`, `faithfulness.py`), orchestrated by `answer_question` in `rag/answer.py`. Verdicts and status events thread through the existing agent → API → SSE → frontend path. Spec: `docs/superpowers/specs/2026-07-10-phase-5-corrective-rag-faithfulness-design.md`.

**Tech Stack:** Python 3.12, FastAPI, LangGraph, pydantic, pytest (async tests run bare — asyncio auto mode), vanilla JS frontend. Run everything with `uv run`.

## Global Constraints

- No new dependencies. `llm/` never imports `rag/` (contexts are plain dicts `{paper_id, title, text}`).
- `messages`/`tools` use the Anthropic shape everywhere; provider adapters translate.
- Guardrails are fail-open and advisory: a grader/checker error must never fail the request or shrink results below the ungraded set.
- New flags `grading_enabled: bool = True`, `faithfulness_enabled: bool = True` in `config.py` (matches `rerank_enabled` pattern).
- Grader output is LINE format (`1: yes`), never JSON — 3B local models follow it more reliably.
- Per-request `provider` threads through every new LLM call (grade, retry rewrite, faithfulness).
- Retry after grading is capped at exactly one; retried chunks are graded against the ORIGINAL question (same philosophy as rerank).
- Faithfulness verdict is `bool | None`; `None` (errored/unparseable) renders nothing in the UI.
- Unit tests are mocked, need no keys, and run with `uv run pytest`.
- Commit messages: imperative conventional style (`feat:`, `test:`, `docs:`), matching `git log`.

---

### Task 1: `rag/grade.py` — batched chunk relevance grading

**Files:**
- Create: `rag/grade.py`
- Create: `tests/test_grade.py`
- Modify: `llm/prompts.py` (append `GRADE_SYSTEM_PROMPT`)

**Interfaces:**
- Consumes: `llm.base.generate(messages, *, system=..., provider=...) -> LLMResponse` (`.text: str`); `rag.store.ScoredChunk` (`paper_id, title, text, score`).
- Produces: `grade_chunks(question: str, chunks: list[ScoredChunk], provider: str | None = None) -> list[ScoredChunk]` — Task 4 calls this from `rag/answer.py`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_grade.py`:

```python
from llm.base import LLMResponse
from rag.store import ScoredChunk


def _chunk(pid="1706.03762", title="Attention", text="self-attention", score=0.9):
    return ScoredChunk(paper_id=pid, title=title, text=text, score=score)


def _patch_generate(monkeypatch, text=None, exc=None):
    import rag.grade as grade_mod

    calls = []

    def fake_generate(messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        if exc is not None:
            raise exc
        return LLMResponse(text=text)

    monkeypatch.setattr(grade_mod, "generate", fake_generate)
    return calls


def test_grade_keeps_relevant_chunks_in_order(monkeypatch):
    from rag.grade import grade_chunks

    calls = _patch_generate(monkeypatch, text="1: yes\n2: no\n3: yes")
    chunks = [_chunk(pid="p1"), _chunk(pid="p2"), _chunk(pid="p3")]
    kept = grade_chunks("q", chunks)
    assert [c.paper_id for c in kept] == ["p1", "p3"]
    assert len(calls) == 1  # one batched call, not one per chunk


def test_grade_prompt_carries_question_and_all_chunks(monkeypatch):
    from rag.grade import grade_chunks

    calls = _patch_generate(monkeypatch, text="1: yes\n2: yes")
    grade_chunks("what is attention?",
                 [_chunk(pid="p1", text="alpha"), _chunk(pid="p2", text="beta")])
    prompt = calls[0]["messages"][0]["content"]
    assert "alpha" in prompt and "beta" in prompt
    assert "what is attention?" in prompt


def test_grade_threads_provider(monkeypatch):
    from rag.grade import grade_chunks

    calls = _patch_generate(monkeypatch, text="1: yes")
    grade_chunks("q", [_chunk()], provider="local")
    assert calls[0]["provider"] == "local"


def test_grade_missing_verdict_fails_open_per_chunk(monkeypatch):
    from rag.grade import grade_chunks

    _patch_generate(monkeypatch, text="1: no")  # chunk 2 never mentioned
    kept = grade_chunks("q", [_chunk(pid="p1"), _chunk(pid="p2")])
    assert [c.paper_id for c in kept] == ["p2"]  # unmentioned chunk passes


def test_grade_garbage_output_returns_all_chunks(monkeypatch):
    from rag.grade import grade_chunks

    _patch_generate(monkeypatch, text="I think they all look great!")
    chunks = [_chunk(pid="p1"), _chunk(pid="p2")]
    assert grade_chunks("q", chunks) == chunks


def test_grade_llm_error_returns_all_chunks(monkeypatch):
    from rag.grade import grade_chunks

    _patch_generate(monkeypatch, exc=RuntimeError("api down"))
    chunks = [_chunk(pid="p1")]
    assert grade_chunks("q", chunks) == chunks


def test_grade_empty_input_makes_no_llm_call(monkeypatch):
    from rag.grade import grade_chunks

    calls = _patch_generate(monkeypatch, text="unused")
    assert grade_chunks("q", []) == []
    assert calls == []


def test_grade_tolerates_format_variants(monkeypatch):
    from rag.grade import grade_chunks

    _patch_generate(monkeypatch, text="1. YES\n 2) no\n3 - yes")
    kept = grade_chunks("q", [_chunk(pid="p1"), _chunk(pid="p2"), _chunk(pid="p3")])
    assert [c.paper_id for c in kept] == ["p1", "p3"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_grade.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.grade'`

- [ ] **Step 3: Append the grade prompt to `llm/prompts.py`**

Append after `SYNTHESIZER_SYSTEM_PROMPT` (end of file):

```python
GRADE_SYSTEM_PROMPT = """You judge whether paper excerpts are relevant to a question.

You get a question and numbered excerpts. For EACH excerpt output exactly one line:
<number>: yes    (the excerpt helps answer the question)
<number>: no     (it does not)

Output ONLY those lines, one per excerpt, in order. No other text."""
```

- [ ] **Step 4: Write `rag/grade.py`**

```python
"""LLM relevance grading of retrieved chunks (corrective RAG, phase 5).

One batched LLM call grades every chunk. Line-format output ("1: yes") rather
than JSON — small local models follow it far more reliably. Fails open: any
LLM or parse failure keeps all chunks — a broken grader must never make
retrieval worse than no grader.
"""

import logging
import re

from llm.base import generate
from llm.prompts import GRADE_SYSTEM_PROMPT
from rag.store import ScoredChunk

logger = logging.getLogger(__name__)

_VERDICT_LINE = re.compile(r"^\s*(\d+)\s*[:.)\-]\s*(yes|no)\b",
                           re.IGNORECASE | re.MULTILINE)


def _build_prompt(question: str, chunks: list[ScoredChunk]) -> str:
    parts = [f"{i}. [paper {c.paper_id} — {c.title}]\n{c.text}"
             for i, c in enumerate(chunks, start=1)]
    return f"Question: {question}\n\nExcerpts:\n\n" + "\n\n---\n\n".join(parts)


def grade_chunks(question: str, chunks: list[ScoredChunk],
                 provider: str | None = None) -> list[ScoredChunk]:
    """Keep only the chunks the grader marks relevant, original order preserved.

    Fail-open: a chunk with a missing/unparseable verdict passes; a grader
    exception or fully unparseable output returns all chunks unchanged.
    """
    if not chunks:
        return []
    try:
        resp = generate(
            [{"role": "user", "content": _build_prompt(question, chunks)}],
            system=GRADE_SYSTEM_PROMPT, provider=provider,
        )
    except Exception:
        logger.warning("Chunk grading failed; keeping all chunks", exc_info=True)
        return chunks
    verdicts = {int(n): v.lower() == "yes" for n, v in _VERDICT_LINE.findall(resp.text)}
    if not verdicts:
        logger.warning("Grader output unparseable; keeping all chunks: %r",
                       resp.text[:200])
        return chunks
    return [c for i, c in enumerate(chunks, start=1) if verdicts.get(i, True)]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_grade.py -v`
Expected: 8 passed

- [ ] **Step 6: Run the full suite, then commit**

Run: `uv run pytest`
Expected: all pass (no existing behavior touched)

```bash
git add rag/grade.py tests/test_grade.py llm/prompts.py
git commit -m "feat: batched LLM relevance grading of retrieved chunks"
```

---

### Task 2: `rag/faithfulness.py` — citation-faithfulness check

**Files:**
- Create: `rag/faithfulness.py`
- Create: `tests/test_faithfulness.py`
- Modify: `llm/prompts.py` (append `FAITHFULNESS_SYSTEM_PROMPT`)

**Interfaces:**
- Consumes: `llm.base.generate`; `llm.prompts.format_context(contexts: list[dict]) -> str`.
- Produces: `check_faithfulness(question: str, answer: str, contexts: list[dict], provider: str | None = None) -> bool | None` — Task 4 calls this. `contexts` items are `{paper_id, title, text}` dicts.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_faithfulness.py`:

```python
from llm.base import LLMResponse

CONTEXTS = [{"paper_id": "1706.03762", "title": "Attention", "text": "self-attention"}]


def _patch_generate(monkeypatch, text=None, exc=None):
    import rag.faithfulness as faith_mod

    calls = []

    def fake_generate(messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        if exc is not None:
            raise exc
        return LLMResponse(text=text)

    monkeypatch.setattr(faith_mod, "generate", fake_generate)
    return calls


def test_yes_means_supported(monkeypatch):
    from rag.faithfulness import check_faithfulness

    _patch_generate(monkeypatch, text="Yes")
    assert check_faithfulness("q", "a [1706.03762]", CONTEXTS) is True


def test_no_means_unsupported_even_with_trailing_prose(monkeypatch):
    from rag.faithfulness import check_faithfulness

    _patch_generate(monkeypatch, text="no. The answer cites a paper not in the excerpts.")
    assert check_faithfulness("q", "a [9999.00001]", CONTEXTS) is False


def test_garbage_verdict_is_none(monkeypatch):
    from rag.faithfulness import check_faithfulness

    _patch_generate(monkeypatch, text="Maybe? Hard to say.")
    assert check_faithfulness("q", "a", CONTEXTS) is None


def test_llm_error_is_none(monkeypatch):
    from rag.faithfulness import check_faithfulness

    _patch_generate(monkeypatch, exc=RuntimeError("api down"))
    assert check_faithfulness("q", "a", CONTEXTS) is None


def test_prompt_carries_excerpts_question_answer_and_provider(monkeypatch):
    from rag.faithfulness import check_faithfulness

    calls = _patch_generate(monkeypatch, text="yes")
    check_faithfulness("what is attention?", "It is attention [1706.03762].",
                       CONTEXTS, provider="local")
    prompt = calls[0]["messages"][0]["content"]
    assert "self-attention" in prompt              # excerpt text
    assert "what is attention?" in prompt          # question
    assert "It is attention [1706.03762]." in prompt  # answer under test
    assert calls[0]["provider"] == "local"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_faithfulness.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.faithfulness'`

- [ ] **Step 3: Append the faithfulness prompt to `llm/prompts.py`**

Append at end of file:

```python
FAITHFULNESS_SYSTEM_PROMPT = """You verify that an answer is supported by paper excerpts.

You get excerpts, a question, and an answer. Reply with exactly one word:
yes — every claim in the answer is supported by the excerpts
no — any claim lacks support or cites a paper the excerpts do not back"""
```

- [ ] **Step 4: Write `rag/faithfulness.py`**

```python
"""Post-answer citation-faithfulness guardrail (phase 5).

One LLM call asks whether the cited excerpts support the answer's claims.
Verdicts: True (supported), False (unsupported), None (check errored or
output unparseable). Advisory only — a failed check never fails the request.
"""

import logging

from llm.base import generate
from llm.prompts import FAITHFULNESS_SYSTEM_PROMPT, format_context

logger = logging.getLogger(__name__)


def check_faithfulness(question: str, answer: str, contexts: list[dict],
                       provider: str | None = None) -> bool | None:
    prompt = (f"Paper excerpts:\n\n{format_context(contexts)}\n\n"
              f"Question: {question}\n\nAnswer:\n{answer}")
    try:
        resp = generate([{"role": "user", "content": prompt}],
                        system=FAITHFULNESS_SYSTEM_PROMPT, provider=provider)
    except Exception:
        logger.warning("Faithfulness check failed", exc_info=True)
        return None
    words = resp.text.strip().lower().split()
    token = words[0].strip(".,!—:;") if words else ""
    if token == "yes":
        return True
    if token == "no":
        return False
    logger.warning("Faithfulness verdict unparseable: %r", resp.text[:100])
    return None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_faithfulness.py -v`
Expected: 5 passed

- [ ] **Step 6: Run the full suite, then commit**

Run: `uv run pytest`
Expected: all pass

```bash
git add rag/faithfulness.py tests/test_faithfulness.py llm/prompts.py
git commit -m "feat: citation-faithfulness check with yes/no/unknown verdicts"
```

---

### Task 3: `retry_rewrite_query` — alternative query after a failed retrieval

**Files:**
- Modify: `rag/rewrite.py`
- Modify: `tests/test_rewrite.py` (append tests)
- Modify: `llm/prompts.py` (append `RETRY_REWRITE_SYSTEM_PROMPT`)

**Interfaces:**
- Consumes: existing `RewrittenQuery` pydantic model and `generate` in `rag/rewrite.py`.
- Produces: `retry_rewrite_query(question: str, provider: str | None = None) -> str` — Task 4 calls this. Fails open by returning the ORIGINAL question; callers use `retry_query == question` to skip a pointless identical retry.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_rewrite.py`:

```python
def test_retry_rewrite_returns_alternative_query(monkeypatch):
    import rag.rewrite as rewrite_mod
    from llm.base import LLMResponse
    from rag.rewrite import RewrittenQuery

    def fake_generate(messages, **kwargs):
        assert kwargs["system"] == rewrite_mod.RETRY_REWRITE_SYSTEM_PROMPT
        assert kwargs["provider"] == "local"
        return LLMResponse(parsed=RewrittenQuery(query="transformer self-attention architecture"))

    monkeypatch.setattr(rewrite_mod, "generate", fake_generate)
    result = rewrite_mod.retry_rewrite_query("how do transformers work?", provider="local")
    assert result == "transformer self-attention architecture"


def test_retry_rewrite_fails_open_to_original(monkeypatch):
    import rag.rewrite as rewrite_mod

    monkeypatch.setattr(rewrite_mod, "generate",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    assert rewrite_mod.retry_rewrite_query("original q") == "original q"


def test_retry_rewrite_empty_result_falls_back(monkeypatch):
    import rag.rewrite as rewrite_mod
    from llm.base import LLMResponse
    from rag.rewrite import RewrittenQuery

    monkeypatch.setattr(rewrite_mod, "generate",
                        lambda *a, **k: LLMResponse(parsed=RewrittenQuery(query="  ")))
    assert rewrite_mod.retry_rewrite_query("original q") == "original q"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_rewrite.py -v`
Expected: new tests FAIL — `AttributeError: ... has no attribute 'retry_rewrite_query'`; existing tests still pass

- [ ] **Step 3: Append the retry prompt to `llm/prompts.py`**

Append at end of file:

```python
RETRY_REWRITE_SYSTEM_PROMPT = """A vector-database search over research papers found no relevant excerpts for the user's question.

Write ONE alternative search query that might match better: use synonyms, expand acronyms, or generalize overly specific phrasing. Return only the query."""
```

- [ ] **Step 4: Implement `retry_rewrite_query` in `rag/rewrite.py`**

Change the import line to include the new prompt:

```python
from llm.prompts import RETRY_REWRITE_SYSTEM_PROMPT, REWRITE_SYSTEM_PROMPT
```

Append at end of file:

```python
def retry_rewrite_query(question: str, provider: str | None = None) -> str:
    """Alternative query after grading rejected every retrieved chunk (phase 5).

    Fails open to the original question — the caller detects that (retry_query
    == question) and skips a retry that would just repeat itself.
    """
    try:
        resp = generate(
            [{"role": "user", "content": question}],
            system=RETRY_REWRITE_SYSTEM_PROMPT,
            structured_schema=RewrittenQuery,
            provider=provider,
        )
        rewritten = resp.parsed.query.strip()
        return rewritten or question
    except Exception:
        logger.warning("Retry rewrite failed; using the original question",
                       exc_info=True)
        return question
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_rewrite.py -v`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add rag/rewrite.py tests/test_rewrite.py llm/prompts.py
git commit -m "feat: retry query rewrite for corrective retrieval"
```

---

### Task 4: Corrective loop + faithfulness in `rag/answer.py`, config flags

**Files:**
- Modify: `config.py`
- Modify: `rag/answer.py` (full rewrite shown below)
- Modify: `tests/test_retrieve_answer.py` (update 3 existing tests, append new ones)
- Modify: `tests/test_config.py` (append flag test)

**Interfaces:**
- Consumes: `grade_chunks` (Task 1), `check_faithfulness` (Task 2), `retry_rewrite_query` (Task 3), existing `retrieve`/`build_rag_prompt`/`generate`.
- Produces: `answer_question(question, store=None, provider=None, on_status: Callable[[str], None] | None = None) -> RagAnswer`; `RagAnswer` gains `faithful: bool | None = None`. Task 5 passes `on_status` and reads `.faithful`. Status strings emitted, in order when applicable: `"grading N chunks…"`, `"M of N chunks relevant"`, `"retrying with rewritten query…"`, `"verifying citations…"`.

- [ ] **Step 1: Add the config flags**

In `config.py`, after the `rewrite_enabled` line in the retrieval-pipeline block:

```python
    # Guardrails (phase 5) — same flag pattern so eval ablation can isolate each
    grading_enabled: bool = True
    faithfulness_enabled: bool = True
```

Append to `tests/test_config.py`:

```python
def test_phase5_guardrail_flags_default_on():
    from config import Settings

    s = Settings(_env_file=None)
    assert s.grading_enabled is True
    assert s.faithfulness_enabled is True
```

Run: `uv run pytest tests/test_config.py -v` — expected: pass.

- [ ] **Step 2: Update the 3 existing answer tests (guardrails off) and write the new failing tests**

In `tests/test_retrieve_answer.py`, add a fixture near the top (after the imports):

```python
import pytest


@pytest.fixture
def guardrails_off(monkeypatch):
    """Phase-5 guardrails default on; these tests exercise the pre-existing path."""
    monkeypatch.setattr(settings, "grading_enabled", False)
    monkeypatch.setattr(settings, "faithfulness_enabled", False)
```

Add the `guardrails_off` parameter to the three existing answer tests (signatures only — bodies unchanged):

```python
def test_answer_question_builds_grounded_prompt(monkeypatch, guardrails_off):
def test_answer_question_threads_provider_to_generate(monkeypatch, guardrails_off):
def test_answer_question_empty_store(monkeypatch, guardrails_off):
```

Append the new tests:

```python
def _fake_llm(text):
    from llm.base import LLMResponse

    return LLMResponse(text=text, usage={"cache_read_input_tokens": 0})


def test_grading_filters_chunks_before_prompt(monkeypatch):
    import rag.answer as answer_mod

    monkeypatch.setattr(settings, "grading_enabled", True)
    monkeypatch.setattr(settings, "faithfulness_enabled", False)
    monkeypatch.setattr(answer_mod, "retrieve", lambda q, store=None: [
        _chunk(), _chunk(pid="1810.04805", title="BERT", text="bert stuff")])
    monkeypatch.setattr(answer_mod, "grade_chunks",
                        lambda q, chunks, provider=None: chunks[:1])
    captured = {}

    def fake_generate(messages, **kwargs):
        captured.update(kwargs)
        return _fake_llm("A [1706.03762].")

    monkeypatch.setattr(answer_mod, "generate", fake_generate)
    result = answer_mod.answer_question("q")
    assert result.sources == ["1706.03762"]              # BERT graded out
    assert "bert stuff" not in captured["system"][1]["text"]


def test_retry_fires_once_on_zero_survivors(monkeypatch):
    import rag.answer as answer_mod

    monkeypatch.setattr(settings, "grading_enabled", True)
    monkeypatch.setattr(settings, "faithfulness_enabled", False)
    retrievals = []
    monkeypatch.setattr(answer_mod, "retrieve",
                        lambda q, store=None: retrievals.append(q) or [_chunk()])
    grades = iter([[], [_chunk()]])  # first grade: nothing; retry grade: survivor
    monkeypatch.setattr(answer_mod, "grade_chunks",
                        lambda q, chunks, provider=None: next(grades))
    monkeypatch.setattr(answer_mod, "retry_rewrite_query",
                        lambda q, provider=None: "alternative query")
    monkeypatch.setattr(answer_mod, "generate",
                        lambda *a, **k: _fake_llm("A [1706.03762]."))
    result = answer_mod.answer_question("original q")
    assert retrievals == ["original q", "alternative query"]
    assert result.sources == ["1706.03762"]


def test_honest_degradation_skips_generate(monkeypatch):
    import rag.answer as answer_mod

    monkeypatch.setattr(settings, "grading_enabled", True)
    monkeypatch.setattr(settings, "faithfulness_enabled", False)
    monkeypatch.setattr(answer_mod, "retrieve", lambda q, store=None: [_chunk()])
    monkeypatch.setattr(answer_mod, "grade_chunks",
                        lambda q, chunks, provider=None: [])
    monkeypatch.setattr(answer_mod, "retry_rewrite_query",
                        lambda q, provider=None: "alt")
    monkeypatch.setattr(answer_mod, "generate",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no LLM call")))
    result = answer_mod.answer_question("q")
    assert result.sources == []
    assert result.faithful is None
    assert "ingest" in result.text.lower()


def test_retry_skipped_when_rewrite_fails_open(monkeypatch):
    import rag.answer as answer_mod

    monkeypatch.setattr(settings, "grading_enabled", True)
    monkeypatch.setattr(settings, "faithfulness_enabled", False)
    retrievals = []
    monkeypatch.setattr(answer_mod, "retrieve",
                        lambda q, store=None: retrievals.append(q) or [_chunk()])
    monkeypatch.setattr(answer_mod, "grade_chunks",
                        lambda q, chunks, provider=None: [])
    monkeypatch.setattr(answer_mod, "retry_rewrite_query",
                        lambda q, provider=None: q)  # failed open → identical query
    result = answer_mod.answer_question("q")
    assert retrievals == ["q"]  # no pointless second retrieval
    assert result.sources == []


def test_grading_disabled_never_calls_grader(monkeypatch):
    import rag.answer as answer_mod

    monkeypatch.setattr(settings, "grading_enabled", False)
    monkeypatch.setattr(settings, "faithfulness_enabled", False)
    monkeypatch.setattr(answer_mod, "retrieve", lambda q, store=None: [_chunk()])
    monkeypatch.setattr(answer_mod, "grade_chunks",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("grader must not run")))
    monkeypatch.setattr(answer_mod, "generate",
                        lambda *a, **k: _fake_llm("A [1706.03762]."))
    result = answer_mod.answer_question("q")
    assert result.sources == ["1706.03762"]


def test_faithfulness_verdict_attached(monkeypatch):
    import rag.answer as answer_mod

    monkeypatch.setattr(settings, "grading_enabled", False)
    monkeypatch.setattr(settings, "faithfulness_enabled", True)
    monkeypatch.setattr(answer_mod, "retrieve", lambda q, store=None: [_chunk()])
    monkeypatch.setattr(answer_mod, "generate",
                        lambda *a, **k: _fake_llm("A [1706.03762]."))
    captured = {}

    def fake_check(question, answer, contexts, provider=None):
        captured.update(question=question, answer=answer,
                        n=len(contexts), provider=provider)
        return False

    monkeypatch.setattr(answer_mod, "check_faithfulness", fake_check)
    result = answer_mod.answer_question("q", provider="local")
    assert result.faithful is False
    assert captured["n"] == 1
    assert captured["provider"] == "local"


def test_on_status_receives_pipeline_events(monkeypatch):
    import rag.answer as answer_mod

    monkeypatch.setattr(settings, "grading_enabled", True)
    monkeypatch.setattr(settings, "faithfulness_enabled", True)
    monkeypatch.setattr(answer_mod, "retrieve",
                        lambda q, store=None: [_chunk(), _chunk(pid="p2")])
    monkeypatch.setattr(answer_mod, "grade_chunks",
                        lambda q, chunks, provider=None: chunks[:1])
    monkeypatch.setattr(answer_mod, "check_faithfulness", lambda *a, **k: True)
    monkeypatch.setattr(answer_mod, "generate", lambda *a, **k: _fake_llm("A."))
    statuses = []
    answer_mod.answer_question("q", on_status=statuses.append)
    assert statuses == ["grading 2 chunks…", "1 of 2 chunks relevant",
                        "verifying citations…"]
```

- [ ] **Step 3: Run tests to verify the new ones fail**

Run: `uv run pytest tests/test_retrieve_answer.py -v`
Expected: new tests FAIL (`TypeError: answer_question() got an unexpected keyword argument 'on_status'`, missing `faithful`, etc.); updated existing tests still pass

- [ ] **Step 4: Rewrite `rag/answer.py`**

Full new content:

```python
import logging
from typing import Callable

from pydantic import BaseModel

from config import settings
from llm.base import generate
from llm.prompts import build_rag_prompt
from rag.faithfulness import check_faithfulness
from rag.grade import grade_chunks
from rag.retrieve import retrieve
from rag.rewrite import retry_rewrite_query
from rag.store import VectorStore

logger = logging.getLogger(__name__)

EMPTY_STORE_ANSWER = ("I don't have any ingested papers to answer from yet. "
                      "Ingest some papers first.")
NO_INFO_ANSWER = ("I don't have enough information in the ingested papers to "
                  "answer this. Try ingesting more papers on the topic.")


class RagAnswer(BaseModel):
    text: str
    sources: list[str]
    faithful: bool | None = None  # None = check disabled, skipped, or failed


def answer_question(
    question: str, store: VectorStore | None = None, provider: str | None = None,
    on_status: Callable[[str], None] | None = None,
) -> RagAnswer:
    """RAG query flow: retrieve → [grade → retry once] → grounded prompt →
    generate → [faithfulness].

    `provider` threads through the phase-5 LLM calls (grading, retry rewrite,
    faithfulness) and the final generate; retrieval's rewrite stage stays on
    the global setting. `on_status` (optional) receives human-readable progress
    lines for the UI activity feed. Guardrails fail open — see rag/grade.py
    and rag/faithfulness.py.
    """
    notify = on_status or (lambda text: None)
    chunks = retrieve(question, store=store)
    if not chunks:
        return RagAnswer(text=EMPTY_STORE_ANSWER, sources=[])

    if settings.grading_enabled:
        notify(f"grading {len(chunks)} chunks…")
        graded = grade_chunks(question, chunks, provider=provider)
        notify(f"{len(graded)} of {len(chunks)} chunks relevant")
        if not graded:
            retry_query = retry_rewrite_query(question, provider=provider)
            if retry_query != question:  # rewrite failed open → retry would repeat
                notify("retrying with rewritten query…")
                retried = retrieve(retry_query, store=store)
                # graded against the ORIGINAL question, like reranking
                graded = grade_chunks(question, retried, provider=provider)
                notify(f"{len(graded)} of {len(retried)} chunks relevant")
        if not graded:
            return RagAnswer(text=NO_INFO_ANSWER, sources=[])
        chunks = graded

    contexts = [{"paper_id": c.paper_id, "title": c.title, "text": c.text}
                for c in chunks]
    system, messages = build_rag_prompt(question, contexts)
    resp = generate(messages, system=system, provider=provider)
    logger.info(
        "answer usage: cache_read=%s cache_creation=%s",
        resp.usage.get("cache_read_input_tokens"),
        resp.usage.get("cache_creation_input_tokens"),
    )
    faithful = None
    if settings.faithfulness_enabled:
        notify("verifying citations…")
        faithful = check_faithfulness(question, resp.text, contexts,
                                      provider=provider)
    return RagAnswer(text=resp.text, sources=sorted({c.paper_id for c in chunks}),
                     faithful=faithful)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_retrieve_answer.py tests/test_config.py -v`
Expected: all pass

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest`
Expected: all pass. If `tests/test_eval_run.py` or `tests/test_graph.py` fail because a fake `answer_question` lacks the new kwarg, they are NOT expected to — `on_status` is passed only by Task 5's code. Investigate before proceeding; do not blanket-patch.

- [ ] **Step 7: Commit**

```bash
git add config.py rag/answer.py tests/test_retrieve_answer.py tests/test_config.py
git commit -m "feat: corrective retrieval loop and faithfulness verdict in answer_question"
```

---

### Task 5: Agent plumbing — status forwarding + verdict channel (`agents/graph.py`)

**Files:**
- Modify: `agents/graph.py`
- Modify: `tests/test_graph.py` (update 1 existing test fake, append new tests)

**Interfaces:**
- Consumes: `answer_question(..., on_status=...)` and `RagAnswer.faithful` (Task 4).
- Produces: `AgentResult` gains `faithful: bool | None = None` (4th NamedTuple field, after `checkpointed`); `_combine_verdicts(verdicts: list[bool | None]) -> bool | None` (importable — Task 6 uses it); `AgentState` gains `verdicts: Annotated[list, operator.add]`. Tasks 6–7 read `result.faithful`.

- [ ] **Step 1: Update the one existing test whose fake receives the new kwarg**

In `tests/test_graph.py::test_on_event_streams_deltas_and_statuses`, the fake is called with `on_status` (because `on_event` is set). Change its lambda:

```python
    monkeypatch.setattr(graph_mod, "answer_question",
                        lambda q, store=None, provider=None, on_status=None: RagAnswer(
                            text="A.", sources=["1706.03762"]))
```

(The other `answer_question` fakes in this file run without `on_event` and are never passed `on_status` — leave them.)

- [ ] **Step 2: Append the new failing tests**

Append to `tests/test_graph.py`:

```python
def test_combine_verdicts():
    from agents.graph import _combine_verdicts

    assert _combine_verdicts([]) is None
    assert _combine_verdicts([True, True]) is True
    assert _combine_verdicts([True, None]) is None
    assert _combine_verdicts([True, None, False]) is False


async def test_faithful_verdicts_collected_per_rag_query(monkeypatch):
    import agents.graph as graph_mod
    from rag.answer import RagAnswer

    verdicts = iter([True, False])

    def fake_answer(q, store=None, provider=None, on_status=None):
        return RagAnswer(text="A [p].", sources=["p"], faithful=next(verdicts))

    monkeypatch.setattr(graph_mod, "answer_question", fake_answer)
    _scripted_generate(monkeypatch, [
        LLMResponse(tool_calls=[
            ToolCall(id="tu_1", name="rag_query", input={"question": "a"}),
            ToolCall(id="tu_2", name="rag_query", input={"question": "b"}),
        ]),
        LLMResponse(text="done"),
    ])
    graph = graph_mod.build_graph(FakeToolbox())
    state = await graph.ainvoke({"messages": [{"role": "user", "content": "q"}],
                                 "steps": 0})
    assert state["verdicts"] == [True, False]


async def test_run_agent_returns_anded_faithful(monkeypatch, tmp_path):
    import agents.graph as graph_mod
    from config import settings
    from rag.answer import RagAnswer

    monkeypatch.setattr(settings, "checkpoint_db", str(tmp_path / "cp.db"))

    def fake_answer(q, store=None, provider=None, on_status=None):
        return RagAnswer(text="A [p].", sources=["p"], faithful=True)

    monkeypatch.setattr(graph_mod, "answer_question", fake_answer)
    _scripted_generate(monkeypatch, [
        LLMResponse(tool_calls=[ToolCall(id="tu_1", name="rag_query",
                                         input={"question": "q"})]),
        LLMResponse(text="grounded answer"),
    ])

    class FakeToolboxCM:
        async def __aenter__(self):
            return FakeToolbox()

        async def __aexit__(self, *exc):
            return None

    monkeypatch.setattr(graph_mod, "MCPToolbox", FakeToolboxCM)
    result = await graph_mod.run_agent("q")
    assert result.faithful is True
    assert result.citations == ["p"]


async def test_rag_query_statuses_forwarded_to_on_event(monkeypatch):
    import agents.graph as graph_mod
    from rag.answer import RagAnswer

    def fake_answer(q, store=None, provider=None, on_status=None):
        on_status("grading 2 chunks…")
        return RagAnswer(text="A.", sources=[], faithful=True)

    monkeypatch.setattr(graph_mod, "answer_question", fake_answer)
    script = [
        LLMResponse(tool_calls=[ToolCall(id="tu_1", name="rag_query",
                                         input={"question": "q"})]),
        LLMResponse(text="Final."),
    ]

    def fake_generate_stream(messages, **kwargs):
        resp = script.pop(0)
        if resp.text:
            kwargs["on_delta"](resp.text)
        return resp

    monkeypatch.setattr(graph_mod, "generate_stream", fake_generate_stream)
    events = []
    graph = graph_mod.build_graph(FakeToolbox(), on_event=events.append)
    await graph.ainvoke({"messages": [{"role": "user", "content": "q"}],
                         "steps": 0, "citations": []})
    status_texts = [e["text"] for e in events if e["event"] == "status"]
    assert "grading 2 chunks…" in status_texts
```

- [ ] **Step 3: Run tests to verify the new ones fail**

Run: `uv run pytest tests/test_graph.py -v`
Expected: new tests FAIL (`ImportError: cannot import name '_combine_verdicts'`, `KeyError: 'verdicts'`, `AttributeError: 'AgentResult' object has no attribute 'faithful'`); existing tests pass

- [ ] **Step 4: Implement in `agents/graph.py`**

4a. Extend `AgentResult` (keep field order — `faithful` LAST so existing positional constructions stay valid):

```python
class AgentResult(NamedTuple):
    text: str
    citations: list[str]
    # False when no checkpoint exists under the caller's thread_id (multi-mode
    # decomposed plans) — the API skips thread registration in that case.
    checkpointed: bool = True
    # AND of per-rag_query faithfulness verdicts for the whole thread
    # (verdicts, like citations, accumulate across a thread's turns):
    # any False → False; else any None → None; else True. None when no
    # rag_query ran or the check is disabled.
    faithful: bool | None = None
```

4b. Add the combiner after `_dedupe`:

```python
def _combine_verdicts(verdicts: list[bool | None]) -> bool | None:
    """AND with unknowns: False dominates, then None, else True. Empty → None."""
    if not verdicts:
        return None
    if any(v is False for v in verdicts):
        return False
    if any(v is None for v in verdicts):
        return None
    return True
```

4c. Extend `AgentState`:

```python
class AgentState(TypedDict):
    messages: Annotated[list[dict], operator.add]
    steps: int
    summary: str
    citations: Annotated[list[str], operator.add]
    verdicts: Annotated[list, operator.add]  # bool | None per rag_query call
```

4d. In `tools_node`, collect verdicts and forward statuses. Replace the `rag_query` branch and the return:

```python
    async def tools_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        results: list[dict] = []
        sources: list[str] = []
        verdicts: list[bool | None] = []
        for block in last["content"]:
            if block["type"] != "tool_use":
                continue
            name, args = block["name"], block["input"]
            if on_event is not None:
                on_event({"event": "status", "text": f"calling {name}…"})
            logger.info("Tool call: %s(%s)", name, args)
            if name == "rag_query":
                try:
                    kwargs = {"provider": provider}
                    if on_event is not None:
                        kwargs["on_status"] = (
                            lambda t: on_event({"event": "status", "text": t}))
                    ans = await asyncio.to_thread(
                        functools.partial(answer_question, args["question"],
                                          **kwargs))
                    content = f"{ans.text}\n\nSources: {', '.join(ans.sources) or 'none'}"
                    is_error = False
                    sources.extend(ans.sources)
                    verdicts.append(ans.faithful)
                except Exception as exc:  # e.g. Qdrant down — agent decides what to do
                    content, is_error = f"rag_query failed: {exc}", True
            else:
                content, is_error = await toolbox.call_tool(name, args)
            results.append({"type": "tool_result", "tool_use_id": block["id"],
                            "content": content, "is_error": is_error})
        return {
            "messages": [{"role": "user", "content": results}],
            "steps": state["steps"] + 1,
            "citations": sources,
            "verdicts": verdicts,
        }
```

(`on_status` is passed only when `on_event` exists, so existing no-event fakes keep their 3-arg signatures.)

4e. In `run_agent`, seed the channel and return the verdict — the `ainvoke` input dict gains `"verdicts": []` and the return becomes:

```python
        state = await graph.ainvoke(
            {"messages": [{"role": "user", "content": question}], "steps": 0,
             "citations": [], "verdicts": []},
            config={"recursion_limit": settings.agent_max_steps * 2 + 6,
                    "configurable": {"thread_id": thread_id}},
        )
        text = final_text(state)
        return AgentResult(text=text or STEP_LIMIT_MESSAGE,
                           citations=_dedupe(state.get("citations", [])),
                           faithful=_combine_verdicts(state.get("verdicts", [])))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_graph.py -v`
Expected: all pass

- [ ] **Step 6: Run the full suite, then commit**

Run: `uv run pytest`
Expected: all pass

```bash
git add agents/graph.py tests/test_graph.py
git commit -m "feat: collect faithfulness verdicts through agent state; forward grading statuses"
```

---

### Task 6: Multi-agent verdict propagation (`agents/multi.py`)

**Files:**
- Modify: `agents/multi.py`
- Modify: `tests/test_multi.py` (append tests)

**Interfaces:**
- Consumes: `AgentResult.faithful` and `_combine_verdicts` (Task 5).
- Produces: `run_multi_agent` returns `AgentResult` with `faithful` = AND over researcher verdicts (a failed researcher contributes `None`). The synthesizer's own output is NOT re-checked — the verdict covers the underlying rag answers (documented in the module docstring).

- [ ] **Step 1: Append the failing tests**

Append to `tests/test_multi.py`:

```python
async def test_multi_ands_researcher_verdicts(monkeypatch):
    import agents.multi as multi_mod
    from agents.graph import AgentResult

    monkeypatch.setattr(multi_mod, "_plan", lambda q, provider=None: multi_mod.Plan(
        simple=False, sub_questions=["a", "b"]))
    results = iter([
        AgentResult(text="fa", citations=["p1"], faithful=True),
        AgentResult(text="fb", citations=["p2"], faithful=False),
    ])

    async def fake_run_agent(q, thread_id=None, provider=None, on_event=None):
        return next(results)

    monkeypatch.setattr(multi_mod, "run_agent", fake_run_agent)
    monkeypatch.setattr(multi_mod, "_synthesize",
                        lambda q, f, provider=None, on_delta=None: "combined")
    result = await multi_mod.run_multi_agent("q")
    assert result.faithful is False
    assert result.checkpointed is False


async def test_multi_failed_researcher_yields_unknown_verdict(monkeypatch):
    import agents.multi as multi_mod
    from agents.graph import AgentResult

    monkeypatch.setattr(multi_mod, "_plan", lambda q, provider=None: multi_mod.Plan(
        simple=False, sub_questions=["a", "b"]))
    calls = {"n": 0}

    async def fake_run_agent(q, thread_id=None, provider=None, on_event=None):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("researcher exploded")
        return AgentResult(text="fa", citations=["p1"], faithful=True)

    monkeypatch.setattr(multi_mod, "run_agent", fake_run_agent)
    monkeypatch.setattr(multi_mod, "_synthesize",
                        lambda q, f, provider=None, on_delta=None: "combined")
    result = await multi_mod.run_multi_agent("q")
    assert result.faithful is None  # [True, None] → unknown, not verified
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_multi.py -v`
Expected: new tests FAIL (`result.faithful` is `None` in the first test where `False` is expected — the field defaults); existing tests pass

- [ ] **Step 3: Implement in `agents/multi.py`**

Change the import from `agents.graph`:

```python
from agents.graph import AgentResult, _combine_verdicts, _dedupe, run_agent
```

In `run_multi_agent`, track verdicts alongside citations:

```python
    findings: list[tuple[str, str]] = []
    citations: list[str] = []
    verdicts: list[bool | None] = []
    for sub_question in plan.sub_questions[:4]:
        if on_event is not None:
            on_event({"event": "status", "text": f"researching: {sub_question}"})
        try:
            # researchers run silently — only the synthesizer token-streams
            result = await run_agent(sub_question, provider=provider)
            findings.append((sub_question, result.text))
            citations.extend(result.citations)
            verdicts.append(result.faithful)
        except Exception as exc:
            logger.exception("Researcher failed for %r", sub_question)
            findings.append((sub_question, f"FAILED: {exc}"))
            verdicts.append(None)
```

And the return:

```python
    return AgentResult(text=text, citations=_dedupe(citations),
                       checkpointed=False,
                       faithful=_combine_verdicts(verdicts))
```

Append to the module docstring's last paragraph: `The faithfulness verdict is the AND of the researchers' verdicts — the synthesizer's own composition is not re-checked.`

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_multi.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add agents/multi.py tests/test_multi.py
git commit -m "feat: propagate faithfulness verdicts through multi-agent supervisor"
```

---

### Task 7: API — `faithful` in ChatResponse and SSE done event

**Files:**
- Modify: `api/main.py`
- Modify: `tests/test_api.py` (update 1 existing test, append 1)
- Modify: `tests/test_api_stream.py` (append 1)

**Interfaces:**
- Consumes: `AgentResult.faithful` (Task 5).
- Produces: `ChatResponse.faithful: bool | None = None`; SSE `done` event JSON gains `"faithful"`. Task 8's frontend reads `data.faithful` from the `done` event.

- [ ] **Step 1: Update the exact-equality test and append the new tests**

In `tests/test_api.py::test_chat_reuses_given_thread_id`, the response JSON gains a field — update the expected dict:

```python
    assert resp.json() == {"reply": "echo: follow-up [t-42]", "thread_id": "t-42",
                           "citations": [], "faithful": None}
```

Append to `tests/test_api.py`:

```python
def test_chat_returns_faithful(monkeypatch):
    import api.main as api_main
    from agents.graph import AgentResult

    async def fake_run_chat(question, thread_id=None, provider=None):
        return AgentResult(text="grounded", citations=["1706.03762"], faithful=False)

    monkeypatch.setattr(api_main, "run_chat", fake_run_chat)
    with _client(monkeypatch) as client:
        resp = client.post("/api/chat", json={"message": "hi"})
    assert resp.json()["faithful"] is False
```

Append to `tests/test_api_stream.py`:

```python
def test_stream_done_carries_faithful(monkeypatch):
    import api.main as api_main
    from agents.graph import AgentResult
    from fastapi.testclient import TestClient

    class FakeStore:
        def ping(self):
            pass

        def check_schema(self):
            pass

    async def fake_run_chat(question, thread_id=None, provider=None, on_event=None):
        return AgentResult(text="grounded", citations=["1706.03762"], faithful=False)

    monkeypatch.setattr(api_main, "VectorStore", FakeStore)
    monkeypatch.setattr(api_main, "run_chat", fake_run_chat)
    with TestClient(api_main.app) as client:
        resp = client.post("/api/chat/stream", json={"message": "hi"})
    assert resp.status_code == 200
    assert "event: done" in resp.text
    assert '"faithful": false' in resp.text
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `uv run pytest tests/test_api.py tests/test_api_stream.py -v`
Expected: `test_chat_returns_faithful` FAILS (`KeyError`/`None`), `test_stream_done_carries_faithful` FAILS, `test_chat_reuses_given_thread_id` FAILS until implementation lands

- [ ] **Step 3: Implement in `api/main.py`**

`ChatResponse`:

```python
class ChatResponse(BaseModel):
    reply: str
    thread_id: str
    citations: list[str] = []
    faithful: bool | None = None  # False → UI shows "citations unverified"
```

`chat` endpoint return:

```python
    return ChatResponse(reply=result.text, thread_id=thread_id,
                        citations=result.citations, faithful=result.faithful)
```

`chat_stream` worker `done` event:

```python
            await queue.put({"event": "done", "reply": result.text,
                             "thread_id": thread_id, "citations": result.citations,
                             "faithful": result.faithful})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_api.py tests/test_api_stream.py -v`
Expected: all pass

- [ ] **Step 5: Run the full suite, then commit**

Run: `uv run pytest`
Expected: all pass

```bash
git add api/main.py tests/test_api.py tests/test_api_stream.py
git commit -m "feat: expose faithfulness verdict on chat response and SSE done event"
```

---

### Task 8: UI — "citations unverified" badge

**Files:**
- Modify: `api/static/app.js`
- Modify: `api/static/index.html`

**Interfaces:**
- Consumes: `data.faithful` on the SSE `done` event (Task 7).
- Produces: amber `.verdict` badge rendered after the citation chips ONLY when `faithful === false`. `true`/`null` render nothing. Restored transcripts never show it (live-only by design).

- [ ] **Step 1: Add the badge renderer to `api/static/app.js`**

After the `addCitations` function:

```javascript
function addVerdict(faithful) {
  // Only an explicit false is worth a warning; true/null stay quiet.
  if (faithful !== false) return;
  log.appendChild(el("div", "verdict", "⚠ citations unverified"));
}
```

In the `done` branch of `sendMessage`'s SSE handler, after `addCitations(data.citations);`:

```javascript
          addCitations(data.citations);
          addVerdict(data.faithful);
```

- [ ] **Step 2: Add the badge style to `api/static/index.html`**

After the `.citations a:hover` rule in the `<style>` block:

```css
    .verdict { display: inline-block; font-size: .75rem; background: #fff3cd; color: #8a6d1a; border-radius: 10px; padding: .1rem .5rem; margin: .25rem 0 .75rem; }
```

- [ ] **Step 3: Verify statics still serve and nothing regressed**

Run: `uv run pytest tests/test_api.py -q`
Expected: pass (includes `test_index_served`)

Optional manual smoke (needs Qdrant + a provider): `uv run uvicorn api.main:app --reload`, ask a question, temporarily set `FAITHFULNESS_ENABLED=true` with a weak local model to see the badge, or fake it by returning `false` from the API.

- [ ] **Step 4: Commit**

```bash
git add api/static/app.js api/static/index.html
git commit -m "feat: unverified-citations warning badge in chat UI"
```

---

### Task 9: Eval — grading ablation preset + faithfulness rate

**Files:**
- Modify: `eval/run.py`
- Modify: `tests/test_eval_run.py` (append tests)

**Interfaces:**
- Consumes: `RagAnswer.faithful` (Task 4); existing `run_eval`/`PRESETS`/`run_ablation`.
- Produces: `PRESETS` rows all pin `grading_enabled` and `faithfulness_enabled`; new preset `hybrid+rerank+grade`; `_faithfulness_rate(rows) -> float | None`; `run_eval` rows gain `"faithful"`, summary gains `"faithfulness_rate"`.

- [ ] **Step 1: Append the failing tests**

Append to `tests/test_eval_run.py`:

```python
def test_ablation_presets_pin_phase5_flags():
    from eval.run import PRESETS

    assert "hybrid+rerank+grade" in PRESETS
    for name, preset in PRESETS.items():
        # every preset must pin both new flags so rows stay comparable
        assert "grading_enabled" in preset, name
        assert preset["faithfulness_enabled"] is False, name  # not a retrieval technique
    assert PRESETS["hybrid+rerank+grade"]["grading_enabled"] is True
    assert PRESETS["hybrid+rerank"]["grading_enabled"] is False
    assert PRESETS["full"]["grading_enabled"] is True


def test_faithfulness_rate():
    import pytest

    from eval.run import _faithfulness_rate

    assert _faithfulness_rate([]) is None
    assert _faithfulness_rate([{"faithful": None}]) is None
    rows = [{"faithful": True}, {"faithful": False},
            {"faithful": None}, {"faithful": True}]
    assert _faithfulness_rate(rows) == pytest.approx(2 / 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_eval_run.py -v`
Expected: new tests FAIL (`KeyError`, `ImportError: cannot import name '_faithfulness_rate'`); existing tests pass

- [ ] **Step 3: Implement in `eval/run.py`**

3a. Add the helper above `run_eval`:

```python
def _faithfulness_rate(rows: list[dict]) -> float | None:
    """Fraction of non-None verdicts that are True; None when nothing was checked."""
    verdicts = [r["faithful"] for r in rows if r.get("faithful") is not None]
    if not verdicts:
        return None
    return sum(1 for v in verdicts if v) / len(verdicts)
```

3b. In `run_eval`, record the verdict per row — inside the `rows.append({...})` dict, after `"recall": recall,`:

```python
            "faithful": answer.faithful,
```

3c. Add the rate to the summary dict, after `"avg_citation_accuracy"`:

```python
        "faithfulness_rate": _faithfulness_rate(rows),
```

3d. Replace `PRESETS` (every row pins the phase-5 flags; faithfulness stays off during ablation — it's an answer-side guardrail, not a retrieval technique, and it would add one LLM call per item per preset):

```python
PRESETS: dict[str, dict] = {
    "baseline-dense": {"retrieval_mode": "dense", "rerank_enabled": False,
                       "rewrite_enabled": False, "grading_enabled": False,
                       "faithfulness_enabled": False},
    "sparse": {"retrieval_mode": "sparse", "rerank_enabled": False,
               "rewrite_enabled": False, "grading_enabled": False,
               "faithfulness_enabled": False},
    "hybrid": {"retrieval_mode": "hybrid", "rerank_enabled": False,
               "rewrite_enabled": False, "grading_enabled": False,
               "faithfulness_enabled": False},
    "hybrid+rerank": {"retrieval_mode": "hybrid", "rerank_enabled": True,
                      "rewrite_enabled": False, "grading_enabled": False,
                      "faithfulness_enabled": False},
    "hybrid+rerank+grade": {"retrieval_mode": "hybrid", "rerank_enabled": True,
                            "rewrite_enabled": False, "grading_enabled": True,
                            "faithfulness_enabled": False},
    "full": {"retrieval_mode": "hybrid", "rerank_enabled": True,
             "rewrite_enabled": True, "grading_enabled": True,
             "faithfulness_enabled": False},
}
```

3e. In `run_ablation`, extend the save/restore field list:

```python
    fields = ["retrieval_mode", "rerank_enabled", "rewrite_enabled",
              "grading_enabled", "faithfulness_enabled"]
```

3f. In `main()`, print the rate after the citation-accuracy line:

```python
    if s["faithfulness_rate"] is not None:
        print(f"  verified answers    : {s['faithfulness_rate']:.0%}")
```

(`_print_ablation` columns stay unchanged — `faithfulness_rate` is None for all presets.)

- [ ] **Step 4: Run the eval tests, then the full suite**

Run: `uv run pytest tests/test_eval_run.py -v` then `uv run pytest`
Expected: all pass. If an existing test in `test_eval_run.py` asserts the exact summary key set, add `"faithfulness_rate": None` to its expectation — that is the only legitimate delta.

- [ ] **Step 5: Commit**

```bash
git add eval/run.py tests/test_eval_run.py
git commit -m "feat: grading ablation preset and faithfulness rate in eval harness"
```

---

### Task 10: Local-model grading test + README

**Files:**
- Create: `tests/test_local_grade.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `grade_chunks` (Task 1). Requires Ollama running for the `local` marker; skipped in default runs.

- [ ] **Step 1: Write the local marker test**

Create `tests/test_local_grade.py`:

```python
"""Real-Ollama grading smoke test: uv run pytest -m local"""

import pytest

from rag.grade import grade_chunks
from rag.store import ScoredChunk

pytestmark = pytest.mark.local


def test_local_grading_returns_subset_and_never_errors():
    chunks = [
        ScoredChunk(paper_id="1706.03762", title="Attention Is All You Need",
                    text="The Transformer relies entirely on self-attention, "
                         "dispensing with recurrence and convolutions.", score=1.0),
        ScoredChunk(paper_id="9999.00001", title="Sourdough Baking Basics",
                    text="Preheat the oven to 230C and score the loaf before "
                         "baking for an open crumb.", score=0.5),
    ]
    kept = grade_chunks("What architecture does the Transformer use?", chunks,
                        provider="local")
    # fail-open means a flaky 3B grade may keep everything; the hard guarantees
    # are: no exception, output is a subset, order preserved
    assert all(c in chunks for c in kept)
    assert [c.paper_id for c in kept] == [c.paper_id for c in chunks
                                          if c in kept]
```

Run (only if Ollama is up, otherwise skip this run): `uv run pytest -m local tests/test_local_grade.py -v`
Also run: `uv run pytest tests/test_local_grade.py -q` — expected: deselected/skipped under the default marker filter, 0 failures.

- [ ] **Step 2: Update `README.md`**

2a. Status table — add a row after phase 4:

```markdown
| 5 | Corrective RAG (LLM chunk grading + one rewritten-query retry + honest degradation) and a citation-faithfulness guardrail with an "unverified citations" badge in the UI |
```

2b. Change the status line `## Status: complete (all 4 phases shipped)` to:

```markdown
## Status: complete (all 5 phases shipped)
```

2c. In the retrieval-pipeline paragraph (the one describing `[rewrite] → embed → search…`), update the stage list and flags sentence to:

```markdown
Retrieval is a staged pipeline — `[rewrite] → embed → search (dense|sparse|hybrid) → [rerank] → [grade → retry once]` —
controlled by `.env` flags (`RETRIEVAL_MODE`, `RERANK_ENABLED`, `REWRITE_ENABLED`, `GRADING_ENABLED`; see `config.py`).
After answering, `FAITHFULNESS_ENABLED` runs a citation-faithfulness check; an
unsupported answer gets an "⚠ citations unverified" badge in the UI (live
responses only — verdicts aren't persisted, so restored threads never show it).
Both guardrails fail open and add 1–3 LLM calls per request — turn them off on
slow local models if latency hurts.
```

- [ ] **Step 3: Full suite, then commit**

Run: `uv run pytest`
Expected: all pass

```bash
git add tests/test_local_grade.py README.md
git commit -m "test: real-Ollama grading smoke test; docs: phase 5 README"
```
