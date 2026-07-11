# Phase 6: Synthetic Eval Data + Bootstrap CIs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Grow the 3-item golden dataset via LLM-generated, self-check-filtered synthetic questions, and put 95% bootstrap confidence intervals on every eval metric and ablation cell.

**Architecture:** Two new stdlib-only modules — `eval/generate.py` (sample chunks → generate question/gist → fail-closed self-check → `eval/golden-synthetic.json`) and `eval/stats.py` (`bootstrap_ci`, percentile method). `eval/run.py` auto-concatenates the synthetic file and renders CIs. `VectorStore` gains `sample_chunks`. Spec: `docs/superpowers/specs/2026-07-12-phase-6-synthetic-eval-and-cis-design.md`.

**Tech Stack:** Python 3.12, pydantic structured output, qdrant scroll API, stdlib `random`/`statistics`, pytest. Run everything with `uv run`.

## Global Constraints

- No new dependencies. No numpy/scipy/RAGAS — bootstrap uses stdlib `random`; the "no eval framework" stance stands.
- **Fail-closed filter** (inverse of phase 5's fail-open guardrails): a synthetic candidate is DROPPED on self-check "no", parse failure, or any LLM error. A bad kept item poisons every future metric run.
- Self-check output is LINE format (`answerable: yes`), never JSON — 3B local models follow it more reliably.
- Synthetic items live in `eval/golden-synthetic.json` (full overwrite on regeneration), shape = golden.json item + `"synthetic": true`, `expected_paper_ids` = exactly one paper id (the source chunk's).
- Dataset resolution in eval: explicit `dataset_path` → that file ONLY; default (`None`) → `eval/golden.json` + `eval/golden-synthetic.json` when the latter exists.
- `bootstrap_ci(values, n_resamples=1000, alpha=0.05, seed=0) -> tuple[float, float]`, percentile method, deterministic per seed; `len(values) < 2` → zero-width `(mean, mean)`; empty list raises ValueError.
- CI display: summary print `0.67 [0.51, 0.82]`; ablation cells `0.67 ±0.08` (half-width); report JSON stores `[lo, hi]` lists under `<metric>_ci` keys.
- Per-request `provider` threads through both generation LLM calls.
- Unit tests are mocked, need no keys, run with `uv run pytest`.
- Commit messages: imperative conventional style (`feat:`, `test:`, `docs:`), matching `git log`.

---

### Task 1: `eval/stats.py` — bootstrap confidence intervals

**Files:**
- Create: `eval/stats.py`
- Create: `tests/test_stats.py`

**Interfaces:**
- Consumes: nothing (stdlib only).
- Produces: `bootstrap_ci(values: list[float], n_resamples: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]` — Task 4 calls this from `eval/run.py`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_stats.py`:

```python
import pytest


def test_ci_brackets_the_mean():
    from statistics import mean

    from eval.stats import bootstrap_ci

    values = [0.2, 0.4, 0.6, 0.8, 1.0, 0.0, 0.5, 0.7]
    lo, hi = bootstrap_ci(values, seed=0)
    assert lo <= mean(values) <= hi
    assert lo < hi  # spread data → non-degenerate interval


def test_ci_deterministic_for_fixed_seed():
    from eval.stats import bootstrap_ci

    values = [0.1, 0.9, 0.4, 0.6, 0.5]
    assert bootstrap_ci(values, seed=7) == bootstrap_ci(values, seed=7)
    # a different seed is allowed to differ (not asserted — could collide)


def test_single_value_zero_width():
    from eval.stats import bootstrap_ci

    assert bootstrap_ci([0.5]) == (0.5, 0.5)


def test_identical_values_zero_width():
    from eval.stats import bootstrap_ci

    lo, hi = bootstrap_ci([0.7, 0.7, 0.7, 0.7], seed=0)
    assert lo == hi == 0.7


def test_empty_values_raise():
    from eval.stats import bootstrap_ci

    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_narrower_alpha_widens_interval():
    from eval.stats import bootstrap_ci

    values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.3, 0.7, 0.5, 0.9]
    lo95, hi95 = bootstrap_ci(values, alpha=0.05, seed=0)
    lo50, hi50 = bootstrap_ci(values, alpha=0.50, seed=0)
    # 95% interval must contain the 50% interval
    assert lo95 <= lo50 and hi50 <= hi95
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_stats.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval.stats'`

- [ ] **Step 3: Write `eval/stats.py`**

```python
"""Bootstrap confidence intervals — stdlib only, no numpy/scipy.

Percentile method: resample the per-row values with replacement, take each
resample's mean, and read the interval straight off the sorted means. Good
enough for eval-report error bars; not a substitute for a real power
analysis.
"""

import random
from statistics import mean


def bootstrap_ci(values: list[float], n_resamples: int = 1000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """(1 - alpha) CI of the mean. Deterministic for a given seed.

    len(values) < 2 carries no spread information → zero-width (mean, mean).
    An empty list has no mean at all → ValueError.
    """
    if not values:
        raise ValueError("bootstrap_ci needs at least one value")
    if len(values) < 2:
        return (values[0], values[0])
    rng = random.Random(seed)
    means = sorted(mean(rng.choices(values, k=len(values)))
                   for _ in range(n_resamples))
    lo_i = int((alpha / 2) * n_resamples)
    hi_i = int((1 - alpha / 2) * n_resamples) - 1
    return (means[lo_i], means[hi_i])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_stats.py -v`
Expected: 6 passed

- [ ] **Step 5: Run the full suite, then commit**

Run: `uv run pytest`
Expected: all pass (nothing existing touched)

```bash
git add eval/stats.py tests/test_stats.py
git commit -m "feat: stdlib bootstrap confidence intervals for eval metrics"
```

---

### Task 2: `VectorStore.sample_chunks` — round-robin chunk sampling

**Files:**
- Modify: `rag/store.py` (add `import random` and one method)
- Modify: `tests/test_store.py` (append tests)

**Interfaces:**
- Consumes: qdrant client `scroll(collection_name, limit, offset, with_payload, with_vectors)` returning `(points, next_offset)`; point payloads carry `paper_id`, `title`, `chunk_text`.
- Produces: `VectorStore.sample_chunks(n: int, seed: int = 0) -> list[dict]` — dicts `{paper_id, title, text}`, shuffled, round-robin across papers, deterministic per seed. Task 3 consumes this.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_store.py` (a `FakeQdrant` class exists at the top of the file; these tests use their own minimal fake to control scroll pagination precisely):

```python
class ScrollFake:
    """Minimal qdrant fake for scroll pagination: two pages, then done."""

    def __init__(self, points):
        self._points = points

    def scroll(self, collection_name, limit=None, offset=None,
               with_payload=None, with_vectors=None, scroll_filter=None):
        start = offset or 0
        page = self._points[start:start + limit]
        next_offset = start + limit if start + limit < len(self._points) else None
        return page, next_offset


def _point(pid, idx):
    from types import SimpleNamespace

    return SimpleNamespace(payload={"paper_id": pid, "title": f"T-{pid}",
                                    "chunk_text": f"text-{pid}-{idx}",
                                    "chunk_index": idx, "section": ""})


def test_sample_chunks_round_robin_across_papers():
    from rag.store import VectorStore

    points = [_point("paperA", i) for i in range(5)] + [_point("paperB", 0)]
    store = VectorStore(client=ScrollFake(points))
    out = store.sample_chunks(2, seed=0)
    # one heavily-chunked paper must not take both slots
    assert sorted({c["paper_id"] for c in out}) == ["paperA", "paperB"]
    assert set(out[0]) == {"paper_id", "title", "text"}


def test_sample_chunks_deterministic_per_seed():
    from rag.store import VectorStore

    points = [_point("paperA", i) for i in range(6)] + \
             [_point("paperB", i) for i in range(6)]
    a = VectorStore(client=ScrollFake(points)).sample_chunks(4, seed=3)
    b = VectorStore(client=ScrollFake(points)).sample_chunks(4, seed=3)
    assert a == b


def test_sample_chunks_paginates_and_caps_at_n():
    from rag.store import VectorStore

    points = [_point(f"p{i}", 0) for i in range(300)]  # > one 256 scroll page
    store = VectorStore(client=ScrollFake(points))
    out = store.sample_chunks(10, seed=0)
    assert len(out) == 10
    assert len({c["paper_id"] for c in out}) == 10  # all distinct papers


def test_sample_chunks_exhausted_supply_returns_all():
    from rag.store import VectorStore

    points = [_point("paperA", 0), _point("paperB", 0)]
    store = VectorStore(client=ScrollFake(points))
    assert len(store.sample_chunks(50, seed=0)) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_store.py -v -k sample_chunks`
Expected: FAIL — `AttributeError: 'VectorStore' object has no attribute 'sample_chunks'`

- [ ] **Step 3: Implement in `rag/store.py`**

Add `import random` at the top (after `import uuid`). Append this method to `VectorStore` (after `has_paper`):

```python
    def sample_chunks(self, n: int, seed: int = 0) -> list[dict]:
        """Up to n chunk payloads for synthetic eval generation (phase 6).

        Scrolls the whole collection (payloads only), shuffles with the given
        seed, then draws round-robin across paper_ids so a heavily-chunked
        paper cannot dominate the sample. Deterministic per seed.
        """
        points, offset = [], None
        while True:
            batch, offset = self.client.scroll(
                collection_name=self.collection, limit=256, offset=offset,
                with_payload=True, with_vectors=False,
            )
            points.extend(batch)
            if offset is None:
                break
        rng = random.Random(seed)
        rng.shuffle(points)
        by_paper: dict[str, list] = {}
        for p in points:
            by_paper.setdefault(p.payload["paper_id"], []).append(p)
        ordered_papers = sorted(by_paper)  # stable draw order across runs
        out: list[dict] = []
        while len(out) < n and any(by_paper.values()):
            for pid in ordered_papers:
                if by_paper[pid] and len(out) < n:
                    p = by_paper[pid].pop()
                    out.append({"paper_id": p.payload["paper_id"],
                                "title": p.payload["title"],
                                "text": p.payload["chunk_text"]})
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_store.py -v`
Expected: all pass (existing store tests untouched)

- [ ] **Step 5: Run the full suite, then commit**

Run: `uv run pytest`
Expected: all pass

```bash
git add rag/store.py tests/test_store.py
git commit -m "feat: seeded round-robin chunk sampling for synthetic eval generation"
```

---

### Task 3: `eval/generate.py` — synthetic item generation with fail-closed self-check

**Files:**
- Create: `eval/generate.py`
- Create: `tests/test_generate.py`
- Modify: `llm/prompts.py` (append 2 prompts)

**Interfaces:**
- Consumes: `VectorStore.sample_chunks(n, seed)` (Task 2); `llm.base.generate(messages, *, system=..., structured_schema=..., provider=...)`.
- Produces: `generate_dataset(count: int, provider: str | None = None, seed: int = 0, out_path: str = "eval/golden-synthetic.json", store=None) -> dict` returning `{"kept": int, "rejected": int, "requested": int}`; CLI `python -m eval.generate --count N [--provider P] [--seed S]`. Output file consumed by Task 4's dataset resolution.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_generate.py`:

```python
import json

from llm.base import LLMResponse

CHUNKS = [
    {"paper_id": "1706.03762", "title": "Attention", "text": "self-attention text"},
    {"paper_id": "1810.04805", "title": "BERT", "text": "masked LM text"},
    {"paper_id": "2005.14165", "title": "GPT-3", "text": "few-shot text"},
]


class FakeStore:
    def __init__(self, chunks=CHUNKS):
        self._chunks = chunks

    def ping(self):
        pass

    def check_schema(self):
        pass

    def sample_chunks(self, n, seed=0):
        return list(self._chunks)[:n]


def _patch_generate(monkeypatch, check_text="answerable: yes\nfaithful: yes",
                    gen_exc=None, check_exc=None):
    """Fake llm.base.generate: dispatches on which system prompt arrives."""
    import eval.generate as gen_mod
    from eval.generate import Candidate
    from llm import prompts

    calls = []

    def fake_generate(messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        if kwargs.get("system") == prompts.SYNTH_QUESTION_SYSTEM_PROMPT:
            if gen_exc is not None:
                raise gen_exc
            return LLMResponse(parsed=Candidate(
                question="What mechanism does the Transformer rely on?",
                expected_answer_gist="It relies on self-attention."))
        assert kwargs.get("system") == prompts.SYNTH_CHECK_SYSTEM_PROMPT
        if check_exc is not None:
            raise check_exc
        return LLMResponse(text=check_text)

    monkeypatch.setattr(gen_mod, "generate", fake_generate)
    return calls


def test_kept_item_has_golden_shape(monkeypatch, tmp_path):
    from eval.generate import generate_dataset

    _patch_generate(monkeypatch)
    out = tmp_path / "synth.json"
    stats = generate_dataset(count=2, store=FakeStore(), out_path=str(out))
    items = json.loads(out.read_text())
    assert stats == {"kept": 2, "rejected": 0, "requested": 2}
    assert len(items) == 2
    item = items[0]
    assert set(item) == {"question", "expected_paper_ids",
                         "expected_answer_gist", "synthetic"}
    assert item["synthetic"] is True
    assert item["expected_paper_ids"] == ["1706.03762"]  # single source paper
    assert item["question"] and item["expected_answer_gist"]


def test_self_check_no_drops_item(monkeypatch, tmp_path):
    from eval.generate import generate_dataset

    _patch_generate(monkeypatch, check_text="answerable: yes\nfaithful: no")
    out = tmp_path / "synth.json"
    stats = generate_dataset(count=3, store=FakeStore(), out_path=str(out))
    assert stats["kept"] == 0
    assert stats["rejected"] == 3
    assert json.loads(out.read_text()) == []


def test_self_check_garbage_drops_item(monkeypatch, tmp_path):
    from eval.generate import generate_dataset

    _patch_generate(monkeypatch, check_text="Looks great to me!")
    stats = generate_dataset(count=1, store=FakeStore(),
                             out_path=str(tmp_path / "s.json"))
    assert stats["kept"] == 0 and stats["rejected"] >= 1


def test_generation_error_drops_item_fail_closed(monkeypatch, tmp_path):
    from eval.generate import generate_dataset

    _patch_generate(monkeypatch, gen_exc=RuntimeError("api down"))
    stats = generate_dataset(count=2, store=FakeStore(),
                             out_path=str(tmp_path / "s.json"))
    assert stats["kept"] == 0


def test_check_error_drops_item_fail_closed(monkeypatch, tmp_path):
    from eval.generate import generate_dataset

    _patch_generate(monkeypatch, check_exc=RuntimeError("api down"))
    stats = generate_dataset(count=2, store=FakeStore(),
                             out_path=str(tmp_path / "s.json"))
    assert stats["kept"] == 0


def test_count_honored_and_stops_early(monkeypatch, tmp_path):
    from eval.generate import generate_dataset

    calls = _patch_generate(monkeypatch)
    stats = generate_dataset(count=1, store=FakeStore(),
                             out_path=str(tmp_path / "s.json"))
    assert stats["kept"] == 1
    # 1 kept item = exactly 2 LLM calls (generate + check); no work on chunk 2+
    assert len(calls) == 2


def test_exhausted_supply_writes_partial_set(monkeypatch, tmp_path):
    from eval.generate import generate_dataset

    _patch_generate(monkeypatch)
    out = tmp_path / "s.json"
    stats = generate_dataset(count=50, store=FakeStore(),  # only 3 chunks exist
                             out_path=str(out))
    assert stats["kept"] == 3
    assert len(json.loads(out.read_text())) == 3


def test_provider_threads_through_both_calls(monkeypatch, tmp_path):
    from eval.generate import generate_dataset

    calls = _patch_generate(monkeypatch)
    generate_dataset(count=1, provider="local", store=FakeStore(),
                     out_path=str(tmp_path / "s.json"))
    assert all(c["provider"] == "local" for c in calls)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_generate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval.generate'`

- [ ] **Step 3: Append the prompts to `llm/prompts.py`**

Append at end of file:

```python
SYNTH_QUESTION_SYSTEM_PROMPT = """You write evaluation questions for a research-paper QA system.

Given one paper excerpt, produce:
- question: ONE specific, self-contained exam question answerable from the excerpt alone. Never say "the excerpt", "the text", or "this paper" — name the concepts instead.
- expected_answer_gist: 1-2 sentences stating what a correct answer must convey, based only on the excerpt."""


SYNTH_CHECK_SYSTEM_PROMPT = """You review a candidate evaluation question against the excerpt it was generated from.

Output exactly two lines and nothing else:
answerable: yes|no   (can the question be fully answered from the excerpt alone?)
faithful: yes|no     (is the expected gist supported by the excerpt?)"""
```

- [ ] **Step 4: Write `eval/generate.py`**

```python
"""Synthetic eval-item generator: uv run python -m eval.generate --count 50

Samples ingested chunks, has the LLM write one exam question + expected gist
per chunk, then self-checks each candidate against its source chunk. The
filter is FAIL-CLOSED — a candidate is dropped on a "no" verdict, a parse
failure, or any LLM error. That is the inverse of the phase-5 runtime
guardrails (fail-open): a dropped candidate costs one retry, but a bad item
that slips through poisons every future metric run.

Output: eval/golden-synthetic.json (full overwrite — regeneration is
stateless). eval.run picks the file up automatically when it exists.
"""

import argparse
import json
import logging
import re
from pathlib import Path

from pydantic import BaseModel

from llm.base import generate
from llm.prompts import SYNTH_CHECK_SYSTEM_PROMPT, SYNTH_QUESTION_SYSTEM_PROMPT
from rag.store import VectorStore

logger = logging.getLogger(__name__)

DEFAULT_OUT_PATH = "eval/golden-synthetic.json"

_CHECK_LINE = re.compile(r"^\s*(answerable|faithful)\s*[:=]\s*(yes|no)\b",
                         re.IGNORECASE | re.MULTILINE)


class Candidate(BaseModel):
    question: str
    expected_answer_gist: str


def _generate_candidate(chunk: dict, provider: str | None) -> Candidate | None:
    prompt = f"[paper {chunk['paper_id']} — {chunk['title']}]\n{chunk['text']}"
    try:
        resp = generate([{"role": "user", "content": prompt}],
                        system=SYNTH_QUESTION_SYSTEM_PROMPT,
                        structured_schema=Candidate, provider=provider)
        candidate = resp.parsed
        if not candidate.question.strip() or not candidate.expected_answer_gist.strip():
            return None
        return candidate
    except Exception:
        logger.warning("Candidate generation failed for %s; dropping",
                       chunk["paper_id"], exc_info=True)
        return None


def _self_check(chunk: dict, candidate: Candidate, provider: str | None) -> bool:
    """Fail-closed: only an explicit yes on BOTH verdicts keeps the item."""
    user = (f"Excerpt:\n{chunk['text']}\n\n"
            f"Question: {candidate.question}\n\n"
            f"Expected gist: {candidate.expected_answer_gist}")
    try:
        resp = generate([{"role": "user", "content": user}],
                        system=SYNTH_CHECK_SYSTEM_PROMPT, provider=provider)
    except Exception:
        logger.warning("Self-check failed; dropping candidate", exc_info=True)
        return False
    verdicts = {k.lower(): v.lower() == "yes"
                for k, v in _CHECK_LINE.findall(resp.text)}
    return verdicts.get("answerable") is True and verdicts.get("faithful") is True


def generate_dataset(count: int, provider: str | None = None, seed: int = 0,
                     out_path: str = DEFAULT_OUT_PATH, store=None) -> dict:
    store = store or VectorStore()
    store.ping()  # fail fast with a clear message when Qdrant is down
    store.check_schema()
    # Over-sample 3x: rejections are expected, exhaustion is handled below.
    chunks = store.sample_chunks(count * 3, seed=seed)
    if not chunks:
        raise RuntimeError("No ingested chunks to generate from — "
                           "ingest some papers first.")
    kept: list[dict] = []
    rejected = 0
    for chunk in chunks:
        if len(kept) >= count:
            break
        candidate = _generate_candidate(chunk, provider)
        if candidate is None or not _self_check(chunk, candidate, provider):
            rejected += 1
            continue
        kept.append({
            "question": candidate.question.strip(),
            "expected_paper_ids": [chunk["paper_id"]],
            "expected_answer_gist": candidate.expected_answer_gist.strip(),
            "synthetic": True,
        })
    Path(out_path).write_text(json.dumps(kept, indent=2))
    return {"kept": len(kept), "rejected": rejected, "requested": count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=50,
                        help="target number of kept synthetic items")
    parser.add_argument("--provider", choices=["anthropic", "openai", "local"],
                        default=None, help="LLM provider (default: configured)")
    parser.add_argument("--seed", type=int, default=0,
                        help="sampling seed (reproducible chunk draw)")
    args = parser.parse_args()
    stats = generate_dataset(count=args.count, provider=args.provider,
                             seed=args.seed)
    print(f"kept {stats['kept']} / requested {stats['requested']} "
          f"({stats['rejected']} rejected) -> {DEFAULT_OUT_PATH}")
    if stats["kept"] < stats["requested"]:
        print("Chunk supply exhausted before reaching the target — "
              "ingest more papers or lower --count.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_generate.py -v`
Expected: 8 passed

- [ ] **Step 6: Run the full suite, then commit**

Run: `uv run pytest`
Expected: all pass

```bash
git add eval/generate.py tests/test_generate.py llm/prompts.py
git commit -m "feat: synthetic eval-item generator with fail-closed self-check"
```

---

### Task 4: `eval/run.py` — dataset concat + CIs in summary and ablation table

**Files:**
- Modify: `eval/run.py`
- Modify: `tests/test_eval_run.py` (append tests)

**Interfaces:**
- Consumes: `bootstrap_ci` (Task 1); `eval/golden-synthetic.json` written by Task 3.
- Produces: `run_eval(dataset_path: str | None = None, ...)` and `run_ablation(dataset_path: str | None = None, ...)` — `None` means golden + synthetic-if-exists; summary gains `precision_ci`, `recall_ci`, `faithfulness_ci`, `relevance_ci`, `citation_accuracy_ci` (each `[lo, hi]` list) and `faithfulness_rate_ci` (`[lo, hi]` or `None`); CLI gains `--dataset PATH`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_eval_run.py`:

```python
def _fake_eval_env(monkeypatch, tmp_path):
    """Shared mocked environment for run_eval dataset/CI tests."""
    import eval.run as run_mod
    from config import settings
    from eval.judge import JudgeScores
    from rag.answer import RagAnswer
    from rag.store import ScoredChunk

    class FakeVectorStore:
        def __init__(self, *args, **kwargs):
            pass

        def ping(self):
            pass

        def check_schema(self):
            pass

    monkeypatch.setattr(run_mod, "VectorStore", FakeVectorStore)
    monkeypatch.setattr(settings, "grading_enabled", False)
    chunk = ScoredChunk(paper_id="1706.03762", title="Attention", text="ctx", score=0.9)
    monkeypatch.setattr(run_mod, "retrieve", lambda q: [chunk])
    monkeypatch.setattr(run_mod, "answer_question",
                        lambda q: RagAnswer(text="ans [1706.03762]", sources=["1706.03762"]))
    monkeypatch.setattr(
        run_mod, "judge_answer",
        lambda question, answer, expected_gist, contexts: JudgeScores(
            faithfulness=4, relevance=5, citation_accuracy=3, reasoning="r"),
    )
    return run_mod


def _item(q):
    return {"question": q, "expected_paper_ids": ["1706.03762"],
            "expected_answer_gist": "g"}


def test_default_dataset_concats_synthetic_when_present(monkeypatch, tmp_path):
    run_mod = _fake_eval_env(monkeypatch, tmp_path)

    golden = tmp_path / "golden.json"
    synth = tmp_path / "golden-synthetic.json"
    golden.write_text(json.dumps([_item("g1"), _item("g2")]))
    synth.write_text(json.dumps([_item("s1"), _item("s2"), _item("s3")]))
    monkeypatch.setattr(run_mod, "DEFAULT_DATASET", str(golden))
    monkeypatch.setattr(run_mod, "SYNTHETIC_DATASET", str(synth))

    report = run_mod.run_eval(report_path=str(tmp_path / "r.json"))
    assert report["summary"]["n"] == 5  # 2 golden + 3 synthetic


def test_default_dataset_without_synthetic_file(monkeypatch, tmp_path):
    run_mod = _fake_eval_env(monkeypatch, tmp_path)

    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps([_item("g1")]))
    monkeypatch.setattr(run_mod, "DEFAULT_DATASET", str(golden))
    monkeypatch.setattr(run_mod, "SYNTHETIC_DATASET", str(tmp_path / "nope.json"))

    report = run_mod.run_eval(report_path=str(tmp_path / "r.json"))
    assert report["summary"]["n"] == 1


def test_explicit_dataset_path_wins_over_concat(monkeypatch, tmp_path):
    run_mod = _fake_eval_env(monkeypatch, tmp_path)

    only = tmp_path / "only.json"
    only.write_text(json.dumps([_item("o1")]))
    synth = tmp_path / "golden-synthetic.json"
    synth.write_text(json.dumps([_item("s1")]))
    monkeypatch.setattr(run_mod, "SYNTHETIC_DATASET", str(synth))

    report = run_mod.run_eval(dataset_path=str(only),
                              report_path=str(tmp_path / "r.json"))
    assert report["summary"]["n"] == 1  # synthetic NOT concatenated


def test_summary_carries_ci_keys(monkeypatch, tmp_path):
    run_mod = _fake_eval_env(monkeypatch, tmp_path)

    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps([_item("g1"), _item("g2"), _item("g3")]))
    monkeypatch.setattr(run_mod, "DEFAULT_DATASET", str(golden))
    monkeypatch.setattr(run_mod, "SYNTHETIC_DATASET", str(tmp_path / "nope.json"))

    s = run_mod.run_eval(report_path=str(tmp_path / "r.json"))["summary"]
    for key in ["precision_ci", "recall_ci", "faithfulness_ci",
                "relevance_ci", "citation_accuracy_ci"]:
        lo, hi = s[key]
        assert lo <= hi
    # identical rows → zero-width interval around the mean
    assert s["precision_ci"] == [s["avg_precision"], s["avg_precision"]]
    # mocked answers carry faithful=None → no verdicts → no rate CI
    assert s["faithfulness_rate_ci"] is None


def test_ablation_cell_renders_half_width(capsys):
    from eval.run import _print_ablation

    report = {"presets": {"demo": {
        "avg_precision": 0.67, "precision_ci": [0.59, 0.83],
        "avg_recall": 1.0, "recall_ci": [1.0, 1.0],
        "avg_faithfulness": 4.0, "faithfulness_ci": [3.5, 4.5],
        "avg_relevance": 5.0, "relevance_ci": [5.0, 5.0],
        "avg_citation_accuracy": 3.0, "citation_accuracy_ci": [2.0, 4.0],
    }}}
    _print_ablation(report)
    out = capsys.readouterr().out
    assert "0.67 ±0.12" in out   # (0.83 - 0.59) / 2
    assert "1.00 ±0.00" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_eval_run.py -v`
Expected: new tests FAIL (`AttributeError: ... no attribute 'DEFAULT_DATASET'`, missing `_ci` keys, `±` absent); existing tests still pass

- [ ] **Step 3: Implement in `eval/run.py`**

3a. Imports — add after the existing `from eval.metrics import precision_recall` line:

```python
from eval.stats import bootstrap_ci
```

3b. Module constants + dataset resolver — add above `_faithfulness_rate`:

```python
DEFAULT_DATASET = "eval/golden.json"
SYNTHETIC_DATASET = "eval/golden-synthetic.json"


def _load_dataset(dataset_path: str | None) -> list[dict]:
    """Explicit path → exactly that file. Default (None) → the hand-written
    golden set plus the synthetic set when it exists (the phase-6 generator
    pipeline is fully automatic — no human gate between generate and use)."""
    if dataset_path is not None:
        return json.loads(Path(dataset_path).read_text())
    items = json.loads(Path(DEFAULT_DATASET).read_text())
    synthetic = Path(SYNTHETIC_DATASET)
    if synthetic.exists():
        items = items + json.loads(synthetic.read_text())
    return items
```

3c. `run_eval` — change the signature and the dataset line:

```python
def run_eval(dataset_path: str | None = None,
             report_path: str = "report.json") -> dict:
```

and replace `dataset = json.loads(Path(dataset_path).read_text())` with:

```python
    dataset = _load_dataset(dataset_path)
```

3d. `run_eval` summary — after the existing `summary = {...}` dict, add:

```python
    for metric in ["precision", "recall", "faithfulness", "relevance",
                   "citation_accuracy"]:
        summary[f"{metric}_ci"] = list(bootstrap_ci([r[metric] for r in rows]))
    verdict_values = [1.0 if r["faithful"] else 0.0
                      for r in rows if r.get("faithful") is not None]
    summary["faithfulness_rate_ci"] = (
        list(bootstrap_ci(verdict_values)) if verdict_values else None)
```

3e. `run_ablation` — change the signature default:

```python
def run_ablation(dataset_path: str | None = None,
                 report_path: str = "report-ablation.json") -> dict:
```

3f. `_print_ablation` — replace entirely (wider columns: `hybrid+rerank+grade` is 19 chars and already overflowed the old 16-char gutter):

```python
def _print_ablation(report: dict) -> None:
    cols = ["avg_precision", "avg_recall", "avg_faithfulness",
            "avg_relevance", "avg_citation_accuracy"]
    print(f"\n{'preset':<22}" + "".join(f"{c.removeprefix('avg_'):>22}" for c in cols))
    for name, s in report["presets"].items():
        cells = []
        for c in cols:
            ci = s.get(c.removeprefix("avg_") + "_ci")
            half_width = (ci[1] - ci[0]) / 2 if ci else 0.0
            cells.append(f"{s[c]:.2f} ±{half_width:.2f}")
        print(f"{name:<22}" + "".join(f"{cell:>22}" for cell in cells))
```

3g. `main()` — add the flag and CI rendering. Replace the whole function:

```python
def main() -> None:
    parser = argparse.ArgumentParser(description="Offline eval harness")
    parser.add_argument("--ablation", action="store_true",
                        help="sweep retrieval presets and print a comparison table")
    parser.add_argument("--dataset", default=None,
                        help="use exactly this dataset file (default: golden.json "
                             "+ golden-synthetic.json when present)")
    args = parser.parse_args()
    if args.ablation:
        _print_ablation(run_ablation(dataset_path=args.dataset))
        return
    report = run_eval(dataset_path=args.dataset)
    s = report["summary"]

    def fmt(value: float, ci: list[float]) -> str:
        return f"{value:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"

    print(f"\nEvaluated {s['n']} questions -> report.json")
    print(f"  retrieval precision : {fmt(s['avg_precision'], s['precision_ci'])}")
    print(f"  retrieval recall    : {fmt(s['avg_recall'], s['recall_ci'])}")
    print(f"  faithfulness        : {fmt(s['avg_faithfulness'], s['faithfulness_ci'])} / 5")
    print(f"  relevance           : {fmt(s['avg_relevance'], s['relevance_ci'])} / 5")
    print(f"  citation accuracy   : {fmt(s['avg_citation_accuracy'], s['citation_accuracy_ci'])} / 5")
    if s["faithfulness_rate"] is not None:
        line = f"  verified answers    : {s['faithfulness_rate']:.0%}"
        if s.get("faithfulness_rate_ci"):
            ci = s["faithfulness_rate_ci"]
            line += f" [{ci[0]:.0%}, {ci[1]:.0%}]"
        print(line)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_eval_run.py -v`
Expected: all pass, including the pre-existing tests (they pass explicit `dataset_path=...`, which bypasses concat; summary equality checks compare dicts that now both carry `_ci` lists — JSON round-trips lists unchanged)

- [ ] **Step 5: Run the full suite, then commit**

Run: `uv run pytest`
Expected: all pass

```bash
git add eval/run.py tests/test_eval_run.py
git commit -m "feat: synthetic dataset concat and bootstrap CIs in eval summary and ablation table"
```

---

### Task 5: Local generation smoke test + README

**Files:**
- Create: `tests/test_local_generate.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `generate_dataset` (Task 3). The local test injects a fake store (real chunks hardcoded) so it needs Ollama but NOT Qdrant.

- [ ] **Step 1: Write the local marker test**

Create `tests/test_local_generate.py`:

```python
"""Real-Ollama synthetic-generation smoke test: uv run pytest -m local

Injects a fake store so only the LLM is real — no Qdrant needed."""

import json

import pytest

from eval.generate import generate_dataset

pytestmark = pytest.mark.local

REAL_CHUNKS = [
    {"paper_id": "1706.03762", "title": "Attention Is All You Need",
     "text": "The Transformer is the first sequence transduction model based "
             "entirely on attention, replacing the recurrent layers most "
             "commonly used in encoder-decoder architectures with multi-headed "
             "self-attention."},
    {"paper_id": "1810.04805", "title": "BERT",
     "text": "BERT is designed to pre-train deep bidirectional representations "
             "from unlabeled text by jointly conditioning on both left and "
             "right context in all layers, using a masked language model "
             "pre-training objective."},
]


class FakeStore:
    def ping(self):
        pass

    def check_schema(self):
        pass

    def sample_chunks(self, n, seed=0):
        return list(REAL_CHUNKS)[:n]


def test_local_generation_produces_schema_valid_items(tmp_path):
    out = tmp_path / "synth.json"
    stats = generate_dataset(count=2, provider="local", store=FakeStore(),
                             out_path=str(out))
    items = json.loads(out.read_text())
    # a flaky 3B check may reject; the hard guarantees are: no exception,
    # kept ≤ requested, and every kept item is schema-valid
    assert stats["kept"] == len(items) <= 2
    for item in items:
        assert set(item) == {"question", "expected_paper_ids",
                             "expected_answer_gist", "synthetic"}
        assert item["synthetic"] is True
        assert len(item["expected_paper_ids"]) == 1
```

Run: `uv run pytest tests/test_local_generate.py -q`
Expected: deselected under the default marker filter (`-m 'not integration and not local'`), 0 failures.
(Only if Ollama is already running: `uv run pytest -m local tests/test_local_generate.py -v` — expected pass.)

- [ ] **Step 2: Update `README.md`**

2a. Status table — add a row after phase 5:

```markdown
| 6 | Synthetic eval generation (LLM question/gist from ingested chunks, fail-closed self-check, grows golden set 3 → 50+) and 95% bootstrap CIs on every eval metric and ablation cell |
```

2b. Status line `## Status: complete (all 5 phases shipped)` →

```markdown
## Status: complete (all 6 phases shipped)
```

2c. In the `## Use` section, after the ablation command block entry, add:

```markdown
# Generate synthetic eval items from ingested chunks (grows the golden set;
# eval.run picks the file up automatically)
uv run python -m eval.generate --count 50

# Run eval on ONLY the hand-written set
uv run python -m eval.run --dataset eval/golden.json
```

2d. After the eval commands, add one explanatory line:

```markdown
Eval metrics print with 95% bootstrap confidence intervals — `0.67 [0.51, 0.82]`
in the summary, `0.67 ±0.08` per ablation cell — so preset differences can be
read against their noise floor.
```

- [ ] **Step 3: Full suite, then commit**

Run: `uv run pytest`
Expected: all pass, local test deselected

```bash
git add tests/test_local_generate.py README.md
git commit -m "test: real-Ollama synthetic-generation smoke test; docs: phase 6 README"
```
