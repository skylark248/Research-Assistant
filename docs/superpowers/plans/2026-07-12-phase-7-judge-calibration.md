# Phase 7: Judge Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the LLM judge itself: an interactive blind-labeling CLI produces human ground truth, then quadratic-weighted Cohen's kappa + MAE (with bootstrap CIs) and an opt-in test-retest consistency mode quantify how far the judge can be trusted.

**Architecture:** One new CLI module `eval/calibrate.py` (subcommands `label` / `report`), one new ordinal-agreement function `weighted_kappa` in `eval/stats.py`. Read-only over `report.json`; human labels live in committed, self-contained `eval/human-labels.json`. Spec: `docs/superpowers/specs/2026-07-12-phase-7-judge-calibration-design.md`.

**Tech Stack:** Python 3.12, argparse subparsers, stdlib `random`/`statistics`/`collections`, pytest with monkeypatched `input`. Run everything with `uv run`.

## Global Constraints

- No new dependencies (no scikit-learn — weighted kappa is stdlib).
- **Blind labeling:** the judge's scores and its `reasoning` text must never be printed during the `label` flow — seeing them anchors the human.
- Rubric text shown before each score prompt comes from ONE source: `eval.judge.JudgeScores.model_fields[dim].description`.
- Labels file `eval/human-labels.json` is self-contained rows `{question, answer, judge: {faithfulness, relevance, citation_accuracy}, human: {…}, labeled_at}`, written after EVERY item (crash-safe resume).
- Sampling: `random.Random(seed).sample(rows, min(n, len(rows)))`, defaults `--n 20 --seed 0`; already-labeled questions skipped on re-run.
- `report` needs no LLM; `report --consistency` re-judges each labeled item once (contexts re-retrieved via `rag.retrieve.retrieve` — output must state that retrieval drift folds into consistency).
- Interpretation bands: κ≥0.8 near-perfect · ≥0.6 substantial · ≥0.4 moderate · ≥0.2 fair · else poor.
- Fail fast: missing `report.json` → "run `uv run python -m eval.run` first"; fewer than 2 labels at report time → error with count.
- `weighted_kappa(a, b, n_categories=5)`: quadratic weights; length mismatch or n<2 → ValueError; zero expected disagreement (both raters constant and equal) → 1.0 by convention.
- Caveat block always printed by `report`: small n, single annotator, consistency ≠ correctness.
- Unit tests mocked, no keys, `uv run pytest`. Commit style: imperative conventional.

---

### Task 1: `weighted_kappa` in `eval/stats.py`

**Files:**
- Modify: `eval/stats.py` (append one function)
- Modify: `tests/test_stats.py` (append tests)

**Interfaces:**
- Consumes: nothing (stdlib).
- Produces: `weighted_kappa(a: list[int], b: list[int], n_categories: int = 5) -> float` — Tasks 2–3 import it from `eval.stats`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_stats.py`:

```python
def test_kappa_perfect_agreement_is_one():
    from eval.stats import weighted_kappa

    assert weighted_kappa([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) == pytest.approx(1.0)


def test_kappa_observed_equals_expected_is_zero():
    from eval.stats import weighted_kappa

    # observed disagreement exactly matches chance for these marginals
    assert weighted_kappa([1, 1, 2, 2], [1, 2, 1, 2]) == pytest.approx(0.0)


def test_kappa_quadratic_weighting_distinguishes_near_from_far_miss():
    from eval.stats import weighted_kappa

    a = [1, 5, 1, 5]
    near = weighted_kappa(a, [2, 4, 2, 4])  # off by one each time
    far = weighted_kappa(a, [5, 1, 5, 1])   # maximally wrong each time
    assert near == pytest.approx(0.8)
    assert far == pytest.approx(-1.0)
    # unweighted agreement would score both identically (zero exact matches);
    # the quadratic weights are what separate them
    assert near > far


def test_kappa_constant_equal_raters_is_one_by_convention():
    from eval.stats import weighted_kappa

    # zero expected disagreement → denominator 0 → 1.0 by convention
    assert weighted_kappa([3, 3, 3], [3, 3, 3]) == pytest.approx(1.0)


def test_kappa_length_mismatch_raises():
    from eval.stats import weighted_kappa

    with pytest.raises(ValueError):
        weighted_kappa([1, 2], [1, 2, 3])


def test_kappa_too_few_pairs_raises():
    from eval.stats import weighted_kappa

    with pytest.raises(ValueError):
        weighted_kappa([3], [3])
```

(`tests/test_stats.py` already imports `pytest` at the top — no import changes needed.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_stats.py -v`
Expected: new tests FAIL — `ImportError: cannot import name 'weighted_kappa'`; existing 6 pass

- [ ] **Step 3: Append to `eval/stats.py`**

```python
def weighted_kappa(a: list[int], b: list[int], n_categories: int = 5) -> float:
    """Quadratic-weighted Cohen's kappa for ordinal scores 1..n_categories.

    Weights disagreements by squared distance, so a 4-vs-5 near-miss costs
    far less than a 1-vs-5 blunder — the standard choice for rubric scales.
    Zero expected disagreement (both raters constant and equal) → 1.0 by
    convention.
    """
    if len(a) != len(b):
        raise ValueError("rating lists must have the same length")
    if len(a) < 2:
        raise ValueError("weighted_kappa needs at least two rating pairs")
    n = n_categories
    observed = [[0.0] * n for _ in range(n)]
    for x, y in zip(a, b):
        observed[x - 1][y - 1] += 1
    total = len(a)
    hist_a = [sum(row) for row in observed]
    hist_b = [sum(observed[i][j] for i in range(n)) for j in range(n)]
    disagreement = 0.0
    expected_disagreement = 0.0
    for i in range(n):
        for j in range(n):
            weight = ((i - j) / (n - 1)) ** 2
            disagreement += weight * observed[i][j]
            expected_disagreement += weight * hist_a[i] * hist_b[j] / total
    if expected_disagreement == 0:
        return 1.0
    return 1.0 - disagreement / expected_disagreement
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_stats.py -v`
Expected: 12 passed

- [ ] **Step 5: Run the full suite, then commit**

Run: `uv run pytest`
Expected: all pass

```bash
git add eval/stats.py tests/test_stats.py
git commit -m "feat: quadratic-weighted Cohen's kappa for ordinal judge scores"
```

---

### Task 2: `eval/calibrate.py` — `label` subcommand (blind, resumable)

**Files:**
- Create: `eval/calibrate.py`
- Create: `tests/test_calibrate.py`

**Interfaces:**
- Consumes: `eval.judge.JudgeScores.model_fields[d].description` (rubric text); `eval.run._load_dataset(None)` (gist lookup by question); report rows carry flat keys `question`, `answer`, `retrieved_paper_ids`, `faithfulness`, `relevance`, `citation_accuracy`, `reasoning`.
- Produces: `run_label(report_path: str, labels_path: str, n: int, seed: int) -> dict` returning `{"labeled": int, "skipped": int, "total": int}`; module constants `DIMENSIONS = ["faithfulness", "relevance", "citation_accuracy"]`, `LABELS_PATH = "eval/human-labels.json"`; `_load_labels(path) -> list[dict]`; CLI `python -m eval.calibrate label`. Task 3 adds `run_report` to this module.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_calibrate.py`:

```python
import json


def _report_file(tmp_path, rows):
    p = tmp_path / "report.json"
    p.write_text(json.dumps({"summary": {}, "rows": rows}))
    return str(p)


def _row(q="q1", answer="answer text [1706.03762]", f=4, r=5, c=2):
    return {"question": q, "answer": answer,
            "expected_paper_ids": ["1706.03762"],
            "retrieved_paper_ids": ["1706.03762"],
            "precision": 1.0, "recall": 1.0,
            "faithfulness": f, "relevance": r, "citation_accuracy": c,
            "reasoning": "JUDGE_REASONING_SENTINEL", "faithful": True}


def _feed_input(monkeypatch, answers):
    it = iter(answers)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(it))


def _no_gists(monkeypatch):
    import eval.calibrate as cal

    monkeypatch.setattr(cal, "_load_dataset", lambda path: [])


def test_label_writes_selfcontained_rows(monkeypatch, tmp_path, capsys):
    from eval.calibrate import run_label

    _no_gists(monkeypatch)
    report = _report_file(tmp_path, [_row()])
    labels_path = str(tmp_path / "labels.json")
    _feed_input(monkeypatch, ["3", "5", "2"])  # one score per dimension

    stats = run_label(report, labels_path, n=20, seed=0)

    assert stats == {"labeled": 1, "skipped": 0, "total": 1}
    rows = json.loads((tmp_path / "labels.json").read_text())
    assert rows[0]["question"] == "q1"
    assert rows[0]["answer"] == "answer text [1706.03762]"
    assert rows[0]["judge"] == {"faithfulness": 4, "relevance": 5,
                                "citation_accuracy": 2}
    assert rows[0]["human"] == {"faithfulness": 3, "relevance": 5,
                                "citation_accuracy": 2}
    assert "labeled_at" in rows[0]


def test_label_is_blind(monkeypatch, tmp_path, capsys):
    from eval.calibrate import run_label

    _no_gists(monkeypatch)
    report = _report_file(tmp_path, [_row()])
    _feed_input(monkeypatch, ["3", "5", "2"])
    run_label(report, str(tmp_path / "labels.json"), n=20, seed=0)
    out = capsys.readouterr().out
    assert "JUDGE_REASONING_SENTINEL" not in out  # judge reasoning hidden
    assert "judge" not in out.lower()             # no judge-score leakage


def test_label_resume_skips_already_labeled(monkeypatch, tmp_path):
    from eval.calibrate import run_label

    _no_gists(monkeypatch)
    report = _report_file(tmp_path, [_row(q="q1"), _row(q="q2")])
    labels_path = str(tmp_path / "labels.json")
    _feed_input(monkeypatch, ["1", "1", "1", "2", "2", "2"])
    run_label(report, labels_path, n=20, seed=0)

    # second session: nothing left to label, input never consulted
    monkeypatch.setattr("builtins.input",
                        lambda prompt="": (_ for _ in ()).throw(
                            AssertionError("input must not be called")))
    stats = run_label(report, labels_path, n=20, seed=0)
    assert stats["labeled"] == 0
    assert stats["total"] == 2


def test_label_reprompts_on_invalid_input(monkeypatch, tmp_path):
    from eval.calibrate import run_label

    _no_gists(monkeypatch)
    report = _report_file(tmp_path, [_row()])
    # garbage, out-of-range, then valid — for the first dimension only
    _feed_input(monkeypatch, ["x", "9", "4", "5", "5"])
    stats = run_label(report, str(tmp_path / "labels.json"), n=20, seed=0)
    assert stats["labeled"] == 1
    rows = json.loads((tmp_path / "labels.json").read_text())
    assert rows[0]["human"]["faithfulness"] == 4


def test_label_skip_and_quit(monkeypatch, tmp_path):
    from eval.calibrate import run_label

    _no_gists(monkeypatch)
    report = _report_file(tmp_path, [_row(q="q1"), _row(q="q2"), _row(q="q3")])
    labels_path = str(tmp_path / "labels.json")
    # item1: skip; item2: label fully; item3: quit on first prompt
    _feed_input(monkeypatch, ["s", "2", "3", "4", "q"])
    stats = run_label(report, labels_path, n=20, seed=0)
    assert stats["skipped"] == 1
    assert stats["labeled"] == 1
    rows = json.loads((tmp_path / "labels.json").read_text())
    assert len(rows) == 1  # quit kept prior progress, item3 unlabeled


def test_label_sampling_deterministic_per_seed(monkeypatch, tmp_path):
    import eval.calibrate as cal

    _no_gists(monkeypatch)
    rows = [_row(q=f"q{i}") for i in range(30)]
    report = _report_file(tmp_path, rows)
    monkeypatch.setattr("builtins.input", lambda prompt="": "q")  # quit at once
    # capture which question is shown first for two identical runs
    firsts = []
    for _ in range(2):
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            cal.run_label(report, str(tmp_path / f"l{len(firsts)}.json"),
                          n=5, seed=7)
        first_q = [l for l in buf.getvalue().splitlines()
                   if l.startswith("Question:")][0]
        firsts.append(first_q)
    assert firsts[0] == firsts[1]


def test_label_missing_report_fails_fast(tmp_path):
    import pytest

    from eval.calibrate import run_label

    with pytest.raises(SystemExit, match="eval.run"):
        run_label(str(tmp_path / "nope.json"), str(tmp_path / "l.json"),
                  n=20, seed=0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_calibrate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval.calibrate'`

- [ ] **Step 3: Write `eval/calibrate.py`**

```python
"""Judge calibration: who judges the judge?

`label`  — blind, resumable CLI to hand-score sampled answers on the judge's
           own rubric (the judge's scores and reasoning are never shown).
`report` — judge-vs-human agreement: quadratic-weighted Cohen's kappa with a
           bootstrap CI, MAE, and score histograms per dimension; optional
           --consistency re-judges each labeled item to measure test-retest
           stability.

Read-only over report.json. Human labels are committed in
eval/human-labels.json and are self-contained, so agreement stays
recomputable after report.json is regenerated.
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

from eval.judge import JudgeScores
from eval.run import _load_dataset
from eval.stats import weighted_kappa

LABELS_PATH = "eval/human-labels.json"
DIMENSIONS = ["faithfulness", "relevance", "citation_accuracy"]

CAVEAT = """
Caveats: n is small (CIs are wide); one annotator (no inter-annotator
baseline); consistency measures stability, not correctness."""


class _SkipItem(Exception):
    pass


class _QuitLabeling(Exception):
    pass


def _load_labels(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return json.loads(p.read_text())


def _prompt_score(dimension: str) -> int:
    while True:
        raw = input(f"  {dimension} (1-5, s=skip item, q=quit): ").strip().lower()
        if raw == "s":
            raise _SkipItem()
        if raw == "q":
            raise _QuitLabeling()
        if raw in {"1", "2", "3", "4", "5"}:
            return int(raw)
        print("  enter an integer 1-5, or s / q")


def run_label(report_path: str, labels_path: str, n: int, seed: int) -> dict:
    report_file = Path(report_path)
    if not report_file.exists():
        raise SystemExit(f"{report_path} not found — run "
                         "`uv run python -m eval.run` first")
    rows = json.loads(report_file.read_text())["rows"]
    labels = _load_labels(labels_path)
    done = {label["question"] for label in labels}
    sample = random.Random(seed).sample(rows, min(n, len(rows)))
    todo = [r for r in sample if r["question"] not in done]
    gists = {item["question"]: item.get("expected_answer_gist", "")
             for item in _load_dataset(None)}
    rubrics = {d: JudgeScores.model_fields[d].description for d in DIMENSIONS}
    labeled = skipped = 0
    print(f"{len(todo)} to label ({len(sample) - len(todo)} already done)")
    try:
        for i, row in enumerate(todo, start=1):
            print(f"\n--- item {i}/{len(todo)} ---")
            print(f"Question: {row['question']}")
            print(f"Retrieved papers: {', '.join(row['retrieved_paper_ids'])}")
            print(f"Expected gist: {gists.get(row['question'], '(unavailable)')}")
            print(f"\nAnswer:\n{row['answer']}\n")
            try:
                human = {}
                for d in DIMENSIONS:
                    print(f"  rubric: {rubrics[d]}")
                    human[d] = _prompt_score(d)
            except _SkipItem:
                skipped += 1
                continue
            labels.append({
                "question": row["question"],
                "answer": row["answer"],
                "judge": {d: row[d] for d in DIMENSIONS},
                "human": human,
                "labeled_at": datetime.now().isoformat(timespec="seconds"),
            })
            Path(labels_path).write_text(json.dumps(labels, indent=2))
            labeled += 1
    except _QuitLabeling:
        print("\nquitting — progress saved")
    print(f"labeled {labeled}, skipped {skipped}, total on file {len(labels)}")
    return {"labeled": labeled, "skipped": skipped, "total": len(labels)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_label = sub.add_parser("label", help="hand-label sampled answers (blind)")
    p_label.add_argument("--n", type=int, default=20)
    p_label.add_argument("--seed", type=int, default=0)
    p_label.add_argument("--report", default="report.json")
    p_label.add_argument("--labels", default=LABELS_PATH)
    args = parser.parse_args()
    if args.cmd == "label":
        run_label(args.report, args.labels, args.n, args.seed)


if __name__ == "__main__":
    main()
```

(Task 3 extends `main()` with the `report` subparser — this task ships `label` only.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_calibrate.py -v`
Expected: 8 passed

- [ ] **Step 5: Run the full suite, then commit**

Run: `uv run pytest`
Expected: all pass

```bash
git add eval/calibrate.py tests/test_calibrate.py
git commit -m "feat: blind resumable labeling CLI for judge calibration"
```

---

### Task 3: `report` subcommand — agreement metrics + test-retest consistency

**Files:**
- Modify: `eval/calibrate.py`
- Modify: `tests/test_calibrate.py` (append tests)

**Interfaces:**
- Consumes: `weighted_kappa` (Task 1); `_load_labels`, `DIMENSIONS`, `CAVEAT` (Task 2); `eval.judge.judge_answer(question, answer, expected_gist, contexts) -> JudgeScores`; `rag.retrieve.retrieve(question) -> list[ScoredChunk]`.
- Produces: `run_report(labels_path: str, consistency: bool) -> dict` — per dimension `{"kappa": float, "kappa_ci": [lo, hi], "mae": float}`, plus `"consistency"` dict when enabled; `_kappa_ci(a, b, n_resamples=1000, seed=0) -> tuple[float, float]`; `_band(kappa) -> str`; CLI `python -m eval.calibrate report [--consistency]`. Task 4's local test calls `run_report(..., consistency=True)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_calibrate.py`:

```python
def _labels_file(tmp_path, rows):
    p = tmp_path / "labels.json"
    p.write_text(json.dumps(rows))
    return str(p)


def _label_row(q, jf, jr, jc, hf, hr, hc):
    return {"question": q, "answer": f"a-{q}",
            "judge": {"faithfulness": jf, "relevance": jr,
                      "citation_accuracy": jc},
            "human": {"faithfulness": hf, "relevance": hr,
                      "citation_accuracy": hc},
            "labeled_at": "2026-07-12T14:00:00"}


PERFECT = [
    _label_row("q1", 1, 2, 3, 1, 2, 3),
    _label_row("q2", 5, 4, 3, 5, 4, 3),
    _label_row("q3", 2, 2, 2, 2, 2, 2),
]


def test_report_perfect_agreement(tmp_path, capsys):
    from eval.calibrate import run_report

    out = run_report(_labels_file(tmp_path, PERFECT), consistency=False)
    for d in ["faithfulness", "relevance", "citation_accuracy"]:
        assert out[d]["kappa"] == 1.0
        assert out[d]["mae"] == 0.0
    printed = capsys.readouterr().out
    assert "near-perfect" in printed
    assert "Caveats" in printed
    assert "weighted kappa" in printed


def test_report_bands(tmp_path):
    from eval.calibrate import _band

    assert _band(0.85) == "near-perfect"
    assert _band(0.7) == "substantial"
    assert _band(0.5) == "moderate"
    assert _band(0.3) == "fair"
    assert _band(0.1) == "poor"
    assert _band(-0.5) == "poor"


def test_report_fails_fast_below_two_labels(tmp_path):
    import pytest

    from eval.calibrate import run_report

    path = _labels_file(tmp_path, PERFECT[:1])
    with pytest.raises(SystemExit, match="2"):
        run_report(path, consistency=False)


def test_kappa_ci_deterministic_and_brackets_point_estimate(tmp_path):
    from eval.calibrate import _kappa_ci
    from eval.stats import weighted_kappa

    judge = [1, 2, 3, 4, 5, 3, 2, 4]
    human = [2, 2, 3, 5, 4, 3, 1, 4]
    lo, hi = _kappa_ci(judge, human, seed=0)
    assert (lo, hi) == _kappa_ci(judge, human, seed=0)
    assert lo <= weighted_kappa(judge, human) <= hi


def test_consistency_rejudges_and_reports_kappa(monkeypatch, tmp_path, capsys):
    import eval.calibrate as cal
    from eval.judge import JudgeScores
    from rag.store import ScoredChunk

    labels = [
        _label_row("q1", 4, 5, 2, 3, 5, 2),
        _label_row("q2", 3, 4, 2, 3, 4, 3),
        _label_row("q3", 5, 5, 1, 4, 5, 1),
    ]
    path = _labels_file(tmp_path, labels)
    monkeypatch.setattr(cal, "_load_dataset", lambda p: [])
    chunk = ScoredChunk(paper_id="1706.03762", title="T", text="ctx", score=0.9)
    monkeypatch.setattr(cal, "retrieve", lambda q: [chunk])
    calls = []

    def fake_judge(question, answer, expected_gist, contexts):
        calls.append(question)
        if question == "q2":
            raise RuntimeError("judge exploded")  # skipped, counted
        return JudgeScores(faithfulness=4, relevance=5, citation_accuracy=2,
                           reasoning="r")

    monkeypatch.setattr(cal, "judge_answer", fake_judge)
    out = cal.run_report(path, consistency=True)
    assert calls == ["q1", "q2", "q3"]
    assert out["consistency"]["failures"] == 1
    # q1 and q3 re-judged: relevance run-1 [5, 5] vs run-2 [5, 5] → kappa 1.0
    assert out["consistency"]["relevance"] == 1.0
    printed = capsys.readouterr().out
    assert "re-retrieved" in printed  # retrieval-drift disclosure
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_calibrate.py -v`
Expected: new tests FAIL — `ImportError: cannot import name 'run_report'`; Task 2 tests pass

- [ ] **Step 3: Extend `eval/calibrate.py`**

3a. Add to the imports block:

```python
from collections import Counter

from eval.judge import JudgeScores, judge_answer
from rag.retrieve import retrieve
```

(replacing the existing `from eval.judge import JudgeScores` line — `judge_answer` and `retrieve` are new.)

3b. Append after `run_label`:

```python
def _band(kappa: float) -> str:
    if kappa >= 0.8:
        return "near-perfect"
    if kappa >= 0.6:
        return "substantial"
    if kappa >= 0.4:
        return "moderate"
    if kappa >= 0.2:
        return "fair"
    return "poor"


def _kappa_ci(a: list[int], b: list[int], n_resamples: int = 1000,
              seed: int = 0) -> tuple[float, float]:
    """Bootstrap CI on kappa: resample the PAIRS, recompute kappa each time."""
    pairs = list(zip(a, b))
    rng = random.Random(seed)
    kappas = []
    for _ in range(n_resamples):
        resample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        xs, ys = zip(*resample)
        kappas.append(weighted_kappa(list(xs), list(ys)))
    kappas.sort()
    lo_i = int(0.025 * len(kappas))
    hi_i = int(0.975 * len(kappas)) - 1
    return (kappas[lo_i], kappas[hi_i])


def _hist(scores: list[int]) -> str:
    counts = Counter(scores)
    return "  ".join(f"{s}:{counts.get(s, 0)}" for s in range(1, 6))


def _run_consistency(labels: list[dict]) -> dict:
    gists = {item["question"]: item.get("expected_answer_gist", "")
             for item in _load_dataset(None)}
    first: dict[str, list[int]] = {d: [] for d in DIMENSIONS}
    rerun: dict[str, list[int]] = {d: [] for d in DIMENSIONS}
    failures = 0
    print(f"\nre-judging {len(labels)} items for test-retest consistency…")
    for label in labels:
        try:
            chunks = retrieve(label["question"])
            contexts = [{"paper_id": c.paper_id, "text": c.text}
                        for c in chunks]
            scores = judge_answer(label["question"], label["answer"],
                                  gists.get(label["question"], ""), contexts)
        except Exception as exc:
            failures += 1
            print(f"  re-judge failed for {label['question'][:60]!r}: {exc}")
            continue
        for d in DIMENSIONS:
            first[d].append(label["judge"][d])
            rerun[d].append(getattr(scores, d))
    print(f"  (contexts re-retrieved — retrieval drift folds into "
          f"consistency; {failures} failed re-judgements skipped)")
    out: dict = {"failures": failures}
    for d in DIMENSIONS:
        if len(first[d]) < 2:
            print(f"  {d}: not enough successful re-judgements")
            continue
        kappa = weighted_kappa(first[d], rerun[d])
        out[d] = kappa
        print(f"  {d} test-retest kappa: {kappa:.2f} — {_band(kappa)}")
    return out


def run_report(labels_path: str, consistency: bool) -> dict:
    labels = _load_labels(labels_path)
    if len(labels) < 2:
        raise SystemExit(f"need at least 2 labels, found {len(labels)} — "
                         "run `uv run python -m eval.calibrate label` first")
    out: dict = {}
    print(f"\nJudge calibration over {len(labels)} human-labeled items")
    for d in DIMENSIONS:
        judge_scores = [label["judge"][d] for label in labels]
        human_scores = [label["human"][d] for label in labels]
        kappa = weighted_kappa(judge_scores, human_scores)
        lo, hi = _kappa_ci(judge_scores, human_scores)
        mae = (sum(abs(j - h) for j, h in zip(judge_scores, human_scores))
               / len(labels))
        out[d] = {"kappa": kappa, "kappa_ci": [lo, hi], "mae": mae}
        print(f"\n{d}")
        print(f"  weighted kappa : {kappa:.2f} [{lo:.2f}, {hi:.2f}] "
              f"— {_band(kappa)}")
        print(f"  MAE            : {mae:.2f}")
        print(f"  judge scores   : {_hist(judge_scores)}")
        print(f"  human scores   : {_hist(human_scores)}")
    if consistency:
        out["consistency"] = _run_consistency(labels)
    print(CAVEAT)
    return out
```

3c. In `main()`, add the `report` subparser and dispatch — full new `main()`:

```python
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_label = sub.add_parser("label", help="hand-label sampled answers (blind)")
    p_label.add_argument("--n", type=int, default=20)
    p_label.add_argument("--seed", type=int, default=0)
    p_label.add_argument("--report", default="report.json")
    p_label.add_argument("--labels", default=LABELS_PATH)
    p_report = sub.add_parser("report", help="judge-vs-human agreement")
    p_report.add_argument("--labels", default=LABELS_PATH)
    p_report.add_argument("--consistency", action="store_true",
                          help="re-judge each labeled item once (live LLM)")
    args = parser.parse_args()
    if args.cmd == "label":
        run_label(args.report, args.labels, args.n, args.seed)
    else:
        run_report(args.labels, args.consistency)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_calibrate.py -v`
Expected: 13 passed

- [ ] **Step 5: Run the full suite, then commit**

Run: `uv run pytest`
Expected: all pass

```bash
git add eval/calibrate.py tests/test_calibrate.py
git commit -m "feat: judge-vs-human agreement report with kappa CIs and test-retest consistency"
```

---

### Task 4: Local consistency smoke test + README

**Files:**
- Create: `tests/test_local_calibrate.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `run_report(labels_path, consistency=True)` (Task 3). Local test needs Ollama AND Qdrant (consistency re-retrieves contexts).

- [ ] **Step 1: Write the local marker test**

Create `tests/test_local_calibrate.py`:

```python
"""Real-Ollama consistency smoke test: uv run pytest -m local
Needs Ollama AND Qdrant (contexts are re-retrieved)."""

import json

import pytest

from eval.calibrate import run_report

pytestmark = pytest.mark.local

LABELS = [
    {"question": "What attention mechanism does the Transformer architecture "
                 "rely on, and why does it help with long-range dependencies?",
     "answer": "The Transformer relies on multi-head self-attention, which "
               "connects any two positions in constant sequential operations "
               "[1706.03762].",
     "judge": {"faithfulness": 4, "relevance": 5, "citation_accuracy": 4},
     "human": {"faithfulness": 4, "relevance": 5, "citation_accuracy": 4},
     "labeled_at": "2026-07-12T14:00:00"},
    {"question": "What pre-training objective does BERT use?",
     "answer": "BERT uses a masked language model objective, predicting "
               "randomly masked tokens from both directions [1810.04805].",
     "judge": {"faithfulness": 4, "relevance": 5, "citation_accuracy": 4},
     "human": {"faithfulness": 4, "relevance": 4, "citation_accuracy": 4},
     "labeled_at": "2026-07-12T14:01:00"},
]


def test_local_consistency_mode_runs(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(LABELS))
    out = run_report(str(path), consistency=True)
    cons = out["consistency"]
    # a flaky 3B judge may fail an item; hard guarantees: no exception,
    # failure count sane, any reported kappa within [-1, 1]
    assert 0 <= cons["failures"] <= 2
    for d in ["faithfulness", "relevance", "citation_accuracy"]:
        if d in cons:
            assert -1.0 <= cons[d] <= 1.0
```

Run: `uv run pytest tests/test_local_calibrate.py -q`
Expected: deselected under the default marker filter, 0 failures.
(Only if Ollama + Qdrant are up: `uv run pytest -m local tests/test_local_calibrate.py -v` — expected pass.)

- [ ] **Step 2: Update `README.md`**

2a. Status line → `## Status: complete (all 7 phases shipped)`

2b. Status table — add after the phase-6 row:

```markdown
| 7 | Judge calibration: blind human-labeling CLI, judge-vs-human agreement (quadratic-weighted kappa + MAE with bootstrap CIs), test-retest consistency mode |
```

2c. In the `## Use` section, after the `--dataset` command block, add:

```markdown
# Calibrate the judge: hand-label ~20 sampled answers (blind), then measure
# judge-vs-human agreement; --consistency re-judges each item (live LLM)
uv run python -m eval.calibrate label
uv run python -m eval.calibrate report
uv run python -m eval.calibrate report --consistency
```

2d. After the synthetic-caveat paragraph (ends "trust the ablation deltas, not the absolute numbers."), append one sentence to the same paragraph:

```markdown
`eval.calibrate` quantifies the judge itself — with one annotator and small n,
treat its kappa as a sanity check, not a certification.
```

- [ ] **Step 3: Full suite, then commit**

Run: `uv run pytest`
Expected: all pass, local test deselected (deselected count 16 → 17)

```bash
git add tests/test_local_calibrate.py README.md
git commit -m "test: real-Ollama consistency smoke test; docs: phase 7 README"
```

---

### Post-merge user action (not an implementation task)

The labeling session itself is the human's job: `uv run python -m eval.calibrate label` (~15 min for 20 items), then `git add eval/human-labels.json` + commit, then `uv run python -m eval.calibrate report`. The executor does NOT fabricate labels.
