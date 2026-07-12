# Phase 7 Design: Judge Calibration

Date: 2026-07-12
Status: approved

## Goal

Every eval number this project reports rests on an uncalibrated 3B LLM judge
(citation accuracy 2.45/5 on the 53-item run — but is the JUDGE right?).
Phase 7 measures the judge itself:

1. **Human ground truth** — an interactive CLI to hand-label ~20 sampled
   answers on the judge's own rubric.
2. **Agreement metrics** — quadratic-weighted Cohen's kappa + MAE per
   dimension, judge vs human, with bootstrap CIs.
3. **Test-retest consistency** — re-judge the same items and measure the
   judge's agreement with itself (the correct robustness check for an
   absolute 1-5 judge; position bias is a pairwise-judging concept and is
   out of scope here).

Learning targets: meta-evaluation ("who judges the judge"), ordinal
agreement statistics (weighted kappa), and honest reporting of what n=20
single-annotator labels can and cannot support.

## Non-goals

- Pairwise/Elo judging, multi-annotator agreement (one human = no
  inter-annotator baseline; stated in output).
- Judge prompt tuning in response to results — that is the natural
  follow-up phase once the kappa is known, not this phase.
- Any change to the existing eval flow. `eval/calibrate.py` is read-only
  over `report.json`; it never mutates eval behavior or settings.
- New dependencies (no scikit-learn — weighted kappa is ~20 stdlib lines).

## Decisions (locked)

- **Interactive CLI labeling** (`uv run python -m eval.calibrate label`),
  resumable, saves after every item.
- **Labels are committed** in `eval/human-labels.json` and self-contained
  (question + answer + judge scores + human scores), so agreement is
  recomputable even after `report.json` (gitignored) is regenerated.
- **Sample**: `--n 20 --seed 0`, deterministic draw from report rows;
  already-labeled questions are skipped on re-run (resume).
- **Consistency mode is opt-in** (`report --consistency`): ~1 live judge
  call per labeled item (needs Ollama or a cloud key; few minutes on 3B).
  Plain `report` needs no LLM at all.

## Architecture

### New module: `eval/calibrate.py`

CLI with two subcommands (argparse `subparsers`, pattern: `python -m
eval.calibrate <cmd>`):

**`label [--n 20] [--seed 0] [--report report.json]`**

1. Load rows from `report.json` (fail fast with a clear message if missing:
   "run `uv run python -m eval.run` first").
2. Deterministic sample: `random.Random(seed).sample(rows, min(n, len(rows)))`.
3. Skip rows whose `question` already appears in `eval/human-labels.json`.
4. Per item, print: question, retrieved paper ids, expected gist, the
   answer text, and the judge's three scores are NOT shown (blind labeling —
   seeing the judge's score anchors the human). Then prompt three 1-5
   integers, each preceded by the SAME rubric text as the corresponding
   `JudgeScores` field description (single source: import the field
   descriptions from `eval.judge.JudgeScores.model_fields`).
5. Input loop accepts only 1-5 (re-prompts otherwise); `s` skips an item;
   `q` quits early (progress kept).
6. Append `{question, answer, judge: {faithfulness, relevance,
   citation_accuracy}, human: {…}, labeled_at}` to `eval/human-labels.json`
   after EACH item (crash-safe resume).

**`report [--consistency] [--labels eval/human-labels.json]`**

1. Load labels (fail fast if fewer than 2).
2. Per dimension (faithfulness, relevance, citation_accuracy):
   - quadratic-weighted Cohen's kappa, judge vs human
   - bootstrap CI on kappa (resample label pairs, reuse `eval.stats.bootstrap_ci`
     machinery pattern — resampling the paired rows, recomputing kappa per
     resample, percentile interval)
   - mean absolute error
   - score distribution lines (judge vs human histogram, 1-5)
   - interpretation band: κ<0.2 poor · 0.2-0.4 fair · 0.4-0.6 moderate ·
     0.6-0.8 substantial · >0.8 near-perfect
3. `--consistency`: for each labeled item, call `judge_answer` once more
   (live LLM) and report test-retest weighted kappa per dimension
   (judge-run-1, from the stored labels, vs judge-run-2). Uses stored
   question/answer; contexts are re-retrieved via `rag.retrieve.retrieve`
   (same path `eval.run` uses) — noted in output as "contexts re-retrieved;
   retrieval drift folds into consistency".
4. Caveat block always printed: n is small (CIs are wide), one annotator,
   consistency ≠ correctness.

### `eval/stats.py` gains

```python
def weighted_kappa(a: list[int], b: list[int], n_categories: int = 5) -> float
```

Quadratic-weighted Cohen's kappa for ordinal scores (1..n_categories).
Stdlib only. Edge cases: `len(a) != len(b)` or `len(a) < 2` → ValueError;
perfect agreement → 1.0; when expected disagreement is 0 (both raters
constant and equal) → 1.0 by convention; observed == expected → 0.0.

### Label file: `eval/human-labels.json` (committed)

List of objects:

```json
{
  "question": "...",
  "answer": "...",
  "judge": {"faithfulness": 4, "relevance": 5, "citation_accuracy": 2},
  "human": {"faithfulness": 3, "relevance": 5, "citation_accuracy": 3},
  "labeled_at": "2026-07-12T14:05:00"
}
```

## Error handling

| Failure | Behavior |
|---|---|
| report.json missing | Fail fast: "run eval.run first" |
| < 2 labels at report time | Fail fast with count |
| Non-integer / out-of-range label input | Re-prompt (loop) |
| Ctrl-C or `q` mid-labeling | Progress already saved per item; exit 0 |
| Consistency judge call fails on an item | Skip item, count reported, kappa over the rest |
| Both raters constant & equal (zero expected disagreement) | kappa 1.0 by convention |

## Cost envelope

`label`: zero LLM calls. `report`: zero. `report --consistency`: one judge
call per labeled item (~20 calls, few minutes on qwen2.5:3b — respect the
OLLAMA_NUM_PARALLEL=1 serve flags from the README).

## Testing

Mocked unit tests (no keys):

- `weighted_kappa`: hand-computed fixtures — perfect agreement → 1.0;
  known partial-agreement case with precomputed value; constant-vs-varying
  raters; length-mismatch and n<2 raise; quadratic weighting distinguishes
  near-miss (4 vs 5) from far-miss (1 vs 5) where unweighted would not.
- `label`: sampling determinism per seed; resume skips already-labeled
  questions; input loop via monkeypatched `input` (valid, invalid-then-valid,
  skip, quit); file written after each item; blind (judge scores absent
  from printed output).
- `report`: output carries kappa + MAE + band per dimension; fails fast on
  <2 labels; consistency mode with mocked `judge_answer` (including one
  failing item → skipped and counted).
- One `-m local` test: consistency mode over 2 synthetic labels against
  real Ollama — no exception, kappa in [-1, 1].

## File touch list

| File | Change |
|---|---|
| `eval/calibrate.py` | new (label + report subcommands) |
| `eval/stats.py` | `weighted_kappa` |
| `eval/human-labels.json` | created by labeling session, committed |
| `tests/test_calibrate.py`, `tests/test_stats.py` | new tests |
| `tests/test_local_calibrate.py` | 1 local test |
| `README.md` | phase 7 row, calibrate commands, caveats line |
