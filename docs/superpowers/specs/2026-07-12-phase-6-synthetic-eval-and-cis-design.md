# Phase 6 Design: Synthetic Eval Data + Bootstrap Confidence Intervals

Date: 2026-07-12
Status: approved

## Goal

The golden dataset has 3 questions — every eval metric is statistically
meaningless. Phase 6 fixes that end to end:

1. **Synthetic eval generation** — LLM generates question/gist pairs from
   ingested chunks, self-check filtered, growing the dataset 3 → 50+.
2. **Bootstrap confidence intervals** — every summary metric and every
   ablation-table cell carries a 95% CI, answering "is 0.50 → 0.67 real or
   noise?" directly.

Learning targets: synthetic data generation with quality filtering
(fail-closed, the inverse of phase 5's fail-open guardrails), and statistical
rigor (bootstrap percentile method) without any framework or new dependency.

## Non-goals

- Human review gate on generated items (chosen: fully automatic; the
  self-check filter is the only gate).
- Pairwise/Elo judging, judge calibration, trajectory eval, adversarial
  sets, CI regression gating — future-phase candidates.
- Multi-chunk or multi-paper questions. Single-chunk provenance keeps
  `expected_paper_ids` trustworthy. Known limitation: with one expected id
  and retrieval_top_k=5, per-item retrieval precision is capped at 1/5 —
  already true of the hand-written items.
- New dependencies. Bootstrap uses stdlib `random`; no numpy/scipy/RAGAS
  (the project's "no eval framework" stance, per eval/judge.py docstring,
  stands).

## Decisions (locked)

- **Fully automatic pipeline.** Generated + self-checked items are used
  directly; no human pruning step.
- **Fail-closed filter.** A candidate is DROPPED on self-check "no", parse
  failure, or LLM error. A quality filter is the opposite of a runtime
  guardrail: a silently bad item poisons every future metric run, whereas a
  dropped item just costs one more generation attempt.
- **CIs everywhere.** Single-run summary and ablation table both display
  them; report JSON stores `[lo, hi]`.
- **Separate file** `eval/golden-synthetic.json`, auto-concatenated with
  `eval/golden.json` when present. `golden.json` stays hand-written.

## Architecture

### New module: `eval/generate.py`

CLI: `uv run python -m eval.generate --count 50 [--provider anthropic|openai|local] [--seed 0]`

Pipeline per candidate item:

1. **Sample chunks**: new `VectorStore.sample_chunks(n, seed)` scrolls the
   Qdrant collection, shuffles with `random.Random(seed)`, and yields chunk
   payloads round-robin across `paper_id`s so a heavily-chunked paper cannot
   dominate the dataset.
2. **Generate** (LLM call 1): from the chunk text, write one exam-style
   `question` and an `expected_answer_gist` (structured output — pydantic
   schema, same pattern as `rag/rewrite.py`).
3. **Self-check** (LLM call 2): given ONLY the chunk and the candidate,
   answer two line-format verdicts (3B-safe, same pattern as
   `rag/grade.py`): `answerable: yes|no` (question answerable from this
   excerpt alone) and `faithful: yes|no` (gist supported by the excerpt).
   Both must be `yes` to keep. Any parse failure or exception → drop
   (fail-closed).
4. **Emit** item: `{question, expected_paper_ids: [chunk.paper_id],
   expected_answer_gist, synthetic: true}` — golden.json shape plus the
   `synthetic` flag.

Loop until `--count` kept items or the chunk supply is exhausted. Write
`eval/golden-synthetic.json` (full overwrite — regeneration is stateless).
Print kept/rejected/exhausted counts.

### New module: `eval/stats.py`

```python
def bootstrap_ci(values: list[float], n_resamples: int = 1000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float]
```

Percentile method: resample `values` with replacement `n_resamples` times
via `random.Random(seed)`, take the mean of each resample, return the
(alpha/2, 1-alpha/2) percentiles. Guards: `len(values) < 2` → `(mean, mean)`
(zero-width, honest about "no spread information"). Deterministic for a
given seed.

### Changes: `eval/run.py`

- **Dataset resolution**: default = `eval/golden.json` concatenated with
  `eval/golden-synthetic.json` when that file exists; `--dataset PATH`
  overrides to exactly that one file.
- **Summary**: alongside each `avg_*` value and `faithfulness_rate`, a
  `ci` entry: `"precision_ci": [lo, hi]` etc. (CI of the mean via
  `bootstrap_ci` over the per-row values; `faithfulness_rate` CI over the
  0/1 verdicts, skipping Nones).
- **Printout**: `retrieval precision : 0.67 [0.51, 0.82]`.
- **Ablation table**: each cell rendered `0.67 ±0.08` (CI half-width);
  `_print_ablation` column widths adjusted.
- Report JSON rows unchanged.

### Changes: `rag/store.py`

`sample_chunks(n: int, seed: int = 0) -> list[dict]` — qdrant `scroll` over
the collection (payload only, no vectors), shuffle, round-robin by
`paper_id`, return up to `n` payload dicts `{paper_id, title, text}`.

### Prompts (`llm/prompts.py`)

`SYNTH_QUESTION_SYSTEM_PROMPT` (write one specific, self-contained exam
question + gist from an excerpt; question must not reference "the excerpt")
and `SYNTH_CHECK_SYSTEM_PROMPT` (two line-format yes/no verdicts:
answerable, faithful).

## Error handling

| Failure | Behavior |
|---|---|
| Generation LLM error/parse failure | Drop candidate, count as rejected, continue |
| Self-check "no" / unparseable / error | Drop candidate (fail-closed) |
| Chunk supply exhausted before --count | Write what was kept, report shortfall, exit 0 |
| Qdrant down / empty collection | Fail fast with clear message (matches store.ping pattern) |
| bootstrap_ci on n<2 | (mean, mean) zero-width |

## Cost envelope

2 LLM calls per kept item (+2 per rejected attempt). 50 items ≈ 100–140
calls, one-time, keyless-capable on Ollama. Eval runtime unchanged — CIs
are arithmetic on existing rows.

## Testing

Mocked unit tests (no keys):

- `eval/generate.py`: self-check "no" drops item; unparseable check drops;
  LLM exception drops (fail-closed asserted); kept item has golden.json
  shape + `synthetic: true` + single-paper `expected_paper_ids`; `--count`
  honored; round-robin spread across papers; exhaustion writes partial set.
- `eval/stats.py`: deterministic for fixed seed; n=1 → zero-width; all-equal
  values → zero-width; known small-sample sanity (CI brackets the mean).
- `eval/run.py`: dataset concat when synthetic file exists; `--dataset`
  override wins; summary carries `*_ci` keys; ablation cell format `±`.
- `rag/store.py`: sample_chunks round-robin + seed determinism against a
  fake client.
- One `-m local` test: generate 2 items against real Ollama, assert schema.

## File touch list

| File | Change |
|---|---|
| `eval/generate.py` | new |
| `eval/stats.py` | new |
| `eval/run.py` | dataset concat, CIs in summary + table |
| `rag/store.py` | `sample_chunks` |
| `llm/prompts.py` | 2 prompts |
| `tests/*` | new unit tests + 1 local test |
| `README.md` | phase 6 row, generate command, CI notation |
