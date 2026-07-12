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
from collections import Counter
from datetime import datetime
from pathlib import Path

from config import settings
from eval.judge import JudgeScores, judge_answer
from eval.run import _load_dataset
from eval.stats import weighted_kappa
from rag.grade import grade_chunks
from rag.retrieve import retrieve

LABELS_PATH = "eval/human-labels.json"
DIMENSIONS = ["faithfulness", "relevance", "citation_accuracy"]

CAVEAT = """
Caveats: n is small (CIs are wide); one annotator (no inter-annotator
baseline); consistency measures stability, not correctness; with a mostly-
synthetic sample, kappa reflects agreement on generator-shared questions and
range compression can deflate it — read it jointly with MAE and the
histograms."""


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
                "synthetic": bool(row.get("synthetic", False)),
                "labeled_at": datetime.now().isoformat(timespec="seconds"),
            })
            Path(labels_path).write_text(json.dumps(labels, indent=2))
            labeled += 1
    except _QuitLabeling:
        print("\nquitting — progress saved")
    print(f"labeled {labeled}, skipped {skipped}, total on file {len(labels)}")
    return {"labeled": labeled, "skipped": skipped, "total": len(labels)}


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
            if settings.grading_enabled:
                chunks = grade_chunks(label["question"], chunks)
            contexts = [{"paper_id": c.paper_id, "text": c.text}
                        for c in chunks]
            gist = gists.get(label["question"])
            if gist is None:
                print(f"  no gist on file for {label['question'][:60]!r} — "
                      "re-judging with an empty reference")
                gist = ""
            scores = judge_answer(label["question"], label["answer"],
                                  gist, contexts)
        except Exception as exc:
            failures += 1
            print(f"  re-judge failed for {label['question'][:60]!r}: {exc}")
            continue
        for d in DIMENSIONS:
            first[d].append(label["judge"][d])
            rerun[d].append(getattr(scores, d))
    print(f"  (contexts re-retrieved and re-graded when grading is enabled — "
          f"pipeline drift folds into consistency; {failures} failed "
          f"re-judgements skipped)")
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


if __name__ == "__main__":
    main()
