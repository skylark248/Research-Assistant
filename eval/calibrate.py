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
