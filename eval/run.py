"""Offline eval harness. Standalone: uv run python -m eval.run

For each golden item: retrieve → precision/recall vs expected ids; answer via
the RAG pipeline; LLM-judge the answer. Writes report.json + prints a summary.
(answer_question retrieves internally too — the double retrieval is accepted
for simplicity. With rewrite_enabled the two retrievals may rewrite
differently and use different chunks — known limitation affecting the "full"
ablation preset; metrics and judge contexts come from the first retrieval.)
"""

import argparse
import json
from pathlib import Path
from statistics import mean

from config import settings
from eval.judge import judge_answer
from eval.metrics import precision_recall
from eval.stats import bootstrap_ci
from rag.answer import answer_question
from rag.grade import grade_chunks
from rag.retrieve import retrieve
from rag.store import VectorStore


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


def _faithfulness_rate(rows: list[dict]) -> float | None:
    """Fraction of non-None verdicts that are True; None when nothing was checked."""
    verdicts = [r["faithful"] for r in rows if r.get("faithful") is not None]
    if not verdicts:
        return None
    return sum(1 for v in verdicts if v) / len(verdicts)


def run_eval(dataset_path: str | None = None,
             report_path: str = "report.json") -> dict:
    store = VectorStore()
    store.ping()  # fail fast with a clear message when Qdrant is down
    store.check_schema()
    dataset = _load_dataset(dataset_path)
    rows: list[dict] = []
    for item in dataset:
        question = item["question"]
        chunks = retrieve(question)
        if settings.grading_enabled:
            chunks = grade_chunks(question, chunks)
        precision, recall = precision_recall(
            [c.paper_id for c in chunks], item["expected_paper_ids"]
        )
        answer = answer_question(question)
        contexts = [{"paper_id": c.paper_id, "text": c.text} for c in chunks]
        scores = judge_answer(question, answer.text, item["expected_answer_gist"], contexts)
        rows.append({
            "question": question,
            "expected_paper_ids": item["expected_paper_ids"],
            "retrieved_paper_ids": sorted({c.paper_id for c in chunks}),
            "precision": precision,
            "recall": recall,
            **scores.model_dump(),
            "faithful": answer.faithful,
            "answer": answer.text,
        })

    summary = {
        "n": len(rows),
        "avg_precision": mean(r["precision"] for r in rows),
        "avg_recall": mean(r["recall"] for r in rows),
        "avg_faithfulness": mean(r["faithfulness"] for r in rows),
        "avg_relevance": mean(r["relevance"] for r in rows),
        "avg_citation_accuracy": mean(r["citation_accuracy"] for r in rows),
        "faithfulness_rate": _faithfulness_rate(rows),
    }
    for metric in ["precision", "recall", "faithfulness", "relevance",
                   "citation_accuracy"]:
        summary[f"{metric}_ci"] = list(bootstrap_ci([r[metric] for r in rows]))
    verdict_values = [1.0 if r["faithful"] else 0.0
                      for r in rows if r.get("faithful") is not None]
    summary["faithfulness_rate_ci"] = (
        list(bootstrap_ci(verdict_values)) if verdict_values else None)
    report = {"summary": summary, "rows": rows}
    Path(report_path).write_text(json.dumps(report, indent=2))
    return report


# Ablation presets: each row of the comparison table. Order matters — the
# printed table reads as "what did each added technique buy us".
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


def run_ablation(dataset_path: str | None = None,
                 report_path: str = "report-ablation.json") -> dict:
    """Run the golden dataset once per preset; collect summaries side by side.

    Mutates settings per preset and restores the originals afterwards —
    fine for this offline, single-threaded harness.
    """
    fields = ["retrieval_mode", "rerank_enabled", "rewrite_enabled",
              "grading_enabled", "faithfulness_enabled"]
    original = {f: getattr(settings, f) for f in fields}
    summaries: dict[str, dict] = {}
    try:
        for name, overrides in PRESETS.items():
            for field, value in overrides.items():
                setattr(settings, field, value)
            per_preset_path = str(Path(report_path).with_name(f"report-{name}.json"))
            summaries[name] = run_eval(dataset_path=dataset_path,
                                       report_path=per_preset_path)["summary"]
    finally:
        for field, value in original.items():
            setattr(settings, field, value)
    report = {"presets": summaries}
    Path(report_path).write_text(json.dumps(report, indent=2))
    return report


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


if __name__ == "__main__":
    main()
