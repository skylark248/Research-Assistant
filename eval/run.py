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
from rag.answer import answer_question
from rag.retrieve import retrieve
from rag.store import VectorStore


def _faithfulness_rate(rows: list[dict]) -> float | None:
    """Fraction of non-None verdicts that are True; None when nothing was checked."""
    verdicts = [r["faithful"] for r in rows if r.get("faithful") is not None]
    if not verdicts:
        return None
    return sum(1 for v in verdicts if v) / len(verdicts)


def run_eval(dataset_path: str = "eval/golden.json",
             report_path: str = "report.json") -> dict:
    store = VectorStore()
    store.ping()  # fail fast with a clear message when Qdrant is down
    store.check_schema()
    dataset = json.loads(Path(dataset_path).read_text())
    rows: list[dict] = []
    for item in dataset:
        question = item["question"]
        chunks = retrieve(question)
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


def run_ablation(dataset_path: str = "eval/golden.json",
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
    print(f"\n{'preset':<16}" + "".join(f"{c.removeprefix('avg_'):>19}" for c in cols))
    for name, s in report["presets"].items():
        print(f"{name:<16}" + "".join(f"{s[c]:>19.2f}" for c in cols))


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline eval harness")
    parser.add_argument("--ablation", action="store_true",
                        help="sweep retrieval presets and print a comparison table")
    args = parser.parse_args()
    if args.ablation:
        _print_ablation(run_ablation())
        return
    report = run_eval()
    s = report["summary"]
    print(f"\nEvaluated {s['n']} questions -> report.json")
    print(f"  retrieval precision : {s['avg_precision']:.2f}")
    print(f"  retrieval recall    : {s['avg_recall']:.2f}")
    print(f"  faithfulness        : {s['avg_faithfulness']:.2f} / 5")
    print(f"  relevance           : {s['avg_relevance']:.2f} / 5")
    print(f"  citation accuracy   : {s['avg_citation_accuracy']:.2f} / 5")
    if s["faithfulness_rate"] is not None:
        print(f"  verified answers    : {s['faithfulness_rate']:.0%}")


if __name__ == "__main__":
    main()
