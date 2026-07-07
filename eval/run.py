"""Offline eval harness. Standalone: uv run python -m eval.run

For each golden item: retrieve → precision/recall vs expected ids; answer via
the RAG pipeline; LLM-judge the answer. Writes report.json + prints a summary.
(answer_question retrieves internally too — the double retrieval is accepted
for simplicity; both calls hit the same store deterministically.)
"""

import json
from pathlib import Path
from statistics import mean

from eval.judge import judge_answer
from eval.metrics import precision_recall
from rag.answer import answer_question
from rag.retrieve import retrieve
from rag.store import VectorStore


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
            "answer": answer.text,
        })

    summary = {
        "n": len(rows),
        "avg_precision": mean(r["precision"] for r in rows),
        "avg_recall": mean(r["recall"] for r in rows),
        "avg_faithfulness": mean(r["faithfulness"] for r in rows),
        "avg_relevance": mean(r["relevance"] for r in rows),
        "avg_citation_accuracy": mean(r["citation_accuracy"] for r in rows),
    }
    report = {"summary": summary, "rows": rows}
    Path(report_path).write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    report = run_eval()
    s = report["summary"]
    print(f"\nEvaluated {s['n']} questions -> report.json")
    print(f"  retrieval precision : {s['avg_precision']:.2f}")
    print(f"  retrieval recall    : {s['avg_recall']:.2f}")
    print(f"  faithfulness        : {s['avg_faithfulness']:.2f} / 5")
    print(f"  relevance           : {s['avg_relevance']:.2f} / 5")
    print(f"  citation accuracy   : {s['avg_citation_accuracy']:.2f} / 5")


if __name__ == "__main__":
    main()
