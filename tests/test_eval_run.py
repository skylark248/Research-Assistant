import json


def test_run_eval_writes_report(monkeypatch, tmp_path):
    import eval.run as run_mod
    from eval.judge import JudgeScores
    from rag.answer import RagAnswer
    from rag.store import ScoredChunk

    dataset = [
        {"question": "q1", "expected_paper_ids": ["1706.03762"], "expected_answer_gist": "g1"},
        {"question": "q2", "expected_paper_ids": ["1810.04805"], "expected_answer_gist": "g2"},
    ]
    dataset_path = tmp_path / "golden.json"
    dataset_path.write_text(json.dumps(dataset))
    report_path = tmp_path / "report.json"

    class FakeVectorStore:
        def __init__(self, *args, **kwargs):
            pass

        def ping(self):
            pass

        def check_schema(self):
            pass

    monkeypatch.setattr(run_mod, "VectorStore", FakeVectorStore)

    chunk = ScoredChunk(paper_id="1706.03762", title="Attention", text="ctx", score=0.9)
    monkeypatch.setattr(run_mod, "retrieve", lambda q: [chunk])
    monkeypatch.setattr(run_mod, "answer_question",
                        lambda q: RagAnswer(text="ans [1706.03762]", sources=["1706.03762"]))
    monkeypatch.setattr(
        run_mod, "judge_answer",
        lambda question, answer, expected_gist, contexts: JudgeScores(
            faithfulness=4, relevance=5, citation_accuracy=3, reasoning="r"),
    )

    report = run_mod.run_eval(dataset_path=str(dataset_path), report_path=str(report_path))

    assert report_path.exists()
    on_disk = json.loads(report_path.read_text())
    assert on_disk["summary"] == report["summary"]

    s = report["summary"]
    assert s["n"] == 2
    assert s["avg_precision"] == 0.5  # q1 match, q2 no match → (1.0 + 0.0) / 2
    assert s["avg_recall"] == 0.5  # q2 expected 1810.04805, retrieved 1706.03762
    assert s["avg_faithfulness"] == 4.0
    assert s["avg_relevance"] == 5.0
    assert s["avg_citation_accuracy"] == 3.0

    row = report["rows"][0]
    assert row["question"] == "q1"
    assert row["answer"] == "ans [1706.03762]"
    assert row["reasoning"] == "r"
