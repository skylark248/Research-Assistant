import json


def test_run_eval_writes_report(monkeypatch, tmp_path):
    import eval.run as run_mod
    from config import settings
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

    # Not testing grading here — keep this test's chunks ungraded so it stays
    # hermetic (grading would otherwise hit a real LLM provider).
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


def test_ablation_sweeps_presets_and_restores_settings(monkeypatch, tmp_path):
    import eval.run as run_mod
    from config import settings

    monkeypatch.setattr(settings, "retrieval_mode", "hybrid")
    monkeypatch.setattr(settings, "rerank_enabled", True)
    monkeypatch.setattr(settings, "rewrite_enabled", False)

    seen = []

    def fake_run_eval(dataset_path, report_path):
        seen.append({"preset_report": report_path,
                     "mode": settings.retrieval_mode,
                     "rerank": settings.rerank_enabled,
                     "rewrite": settings.rewrite_enabled})
        return {"summary": {"n": 1, "avg_precision": 1.0, "avg_recall": 1.0,
                            "avg_faithfulness": 5.0, "avg_relevance": 5.0,
                            "avg_citation_accuracy": 5.0}}

    monkeypatch.setattr(run_mod, "run_eval", fake_run_eval)
    report = run_mod.run_ablation(report_path=str(tmp_path / "ablation.json"))

    assert list(report["presets"]) == ["baseline-dense", "sparse", "hybrid",
                                       "hybrid+rerank", "hybrid+rerank+grade", "full"]
    assert [s["mode"] for s in seen] == ["dense", "sparse", "hybrid", "hybrid", "hybrid", "hybrid"]
    assert [s["rerank"] for s in seen] == [False, False, False, True, True, True]
    assert [s["rewrite"] for s in seen] == [False, False, False, False, False, True]
    assert seen[0]["preset_report"] == str(tmp_path / "report-baseline-dense.json")
    # settings restored after the sweep
    assert settings.retrieval_mode == "hybrid"
    assert settings.rerank_enabled is True
    assert settings.rewrite_enabled is False


def test_ablation_restores_settings_on_failure(monkeypatch, tmp_path):
    import pytest

    import eval.run as run_mod
    from config import settings

    monkeypatch.setattr(settings, "retrieval_mode", "hybrid")

    def boom(dataset_path, report_path):
        raise RuntimeError("qdrant down")

    monkeypatch.setattr(run_mod, "run_eval", boom)
    with pytest.raises(RuntimeError):
        run_mod.run_ablation(report_path=str(tmp_path / "ablation.json"))
    assert settings.retrieval_mode == "hybrid"


def test_ablation_presets_pin_phase5_flags():
    from eval.run import PRESETS

    assert "hybrid+rerank+grade" in PRESETS
    for name, preset in PRESETS.items():
        # every preset must pin both new flags so rows stay comparable
        assert "grading_enabled" in preset, name
        assert preset["faithfulness_enabled"] is False, name  # not a retrieval technique
    assert PRESETS["hybrid+rerank+grade"]["grading_enabled"] is True
    assert PRESETS["hybrid+rerank"]["grading_enabled"] is False
    assert PRESETS["full"]["grading_enabled"] is True


def test_run_eval_grades_measured_chunks_when_enabled(monkeypatch, tmp_path):
    import eval.run as run_mod
    from config import settings
    from eval.judge import JudgeScores
    from rag.answer import RagAnswer
    from rag.store import ScoredChunk

    dataset = [
        {"question": "q1", "expected_paper_ids": ["1706.03762"], "expected_answer_gist": "g1"},
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

    kept = ScoredChunk(paper_id="1706.03762", title="Attention", text="ctx", score=0.9)
    dropped = ScoredChunk(paper_id="1810.04805", title="BERT", text="ctx2", score=0.5)
    monkeypatch.setattr(run_mod, "retrieve", lambda q: [kept, dropped])
    monkeypatch.setattr(run_mod, "answer_question",
                        lambda q: RagAnswer(text="ans [1706.03762]", sources=["1706.03762"]))
    monkeypatch.setattr(
        run_mod, "judge_answer",
        lambda question, answer, expected_gist, contexts: JudgeScores(
            faithfulness=4, relevance=5, citation_accuracy=3, reasoning="r"),
    )

    monkeypatch.setattr(settings, "grading_enabled", True)
    monkeypatch.setattr(run_mod, "grade_chunks", lambda question, chunks: [kept])

    report = run_mod.run_eval(dataset_path=str(dataset_path), report_path=str(report_path))
    row = report["rows"][0]
    assert row["retrieved_paper_ids"] == ["1706.03762"]


def test_run_eval_skips_grading_when_disabled(monkeypatch, tmp_path):
    import eval.run as run_mod
    from config import settings
    from eval.judge import JudgeScores
    from rag.answer import RagAnswer
    from rag.store import ScoredChunk

    dataset = [
        {"question": "q1", "expected_paper_ids": ["1706.03762"], "expected_answer_gist": "g1"},
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

    kept = ScoredChunk(paper_id="1706.03762", title="Attention", text="ctx", score=0.9)
    dropped = ScoredChunk(paper_id="1810.04805", title="BERT", text="ctx2", score=0.5)
    monkeypatch.setattr(run_mod, "retrieve", lambda q: [kept, dropped])
    monkeypatch.setattr(run_mod, "answer_question",
                        lambda q: RagAnswer(text="ans [1706.03762]", sources=["1706.03762"]))
    monkeypatch.setattr(
        run_mod, "judge_answer",
        lambda question, answer, expected_gist, contexts: JudgeScores(
            faithfulness=4, relevance=5, citation_accuracy=3, reasoning="r"),
    )

    def _boom(question, chunks):
        raise AssertionError("grade_chunks must not be called when grading is disabled")

    monkeypatch.setattr(settings, "grading_enabled", False)
    monkeypatch.setattr(run_mod, "grade_chunks", _boom)

    report = run_mod.run_eval(dataset_path=str(dataset_path), report_path=str(report_path))
    row = report["rows"][0]
    assert row["retrieved_paper_ids"] == ["1706.03762", "1810.04805"]


def test_faithfulness_rate():
    import pytest

    from eval.run import _faithfulness_rate

    assert _faithfulness_rate([]) is None
    assert _faithfulness_rate([{"faithful": None}]) is None
    rows = [{"faithful": True}, {"faithful": False},
            {"faithful": None}, {"faithful": True}]
    assert _faithfulness_rate(rows) == pytest.approx(2 / 3)


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


def test_rows_carry_synthetic_flag_and_subsets_split(monkeypatch, tmp_path):
    run_mod = _fake_eval_env(monkeypatch, tmp_path)

    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps([_item("g1"),
                                  {**_item("s1"), "synthetic": True}]))
    monkeypatch.setattr(run_mod, "DEFAULT_DATASET", str(golden))
    monkeypatch.setattr(run_mod, "SYNTHETIC_DATASET", str(tmp_path / "nope.json"))

    report = run_mod.run_eval(report_path=str(tmp_path / "r.json"))
    assert [r["synthetic"] for r in report["rows"]] == [False, True]
    subsets = report["summary"]["subsets"]
    assert subsets["hand"]["n"] == 1
    assert subsets["synthetic"]["n"] == 1
    assert set(subsets["hand"]) == {"n", "avg_precision", "avg_recall",
                                    "avg_faithfulness", "avg_relevance",
                                    "avg_citation_accuracy"}


def test_subsets_omit_empty_subset(monkeypatch, tmp_path):
    run_mod = _fake_eval_env(monkeypatch, tmp_path)

    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps([_item("g1"), _item("g2")]))  # hand only
    monkeypatch.setattr(run_mod, "DEFAULT_DATASET", str(golden))
    monkeypatch.setattr(run_mod, "SYNTHETIC_DATASET", str(tmp_path / "nope.json"))

    report = run_mod.run_eval(report_path=str(tmp_path / "r.json"))
    assert list(report["summary"]["subsets"]) == ["hand"]
