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
