import json
from pathlib import Path

from eval.metrics import precision_recall


def test_perfect_retrieval():
    assert precision_recall(["a", "b"], ["a", "b"]) == (1.0, 1.0)


def test_partial_overlap():
    p, r = precision_recall(["a", "b", "c", "d"], ["a", "x"])
    assert p == 0.25  # 1 of 4 retrieved is relevant
    assert r == 0.5   # 1 of 2 expected was found


def test_no_retrieved():
    assert precision_recall([], ["a"]) == (0.0, 0.0)


def test_duplicates_do_not_inflate():
    p, r = precision_recall(["a", "a", "a"], ["a"])
    assert p == 1.0
    assert r == 1.0


def test_golden_dataset_shape():
    items = json.loads(Path("eval/golden.json").read_text())
    assert len(items) >= 3
    for item in items:
        assert item["question"]
        assert isinstance(item["expected_paper_ids"], list) and item["expected_paper_ids"]
        assert item["expected_answer_gist"]
