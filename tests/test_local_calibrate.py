"""Real-Ollama consistency smoke test: uv run pytest -m local
Needs Ollama AND Qdrant (contexts are re-retrieved)."""

import json

import pytest

from eval.calibrate import run_report

pytestmark = pytest.mark.local

LABELS = [
    {"question": "What attention mechanism does the Transformer architecture "
                 "rely on, and why does it help with long-range dependencies?",
     "answer": "The Transformer relies on multi-head self-attention, which "
               "connects any two positions in constant sequential operations "
               "[1706.03762].",
     "judge": {"faithfulness": 4, "relevance": 5, "citation_accuracy": 4},
     "human": {"faithfulness": 4, "relevance": 5, "citation_accuracy": 4},
     "labeled_at": "2026-07-12T14:00:00"},
    {"question": "What pre-training objective does BERT use?",
     "answer": "BERT uses a masked language model objective, predicting "
               "randomly masked tokens from both directions [1810.04805].",
     "judge": {"faithfulness": 4, "relevance": 5, "citation_accuracy": 4},
     "human": {"faithfulness": 4, "relevance": 4, "citation_accuracy": 4},
     "labeled_at": "2026-07-12T14:01:00"},
]


def test_local_consistency_mode_runs(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(LABELS))
    out = run_report(str(path), consistency=True)
    cons = out["consistency"]
    # a flaky 3B judge may fail an item; hard guarantees: no exception,
    # failure count sane, any reported kappa within [-1, 1]
    assert 0 <= cons["failures"] <= 2
    for d in ["faithfulness", "relevance", "citation_accuracy"]:
        if d in cons:
            assert -1.0 <= cons[d] <= 1.0
