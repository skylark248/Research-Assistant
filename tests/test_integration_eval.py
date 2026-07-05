"""Eval smoke test: scores must stay above a floor to catch regressions.

Prereqs: docker compose up -d, real keys, and the golden papers ingested:
  uv run python -c "
from rag.arxiv_client import get_paper
from rag.ingest import ingest_paper
from rag.store import VectorStore
store = VectorStore(); store.ping(); store.ensure_collection()
for pid in ['1706.03762', '1810.04805', '2005.11401']:
    ingest_paper(get_paper(pid), store)
"

Run: uv run pytest tests/test_integration_eval.py -m integration -v
"""

import pytest

pytestmark = pytest.mark.integration


def test_eval_scores_stay_above_floor(tmp_path):
    from eval.run import run_eval

    report = run_eval(report_path=str(tmp_path / "report.json"))
    s = report["summary"]
    assert s["avg_recall"] >= 0.5, s
    assert s["avg_faithfulness"] >= 3.5, s
    assert s["avg_relevance"] >= 3.5, s
    assert s["avg_citation_accuracy"] >= 3.0, s
