"""Real-Ollama grading smoke test: uv run pytest -m local"""

import pytest

from rag.grade import grade_chunks
from rag.store import ScoredChunk

pytestmark = pytest.mark.local


def test_local_grading_returns_subset_and_never_errors():
    chunks = [
        ScoredChunk(paper_id="1706.03762", title="Attention Is All You Need",
                    text="The Transformer relies entirely on self-attention, "
                         "dispensing with recurrence and convolutions.", score=1.0),
        ScoredChunk(paper_id="9999.00001", title="Sourdough Baking Basics",
                    text="Preheat the oven to 230C and score the loaf before "
                         "baking for an open crumb.", score=0.5),
    ]
    kept = grade_chunks("What architecture does the Transformer use?", chunks,
                        provider="local")
    # fail-open means a flaky 3B grade may keep everything; the hard guarantees
    # are: no exception, output is a subset, order preserved
    assert all(c in chunks for c in kept)
    assert [c.paper_id for c in kept] == [c.paper_id for c in chunks
                                          if c in kept]
