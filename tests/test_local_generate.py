"""Real-Ollama synthetic-generation smoke test: uv run pytest -m local

Injects a fake store so only the LLM is real — no Qdrant needed."""

import json

import pytest

from eval.generate import generate_dataset

pytestmark = pytest.mark.local

REAL_CHUNKS = [
    {"paper_id": "1706.03762", "title": "Attention Is All You Need",
     "text": "The Transformer is the first sequence transduction model based "
             "entirely on attention, replacing the recurrent layers most "
             "commonly used in encoder-decoder architectures with multi-headed "
             "self-attention."},
    {"paper_id": "1810.04805", "title": "BERT",
     "text": "BERT is designed to pre-train deep bidirectional representations "
             "from unlabeled text by jointly conditioning on both left and "
             "right context in all layers, using a masked language model "
             "pre-training objective."},
]


class FakeStore:
    def ping(self):
        pass

    def check_schema(self):
        pass

    def sample_chunks(self, n, seed=0):
        return list(REAL_CHUNKS)[:n]


def test_local_generation_produces_schema_valid_items(tmp_path):
    out = tmp_path / "synth.json"
    stats = generate_dataset(count=2, provider="local", store=FakeStore(),
                             out_path=str(out))
    items = json.loads(out.read_text())
    # a flaky 3B check may reject; the hard guarantees are: no exception,
    # kept ≤ requested, and every kept item is schema-valid
    assert stats["kept"] == len(items) <= 2
    for item in items:
        assert set(item) == {"question", "expected_paper_ids",
                             "expected_answer_gist", "synthetic"}
        assert item["synthetic"] is True
        assert len(item["expected_paper_ids"]) == 1
