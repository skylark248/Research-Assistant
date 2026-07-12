import json

from llm.base import LLMResponse

CHUNKS = [
    {"paper_id": "1706.03762", "title": "Attention", "text": "self-attention text"},
    {"paper_id": "1810.04805", "title": "BERT", "text": "masked LM text"},
    {"paper_id": "2005.14165", "title": "GPT-3", "text": "few-shot text"},
]


class FakeStore:
    def __init__(self, chunks=CHUNKS):
        self._chunks = chunks

    def ping(self):
        pass

    def check_schema(self):
        pass

    def sample_chunks(self, n, seed=0):
        return list(self._chunks)[:n]


def _patch_generate(monkeypatch, check_text="answerable: yes\nfaithful: yes",
                    gen_exc=None, check_exc=None):
    """Fake llm.base.generate: dispatches on which system prompt arrives."""
    import eval.generate as gen_mod
    from eval.generate import Candidate
    from llm import prompts

    calls = []

    def fake_generate(messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        if kwargs.get("system") == prompts.SYNTH_QUESTION_SYSTEM_PROMPT:
            if gen_exc is not None:
                raise gen_exc
            return LLMResponse(parsed=Candidate(
                question="What mechanism does the Transformer rely on?",
                expected_answer_gist="It relies on self-attention."))
        assert kwargs.get("system") == prompts.SYNTH_CHECK_SYSTEM_PROMPT
        if check_exc is not None:
            raise check_exc
        return LLMResponse(text=check_text)

    monkeypatch.setattr(gen_mod, "generate", fake_generate)
    return calls


def test_kept_item_has_golden_shape(monkeypatch, tmp_path):
    from eval.generate import generate_dataset

    _patch_generate(monkeypatch)
    out = tmp_path / "synth.json"
    stats = generate_dataset(count=2, store=FakeStore(), out_path=str(out))
    items = json.loads(out.read_text())
    assert stats == {"kept": 2, "rejected": 0, "requested": 2}
    assert len(items) == 2
    item = items[0]
    assert set(item) == {"question", "expected_paper_ids",
                         "expected_answer_gist", "synthetic"}
    assert item["synthetic"] is True
    assert item["expected_paper_ids"] == ["1706.03762"]  # single source paper
    assert item["question"] and item["expected_answer_gist"]


def test_self_check_no_drops_item(monkeypatch, tmp_path):
    from eval.generate import generate_dataset

    _patch_generate(monkeypatch, check_text="answerable: yes\nfaithful: no")
    out = tmp_path / "synth.json"
    stats = generate_dataset(count=3, store=FakeStore(), out_path=str(out))
    assert stats["kept"] == 0
    assert stats["rejected"] == 3
    assert json.loads(out.read_text()) == []


def test_self_check_contradictory_verdicts_drop_item(monkeypatch, tmp_path):
    from eval.generate import generate_dataset

    _patch_generate(monkeypatch,
                    check_text="answerable: no\nfaithful: yes\nanswerable: yes")
    stats = generate_dataset(count=1, store=FakeStore(),
                             out_path=str(tmp_path / "s.json"))
    assert stats["kept"] == 0  # any "no" wins — fail-closed


def test_self_check_garbage_drops_item(monkeypatch, tmp_path):
    from eval.generate import generate_dataset

    _patch_generate(monkeypatch, check_text="Looks great to me!")
    stats = generate_dataset(count=1, store=FakeStore(),
                             out_path=str(tmp_path / "s.json"))
    assert stats["kept"] == 0 and stats["rejected"] >= 1


def test_generation_error_drops_item_fail_closed(monkeypatch, tmp_path):
    from eval.generate import generate_dataset

    _patch_generate(monkeypatch, gen_exc=RuntimeError("api down"))
    stats = generate_dataset(count=2, store=FakeStore(),
                             out_path=str(tmp_path / "s.json"))
    assert stats["kept"] == 0


def test_check_error_drops_item_fail_closed(monkeypatch, tmp_path):
    from eval.generate import generate_dataset

    _patch_generate(monkeypatch, check_exc=RuntimeError("api down"))
    stats = generate_dataset(count=2, store=FakeStore(),
                             out_path=str(tmp_path / "s.json"))
    assert stats["kept"] == 0


def test_count_honored_and_stops_early(monkeypatch, tmp_path):
    from eval.generate import generate_dataset

    calls = _patch_generate(monkeypatch)
    stats = generate_dataset(count=1, store=FakeStore(),
                             out_path=str(tmp_path / "s.json"))
    assert stats["kept"] == 1
    # 1 kept item = exactly 2 LLM calls (generate + check); no work on chunk 2+
    assert len(calls) == 2


def test_exhausted_supply_writes_partial_set(monkeypatch, tmp_path):
    from eval.generate import generate_dataset

    _patch_generate(monkeypatch)
    out = tmp_path / "s.json"
    stats = generate_dataset(count=50, store=FakeStore(),  # only 3 chunks exist
                             out_path=str(out))
    assert stats["kept"] == 3
    assert len(json.loads(out.read_text())) == 3


def test_provider_threads_through_both_calls(monkeypatch, tmp_path):
    from eval.generate import generate_dataset

    calls = _patch_generate(monkeypatch)
    generate_dataset(count=1, provider="local", store=FakeStore(),
                     out_path=str(tmp_path / "s.json"))
    assert all(c["provider"] == "local" for c in calls)
