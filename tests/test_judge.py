import pytest
from pydantic import ValidationError


def test_scores_are_bounded():
    from eval.judge import JudgeScores

    JudgeScores(faithfulness=5, relevance=1, citation_accuracy=3, reasoning="ok")
    with pytest.raises(ValidationError):
        JudgeScores(faithfulness=6, relevance=1, citation_accuracy=3, reasoning="ok")
    with pytest.raises(ValidationError):
        JudgeScores(faithfulness=0, relevance=1, citation_accuracy=3, reasoning="ok")


def test_judge_answer_uses_structured_output(monkeypatch):
    import eval.judge as judge_mod
    from llm.base import LLMResponse

    expected = judge_mod.JudgeScores(faithfulness=4, relevance=5,
                                     citation_accuracy=4, reasoning="solid")
    captured = {}

    def fake_generate(messages, **kwargs):
        captured["messages"] = messages
        captured.update(kwargs)
        return LLMResponse(text="", parsed=expected)

    monkeypatch.setattr(judge_mod, "generate", fake_generate)

    scores = judge_mod.judge_answer(
        question="What is attention?",
        answer="Self-attention [1706.03762].",
        expected_gist="Transformers use self-attention.",
        contexts=[{"paper_id": "1706.03762", "text": "self-attention connects positions"}],
    )

    assert scores == expected
    assert captured["structured_schema"] is judge_mod.JudgeScores
    prompt = captured["messages"][0]["content"]
    assert "What is attention?" in prompt
    assert "Self-attention [1706.03762]." in prompt
    assert "Transformers use self-attention." in prompt
    assert "self-attention connects positions" in prompt
