def test_format_context_labels_papers():
    from llm.prompts import format_context

    out = format_context([
        {"paper_id": "1706.03762", "title": "Attention Is All You Need", "text": "Self-attention..."},
        {"paper_id": "1810.04805", "title": "BERT", "text": "Masked LM..."},
    ])
    assert "[paper 1706.03762 — Attention Is All You Need]" in out
    assert "Self-attention..." in out
    assert "[paper 1810.04805 — BERT]" in out


def test_build_rag_prompt_structure():
    from llm.prompts import CITATION_SYSTEM_PROMPT, FEW_SHOT_MESSAGES, build_rag_prompt

    contexts = [{"paper_id": "1706.03762", "title": "Attention", "text": "chunk text"}]
    system, messages = build_rag_prompt("What is attention?", contexts)

    # system: [instructions, cached context block]
    assert system[0]["text"] == CITATION_SYSTEM_PROMPT
    assert "chunk text" in system[1]["text"]
    assert system[1]["cache_control"] == {"type": "ephemeral"}

    # messages: few-shot pairs first, real question last
    assert messages[: len(FEW_SHOT_MESSAGES)] == FEW_SHOT_MESSAGES
    assert messages[-1]["role"] == "user"
    assert "What is attention?" in messages[-1]["content"]


def test_few_shot_demonstrates_citation_format():
    from llm.prompts import FEW_SHOT_MESSAGES

    assistant_turns = [m for m in FEW_SHOT_MESSAGES if m["role"] == "assistant"]
    assert assistant_turns, "few-shot must include an assistant example"
    assert any("[" in m["content"] and "]" in m["content"] for m in assistant_turns)
