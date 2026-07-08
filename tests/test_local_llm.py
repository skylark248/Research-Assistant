"""Real-model tests against a local Ollama server — no API keys.

Run: uv run pytest -m local
Needs: `ollama serve` with `ollama pull qwen2.5:3b` done (first pull ~1.9GB);
test_local_embeddings downloads the fastembed model (~130MB) on first run.
A 3B model is small: the tool-call and structured tests use deliberately
unambiguous prompts, but occasional flakes are expected — rerun before
treating a failure as a regression.
"""

import pytest
from pydantic import BaseModel

pytestmark = pytest.mark.local


def test_generate_round_trip():
    from llm.base import generate

    resp = generate([{"role": "user", "content": "Reply with exactly one word: hello"}],
                    provider="local", max_tokens=50)
    assert resp.text.strip()


def test_structured_output():
    from llm.base import generate

    class Verdict(BaseModel):
        answer: str
        confident: bool

    resp = generate(
        [{"role": "user", "content": "What is the capital of France? Answer confidently."}],
        provider="local", structured_schema=Verdict, max_tokens=100,
    )
    assert resp.parsed is not None
    assert "paris" in resp.parsed.answer.lower()


def test_tool_call():
    from llm.base import generate

    tools = [{
        "name": "get_current_time",
        "description": "Returns the current time. Use this whenever asked about the time.",
        "input_schema": {"type": "object", "properties": {}},
    }]
    resp = generate(
        [{"role": "user", "content": "What time is it right now? Use the tool."}],
        provider="local", tools=tools, max_tokens=100,
    )
    assert resp.tool_calls, f"expected a tool call, got text: {resp.text!r}"
    assert resp.tool_calls[0].name == "get_current_time"


def test_local_embeddings_shape(monkeypatch):
    import rag.embed as embed
    from config import settings

    monkeypatch.setattr(settings, "embedding_provider", "local")
    monkeypatch.setattr(embed, "_local_model", None)  # force real construction

    vectors = embed.embed_texts(["hybrid retrieval", "sparse vectors"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 384
    assert vectors[0] != vectors[1]
