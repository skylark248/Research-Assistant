"""End-to-end agent loop with real LLM, MCP servers, arXiv, and Qdrant.

Run: uv run pytest tests/test_integration_agent.py -m integration -v
"""

import pytest

pytestmark = pytest.mark.integration


async def test_agent_answers_with_tools():
    from agents.graph import run_agent

    reply = await run_agent(
        "Fetch the arXiv paper 1706.03762 if you don't have it, then tell me "
        "what attention mechanism it introduces."
    )
    assert reply
    assert "1706.03762" in reply  # cited
