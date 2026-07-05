"""Spawns the real MCP servers as subprocesses. Needs network + uvx on PATH.

Run: uv run pytest tests/test_integration_mcp.py -m integration -v
"""

import json

import pytest

pytestmark = pytest.mark.integration


async def test_toolbox_aggregates_both_servers_and_calls_search():
    from agents.mcp_client import MCPToolbox

    async with MCPToolbox() as box:
        names = {t["name"] for t in box.list_tools()}
        assert {"arxiv_search", "arxiv_fetch_paper"} <= names
        assert "fetch" in names  # from mcp-server-fetch

        content, is_error = await box.call_tool(
            "arxiv_search", {"query": "attention is all you need", "max_results": 2}
        )
        assert is_error is False
        assert json.loads(content)  # non-empty JSON list
