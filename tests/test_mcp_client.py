from types import SimpleNamespace

import pytest


class FakeSession:
    def __init__(self, result_text="ok", is_error=False, raise_exc=None):
        self.result_text = result_text
        self.is_error = is_error
        self.raise_exc = raise_exc
        self.calls = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if self.raise_exc:
            raise self.raise_exc
        block = SimpleNamespace(type="text", text=self.result_text)
        return SimpleNamespace(content=[block], isError=self.is_error)


def _toolbox_with(session, tool_name="arxiv_search"):
    from agents.mcp_client import MCPToolbox

    box = MCPToolbox(servers={})
    box._sessions = {tool_name: session}
    box._tools = [{"name": tool_name, "description": "d",
                   "input_schema": {"type": "object", "properties": {}}}]
    return box


def test_list_tools_returns_copies():
    box = _toolbox_with(FakeSession())
    tools = box.list_tools()
    assert tools[0]["name"] == "arxiv_search"
    tools.append("junk")
    assert len(box.list_tools()) == 1  # internal list untouched


async def test_call_tool_happy_path():
    session = FakeSession(result_text="found it")
    box = _toolbox_with(session)
    content, is_error = await box.call_tool("arxiv_search", {"query": "attention"})
    assert content == "found it"
    assert is_error is False
    assert session.calls == [("arxiv_search", {"query": "attention"})]


async def test_call_tool_reports_server_error_flag():
    box = _toolbox_with(FakeSession(result_text="boom", is_error=True))
    content, is_error = await box.call_tool("arxiv_search", {})
    assert is_error is True
    assert "boom" in content


async def test_call_tool_exception_becomes_error_result():
    box = _toolbox_with(FakeSession(raise_exc=ConnectionError("pipe closed")))
    content, is_error = await box.call_tool("arxiv_search", {})
    assert is_error is True
    assert "pipe closed" in content


async def test_call_tool_unknown_tool():
    box = _toolbox_with(FakeSession())
    content, is_error = await box.call_tool("nope", {})
    assert is_error is True
    assert "Unknown tool" in content
