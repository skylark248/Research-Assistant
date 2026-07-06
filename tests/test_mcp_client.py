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


async def test_aenter_failure_names_server_and_propagates(monkeypatch, caplog):
    import agents.mcp_client as mc
    from mcp import StdioServerParameters

    class BrokenCM:
        async def __aenter__(self):
            raise ConnectionError("spawn failed")

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(mc, "stdio_client", lambda params: BrokenCM())
    box = mc.MCPToolbox(servers={"fetch": StdioServerParameters(command="uvx", args=["x"])})

    with caplog.at_level("ERROR"):
        with pytest.raises(ConnectionError, match="spawn failed"):
            async with box:
                pass  # pragma: no cover - never reached

    assert box._stack is None  # unwound, safe to retry
    assert any("fetch" in record.getMessage() for record in caplog.records)


def test_duplicate_tool_names_first_wins(caplog):
    from agents.mcp_client import MCPToolbox

    box = MCPToolbox(servers={})
    first_session = FakeSession()
    second_session = FakeSession()
    dup_tool = SimpleNamespace(name="dup", description="d", inputSchema={"type": "object"})

    with caplog.at_level("WARNING"):
        box._register_tools("server_a", first_session, [dup_tool])
        box._register_tools("server_b", second_session, [dup_tool])

    assert box._sessions["dup"] is first_session
    assert len(box.list_tools()) == 1
    assert any("dup" in record.getMessage() for record in caplog.records)
