"""MCP client aggregating tools from all configured servers.

Servers: our custom arxiv server (stdio subprocess) and the official
mcp-server-fetch (via uvx). Tool failures are returned as (message, True)
results so the agent can decide retry vs give up — never raised.
"""

import logging
import sys
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

SERVERS: dict[str, StdioServerParameters] = {
    "arxiv": StdioServerParameters(command=sys.executable, args=["-m", "agents.mcp_server"]),
    "fetch": StdioServerParameters(command="uvx", args=["mcp-server-fetch"]),
}


class MCPToolbox:
    def __init__(self, servers: dict[str, StdioServerParameters] | None = None):
        self.servers = SERVERS if servers is None else servers
        self._stack: AsyncExitStack | None = None
        self._sessions: dict[str, ClientSession] = {}  # tool name -> owning session
        self._tools: list[dict] = []

    async def __aenter__(self) -> "MCPToolbox":
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        try:
            for server_name, params in self.servers.items():
                try:
                    read, write = await self._stack.enter_async_context(stdio_client(params))
                    session = await self._stack.enter_async_context(ClientSession(read, write))
                    await session.initialize()
                    listing = await session.list_tools()
                except Exception:
                    logger.error(
                        "MCP server %r failed to start (command: %s %s)",
                        server_name, params.command, " ".join(params.args),
                    )
                    raise
                self._register_tools(server_name, session, listing.tools)
        except BaseException:
            # Unwind servers that did start; a cleanup failure here (e.g. anyio
            # cancel-scope errors from half-open stdio pipes) must not mask the
            # startup exception.
            try:
                await self._stack.__aexit__(*sys.exc_info())
            except Exception:
                logger.debug("Cleanup after failed startup raised", exc_info=True)
            self._stack = None
            raise
        return self

    def _register_tools(self, server_name: str, session: ClientSession, tools) -> None:
        """Register tools from a single server. First registration wins on name collision."""
        for tool in tools:
            if tool.name in self._sessions:
                logger.warning(
                    "Skipping duplicate tool %r from server %r: already registered",
                    tool.name, server_name,
                )
                continue
            self._sessions[tool.name] = session
            self._tools.append({
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.inputSchema,
            })

    async def __aexit__(self, *exc) -> None:
        if self._stack is not None:
            await self._stack.__aexit__(*exc)

    def list_tools(self) -> list[dict]:
        """Tools from every server, in the Anthropic tool format."""
        return list(self._tools)

    async def call_tool(self, name: str, arguments: dict) -> tuple[str, bool]:
        """Returns (content, is_error). Never raises."""
        session = self._sessions.get(name)
        if session is None:
            return f"Unknown tool: {name}", True
        try:
            result = await session.call_tool(name, arguments)
        except Exception as exc:
            return f"Tool {name} failed: {exc}", True
        text = "\n".join(
            block.text for block in result.content
            if getattr(block, "type", "") == "text"
        )
        return text, bool(result.isError)
