"""LangGraph agent: an LLM node decides between answering directly, querying
the local RAG store, or calling MCP tools (arxiv_search / arxiv_fetch_paper /
fetch); a tools node executes calls and loops back until a final answer."""

import logging
from typing import TypedDict

from langgraph.graph import END, StateGraph

from agents.mcp_client import MCPToolbox
from config import settings
from llm.base import generate
from llm.prompts import AGENT_SYSTEM_PROMPT
from rag.answer import answer_question

logger = logging.getLogger(__name__)

RAG_QUERY_TOOL = {
    "name": "rag_query",
    "description": (
        "Answer a question from the already-ingested arXiv papers, with [paper_id] "
        "citations. Tells you when it doesn't have enough information."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"question": {"type": "string"}},
        "required": ["question"],
    },
}


class AgentState(TypedDict):
    messages: list[dict]
    steps: int


def build_graph(toolbox):
    tools = [RAG_QUERY_TOOL] + toolbox.list_tools()

    async def agent_node(state: AgentState) -> dict:
        resp = generate(state["messages"], system=AGENT_SYSTEM_PROMPT, tools=tools)
        content: list[dict] = []
        if resp.text:
            content.append({"type": "text", "text": resp.text})
        for tc in resp.tool_calls:
            content.append({"type": "tool_use", "id": tc.id, "name": tc.name,
                            "input": tc.input})
        return {"messages": state["messages"] + [{"role": "assistant", "content": content}]}

    async def tools_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        results: list[dict] = []
        for block in last["content"]:
            if block["type"] != "tool_use":
                continue
            name, args = block["name"], block["input"]
            logger.info("Tool call: %s(%s)", name, args)
            if name == "rag_query":
                try:
                    ans = answer_question(args["question"])
                    content = f"{ans.text}\n\nSources: {', '.join(ans.sources) or 'none'}"
                    is_error = False
                except Exception as exc:  # e.g. Qdrant down — agent decides what to do
                    content, is_error = f"rag_query failed: {exc}", True
            else:
                content, is_error = await toolbox.call_tool(name, args)
            results.append({"type": "tool_result", "tool_use_id": block["id"],
                            "content": content, "is_error": is_error})
        return {
            "messages": state["messages"] + [{"role": "user", "content": results}],
            "steps": state["steps"] + 1,
        }

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        has_tool_use = isinstance(last["content"], list) and any(
            b["type"] == "tool_use" for b in last["content"]
        )
        if has_tool_use and state["steps"] < settings.agent_max_steps:
            return "tools"
        return END

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


def final_text(state: dict) -> str:
    """Text of the last assistant message that has any text."""
    for message in reversed(state["messages"]):
        if message["role"] != "assistant":
            continue
        content = message["content"]
        if isinstance(content, list):
            texts = [b["text"] for b in content if b["type"] == "text"]
            if texts:
                return "\n".join(texts)
        elif content:
            return content
    return ""


async def run_agent(question: str) -> str:
    async with MCPToolbox() as toolbox:
        graph = build_graph(toolbox)
        state = await graph.ainvoke(
            {"messages": [{"role": "user", "content": question}], "steps": 0},
            config={"recursion_limit": settings.agent_max_steps * 2 + 4},
        )
        return final_text(state)
