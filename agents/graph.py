"""LangGraph agent with multi-turn memory.

Flow per invoke: summarize (no-op below the threshold) → agent ⇄ tools → END.
State is checkpointed per thread_id (AsyncSqliteSaver); `messages` uses an
operator.add reducer, so nodes return ONLY their new messages and invoking a
checkpointed thread with one new user message appends to restored history.

Long threads: the full history lives in the checkpoint, but the LLM sees a
trimmed window (_trimmed_history) plus a running summary in the system prompt.
The summarize node re-summarizes everything outside the window each time it
fires — costs some tokens, keeps the bookkeeping trivial.
"""

import asyncio
import logging
import operator
import uuid
from pathlib import Path
from typing import Annotated, NamedTuple, TypedDict

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph

from agents.mcp_client import MCPToolbox
from config import settings
from llm.base import generate, generate_stream
from llm.prompts import AGENT_SYSTEM_PROMPT, SUMMARIZE_SYSTEM_PROMPT
from rag.answer import answer_question

logger = logging.getLogger(__name__)

STEP_LIMIT_MESSAGE = (
    "I hit my tool-step limit before reaching a final answer. "
    "Any papers fetched so far are ingested - try asking again or narrowing the question."
)


class AgentResult(NamedTuple):
    text: str
    citations: list[str]


def _dedupe(items: list[str]) -> list[str]:
    """Order-preserving dedupe (dict keys keep insertion order)."""
    return list(dict.fromkeys(items))


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
    messages: Annotated[list[dict], operator.add]
    steps: int
    summary: str
    citations: Annotated[list[str], operator.add]


def _trimmed_history(messages: list[dict], keep: int) -> list[dict]:
    """Window of recent messages that starts at a plain user turn.

    Walking back to a plain (string-content) user message keeps tool_use /
    tool_result pairs intact — an orphaned tool_result at the front of the
    window would be an API error. Index 0 is always a plain user turn, so the
    walk terminates.
    """
    if len(messages) <= keep:
        return messages
    start = len(messages) - keep
    while start > 0 and not (messages[start]["role"] == "user"
                             and isinstance(messages[start]["content"], str)):
        start -= 1
    return messages[start:]


def _render_for_summary(messages: list[dict]) -> str:
    lines: list[str] = []
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            lines.append(f"{m['role']}: {content}")
            continue
        for block in content:
            if block["type"] == "text":
                lines.append(f"{m['role']}: {block['text']}")
            elif block["type"] == "tool_use":
                lines.append(f"{m['role']} called {block['name']}({block['input']})")
            elif block["type"] == "tool_result":
                lines.append(f"tool result: {str(block['content'])[:300]}")
    return "\n".join(lines)


def build_graph(toolbox, checkpointer=None, provider: str | None = None, on_event=None):
    tools = [RAG_QUERY_TOOL] + toolbox.list_tools()

    async def summarize_node(state: AgentState) -> dict:
        messages = state["messages"]
        if len(messages) <= settings.memory_max_messages:
            return {}
        window = _trimmed_history(messages, settings.memory_keep_messages)
        older = messages[: len(messages) - len(window)]
        if not older:
            return {}
        prior = state.get("summary", "")
        prompt = (f"Previous summary:\n{prior}\n\n" if prior else "") + \
            "Conversation to compress:\n" + _render_for_summary(older)
        if on_event is not None:
            on_event({"event": "status", "text": "summarizing conversation…"})
        resp = await asyncio.to_thread(
            generate, [{"role": "user", "content": prompt}],
            system=SUMMARIZE_SYSTEM_PROMPT, provider=provider,
        )
        return {"summary": resp.text}

    async def agent_node(state: AgentState) -> dict:
        messages = state["messages"]
        if len(messages) <= settings.memory_max_messages:
            history = messages
        else:
            history = _trimmed_history(messages, settings.memory_keep_messages)
        system = AGENT_SYSTEM_PROMPT
        if state.get("summary"):
            system = (f"{AGENT_SYSTEM_PROMPT}\n\n"
                      f"Conversation so far (summarized):\n{state['summary']}")
        if on_event is None:
            resp = await asyncio.to_thread(generate, history, system=system,
                                           tools=tools, provider=provider)
        else:
            def _stream():
                return generate_stream(
                    history, system=system, tools=tools, provider=provider,
                    on_delta=lambda t: on_event({"event": "delta", "text": t}),
                )
            resp = await asyncio.to_thread(_stream)
            on_event({"event": "turn_end", "has_tools": bool(resp.tool_calls)})
        content: list[dict] = []
        if resp.text:
            content.append({"type": "text", "text": resp.text})
        for tc in resp.tool_calls:
            content.append({"type": "tool_use", "id": tc.id, "name": tc.name,
                            "input": tc.input})
        return {"messages": [{"role": "assistant", "content": content}]}

    async def tools_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        results: list[dict] = []
        sources: list[str] = []
        for block in last["content"]:
            if block["type"] != "tool_use":
                continue
            name, args = block["name"], block["input"]
            if on_event is not None:
                on_event({"event": "status", "text": f"calling {name}…"})
            logger.info("Tool call: %s(%s)", name, args)
            if name == "rag_query":
                try:
                    ans = await asyncio.to_thread(answer_question, args["question"])
                    content = f"{ans.text}\n\nSources: {', '.join(ans.sources) or 'none'}"
                    is_error = False
                    sources.extend(ans.sources)
                except Exception as exc:  # e.g. Qdrant down — agent decides what to do
                    content, is_error = f"rag_query failed: {exc}", True
            else:
                content, is_error = await toolbox.call_tool(name, args)
            results.append({"type": "tool_result", "tool_use_id": block["id"],
                            "content": content, "is_error": is_error})
        return {
            "messages": [{"role": "user", "content": results}],
            "steps": state["steps"] + 1,
            "citations": sources,
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
    graph.add_node("summarize", summarize_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.set_entry_point("summarize")
    graph.add_edge("summarize", "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)


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


async def run_agent(question: str, thread_id: str | None = None,
                    provider: str | None = None, on_event=None) -> AgentResult:
    """One agent turn. Same thread_id continues a conversation; omitted → fresh
    single-shot thread (direct callers like eval stay stateless)."""
    thread_id = thread_id or str(uuid.uuid4())
    Path(settings.checkpoint_db).parent.mkdir(parents=True, exist_ok=True)
    async with MCPToolbox() as toolbox, \
            AsyncSqliteSaver.from_conn_string(settings.checkpoint_db) as saver:
        graph = build_graph(toolbox, checkpointer=saver, provider=provider,
                            on_event=on_event)
        state = await graph.ainvoke(
            {"messages": [{"role": "user", "content": question}], "steps": 0,
             "citations": []},
            config={"recursion_limit": settings.agent_max_steps * 2 + 6,
                    "configurable": {"thread_id": thread_id}},
        )
        text = final_text(state)
        return AgentResult(text=text or STEP_LIMIT_MESSAGE,
                           citations=_dedupe(state.get("citations", [])))
