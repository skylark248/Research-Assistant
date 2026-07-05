from llm.base import LLMResponse, ToolCall


class FakeToolbox:
    def __init__(self, tools=None, result=("tool output", False)):
        self._tools = tools or []
        self.result = result
        self.calls = []

    def list_tools(self):
        return list(self._tools)

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return self.result


def _scripted_generate(monkeypatch, responses):
    """Patch agents.graph.generate to pop scripted responses in order."""
    import agents.graph as graph_mod

    script = list(responses)
    seen = []

    def fake_generate(messages, **kwargs):
        seen.append({"messages": messages, **kwargs})
        return script.pop(0)

    monkeypatch.setattr(graph_mod, "generate", fake_generate)
    return seen


async def test_direct_answer_no_tools(monkeypatch):
    import agents.graph as graph_mod

    seen = _scripted_generate(monkeypatch, [LLMResponse(text="Direct answer.")])
    graph = graph_mod.build_graph(FakeToolbox())
    state = await graph.ainvoke({"messages": [{"role": "user", "content": "hi"}], "steps": 0})

    assert graph_mod.final_text(state) == "Direct answer."
    assert len(seen) == 1
    # rag_query is always offered alongside MCP tools
    tool_names = [t["name"] for t in seen[0]["tools"]]
    assert "rag_query" in tool_names


async def test_rag_query_tool_loop(monkeypatch):
    import agents.graph as graph_mod
    from rag.answer import RagAnswer

    monkeypatch.setattr(graph_mod, "answer_question",
                        lambda q: RagAnswer(text="Attention [1706.03762].",
                                            sources=["1706.03762"]))
    seen = _scripted_generate(monkeypatch, [
        LLMResponse(tool_calls=[ToolCall(id="tu_1", name="rag_query",
                                         input={"question": "what is attention?"})]),
        LLMResponse(text="It is attention [1706.03762]."),
    ])
    graph = graph_mod.build_graph(FakeToolbox())
    state = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "what is attention?"}], "steps": 0}
    )

    assert graph_mod.final_text(state) == "It is attention [1706.03762]."
    assert len(seen) == 2
    # Second call saw the tool result appended in canonical format.
    tool_result_msg = seen[1]["messages"][-1]
    assert tool_result_msg["role"] == "user"
    assert tool_result_msg["content"][0]["type"] == "tool_result"
    assert tool_result_msg["content"][0]["tool_use_id"] == "tu_1"
    assert "1706.03762" in tool_result_msg["content"][0]["content"]


async def test_mcp_tool_error_flows_back_to_agent(monkeypatch):
    import agents.graph as graph_mod

    toolbox = FakeToolbox(
        tools=[{"name": "arxiv_search", "description": "d",
                "input_schema": {"type": "object", "properties": {}}}],
        result=("Tool arxiv_search failed: timeout", True),
    )
    _scripted_generate(monkeypatch, [
        LLMResponse(tool_calls=[ToolCall(id="tu_1", name="arxiv_search",
                                         input={"query": "q"})]),
        LLMResponse(text="Search failed, sorry."),
    ])
    graph = graph_mod.build_graph(toolbox)
    state = await graph.ainvoke({"messages": [{"role": "user", "content": "find"}],
                                 "steps": 0})

    assert graph_mod.final_text(state) == "Search failed, sorry."
    tool_result = state["messages"][-2]["content"][0]
    assert tool_result["is_error"] is True


async def test_loop_stops_at_max_steps(monkeypatch):
    import agents.graph as graph_mod
    from config import settings

    monkeypatch.setattr(settings, "agent_max_steps", 2)
    endless = [
        LLMResponse(tool_calls=[ToolCall(id=f"tu_{i}", name="rag_query",
                                         input={"question": "q"})])
        for i in range(10)
    ]
    seen = _scripted_generate(monkeypatch, endless)
    from rag.answer import RagAnswer
    monkeypatch.setattr(graph_mod, "answer_question",
                        lambda q: RagAnswer(text="partial", sources=[]))

    graph = graph_mod.build_graph(FakeToolbox())
    await graph.ainvoke({"messages": [{"role": "user", "content": "q"}], "steps": 0})

    assert len(seen) == 3  # initial + 2 tool rounds, then the guard ends the loop


async def test_run_agent_falls_back_when_step_limit_hit(monkeypatch):
    import agents.graph as graph_mod
    from config import settings
    from rag.answer import RagAnswer

    monkeypatch.setattr(settings, "agent_max_steps", 1)
    endless = [
        LLMResponse(tool_calls=[ToolCall(id=f"tu_{i}", name="rag_query",
                                         input={"question": "q"})])
        for i in range(5)
    ]
    _scripted_generate(monkeypatch, endless)
    monkeypatch.setattr(graph_mod, "answer_question",
                        lambda q: RagAnswer(text="partial", sources=[]))

    class FakeToolboxCM:
        async def __aenter__(self):
            return FakeToolbox()

        async def __aexit__(self, *exc):
            return None

    monkeypatch.setattr(graph_mod, "MCPToolbox", FakeToolboxCM)

    reply = await graph_mod.run_agent("q")

    assert reply == graph_mod.STEP_LIMIT_MESSAGE
