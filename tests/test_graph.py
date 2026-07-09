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
                        lambda q, store=None, provider=None: RagAnswer(
                            text="Attention [1706.03762].", sources=["1706.03762"]))
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
                        lambda q, store=None, provider=None: RagAnswer(
                            text="partial", sources=[]))

    graph = graph_mod.build_graph(FakeToolbox())
    await graph.ainvoke({"messages": [{"role": "user", "content": "q"}], "steps": 0})

    assert len(seen) == 3  # initial + 2 tool rounds, then the guard ends the loop


async def test_run_agent_falls_back_when_step_limit_hit(monkeypatch, tmp_path):
    import agents.graph as graph_mod
    from config import settings
    from rag.answer import RagAnswer

    monkeypatch.setattr(settings, "checkpoint_db", str(tmp_path / "checkpoints.db"))
    monkeypatch.setattr(settings, "agent_max_steps", 1)
    endless = [
        LLMResponse(tool_calls=[ToolCall(id=f"tu_{i}", name="rag_query",
                                         input={"question": "q"})])
        for i in range(5)
    ]
    _scripted_generate(monkeypatch, endless)
    monkeypatch.setattr(graph_mod, "answer_question",
                        lambda q, store=None, provider=None: RagAnswer(
                            text="partial", sources=[]))

    class FakeToolboxCM:
        async def __aenter__(self):
            return FakeToolbox()

        async def __aexit__(self, *exc):
            return None

    monkeypatch.setattr(graph_mod, "MCPToolbox", FakeToolboxCM)

    result = await graph_mod.run_agent("q")

    assert result.text == graph_mod.STEP_LIMIT_MESSAGE


async def test_multi_turn_thread_restores_history(monkeypatch, tmp_path):
    """Second invoke on the same thread sees the first turn's messages."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    import agents.graph as graph_mod

    seen = _scripted_generate(monkeypatch, [
        LLMResponse(text="Paris."),
        LLMResponse(text="About 2 million."),
    ])
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "cp.db")) as saver:
        graph = graph_mod.build_graph(FakeToolbox(), checkpointer=saver)
        config = {"configurable": {"thread_id": "t1"}}
        await graph.ainvoke({"messages": [{"role": "user", "content": "Capital of France?"}],
                             "steps": 0}, config=config)
        state = await graph.ainvoke({"messages": [{"role": "user", "content": "Its population?"}],
                                     "steps": 0}, config=config)

    assert graph_mod.final_text(state) == "About 2 million."
    second_call_messages = seen[1]["messages"]
    contents = [m["content"] for m in second_call_messages if isinstance(m["content"], str)]
    assert "Capital of France?" in contents  # history restored from the checkpoint
    assert "Its population?" in contents


async def test_summarize_compresses_long_history(monkeypatch, tmp_path):
    """Past memory_max_messages, older turns get summarized into the system prompt."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    import agents.graph as graph_mod
    from config import settings

    monkeypatch.setattr(settings, "memory_max_messages", 3)
    monkeypatch.setattr(settings, "memory_keep_messages", 2)

    # Turn 1 and 2: direct answers. Turn 3: history (5 messages) exceeds the
    # threshold → summarize node calls generate first, then the agent node.
    seen = _scripted_generate(monkeypatch, [
        LLMResponse(text="answer one"),
        LLMResponse(text="answer two"),
        LLMResponse(text="THE SUMMARY"),
        LLMResponse(text="answer three"),
    ])
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "cp.db")) as saver:
        graph = graph_mod.build_graph(FakeToolbox(), checkpointer=saver)
        config = {"configurable": {"thread_id": "t1"}}
        for q in ["q1", "q2", "q3"]:
            state = await graph.ainvoke(
                {"messages": [{"role": "user", "content": q}], "steps": 0}, config=config)

    assert graph_mod.final_text(state) == "answer three"
    assert len(seen) == 4
    # the agent call after summarization carries the summary in its system prompt
    assert "THE SUMMARY" in seen[3]["system"]
    # and only the keep-window of messages (trimmed history)
    assert len(seen[3]["messages"]) <= 3


async def test_mid_band_history_not_trimmed(monkeypatch, tmp_path):
    """Between keep-window and summarize threshold, the agent sees FULL history."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    import agents.graph as graph_mod
    from config import settings

    monkeypatch.setattr(settings, "memory_keep_messages", 2)
    monkeypatch.setattr(settings, "memory_max_messages", 20)

    seen = _scripted_generate(monkeypatch, [
        LLMResponse(text="a1"),
        LLMResponse(text="a2"),
        LLMResponse(text="a3"),
    ])
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "cp.db")) as saver:
        graph = graph_mod.build_graph(FakeToolbox(), checkpointer=saver)
        config = {"configurable": {"thread_id": "t1"}}
        for q in ["q1", "q2", "q3"]:
            await graph.ainvoke(
                {"messages": [{"role": "user", "content": q}], "steps": 0}, config=config)

    # 5 messages at third call — above keep=2, below max=20: nothing may be dropped
    contents = [m["content"] for m in seen[2]["messages"] if isinstance(m["content"], str)]
    assert contents == ["q1", "q2", "q3"]
    assert len(seen) == 3  # summarize never fired


async def test_trimmed_history_starts_at_plain_user_turn():
    from agents.graph import _trimmed_history

    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1",
                                           "name": "rag_query", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1",
                                      "content": "result"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "a1"}]},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": [{"type": "text", "text": "a2"}]},
    ]
    # keep=3 starts the window at "assistant a1"; the walk-back must pass the
    # tool_result (list content, not plain) and land on plain user "q1"
    trimmed = _trimmed_history(messages, keep=3)
    assert trimmed[0] == {"role": "user", "content": "q1"}
    # keep=2 starts at "q2" — already a clean boundary, no walk
    trimmed2 = _trimmed_history(messages, keep=2)
    assert trimmed2[0] == {"role": "user", "content": "q2"}
    # short history returned whole
    assert _trimmed_history(messages[:2], keep=8) == messages[:2]


async def test_provider_threads_to_generate(monkeypatch):
    import agents.graph as graph_mod

    seen = _scripted_generate(monkeypatch, [LLMResponse(text="hi")])
    graph = graph_mod.build_graph(FakeToolbox(), provider="local")
    await graph.ainvoke({"messages": [{"role": "user", "content": "q"}], "steps": 0})
    assert seen[0]["provider"] == "local"


async def test_provider_threads_to_answer_question(monkeypatch):
    """Fix 1: the per-request provider passed to build_graph must reach
    answer_question when the agent invokes the rag_query tool."""
    import agents.graph as graph_mod
    from rag.answer import RagAnswer

    captured = {}

    def fake_answer_question(q, store=None, provider=None):
        captured["provider"] = provider
        return RagAnswer(text="Attention [1706.03762].", sources=["1706.03762"])

    monkeypatch.setattr(graph_mod, "answer_question", fake_answer_question)
    _scripted_generate(monkeypatch, [
        LLMResponse(tool_calls=[ToolCall(id="tu_1", name="rag_query",
                                         input={"question": "what is attention?"})]),
        LLMResponse(text="It is attention [1706.03762]."),
    ])
    graph = graph_mod.build_graph(FakeToolbox(), provider="local")
    await graph.ainvoke({"messages": [{"role": "user", "content": "q"}], "steps": 0})

    assert captured["provider"] == "local"


async def test_citations_collected_from_rag_query(monkeypatch):
    import agents.graph as graph_mod
    from rag.answer import RagAnswer

    monkeypatch.setattr(graph_mod, "answer_question",
                        lambda q, store=None, provider=None: RagAnswer(
                            text="Attention [1706.03762].", sources=["1706.03762"]))
    _scripted_generate(monkeypatch, [
        LLMResponse(tool_calls=[ToolCall(id="tu_1", name="rag_query",
                                         input={"question": "what is attention?"})]),
        LLMResponse(text="It is attention [1706.03762]."),
    ])
    graph = graph_mod.build_graph(FakeToolbox())
    state = await graph.ainvoke({"messages": [{"role": "user", "content": "q"}],
                                 "steps": 0, "citations": []})
    assert state["citations"] == ["1706.03762"]


def test_dedupe_preserves_order():
    from agents.graph import _dedupe

    assert _dedupe(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]


async def test_on_event_streams_deltas_and_statuses(monkeypatch):
    import agents.graph as graph_mod
    from rag.answer import RagAnswer

    monkeypatch.setattr(graph_mod, "answer_question",
                        lambda q, store=None, provider=None: RagAnswer(
                            text="A.", sources=["1706.03762"]))
    script = [
        LLMResponse(tool_calls=[ToolCall(id="tu_1", name="rag_query",
                                         input={"question": "q"})]),
        LLMResponse(text="Final answer."),
    ]

    def fake_generate_stream(messages, **kwargs):
        resp = script.pop(0)
        for piece in (resp.text[:3], resp.text[3:]):
            if piece:
                kwargs["on_delta"](piece)
        return resp

    monkeypatch.setattr(graph_mod, "generate_stream", fake_generate_stream)
    events = []
    graph = graph_mod.build_graph(FakeToolbox(), on_event=events.append)
    await graph.ainvoke({"messages": [{"role": "user", "content": "q"}],
                         "steps": 0, "citations": []})
    kinds = [e["event"] for e in events]
    assert kinds == ["turn_end", "status", "delta", "delta", "turn_end"]
    assert events[0]["has_tools"] is True          # tool-reasoning turn
    assert "rag_query" in events[1]["text"]        # status line
    assert events[-1]["has_tools"] is False        # final answer turn
    assert "".join(e["text"] for e in events if e["event"] == "delta") == "Final answer."
