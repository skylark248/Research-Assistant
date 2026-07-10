from llm.base import LLMResponse


def _patch_plan(monkeypatch, plan, synth_text="synthesized"):
    """Patch agents.multi.generate: first call returns the plan, later calls synthesize."""
    import agents.multi as multi_mod

    calls = []

    def fake_generate(messages, **kwargs):
        calls.append({"messages": messages, **kwargs})
        if kwargs.get("structured_schema") is multi_mod.Plan:
            return LLMResponse(parsed=plan)
        return LLMResponse(text=synth_text)

    monkeypatch.setattr(multi_mod, "generate", fake_generate)
    return calls


async def test_simple_question_falls_through_to_single_agent(monkeypatch):
    import agents.multi as multi_mod
    from agents.graph import AgentResult

    _patch_plan(monkeypatch, multi_mod.Plan(simple=True))

    async def fake_run_agent(question, thread_id=None, provider=None, on_event=None):
        return AgentResult(text=f"single: {question} [{thread_id}]", citations=["1706.03762"])

    monkeypatch.setattr(multi_mod, "run_agent", fake_run_agent)
    result = await multi_mod.run_multi_agent("what is attention?", thread_id="t-1")
    assert result.text == "single: what is attention? [t-1]"


async def test_decomposed_question_researches_and_synthesizes(monkeypatch):
    import agents.multi as multi_mod
    from agents.graph import AgentResult

    plan = multi_mod.Plan(simple=False,
                          sub_questions=["what is BERT?", "what is GPT?"])
    calls = _patch_plan(monkeypatch, plan, synth_text="combined answer [1810.04805]")
    researched = []

    async def fake_run_agent(question, thread_id=None, provider=None):
        researched.append((question, thread_id))
        return AgentResult(text=f"finding about {question}", citations=["1810.04805"])

    monkeypatch.setattr(multi_mod, "run_agent", fake_run_agent)
    result = await multi_mod.run_multi_agent("compare BERT and GPT")

    assert result.text == "combined answer [1810.04805]"
    # researchers run per sub-question, single-shot (no thread)
    assert researched == [("what is BERT?", None), ("what is GPT?", None)]
    # synthesizer saw the question and both findings
    synth_input = calls[-1]["messages"][-1]["content"]
    assert "compare BERT and GPT" in synth_input
    assert "finding about what is BERT?" in synth_input


async def test_failed_researcher_reported_to_synthesizer(monkeypatch):
    import agents.multi as multi_mod
    from agents.graph import AgentResult

    plan = multi_mod.Plan(simple=False, sub_questions=["good q", "bad q"])
    calls = _patch_plan(monkeypatch, plan)

    async def flaky_run_agent(question, thread_id=None, provider=None):
        if question == "bad q":
            raise RuntimeError("mcp exploded")
        return AgentResult(text="a finding", citations=[])

    monkeypatch.setattr(multi_mod, "run_agent", flaky_run_agent)
    result = await multi_mod.run_multi_agent("q")

    assert result.text == "synthesized"
    synth_input = calls[-1]["messages"][-1]["content"]
    assert "FAILED: mcp exploded" in synth_input


async def test_run_chat_dispatches_on_agent_mode(monkeypatch):
    import agents.multi as multi_mod
    from agents.graph import AgentResult
    from config import settings

    async def fake_single(question, thread_id=None, provider=None, on_event=None):
        return AgentResult(text="single", citations=[])

    async def fake_multi(question, thread_id=None, provider=None, on_event=None):
        return AgentResult(text="multi", citations=[])

    monkeypatch.setattr(multi_mod, "run_agent", fake_single)
    monkeypatch.setattr(multi_mod, "run_multi_agent", fake_multi)

    monkeypatch.setattr(settings, "agent_mode", "single")
    result = await multi_mod.run_chat("q")
    assert result.text == "single"
    monkeypatch.setattr(settings, "agent_mode", "multi")
    result = await multi_mod.run_chat("q")
    assert result.text == "multi"


async def test_provider_reaches_planner_researchers_synthesizer(monkeypatch):
    import agents.multi as multi_mod
    from agents.graph import AgentResult
    from llm.base import LLMResponse

    seen_generate = []
    seen_agent = []

    def fake_generate(messages, **kwargs):
        seen_generate.append(kwargs)
        if kwargs.get("structured_schema") is not None:
            return LLMResponse(parsed=multi_mod.Plan(simple=False, sub_questions=["a", "b"]))
        return LLMResponse(text="synthesis")

    async def fake_run_agent(question, thread_id=None, provider=None):
        seen_agent.append(provider)
        return AgentResult(text=f"answer to {question}", citations=[])

    monkeypatch.setattr(multi_mod, "generate", fake_generate)
    monkeypatch.setattr(multi_mod, "run_agent", fake_run_agent)
    result = await multi_mod.run_multi_agent("big question", provider="openai")
    assert result.text == "synthesis"
    assert all(k["provider"] == "openai" for k in seen_generate)
    assert seen_agent == ["openai", "openai"]


async def test_multi_unions_researcher_citations(monkeypatch):
    import agents.multi as multi_mod
    from agents.graph import AgentResult
    from llm.base import LLMResponse

    def fake_generate(messages, **kwargs):
        if kwargs.get("structured_schema") is not None:
            return LLMResponse(parsed=multi_mod.Plan(simple=False, sub_questions=["a", "b"]))
        return LLMResponse(text="synthesis")

    calls = iter([AgentResult("ans a", ["1706.03762", "2105.02723"]),
                  AgentResult("ans b", ["2105.02723"])])

    async def fake_run_agent(question, thread_id=None, provider=None):
        return next(calls)

    monkeypatch.setattr(multi_mod, "generate", fake_generate)
    monkeypatch.setattr(multi_mod, "run_agent", fake_run_agent)
    result = await multi_mod.run_multi_agent("big question")
    assert result.text == "synthesis"
    assert result.citations == ["1706.03762", "2105.02723"]


async def test_multi_emits_statuses_and_streams_synthesis(monkeypatch):
    import agents.multi as multi_mod
    from agents.graph import AgentResult
    from llm.base import LLMResponse

    def fake_generate(messages, **kwargs):
        return LLMResponse(parsed=multi_mod.Plan(simple=False, sub_questions=["a"]))

    def fake_generate_stream(messages, **kwargs):
        kwargs["on_delta"]("synth")
        return LLMResponse(text="synth")

    async def fake_run_agent(question, thread_id=None, provider=None, on_event=None):
        return AgentResult(text="ans", citations=[])

    monkeypatch.setattr(multi_mod, "generate", fake_generate)
    monkeypatch.setattr(multi_mod, "generate_stream", fake_generate_stream)
    monkeypatch.setattr(multi_mod, "run_agent", fake_run_agent)
    events = []
    result = await multi_mod.run_multi_agent("q", on_event=events.append)
    assert result.text == "synth"
    kinds = [e["event"] for e in events]
    assert kinds == ["status", "status", "delta", "turn_end"]
    assert events[0]["text"] == "planning…"
    assert events[1]["text"] == "researching: a"


async def test_decomposed_plan_result_not_checkpointed(monkeypatch):
    import agents.multi as multi_mod
    from agents.graph import AgentResult
    from llm.base import LLMResponse

    def fake_generate(messages, **kwargs):
        if kwargs.get("structured_schema") is not None:
            return LLMResponse(parsed=multi_mod.Plan(simple=False, sub_questions=["a"]))
        return LLMResponse(text="synthesis")

    async def fake_run_agent(question, thread_id=None, provider=None, on_event=None):
        return AgentResult(text="ans", citations=[])

    monkeypatch.setattr(multi_mod, "generate", fake_generate)
    monkeypatch.setattr(multi_mod, "run_agent", fake_run_agent)
    result = await multi_mod.run_multi_agent("big question")
    assert result.checkpointed is False


async def test_simple_plan_result_stays_checkpointed(monkeypatch):
    import agents.multi as multi_mod
    from agents.graph import AgentResult
    from llm.base import LLMResponse

    def fake_generate(messages, **kwargs):
        return LLMResponse(parsed=multi_mod.Plan(simple=True))

    async def fake_run_agent(question, thread_id=None, provider=None, on_event=None):
        return AgentResult(text="direct", citations=[])

    monkeypatch.setattr(multi_mod, "generate", fake_generate)
    monkeypatch.setattr(multi_mod, "run_agent", fake_run_agent)
    result = await multi_mod.run_multi_agent("simple question")
    assert result.checkpointed is True
