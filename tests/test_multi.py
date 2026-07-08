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

    _patch_plan(monkeypatch, multi_mod.Plan(simple=True))

    async def fake_run_agent(question, thread_id=None, provider=None):
        return f"single: {question} [{thread_id}]"

    monkeypatch.setattr(multi_mod, "run_agent", fake_run_agent)
    reply = await multi_mod.run_multi_agent("what is attention?", thread_id="t-1")
    assert reply == "single: what is attention? [t-1]"


async def test_decomposed_question_researches_and_synthesizes(monkeypatch):
    import agents.multi as multi_mod

    plan = multi_mod.Plan(simple=False,
                          sub_questions=["what is BERT?", "what is GPT?"])
    calls = _patch_plan(monkeypatch, plan, synth_text="combined answer [1810.04805]")
    researched = []

    async def fake_run_agent(question, thread_id=None, provider=None):
        researched.append((question, thread_id))
        return f"finding about {question}"

    monkeypatch.setattr(multi_mod, "run_agent", fake_run_agent)
    reply = await multi_mod.run_multi_agent("compare BERT and GPT")

    assert reply == "combined answer [1810.04805]"
    # researchers run per sub-question, single-shot (no thread)
    assert researched == [("what is BERT?", None), ("what is GPT?", None)]
    # synthesizer saw the question and both findings
    synth_input = calls[-1]["messages"][-1]["content"]
    assert "compare BERT and GPT" in synth_input
    assert "finding about what is BERT?" in synth_input


async def test_failed_researcher_reported_to_synthesizer(monkeypatch):
    import agents.multi as multi_mod

    plan = multi_mod.Plan(simple=False, sub_questions=["good q", "bad q"])
    calls = _patch_plan(monkeypatch, plan)

    async def flaky_run_agent(question, thread_id=None, provider=None):
        if question == "bad q":
            raise RuntimeError("mcp exploded")
        return "a finding"

    monkeypatch.setattr(multi_mod, "run_agent", flaky_run_agent)
    reply = await multi_mod.run_multi_agent("q")

    assert reply == "synthesized"
    synth_input = calls[-1]["messages"][-1]["content"]
    assert "FAILED: mcp exploded" in synth_input


async def test_run_chat_dispatches_on_agent_mode(monkeypatch):
    import agents.multi as multi_mod
    from config import settings

    async def fake_single(question, thread_id=None, provider=None):
        return "single"

    async def fake_multi(question, thread_id=None, provider=None):
        return "multi"

    monkeypatch.setattr(multi_mod, "run_agent", fake_single)
    monkeypatch.setattr(multi_mod, "run_multi_agent", fake_multi)

    monkeypatch.setattr(settings, "agent_mode", "single")
    assert await multi_mod.run_chat("q") == "single"
    monkeypatch.setattr(settings, "agent_mode", "multi")
    assert await multi_mod.run_chat("q") == "multi"


async def test_provider_reaches_planner_researchers_synthesizer(monkeypatch):
    import agents.multi as multi_mod
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
        return f"answer to {question}"

    monkeypatch.setattr(multi_mod, "generate", fake_generate)
    monkeypatch.setattr(multi_mod, "run_agent", fake_run_agent)
    reply = await multi_mod.run_multi_agent("big question", provider="openai")
    assert reply == "synthesis"
    assert all(k["provider"] == "openai" for k in seen_generate)
    assert seen_agent == ["openai", "openai"]
