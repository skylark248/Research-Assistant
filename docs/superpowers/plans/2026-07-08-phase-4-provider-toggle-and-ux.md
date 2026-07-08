# Phase 4: Provider Toggle, Streaming, Citations, Threads — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Per-request LLM provider selection from the UI (Anthropic / OpenAI / local Ollama), plus provider availability checks, SSE streaming (activity + tokens), source citations, and persistent thread history with a rebuilt frontend.

**Architecture:** `llm/base.py::generate()` already accepts a `provider` override — this phase threads that parameter from a new `ChatRequest.provider` field through `run_chat` → `run_agent`/`run_multi_agent` → every `generate()` call. Streaming uses callback-style `generate_stream()` in each client plus an `on_event` callback threaded the same way, drained by an SSE endpoint via `asyncio.Queue`. Citations ride the LangGraph state. Threads get their own small table beside LangGraph's checkpoint tables in `data/checkpoints.db`.

**Tech Stack:** FastAPI (SSE via `StreamingResponse`), LangGraph + AsyncSqliteSaver, anthropic / openai SDKs, sqlite3 stdlib, vanilla JS + `marked` + `DOMPurify` from CDN.

**Spec:** `docs/superpowers/specs/2026-07-08-phase-4-provider-toggle-and-ux-design.md`

## Global Constraints

- Python >= 3.11; run everything with `uv run …` from the project root.
- **No new runtime dependencies.** The Ollama probe uses `requests>=2.32` (already a runtime dep). `httpx` stays dev-only. (Amends the spec's "promote httpx" line — Task 1 updates the spec.)
- Unit tests must pass with **no API keys and no running services** (`uv run pytest` default deselects `integration` and `local` markers).
- Messages/tools use the **Anthropic shape** everywhere internally; only `llm/openai_client.py` adapts.
- Frontend stays vanilla JS, no build step. CDN scripts allowed: `marked`, `DOMPurify`.
- Provider names are exactly `anthropic`, `openai`, `local` (pydantic `Literal`).
- Commit style: `feat:` / `test:` / `docs:` prefixes, imperative mood (matches git log).

---

### Task 1: `GET /api/providers` — availability check

**Files:**
- Create: `api/providers.py`
- Modify: `api/main.py` (new endpoint)
- Modify: `docs/superpowers/specs/2026-07-08-phase-4-provider-toggle-and-ux-design.md` (httpx → requests)
- Test: `tests/test_providers.py`

**Interfaces:**
- Consumes: `config.settings` (keys, models, `local_base_url`, `llm_provider`).
- Produces: `ProviderStatus(BaseModel)` with fields `provider: str`, `available: bool`, `detail: str = ""`, `model: str`, `is_default: bool = False`; `check_provider(name: str) -> ProviderStatus`; `check_providers() -> list[ProviderStatus]`; `_probe_local() -> tuple[bool, str]`. Task 2 imports `check_provider` and `ProviderStatus` in `api/main.py`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_providers.py
def _client(monkeypatch):
    import api.main as api_main
    from fastapi.testclient import TestClient

    class FakeStore:
        def ping(self):
            pass

        def check_schema(self):
            pass

    monkeypatch.setattr(api_main, "VectorStore", FakeStore)
    return TestClient(api_main.app)


def test_cloud_availability_follows_keys(monkeypatch):
    from api import providers as prov

    monkeypatch.setattr(prov.settings, "anthropic_api_key", "sk-real")
    monkeypatch.setattr(prov.settings, "openai_api_key", "")
    monkeypatch.setattr(prov, "_probe_local", lambda: (False, "Ollama unreachable at http://x"))
    statuses = {s.provider: s for s in prov.check_providers()}
    assert statuses["anthropic"].available is True
    assert statuses["anthropic"].model == prov.settings.anthropic_model
    assert statuses["openai"].available is False
    assert statuses["openai"].detail == "no API key set"
    assert statuses["local"].available is False
    assert statuses["local"].detail == "Ollama unreachable at http://x"


def test_default_flag_follows_settings(monkeypatch):
    from api import providers as prov

    monkeypatch.setattr(prov.settings, "llm_provider", "openai")
    monkeypatch.setattr(prov, "_probe_local", lambda: (True, ""))
    statuses = {s.provider: s for s in prov.check_providers()}
    assert statuses["openai"].is_default is True
    assert statuses["anthropic"].is_default is False


def test_local_probe_hits_models_endpoint(monkeypatch):
    from api import providers as prov

    calls = {}

    class FakeResp:
        def raise_for_status(self):
            pass

    def fake_get(url, timeout):
        calls.update(url=url, timeout=timeout)
        return FakeResp()

    monkeypatch.setattr(prov.requests, "get", fake_get)
    ok, detail = prov._probe_local()
    assert (ok, detail) == (True, "")
    assert calls["url"].endswith("/models")
    assert calls["timeout"] == 1.5


def test_local_probe_down(monkeypatch):
    from api import providers as prov

    def fake_get(url, timeout):
        raise prov.requests.ConnectionError("boom")

    monkeypatch.setattr(prov.requests, "get", fake_get)
    ok, detail = prov._probe_local()
    assert ok is False
    assert prov.settings.local_base_url in detail


def test_providers_endpoint(monkeypatch):
    import api.providers as prov

    monkeypatch.setattr(prov, "_probe_local", lambda: (False, "down"))
    with _client(monkeypatch) as client:
        resp = client.get("/api/providers")
    assert resp.status_code == 200
    body = resp.json()
    assert [s["provider"] for s in body] == ["anthropic", "openai", "local"]
    assert all({"provider", "available", "detail", "model", "is_default"} <= set(s) for s in body)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_providers.py -v`
Expected: FAIL / ERROR with `ModuleNotFoundError: No module named 'api.providers'`

- [ ] **Step 3: Implement `api/providers.py`**

```python
"""Provider availability for the UI toggle.

Cloud providers are "available" when their API key is set (no network call —
a wrong key still fails loudly at chat time, same policy as everywhere else).
Local means Ollama answers /v1/models within 1.5s.
"""

import requests
from pydantic import BaseModel

from config import settings


class ProviderStatus(BaseModel):
    provider: str
    available: bool
    detail: str = ""
    model: str
    is_default: bool = False


def _probe_local() -> tuple[bool, str]:
    url = settings.local_base_url.rstrip("/") + "/models"
    try:
        resp = requests.get(url, timeout=1.5)
        resp.raise_for_status()
        return True, ""
    except requests.RequestException:
        return False, f"Ollama unreachable at {settings.local_base_url}"


def check_provider(name: str) -> ProviderStatus:
    if name == "anthropic":
        available = bool(settings.anthropic_api_key)
        return ProviderStatus(
            provider=name, available=available,
            detail="" if available else "no API key set",
            model=settings.anthropic_model, is_default=settings.llm_provider == name,
        )
    if name == "openai":
        available = bool(settings.openai_api_key)
        return ProviderStatus(
            provider=name, available=available,
            detail="" if available else "no API key set",
            model=settings.openai_model, is_default=settings.llm_provider == name,
        )
    if name == "local":
        available, detail = _probe_local()
        return ProviderStatus(
            provider=name, available=available, detail=detail,
            model=settings.local_model, is_default=settings.llm_provider == name,
        )
    raise ValueError(f"Unknown provider: {name}")


def check_providers() -> list[ProviderStatus]:
    return [check_provider(name) for name in ("anthropic", "openai", "local")]
```

- [ ] **Step 4: Add the endpoint to `api/main.py`**

Add import (with the existing imports):

```python
from api.providers import ProviderStatus, check_provider, check_providers
```

Add endpoint (next to the other routes, BEFORE the static mount at the bottom):

```python
@app.get("/api/providers", response_model=list[ProviderStatus])
async def providers() -> list[ProviderStatus]:
    # check_providers may block up to 1.5s probing Ollama; keep the loop free.
    return await run_in_threadpool(check_providers)
```

(`check_provider` is imported now but first used by Task 2 — that's fine, it keeps the import diff in one place. If the linter complains, import only `ProviderStatus, check_providers` here and let Task 2 add `check_provider`.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_providers.py -v`
Expected: 5 PASS

- [ ] **Step 6: Amend the spec (httpx line)**

In `docs/superpowers/specs/2026-07-08-phase-4-provider-toggle-and-ux-design.md`, replace:

```
- `httpx` moves from dev-only to runtime dependency (used for the Ollama
  probe; async-friendly inside FastAPI).
```

with:

```
- The Ollama probe uses `requests` (already a runtime dependency), executed
  via `run_in_threadpool`; `httpx` stays dev-only.
```

- [ ] **Step 7: Full suite + commit**

Run: `uv run pytest`
Expected: all pass (81+ passed, integration/local deselected)

```bash
git add api/providers.py api/main.py tests/test_providers.py docs/superpowers/specs/2026-07-08-phase-4-provider-toggle-and-ux-design.md
git commit -m "feat: provider availability endpoint GET /api/providers"
```

---

### Task 2: Per-request provider threading

**Files:**
- Modify: `agents/graph.py` (`build_graph`, `run_agent`)
- Modify: `agents/multi.py` (`_plan`, `_synthesize`, `run_multi_agent`, `run_chat`)
- Modify: `api/main.py` (`ChatRequest.provider`, availability guard)
- Test: `tests/test_graph.py`, `tests/test_multi.py`, `tests/test_api.py` (additions)

**Interfaces:**
- Consumes: `llm.base.generate(..., provider=...)` (exists); `api.providers.check_provider` (Task 1).
- Produces: `build_graph(toolbox, checkpointer=None, provider: str | None = None)`; `run_agent(question, thread_id=None, provider=None) -> str`; `run_chat(message, thread_id=None, provider=None) -> str`; `run_multi_agent(question, thread_id=None, provider=None) -> str`; `_plan(question, provider=None)`; `_synthesize(question, findings, provider=None)`; `ChatRequest.provider: Literal["anthropic", "openai", "local"] | None = None`. Tasks 3/6 extend these same signatures.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_graph.py`:

```python
async def test_provider_threads_to_generate(monkeypatch):
    import agents.graph as graph_mod

    seen = _scripted_generate(monkeypatch, [LLMResponse(text="hi")])
    graph = graph_mod.build_graph(FakeToolbox(), provider="local")
    await graph.ainvoke({"messages": [{"role": "user", "content": "q"}], "steps": 0})
    assert seen[0]["provider"] == "local"
```

Append to `tests/test_multi.py` (match the file's existing fake/monkeypatch style):

```python
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
```

Append to `tests/test_api.py`:

```python
def _allow_provider(monkeypatch, available=True, detail=""):
    import api.main as api_main
    from api.providers import ProviderStatus

    monkeypatch.setattr(
        api_main, "check_provider",
        lambda name: ProviderStatus(provider=name, available=available,
                                    detail=detail, model="m"),
    )


def test_chat_forwards_provider(monkeypatch):
    import api.main as api_main

    captured = {}

    async def fake_run_chat(question, thread_id=None, provider=None):
        captured["provider"] = provider
        return "ok"

    monkeypatch.setattr(api_main, "run_chat", fake_run_chat)
    _allow_provider(monkeypatch)
    with _client(monkeypatch) as client:
        resp = client.post("/api/chat", json={"message": "hi", "provider": "local"})
    assert resp.status_code == 200
    assert captured["provider"] == "local"


def test_chat_rejects_unavailable_provider(monkeypatch):
    import api.main as api_main

    async def fake_run_chat(question, thread_id=None, provider=None):
        raise AssertionError("agent must not start")

    monkeypatch.setattr(api_main, "run_chat", fake_run_chat)
    _allow_provider(monkeypatch, available=False, detail="no API key set")
    with _client(monkeypatch) as client:
        resp = client.post("/api/chat", json={"message": "hi", "provider": "openai"})
    assert resp.status_code == 400
    assert resp.json()["detail"] == "no API key set"


def test_chat_rejects_unknown_provider(monkeypatch):
    with _client(monkeypatch) as client:
        resp = client.post("/api/chat", json={"message": "hi", "provider": "gemini"})
    assert resp.status_code == 422
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_graph.py::test_provider_threads_to_generate tests/test_multi.py::test_provider_reaches_planner_researchers_synthesizer tests/test_api.py -v`
Expected: new tests FAIL (`build_graph() got an unexpected keyword argument 'provider'`, etc.)

- [ ] **Step 3: Thread provider through `agents/graph.py`**

Change the `build_graph` signature and both `generate` calls:

```python
def build_graph(toolbox, checkpointer=None, provider: str | None = None):
```

In `summarize_node`:

```python
        resp = await asyncio.to_thread(
            generate, [{"role": "user", "content": prompt}],
            system=SUMMARIZE_SYSTEM_PROMPT, provider=provider,
        )
```

In `agent_node`:

```python
        resp = await asyncio.to_thread(generate, history, system=system, tools=tools,
                                       provider=provider)
```

Change `run_agent`:

```python
async def run_agent(question: str, thread_id: str | None = None,
                    provider: str | None = None) -> str:
    """One agent turn. Same thread_id continues a conversation; omitted → fresh
    single-shot thread (direct callers like eval stay stateless)."""
    thread_id = thread_id or str(uuid.uuid4())
    Path(settings.checkpoint_db).parent.mkdir(parents=True, exist_ok=True)
    async with MCPToolbox() as toolbox, \
            AsyncSqliteSaver.from_conn_string(settings.checkpoint_db) as saver:
        graph = build_graph(toolbox, checkpointer=saver, provider=provider)
        ...  # rest unchanged
```

- [ ] **Step 4: Thread provider through `agents/multi.py`**

```python
def _plan(question: str, provider: str | None = None) -> Plan:
    resp = generate([{"role": "user", "content": question}],
                    system=PLANNER_SYSTEM_PROMPT, structured_schema=Plan,
                    provider=provider)
    return resp.parsed


def _synthesize(question: str, findings: list[tuple[str, str]],
                provider: str | None = None) -> str:
    parts = [f"Sub-question: {sq}\nFinding: {answer}" for sq, answer in findings]
    content = f"Question: {question}\n\n" + "\n\n---\n\n".join(parts)
    resp = generate([{"role": "user", "content": content}],
                    system=SYNTHESIZER_SYSTEM_PROMPT, provider=provider)
    return resp.text


async def run_multi_agent(question: str, thread_id: str | None = None,
                          provider: str | None = None) -> str:
    plan = await asyncio.to_thread(_plan, question, provider)
    if plan.simple or not plan.sub_questions:
        return await run_agent(question, thread_id, provider=provider)
    findings: list[tuple[str, str]] = []
    for sub_question in plan.sub_questions[:4]:
        try:
            findings.append((sub_question, await run_agent(sub_question, provider=provider)))
        except Exception as exc:
            logger.exception("Researcher failed for %r", sub_question)
            findings.append((sub_question, f"FAILED: {exc}"))
    return await asyncio.to_thread(_synthesize, question, findings, provider)


async def run_chat(message: str, thread_id: str | None = None,
                   provider: str | None = None) -> str:
    """Dispatch on agent_mode: the single loop (default) or the supervisor."""
    if settings.agent_mode == "multi":
        return await run_multi_agent(message, thread_id, provider=provider)
    return await run_agent(message, thread_id, provider=provider)
```

- [ ] **Step 5: API — provider field + availability guard in `api/main.py`**

Imports: add `from typing import Literal` and `from fastapi import FastAPI, HTTPException` (extend the existing fastapi import). Ensure `check_provider` is imported from `api.providers`.

```python
class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None  # omit to start a new conversation
    provider: Literal["anthropic", "openai", "local"] | None = None  # None → configured default
```

Add a helper and use it in the chat endpoint:

```python
async def _require_available(provider: str | None) -> None:
    if provider is None:
        return
    status = await run_in_threadpool(check_provider, provider)
    if not status.available:
        raise HTTPException(status_code=400, detail=status.detail)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    await _require_available(req.provider)
    thread_id = req.thread_id or str(uuid.uuid4())
    reply = await run_chat(req.message, thread_id, provider=req.provider)
    return ChatResponse(reply=reply, thread_id=thread_id)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_graph.py tests/test_multi.py tests/test_api.py -v`
Expected: all PASS (old + new)

- [ ] **Step 7: Full suite + commit**

Run: `uv run pytest`
Expected: all pass

```bash
git add agents/graph.py agents/multi.py api/main.py tests/test_graph.py tests/test_multi.py tests/test_api.py
git commit -m "feat: per-request LLM provider selection through chat API"
```

---

### Task 3: Citations through agent state to the API

**Files:**
- Modify: `agents/graph.py` (`AgentState`, `tools_node`, `run_agent`, new `AgentResult`, `_dedupe`)
- Modify: `agents/multi.py` (return `AgentResult`, union citations)
- Modify: `api/main.py` (`ChatResponse.citations`)
- Modify: `tests/test_integration_agent.py`, `tests/test_integration_phase2.py` (`.text` on replies)
- Test: `tests/test_graph.py`, `tests/test_multi.py`, `tests/test_api.py` (additions/updates)

**Interfaces:**
- Consumes: `RagAnswer.sources: list[str]` from `rag/answer.py`.
- Produces: `AgentResult(NamedTuple)` in `agents/graph.py` with `text: str`, `citations: list[str]`; `run_agent(...) -> AgentResult`; `run_multi_agent(...) -> AgentResult`; `run_chat(...) -> AgentResult`; `_dedupe(items: list[str]) -> list[str]` (order-preserving); `ChatResponse.citations: list[str] = []`. Task 6's `done` SSE event reuses `AgentResult`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_graph.py`:

```python
async def test_citations_collected_from_rag_query(monkeypatch):
    import agents.graph as graph_mod
    from rag.answer import RagAnswer

    monkeypatch.setattr(graph_mod, "answer_question",
                        lambda q: RagAnswer(text="Attention [1706.03762].",
                                            sources=["1706.03762"]))
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
```

Update the existing `test_run_agent_falls_back_when_step_limit_hit` final assertion: `reply = await graph_mod.run_agent("q")` becomes

```python
    result = await graph_mod.run_agent("q")
    assert result.text == graph_mod.STEP_LIMIT_MESSAGE
```

(keep whatever the current assertion checks, applied to `result.text`).

In `tests/test_multi.py`: update every fake `run_agent` to return `AgentResult`, e.g.

```python
    from agents.graph import AgentResult

    async def fake_run_agent(question, thread_id=None, provider=None):
        return AgentResult(text=f"answer to {question}", citations=["1706.03762"])
```

and every assertion on `run_chat`/`run_multi_agent` results to use `.text`. Add:

```python
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
```

In `tests/test_api.py`: update fake `run_chat` functions to return `AgentResult(text=..., citations=[...])` (import `from agents.graph import AgentResult`) and add:

```python
def test_chat_returns_citations(monkeypatch):
    import api.main as api_main
    from agents.graph import AgentResult

    async def fake_run_chat(question, thread_id=None, provider=None):
        return AgentResult(text="grounded", citations=["1706.03762"])

    monkeypatch.setattr(api_main, "run_chat", fake_run_chat)
    with _client(monkeypatch) as client:
        resp = client.post("/api/chat", json={"message": "hi"})
    assert resp.json()["citations"] == ["1706.03762"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_graph.py tests/test_multi.py tests/test_api.py -v`
Expected: new tests FAIL (`ImportError: cannot import name 'AgentResult'`, missing `citations` key)

- [ ] **Step 3: Implement in `agents/graph.py`**

Add to imports: `from typing import Annotated, NamedTuple, TypedDict` (extend existing line).

```python
class AgentResult(NamedTuple):
    text: str
    citations: list[str]


def _dedupe(items: list[str]) -> list[str]:
    """Order-preserving dedupe (dict keys keep insertion order)."""
    return list(dict.fromkeys(items))


class AgentState(TypedDict):
    messages: Annotated[list[dict], operator.add]
    steps: int
    summary: str
    citations: Annotated[list[str], operator.add]
```

In `tools_node`, collect sources and include them in the returned delta:

```python
    async def tools_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        results: list[dict] = []
        sources: list[str] = []
        for block in last["content"]:
            if block["type"] != "tool_use":
                continue
            name, args = block["name"], block["input"]
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
```

In `run_agent`: seed the channel and return `AgentResult`:

```python
        state = await graph.ainvoke(
            {"messages": [{"role": "user", "content": question}], "steps": 0,
             "citations": []},
            config={"recursion_limit": settings.agent_max_steps * 2 + 6,
                    "configurable": {"thread_id": thread_id}},
        )
        text = final_text(state)
        return AgentResult(text=text or STEP_LIMIT_MESSAGE,
                           citations=_dedupe(state.get("citations", [])))
```

Signature/docstring: `async def run_agent(question: str, thread_id: str | None = None, provider: str | None = None) -> AgentResult:`

Note: `citations` uses an `operator.add` reducer, so on a continued thread it accumulates across turns — the whole conversation's sources, deduped. That is the wanted behavior (chips reflect everything the answer built on).

- [ ] **Step 4: Implement in `agents/multi.py`**

Import: `from agents.graph import AgentResult, _dedupe, run_agent`.

```python
async def run_multi_agent(question: str, thread_id: str | None = None,
                          provider: str | None = None) -> AgentResult:
    plan = await asyncio.to_thread(_plan, question, provider)
    if plan.simple or not plan.sub_questions:
        return await run_agent(question, thread_id, provider=provider)
    findings: list[tuple[str, str]] = []
    citations: list[str] = []
    for sub_question in plan.sub_questions[:4]:
        try:
            result = await run_agent(sub_question, provider=provider)
            findings.append((sub_question, result.text))
            citations.extend(result.citations)
        except Exception as exc:
            logger.exception("Researcher failed for %r", sub_question)
            findings.append((sub_question, f"FAILED: {exc}"))
    text = await asyncio.to_thread(_synthesize, question, findings, provider)
    return AgentResult(text=text, citations=_dedupe(citations))
```

`run_chat` return type becomes `AgentResult` (body unchanged — both branches already return the new type).

- [ ] **Step 5: Update `api/main.py`**

```python
class ChatResponse(BaseModel):
    reply: str
    thread_id: str
    citations: list[str] = []


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    await _require_available(req.provider)
    thread_id = req.thread_id or str(uuid.uuid4())
    result = await run_chat(req.message, thread_id, provider=req.provider)
    return ChatResponse(reply=result.text, thread_id=thread_id,
                        citations=result.citations)
```

- [ ] **Step 6: Update integration tests (`.text`)**

In `tests/test_integration_agent.py` and `tests/test_integration_phase2.py`, each `reply = await run_agent(...)` now yields `AgentResult`; change downstream string assertions to `reply.text` (e.g. `assert "attention" in reply.text.lower()`). Keep everything else as is.

- [ ] **Step 7: Run tests, full suite, commit**

Run: `uv run pytest tests/test_graph.py tests/test_multi.py tests/test_api.py -v` → all PASS
Run: `uv run pytest` → all pass (integration files still deselected but must import cleanly)

```bash
git add agents/graph.py agents/multi.py api/main.py tests/
git commit -m "feat: collect rag_query citations through agent state into chat response"
```

---

### Task 4: Threads — table, transcript, endpoints

**Files:**
- Create: `api/threads.py`
- Modify: `api/main.py` (upsert on chat + 3 endpoints)
- Test: `tests/test_threads.py`, `tests/test_api.py` (additions)

**Interfaces:**
- Consumes: `settings.checkpoint_db`; LangGraph `AsyncSqliteSaver` checkpoints (read-only via `saver.aget`).
- Produces: `ThreadInfo(BaseModel)` (`thread_id, title, created_at, updated_at`); `TranscriptTurn(BaseModel)` (`role, text`); `upsert_thread(thread_id, first_message)`; `list_threads() -> list[ThreadInfo]`; `delete_thread(thread_id)`; `async get_transcript(thread_id) -> list[TranscriptTurn] | None`; `_turns_from_messages(messages) -> list[TranscriptTurn]`. Endpoints: `GET /api/threads`, `GET /api/threads/{id}` (404 unknown), `DELETE /api/threads/{id}`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_threads.py
def _use_tmp_db(monkeypatch, tmp_path):
    from api import threads as threads_mod

    monkeypatch.setattr(threads_mod.settings, "checkpoint_db", str(tmp_path / "cp.db"))
    return threads_mod


def test_upsert_and_list(monkeypatch, tmp_path):
    tm = _use_tmp_db(monkeypatch, tmp_path)
    tm.upsert_thread("t1", "x" * 100)
    tm.upsert_thread("t2", "second thread")
    tm.upsert_thread("t1", "ignored on update")
    rows = tm.list_threads()
    assert [r.thread_id for r in rows][0] == "t1"  # most recently updated first
    t1 = next(r for r in rows if r.thread_id == "t1")
    assert t1.title == "x" * 80  # truncated, set on first insert only
    assert t1.updated_at >= t1.created_at


def test_delete_thread(monkeypatch, tmp_path):
    tm = _use_tmp_db(monkeypatch, tmp_path)
    tm.upsert_thread("t1", "hello")
    tm.delete_thread("t1")
    assert tm.list_threads() == []


async def test_get_transcript_unknown_thread(monkeypatch, tmp_path):
    tm = _use_tmp_db(monkeypatch, tmp_path)
    assert await tm.get_transcript("nope") is None


def test_turns_from_messages():
    from api.threads import _turns_from_messages

    messages = [
        {"role": "user", "content": "what is attention?"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Let me check."},
            {"type": "tool_use", "id": "tu_1", "name": "rag_query", "input": {}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu_1", "content": "stuff"},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": "It is attention."}]},
    ]
    turns = _turns_from_messages(messages)
    assert [(t.role, t.text) for t in turns] == [
        ("user", "what is attention?"),
        ("assistant", "Let me check."),
        ("assistant", "It is attention."),
    ]
```

Append to `tests/test_api.py`:

```python
def test_thread_endpoints(monkeypatch, tmp_path):
    import api.main as api_main
    import api.threads as threads_mod
    from agents.graph import AgentResult

    monkeypatch.setattr(threads_mod.settings, "checkpoint_db", str(tmp_path / "cp.db"))

    async def fake_run_chat(question, thread_id=None, provider=None):
        return AgentResult(text="ok", citations=[])

    monkeypatch.setattr(api_main, "run_chat", fake_run_chat)
    with _client(monkeypatch) as client:
        client.post("/api/chat", json={"message": "first message"})
        rows = client.get("/api/threads").json()
        assert len(rows) == 1
        assert rows[0]["title"] == "first message"
        tid = rows[0]["thread_id"]
        # no checkpoint written (run_chat faked) → transcript 404
        assert client.get(f"/api/threads/{tid}").status_code == 404
        assert client.delete(f"/api/threads/{tid}").status_code == 200
        assert client.get("/api/threads").json() == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_threads.py tests/test_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'api.threads'`

- [ ] **Step 3: Implement `api/threads.py`**

```python
"""Thread bookkeeping for the UI.

A small `threads` table lives in the same SQLite file as LangGraph's
checkpoint tables (one DB to manage); LangGraph's own tables are never
written here. Transcripts are read from the latest checkpoint, so there is
no duplicate message storage.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel

from config import settings


class ThreadInfo(BaseModel):
    thread_id: str
    title: str
    created_at: str
    updated_at: str


class TranscriptTurn(BaseModel):
    role: str  # "user" | "assistant"
    text: str


def _connect() -> sqlite3.Connection:
    Path(settings.checkpoint_db).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.checkpoint_db)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS threads (
               thread_id  TEXT PRIMARY KEY,
               title      TEXT NOT NULL,
               created_at TEXT NOT NULL,
               updated_at TEXT NOT NULL)"""
    )
    return conn


def upsert_thread(thread_id: str, first_message: str) -> None:
    """Insert with title on first sight; later calls only bump updated_at."""
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            """INSERT INTO threads (thread_id, title, created_at, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(thread_id) DO UPDATE SET updated_at = excluded.updated_at""",
            (thread_id, first_message[:80], now, now),
        )


def list_threads() -> list[ThreadInfo]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT thread_id, title, created_at, updated_at FROM threads "
            "ORDER BY updated_at DESC"
        ).fetchall()
    return [ThreadInfo(thread_id=r[0], title=r[1], created_at=r[2], updated_at=r[3])
            for r in rows]


def delete_thread(thread_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM threads WHERE thread_id = ?", (thread_id,))
        for table in ("checkpoints", "writes"):  # LangGraph's tables
            try:
                conn.execute(f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))
            except sqlite3.OperationalError:
                pass  # table not created yet (no real chat has run)


def _turns_from_messages(messages: list[dict]) -> list[TranscriptTurn]:
    """Plain-text turns only; tool_use/tool_result traffic is omitted."""
    turns: list[TranscriptTurn] = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            if message["role"] == "user" and content:
                turns.append(TranscriptTurn(role="user", text=content))
        elif message["role"] == "assistant":
            texts = [b["text"] for b in content if b["type"] == "text"]
            if texts:
                turns.append(TranscriptTurn(role="assistant", text="\n".join(texts)))
    return turns


async def get_transcript(thread_id: str) -> list[TranscriptTurn] | None:
    """Turns from the latest checkpoint; None when the thread has none."""
    async with AsyncSqliteSaver.from_conn_string(settings.checkpoint_db) as saver:
        checkpoint = await saver.aget({"configurable": {"thread_id": thread_id}})
    if checkpoint is None:
        return None
    return _turns_from_messages(checkpoint["channel_values"].get("messages", []))
```

- [ ] **Step 4: Wire into `api/main.py`**

Import:

```python
from api.threads import (ThreadInfo, TranscriptTurn, delete_thread, get_transcript,
                         list_threads, upsert_thread)
```

In the `chat` endpoint, after a successful reply (only successful chats create/refresh a thread):

```python
    result = await run_chat(req.message, thread_id, provider=req.provider)
    await run_in_threadpool(upsert_thread, thread_id, req.message)
    return ChatResponse(reply=result.text, thread_id=thread_id,
                        citations=result.citations)
```

New endpoints (before the static mount):

```python
@app.get("/api/threads", response_model=list[ThreadInfo])
async def threads() -> list[ThreadInfo]:
    return await run_in_threadpool(list_threads)


@app.get("/api/threads/{thread_id}", response_model=list[TranscriptTurn])
async def thread_transcript(thread_id: str) -> list[TranscriptTurn]:
    turns = await get_transcript(thread_id)
    if turns is None:
        raise HTTPException(status_code=404, detail="unknown thread")
    return turns


@app.delete("/api/threads/{thread_id}")
async def remove_thread(thread_id: str) -> dict:
    await run_in_threadpool(delete_thread, thread_id)
    return {"deleted": thread_id}
```

- [ ] **Step 5: Run tests, full suite, commit**

Run: `uv run pytest tests/test_threads.py tests/test_api.py -v` → all PASS
Run: `uv run pytest` → all pass

```bash
git add api/threads.py api/main.py tests/test_threads.py tests/test_api.py
git commit -m "feat: thread list, transcript restore, and delete endpoints"
```

---

### Task 5: `generate_stream` in all three clients + dispatcher

**Files:**
- Modify: `llm/anthropic_client.py`, `llm/openai_client.py`, `llm/local_client.py`, `llm/base.py`
- Modify: `docs/superpowers/specs/2026-07-08-phase-4-provider-toggle-and-ux-design.md` (generator → callback wording)
- Test: `tests/test_llm_stream.py`

**Interfaces:**
- Consumes: existing `_get_client()` singletons, `convert_messages`, `convert_tools`, `_to_llm_response`.
- Produces (all callback-style — a callback survives `asyncio.to_thread`, a generator doesn't):
  - `llm.base.generate_stream(messages, *, system=None, tools=None, on_delta: Callable[[str], None], provider=None, max_tokens=None) -> LLMResponse`
  - `generate_anthropic_stream(messages, *, system=None, tools=None, max_tokens=4096, on_delta) -> LLMResponse`
  - `generate_openai_stream(messages, *, system=None, tools=None, max_tokens=4096, on_delta, client=None, model=None) -> LLMResponse`
  - `generate_local_stream(...)` — same as openai minus client/model.
  - No `structured_schema` anywhere in the stream path (planner stays on `generate`). `usage` stays `{}` on streamed OpenAI/local responses (chunks don't carry it).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_llm_stream.py
from types import SimpleNamespace

from llm.base import LLMResponse


def _chunk(content=None, tool_calls=None, finish_reason=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta,
                                                    finish_reason=finish_reason)])


def _tc_delta(index, id=None, name=None, arguments=None):
    fn = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=id, function=fn)


class FakeOpenAIClient:
    def __init__(self, chunks):
        completions = SimpleNamespace(create=self._create)
        self.chat = SimpleNamespace(completions=completions)
        self._chunks = chunks
        self.kwargs = None

    def _create(self, **kwargs):
        self.kwargs = kwargs
        return iter(self._chunks)


def test_openai_stream_accumulates_text_and_calls_on_delta():
    from llm.openai_client import generate_openai_stream

    client = FakeOpenAIClient([
        _chunk(content="Hel"),
        _chunk(content="lo"),
        _chunk(finish_reason="stop"),
    ])
    deltas = []
    resp = generate_openai_stream([{"role": "user", "content": "hi"}],
                                  on_delta=deltas.append, client=client, model="m")
    assert deltas == ["Hel", "lo"]
    assert resp.text == "Hello"
    assert resp.stop_reason == "stop"
    assert client.kwargs["stream"] is True


def test_openai_stream_accumulates_tool_calls():
    from llm.openai_client import generate_openai_stream

    client = FakeOpenAIClient([
        _chunk(tool_calls=[_tc_delta(0, id="tc_1", name="rag_query", arguments='{"que')]),
        _chunk(tool_calls=[_tc_delta(0, arguments='stion": "x"}')]),
        _chunk(finish_reason="tool_calls"),
    ])
    resp = generate_openai_stream([{"role": "user", "content": "hi"}],
                                  on_delta=lambda t: None, client=client, model="m")
    assert resp.text == ""
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].id == "tc_1"
    assert resp.tool_calls[0].name == "rag_query"
    assert resp.tool_calls[0].input == {"question": "x"}


def test_anthropic_stream(monkeypatch):
    import llm.anthropic_client as ac

    final_message = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="Hello")],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=1, output_tokens=2),
    )

    class FakeStream:
        text_stream = iter(["Hel", "lo"])

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get_final_message(self):
            return final_message

    class FakeClient:
        messages = SimpleNamespace(stream=lambda **kwargs: FakeStream())

    monkeypatch.setattr(ac, "_get_client", lambda: FakeClient())
    deltas = []
    resp = ac.generate_anthropic_stream([{"role": "user", "content": "hi"}],
                                        on_delta=deltas.append)
    assert deltas == ["Hel", "lo"]
    assert resp.text == "Hello"
    assert resp.stop_reason == "end_turn"


def test_base_dispatch(monkeypatch):
    import llm.anthropic_client as ac
    import llm.base as base

    def fake_stream(messages, **kwargs):
        kwargs["on_delta"]("x")
        return LLMResponse(text="x")

    monkeypatch.setattr(ac, "generate_anthropic_stream", fake_stream)
    deltas = []
    resp = base.generate_stream([{"role": "user", "content": "hi"}],
                                on_delta=deltas.append, provider="anthropic")
    assert resp.text == "x" and deltas == ["x"]


def test_base_dispatch_unknown_provider():
    import pytest

    import llm.base as base

    with pytest.raises(ValueError, match="Unknown provider"):
        base.generate_stream([{"role": "user", "content": "hi"}],
                             on_delta=lambda t: None, provider="gemini")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_llm_stream.py -v`
Expected: FAIL with `ImportError: cannot import name 'generate_openai_stream'`

- [ ] **Step 3: Implement `generate_anthropic_stream`**

Append to `llm/anthropic_client.py` (imports already cover everything):

```python
def generate_anthropic_stream(
    messages: list[dict],
    *,
    system: str | list[dict] | None = None,
    tools: list[dict] | None = None,
    max_tokens: int = 4096,
    on_delta,
) -> LLMResponse:
    """Streaming variant: on_delta(str) per text chunk, returns the full response.

    tool_use blocks are accumulated by the SDK and arrive only on the final
    message — never through on_delta.
    """
    client = _get_client()
    kwargs: dict = {
        "model": settings.anthropic_model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system is not None:
        kwargs["system"] = system
    if tools:
        kwargs["tools"] = tools
    with client.messages.stream(**kwargs) as stream:
        for text in stream.text_stream:
            on_delta(text)
        response = stream.get_final_message()
    return _to_llm_response(response)
```

- [ ] **Step 4: Implement `generate_openai_stream`**

Append to `llm/openai_client.py`:

```python
def generate_openai_stream(
    messages: list[dict],
    *,
    system: str | list[dict] | None = None,
    tools: list[dict] | None = None,
    max_tokens: int = 4096,
    on_delta,
    client=None,
    model: str | None = None,
) -> LLMResponse:
    """Streaming variant: on_delta(str) per text chunk, returns the full response.

    Tool-call fragments are accumulated internally (never sent to on_delta);
    streamed chunks carry no usage, so usage stays {}.
    """
    client = client or _get_client()
    kwargs: dict = {
        "model": model or settings.openai_model,
        "messages": convert_messages(messages, system),
        "max_completion_tokens": max_tokens,
        "stream": True,
    }
    if tools:
        kwargs["tools"] = convert_tools(tools)
    text_parts: list[str] = []
    acc: dict[int, dict] = {}  # index -> {"id", "name", "arguments"}
    finish_reason = None
    for chunk in client.chat.completions.create(**kwargs):
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = choice.delta
        if delta.content:
            text_parts.append(delta.content)
            on_delta(delta.content)
        for tc in (delta.tool_calls or []):
            slot = acc.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
            if tc.id:
                slot["id"] = tc.id
            if tc.function and tc.function.name:
                slot["name"] = tc.function.name
            if tc.function and tc.function.arguments:
                slot["arguments"] += tc.function.arguments
        if choice.finish_reason:
            finish_reason = choice.finish_reason
    tool_calls = [
        ToolCall(id=slot["id"], name=slot["name"],
                 input=json.loads(slot["arguments"] or "{}"))
        for _, slot in sorted(acc.items())
    ]
    return LLMResponse(text="".join(text_parts), tool_calls=tool_calls,
                       stop_reason=finish_reason)
```

- [ ] **Step 5: Implement `generate_local_stream` and the dispatcher**

Append to `llm/local_client.py`:

```python
def generate_local_stream(
    messages: list[dict],
    *,
    system: str | list[dict] | None = None,
    tools: list[dict] | None = None,
    max_tokens: int = 4096,
    on_delta,
) -> LLMResponse:
    from llm.openai_client import generate_openai_stream

    return generate_openai_stream(
        messages, system=system, tools=tools, max_tokens=max_tokens,
        on_delta=on_delta, client=_get_client(), model=settings.local_model,
    )
```

Append to `llm/base.py`:

```python
def generate_stream(
    messages: list[dict],
    *,
    system: str | list[dict] | None = None,
    tools: list[dict] | None = None,
    on_delta,
    provider: str | None = None,
    max_tokens: int | None = None,
) -> LLMResponse:
    """Streaming variant of generate(): on_delta(str) fires per text chunk,
    the complete LLMResponse (with tool_calls) is returned at the end.
    No structured_schema — structured calls stay on generate()."""
    provider = provider or settings.llm_provider
    max_tokens = max_tokens or settings.llm_max_tokens
    if provider == "anthropic":
        import llm.anthropic_client as anthropic_client

        return anthropic_client.generate_anthropic_stream(
            messages, system=system, tools=tools, max_tokens=max_tokens,
            on_delta=on_delta,
        )
    if provider == "openai":
        import llm.openai_client as openai_client

        return openai_client.generate_openai_stream(
            messages, system=system, tools=tools, max_tokens=max_tokens,
            on_delta=on_delta,
        )
    if provider == "local":
        import llm.local_client as local_client

        return local_client.generate_local_stream(
            messages, system=system, tools=tools, max_tokens=max_tokens,
            on_delta=on_delta,
        )
    raise ValueError(f"Unknown provider: {provider}")
```

(Module-attribute access — `anthropic_client.generate_anthropic_stream` — keeps `monkeypatch.setattr(ac, ...)` effective.)

- [ ] **Step 6: Amend the spec (generator → callback)**

In the spec's "generate_stream() in all three clients" bullet, replace the "it is a generator: yields text deltas (strings) as they arrive, then returns" phrasing with: "it takes an `on_delta(str)` callback invoked per text chunk and returns the final `LLMResponse` (callback-style survives `asyncio.to_thread`; a generator would not)".

- [ ] **Step 7: Run tests, full suite, commit**

Run: `uv run pytest tests/test_llm_stream.py -v` → 5 PASS
Run: `uv run pytest` → all pass

```bash
git add llm/ tests/test_llm_stream.py docs/superpowers/specs/2026-07-08-phase-4-provider-toggle-and-ux-design.md
git commit -m "feat: callback-style generate_stream for anthropic, openai, and local clients"
```

---

### Task 6: `on_event` plumbing + SSE endpoint `POST /api/chat/stream`

**Files:**
- Modify: `agents/graph.py` (`build_graph(..., on_event=None)`, `run_agent(..., on_event=None)`)
- Modify: `agents/multi.py` (`_synthesize(..., on_delta=None)`, `run_multi_agent(..., on_event=None)`, `run_chat(..., on_event=None)`)
- Modify: `api/main.py` (SSE endpoint)
- Test: `tests/test_graph.py`, `tests/test_multi.py` (additions), `tests/test_api_stream.py`

**Interfaces:**
- Consumes: `generate_stream` (Task 5), `AgentResult` (Task 3), `upsert_thread` (Task 4), `_require_available` (Task 2).
- Produces: `on_event: Callable[[dict], None] | None` threaded through `run_chat` / `run_agent` / `run_multi_agent` / `build_graph`. Event dicts (exact keys):
  - `{"event": "status", "text": str}`
  - `{"event": "delta", "text": str}`
  - `{"event": "turn_end", "has_tools": bool}`
  - endpoint-added: `{"event": "done", "reply", "thread_id", "citations"}`, `{"event": "error", "message"}`
  - SSE wire format per event: `event: <name>\ndata: <json-of-remaining-keys>\n\n`. Task 7's frontend parses exactly this.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_graph.py`:

```python
async def test_on_event_streams_deltas_and_statuses(monkeypatch):
    import agents.graph as graph_mod
    from rag.answer import RagAnswer

    monkeypatch.setattr(graph_mod, "answer_question",
                        lambda q: RagAnswer(text="A.", sources=["1706.03762"]))
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
```

(The first scripted turn produces no text, so no deltas precede the first `turn_end`.)

Append to `tests/test_multi.py`:

```python
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
```

```python
# tests/test_api_stream.py
def _client(monkeypatch):
    import api.main as api_main
    from fastapi.testclient import TestClient

    class FakeStore:
        def ping(self):
            pass

        def check_schema(self):
            pass

    monkeypatch.setattr(api_main, "VectorStore", FakeStore)
    return TestClient(api_main.app)


def test_chat_stream_sse_sequence(monkeypatch, tmp_path):
    import api.main as api_main
    import api.threads as threads_mod
    from agents.graph import AgentResult

    monkeypatch.setattr(threads_mod.settings, "checkpoint_db", str(tmp_path / "cp.db"))

    async def fake_run_chat(question, thread_id=None, provider=None, on_event=None):
        on_event({"event": "status", "text": "calling rag_query…"})
        on_event({"event": "delta", "text": "Hel"})
        on_event({"event": "delta", "text": "lo"})
        on_event({"event": "turn_end", "has_tools": False})
        return AgentResult(text="Hello", citations=["1706.03762"])

    monkeypatch.setattr(api_main, "run_chat", fake_run_chat)
    with _client(monkeypatch) as client:
        with client.stream("POST", "/api/chat/stream", json={"message": "hi"}) as resp:
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("text/event-stream")
            body = "".join(resp.iter_text())
    assert body.index("event: status") < body.index("event: delta")
    assert body.index("event: delta") < body.index("event: turn_end")
    assert body.index("event: turn_end") < body.index("event: done")
    assert '"reply": "Hello"' in body
    assert "1706.03762" in body
    # successful stream registers the thread
    with _client(monkeypatch) as client:
        assert len(client.get("/api/threads").json()) == 1


def test_chat_stream_error_event(monkeypatch):
    import api.main as api_main

    async def failing_run_chat(question, thread_id=None, provider=None, on_event=None):
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(api_main, "run_chat", failing_run_chat)
    with _client(monkeypatch) as client:
        with client.stream("POST", "/api/chat/stream", json={"message": "hi"}) as resp:
            body = "".join(resp.iter_text())
    assert "event: error" in body
    assert "provider exploded" in body
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_graph.py tests/test_multi.py tests/test_api_stream.py -v`
Expected: new tests FAIL (`unexpected keyword argument 'on_event'`, 404 on `/api/chat/stream`)

- [ ] **Step 3: Plumb `on_event` through `agents/graph.py`**

Import: `from llm.base import generate, generate_stream`.

`build_graph(toolbox, checkpointer=None, provider=None, on_event=None)`.

In `agent_node`, replace the single generate call:

```python
        if on_event is None:
            resp = await asyncio.to_thread(generate, history, system=system,
                                           tools=tools, provider=provider)
        else:
            def _stream() -> "LLMResponse":
                return generate_stream(
                    history, system=system, tools=tools, provider=provider,
                    on_delta=lambda t: on_event({"event": "delta", "text": t}),
                )
            resp = await asyncio.to_thread(_stream)
            on_event({"event": "turn_end", "has_tools": bool(resp.tool_calls)})
```

In `tools_node`, at the top of the per-block loop right after `name, args = ...`:

```python
            if on_event is not None:
                on_event({"event": "status", "text": f"calling {name}…"})
```

In `summarize_node`, right before the generate call:

```python
        if on_event is not None:
            on_event({"event": "status", "text": "summarizing conversation…"})
```

`run_agent` gains and forwards the parameter:

```python
async def run_agent(question: str, thread_id: str | None = None,
                    provider: str | None = None, on_event=None) -> AgentResult:
    ...
        graph = build_graph(toolbox, checkpointer=saver, provider=provider,
                            on_event=on_event)
```

- [ ] **Step 4: Plumb through `agents/multi.py`**

Import: `from llm.base import generate, generate_stream`.

```python
def _synthesize(question: str, findings: list[tuple[str, str]],
                provider: str | None = None, on_delta=None) -> str:
    parts = [f"Sub-question: {sq}\nFinding: {answer}" for sq, answer in findings]
    content = f"Question: {question}\n\n" + "\n\n---\n\n".join(parts)
    messages = [{"role": "user", "content": content}]
    if on_delta is None:
        resp = generate(messages, system=SYNTHESIZER_SYSTEM_PROMPT, provider=provider)
    else:
        resp = generate_stream(messages, system=SYNTHESIZER_SYSTEM_PROMPT,
                               provider=provider, on_delta=on_delta)
    return resp.text


async def run_multi_agent(question: str, thread_id: str | None = None,
                          provider: str | None = None, on_event=None) -> AgentResult:
    if on_event is not None:
        on_event({"event": "status", "text": "planning…"})
    plan = await asyncio.to_thread(_plan, question, provider)
    if plan.simple or not plan.sub_questions:
        return await run_agent(question, thread_id, provider=provider,
                               on_event=on_event)
    findings: list[tuple[str, str]] = []
    citations: list[str] = []
    for sub_question in plan.sub_questions[:4]:
        if on_event is not None:
            on_event({"event": "status", "text": f"researching: {sub_question}"})
        try:
            # researchers run silently — only the synthesizer token-streams
            result = await run_agent(sub_question, provider=provider)
            findings.append((sub_question, result.text))
            citations.extend(result.citations)
        except Exception as exc:
            logger.exception("Researcher failed for %r", sub_question)
            findings.append((sub_question, f"FAILED: {exc}"))
    on_delta = (lambda t: on_event({"event": "delta", "text": t})) if on_event else None
    text = await asyncio.to_thread(_synthesize, question, findings, provider, on_delta)
    if on_event is not None:
        on_event({"event": "turn_end", "has_tools": False})
    return AgentResult(text=text, citations=_dedupe(citations))


async def run_chat(message: str, thread_id: str | None = None,
                   provider: str | None = None, on_event=None) -> AgentResult:
    """Dispatch on agent_mode: the single loop (default) or the supervisor."""
    if settings.agent_mode == "multi":
        return await run_multi_agent(message, thread_id, provider=provider,
                                     on_event=on_event)
    return await run_agent(message, thread_id, provider=provider, on_event=on_event)
```

- [ ] **Step 5: SSE endpoint in `api/main.py`**

Imports: add `import asyncio`, `import json`, `import logging`, `from fastapi.responses import StreamingResponse`. Add `logger = logging.getLogger(__name__)` below the imports.

```python
@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """SSE: status/delta/turn_end events while the agent works, then done|error."""
    await _require_available(req.provider)
    thread_id = req.thread_id or str(uuid.uuid4())
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def on_event(event: dict) -> None:
        # Called from worker threads (generate_stream runs in to_thread).
        loop.call_soon_threadsafe(queue.put_nowait, event)

    async def worker() -> None:
        try:
            result = await run_chat(req.message, thread_id,
                                    provider=req.provider, on_event=on_event)
            await run_in_threadpool(upsert_thread, thread_id, req.message)
            await queue.put({"event": "done", "reply": result.text,
                             "thread_id": thread_id, "citations": result.citations})
        except Exception as exc:
            logger.exception("chat stream failed")
            await queue.put({"event": "error", "message": str(exc)})
        await queue.put(None)  # sentinel: stream complete

    async def sse():
        task = asyncio.create_task(worker())
        try:
            while (event := await queue.get()) is not None:
                name = event.pop("event")
                yield f"event: {name}\ndata: {json.dumps(event)}\n\n"
        finally:
            await task

    return StreamingResponse(sse(), media_type="text/event-stream")
```

- [ ] **Step 6: Run tests, full suite, commit**

Run: `uv run pytest tests/test_graph.py tests/test_multi.py tests/test_api_stream.py -v` → all PASS
Run: `uv run pytest` → all pass

```bash
git add agents/graph.py agents/multi.py api/main.py tests/
git commit -m "feat: SSE chat streaming with agent activity and token deltas"
```

---

### Task 7: Frontend rebuild

**Files:**
- Modify: `api/static/index.html` (full rewrite)
- Modify: `api/static/app.js` (full rewrite)

**Interfaces:**
- Consumes: `GET /api/providers` (Task 1), `POST /api/chat/stream` SSE events (Task 6), `GET/DELETE /api/threads`, `GET /api/threads/{id}` (Task 4), `POST /api/ingest` (existing).
- Produces: UI only — no exports.

- [ ] **Step 1: Rewrite `api/static/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Paper Research Assistant</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: system-ui, sans-serif; margin: 0; display: flex; height: 100vh; }
    #sidebar { width: 260px; border-right: 1px solid #ddd; padding: 1rem; overflow-y: auto; flex-shrink: 0; }
    #sidebar h2 { font-size: .9rem; text-transform: uppercase; color: #777; margin: 0 0 .5rem; }
    .thread { display: flex; align-items: center; gap: .3rem; padding: .35rem .5rem; border-radius: 6px; cursor: pointer; font-size: .85rem; }
    .thread:hover { background: #f0f0f0; }
    .thread.active { background: #e6ecff; }
    .thread .title { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .thread .del { border: none; background: none; color: #999; cursor: pointer; }
    .thread .del:hover { color: #c00; }
    #main { flex: 1; display: flex; flex-direction: column; max-width: 860px; margin: 0 auto; padding: 1rem; min-width: 0; }
    h1 { font-size: 1.3rem; margin: 0 0 1rem; }
    fieldset { border: 1px solid #ccc; border-radius: 6px; margin-bottom: 1rem; }
    input[type=text] { padding: .5rem; }
    button { padding: .5rem 1rem; }
    #log { flex: 1; overflow-y: auto; border: 1px solid #ccc; border-radius: 6px; padding: 1rem; }
    .user { color: #003; background: #eef2ff; border-radius: 8px; padding: .5rem .75rem; margin: .5rem 0; }
    .bot { color: #030; background: #f2fff2; border-radius: 8px; padding: .5rem .75rem; margin: .5rem 0; }
    .bot p:first-child { margin-top: 0; }
    .bot p:last-child { margin-bottom: 0; }
    .status { color: #888; font-style: italic; font-size: .85rem; margin: .2rem 0; }
    .activity { color: #999; font-size: .8rem; margin: .15rem 0 .15rem 1rem; }
    .citations { margin: .25rem 0 .75rem; }
    .citations a { display: inline-block; font-size: .75rem; background: #eee; border-radius: 10px; padding: .1rem .5rem; margin-right: .3rem; color: #336; text-decoration: none; }
    .citations a:hover { background: #ddd; }
    #chat-row { display: flex; gap: .5rem; margin-top: 1rem; }
    #chat-input { flex: 1; }
    #provider { padding: .4rem; }
    #ingest-row input { width: 60%; }
  </style>
</head>
<body>
  <nav id="sidebar">
    <button id="new-conv-btn" style="width:100%; margin-bottom:1rem;">+ New conversation</button>
    <h2>Threads</h2>
    <div id="thread-list"></div>
  </nav>

  <main id="main">
    <h1>Paper Research Assistant</h1>

    <fieldset id="ingest-row">
      <legend>Ingest papers</legend>
      <input id="ingest-query" type="text" placeholder="arXiv search, e.g. attention is all you need">
      <button id="ingest-btn">Ingest</button>
      <div id="ingest-status" class="status"></div>
    </fieldset>

    <div id="log"></div>

    <div id="chat-row">
      <input id="chat-input" type="text" placeholder="Ask about the ingested papers…">
      <select id="provider" title="Reasoning model"></select>
      <button id="chat-btn">Send</button>
    </div>
  </main>

  <script src="https://cdn.jsdelivr.net/npm/marked@12/marked.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/dompurify@3/dist/purify.min.js"></script>
  <script src="/app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Rewrite `api/static/app.js`**

```javascript
const log = document.getElementById("log");
const threadList = document.getElementById("thread-list");
const providerSelect = document.getElementById("provider");
let threadId = null; // set from the first reply; sent back to continue the thread

// ---------- rendering ----------

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

function scrollLog() { log.scrollTop = log.scrollHeight; }

function renderMarkdown(node, text) {
  node.innerHTML = DOMPurify.sanitize(marked.parse(text));
}

function addUser(text) {
  const node = el("div", "user", `You: ${text}`);
  log.appendChild(node);
  scrollLog();
}

function addBotMarkdown(text, citations) {
  const node = el("div", "bot");
  renderMarkdown(node, text);
  log.appendChild(node);
  addCitations(citations);
  scrollLog();
}

function addCitations(citations) {
  if (!citations || !citations.length) return;
  const row = el("div", "citations");
  for (const id of citations) {
    const a = el("a", null, id);
    a.href = `https://arxiv.org/abs/${id}`;
    a.target = "_blank";
    a.rel = "noopener";
    row.appendChild(a);
  }
  log.appendChild(row);
}

function addStatus(text) {
  const node = el("div", "status", text);
  log.appendChild(node);
  scrollLog();
  return node;
}

function addActivity(text) {
  const node = el("div", "activity", text);
  log.appendChild(node);
  scrollLog();
}

// ---------- providers ----------

async function loadProviders() {
  const resp = await fetch("/api/providers");
  const providers = await resp.json();
  providerSelect.replaceChildren();
  let selected = null;
  for (const p of providers) {
    const opt = document.createElement("option");
    opt.value = p.provider;
    opt.textContent = p.available ? `${p.provider} (${p.model})`
                                  : `${p.provider} — ${p.detail}`;
    opt.disabled = !p.available;
    providerSelect.appendChild(opt);
    if (p.available && (selected === null || p.is_default)) selected = p.provider;
  }
  if (selected) providerSelect.value = selected;
}

// ---------- SSE chat ----------

function parseSSE(buffer, onEvent) {
  // Returns the unconsumed tail of buffer; calls onEvent(name, data) per event.
  const events = buffer.split("\n\n");
  const tail = events.pop(); // possibly incomplete
  for (const raw of events) {
    let name = "message";
    let data = "";
    for (const line of raw.split("\n")) {
      if (line.startsWith("event: ")) name = line.slice(7).trim();
      else if (line.startsWith("data: ")) data += line.slice(6);
    }
    if (data) onEvent(name, JSON.parse(data));
  }
  return tail;
}

async function sendMessage() {
  const input = document.getElementById("chat-input");
  const message = input.value.trim();
  if (!message) return;
  input.value = "";
  addUser(message);

  const pending = el("div", "bot"); // live-updating bubble for streamed deltas
  log.appendChild(pending);
  let pendingText = "";
  const thinking = addStatus("thinking…");

  const body = { message, provider: providerSelect.value || null };
  if (threadId) body.thread_id = threadId;

  try {
    const resp = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }
    thinking.remove();
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let finished = false;
    while (!finished) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      buffer = parseSSE(buffer, (name, data) => {
        if (name === "status") {
          addActivity(data.text);
        } else if (name === "delta") {
          pendingText += data.text;
          pending.textContent = pendingText;
          scrollLog();
        } else if (name === "turn_end") {
          if (data.has_tools && pendingText) {
            addActivity(pendingText); // tool-reasoning text → activity feed
          }
          pendingText = "";
          pending.textContent = "";
        } else if (name === "done") {
          threadId = data.thread_id;
          renderMarkdown(pending, data.reply); // authoritative full reply
          addCitations(data.citations);
          finished = true;
          loadThreads();
        } else if (name === "error") {
          pending.remove();
          addStatus(`Chat failed: ${data.message}`);
          finished = true;
        }
      });
    }
    scrollLog();
  } catch (err) {
    thinking.remove();
    pending.remove();
    addStatus(`Chat failed: ${err.message}`);
  }
}

// ---------- threads ----------

async function loadThreads() {
  const resp = await fetch("/api/threads");
  const threads = await resp.json();
  threadList.replaceChildren();
  for (const t of threads) {
    const row = el("div", "thread" + (t.thread_id === threadId ? " active" : ""));
    const title = el("span", "title", t.title);
    title.title = t.title;
    row.appendChild(title);
    const del = el("button", "del", "✕");
    del.title = "Delete thread";
    del.addEventListener("click", async (e) => {
      e.stopPropagation();
      await fetch(`/api/threads/${t.thread_id}`, { method: "DELETE" });
      if (t.thread_id === threadId) startNewConversation();
      loadThreads();
    });
    row.appendChild(del);
    row.addEventListener("click", () => openThread(t.thread_id));
    threadList.appendChild(row);
  }
}

async function openThread(id) {
  const resp = await fetch(`/api/threads/${id}`);
  if (!resp.ok) return;
  const turns = await resp.json();
  threadId = id;
  log.replaceChildren();
  for (const turn of turns) {
    if (turn.role === "user") addUser(turn.text);
    else addBotMarkdown(turn.text, []);
  }
  loadThreads(); // refresh active highlight
}

function startNewConversation() {
  threadId = null;
  log.replaceChildren();
  addStatus("New conversation started.");
  loadThreads();
}

// ---------- ingest (unchanged behavior) ----------

async function post(path, body) {
  const resp = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) throw new Error(`${path} -> ${resp.status}`);
  return resp.json();
}

document.getElementById("ingest-btn").addEventListener("click", async () => {
  const query = document.getElementById("ingest-query").value.trim();
  if (!query) return;
  const status = document.getElementById("ingest-status");
  status.textContent = "Ingesting… (downloads, parses, and embeds PDFs — may take a minute)";
  try {
    const result = await post("/api/ingest", { query, max_results: 3 });
    status.textContent = `Ingested: ${result.ingested.join(", ") || "none"}` +
      (result.skipped.length ? ` | Skipped: ${result.skipped.join(", ")}` : "");
  } catch (err) {
    status.textContent = `Ingest failed: ${err.message}`;
  }
});

// ---------- wiring ----------

document.getElementById("chat-btn").addEventListener("click", sendMessage);
document.getElementById("chat-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendMessage();
});
document.getElementById("new-conv-btn").addEventListener("click", startNewConversation);

loadProviders();
loadThreads();
```

- [ ] **Step 3: Verify existing API tests still pass (static mount serves new files)**

Run: `uv run pytest tests/test_api.py::test_index_served -v`
Expected: PASS (`Paper Research Assistant` still in the page)

Run: `uv run pytest`
Expected: all pass

- [ ] **Step 4: Manual smoke test**

Requires Docker (Qdrant) and Ollama running:

```bash
docker compose up -d
uv run uvicorn api.main:app --port 8000
```

Open http://localhost:8000 and verify:
1. Provider dropdown lists 3 entries; ones without keys/Ollama are disabled with a reason.
2. Send a chat with `local` selected → activity lines appear, answer streams in, citations chips link to arxiv.org.
3. Thread appears in sidebar; click it after "New conversation" → transcript restores.
4. Delete button removes the thread.
5. Markdown (lists/bold) renders in replies.

(If the browse tool is available, use it for this smoke test and screenshot the result.)

- [ ] **Step 5: Commit**

```bash
git add api/static/index.html api/static/app.js
git commit -m "feat: frontend with provider toggle, streaming, citations, and thread sidebar"
```

---

### Task 8 (optional, keys/Ollama required): live verification

Not a code task — run the `local` marker suite against a running Ollama to confirm nothing regressed end-to-end:

```bash
uv run pytest -m local -v
```

Expected: pass (known caveat from phase 3: qwen2.5:3b agent loop can be flaky; rerun or use `ollama pull qwen2.5:7b` + `LOCAL_MODEL=qwen2.5:7b`).

---

## Self-Review Notes

- **Spec coverage:** §1 provider → Task 2; §2 status → Task 1; §3 streaming → Tasks 5–6; §4 citations → Task 3; §5 threads → Task 4; §6 frontend → Task 7; error handling spread across Tasks 2 (400/422), 6 (error event); testing sections per task. Spec amendments (requests instead of httpx, callback instead of generator) are explicit steps in Tasks 1 and 5.
- **Ordering constraint:** Task 3 changes `run_agent`'s return type — Tasks 4/6 test code assumes `AgentResult` exists, so tasks must run in order.
- **Type consistency:** `AgentResult(text, citations)` defined in Task 3, consumed in Tasks 4/6/7 test fakes; event dict shapes defined in Task 6 interfaces and consumed verbatim by Task 7's `parseSSE` handlers; `ProviderStatus.is_default` produced in Task 1, consumed by Task 7 `loadProviders()`.
