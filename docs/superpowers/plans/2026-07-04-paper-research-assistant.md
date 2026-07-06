# Paper Research Assistant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a learning-project research assistant that ingests arXiv papers into a RAG pipeline, answers questions with `[paper_id]` citations, autonomously fetches missing papers via a LangGraph agent + MCP tools, and evaluates itself with an LLM judge.

**Architecture:** Six flat top-level Python packages (`llm`, `rag`, `agents`, `eval`, `api`, `tests`) with a one-way import graph: `api → agents → rag/llm`; `eval → rag/agents/llm`. A provider-neutral `llm.base.generate()` wraps Anthropic + OpenAI. RAG stores OpenAI embeddings in Qdrant (Docker). The agent is a LangGraph `StateGraph` whose tools come from a local `rag_query` function plus two MCP servers (custom arXiv server + external `mcp-server-fetch`).

**Tech Stack:** uv, Python ≥3.11, `anthropic`, `openai`, `qdrant-client`, `langgraph`, `mcp`, `fastapi`+`uvicorn`, `pydantic-settings`, `pypdf`, `arxiv`, `tiktoken`, `pytest`+`pytest-asyncio`.

## Global Constraints

- Package manager: `uv`. Project is not packaged (`[tool.uv] package = false`); everything runs from repo root.
- Packages exactly: `llm/`, `rag/`, `agents/`, `eval/`, `api/`, `tests/` + top-level `config.py`.
- Import direction only: `api → agents → rag/llm`; `eval → rag/agents/llm`. No package imports a consumer.
- LLM entrypoint signature: `generate(messages, *, system=None, tools=None, structured_schema=None, provider=None, max_tokens=None)` in `llm/base.py`.
- Canonical message format is Anthropic-shaped everywhere; the OpenAI client adapts.
- Default Anthropic model: `claude-opus-4-8` (no `temperature`/`top_p` — removed on this model). Default OpenAI model: `gpt-5` (configurable).
- Embeddings: OpenAI `text-embedding-3-small`, dim 1536.
- Vector store: Qdrant at `http://localhost:6333`, collection `papers`, via `docker-compose.yml`. Fail fast with a clear message when down.
- Chunking: recursive splitter, ~500 tokens, 50 token overlap, tokens counted with tiktoken `cl100k_base`.
- Citations: inline `[paper_id]` square-bracket arXiv ids.
- Error handling: LLM retries via SDK `max_retries` (retries 429/5xx with backoff); PDF parse failure → skip + log + continue batch; MCP tool failure → returned to agent as tool error result; Qdrant down → `RuntimeError` at startup.
- Tests: unit tests mock all network/LLM; real-API tests behind `@pytest.mark.integration`; default `pytest` run excludes integration (`addopts = "-m 'not integration'"`).
- Eval runs standalone: `uv run python -m eval.run` → `report.json` + printed summary.
- Out of scope: provider fallback/routing, auth, deployment, frontend build tooling.

---

### Task 1: Project scaffold + config

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.env.example`, `docker-compose.yml`, `conftest.py` (empty), `config.py`
- Create: `llm/__init__.py`, `rag/__init__.py`, `agents/__init__.py`, `eval/__init__.py`, `api/__init__.py`, `api/static/.gitkeep` (empty files)
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `from config import settings, Settings` — `Settings` fields listed in the code below; a singleton `settings` instantiated at import. All later tasks read config from here.

- [ ] **Step 1: Write project metadata files**

`pyproject.toml`:

```toml
[project]
name = "paper-research-assistant"
version = "0.1.0"
description = "Learning project: LLM APIs, RAG, evaluation, agents + MCP"
requires-python = ">=3.11"
dependencies = [
    "anthropic>=0.60",
    "openai>=1.60",
    "qdrant-client>=1.12",
    "langgraph>=0.2",
    "mcp>=1.2",
    "fastapi>=0.115",
    "uvicorn>=0.30",
    "pydantic-settings>=2.5",
    "pypdf>=5.0",
    "arxiv>=2.1",
    "tiktoken>=0.8",
]

[dependency-groups]
dev = [
    "pytest>=8.3",
    "pytest-asyncio>=0.25",
    "httpx>=0.27",
]

[tool.uv]
package = false

[tool.pytest.ini_options]
markers = [
    "integration: needs real API keys and a running Qdrant; run with `pytest -m integration`",
]
addopts = "-m 'not integration'"
asyncio_mode = "auto"
```

`.gitignore`:

```
.venv/
__pycache__/
*.pyc
.env
.pytest_cache/
data/
report.json
```

`.env.example`:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
LLM_PROVIDER=anthropic
```

`docker-compose.yml`:

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  qdrant_data:
```

`conftest.py`: create as an **empty file** at repo root. (Its presence makes pytest put the repo root on `sys.path` so `import llm` etc. work without installing the project.)

Create empty `__init__.py` in `llm/`, `rag/`, `agents/`, `eval/`, `api/`, and an empty `api/static/.gitkeep`. `tests/` gets **no** `__init__.py` (pytest rootdir imports; keep test file basenames unique).

- [ ] **Step 2: Write the failing test**

`tests/test_config.py`:

```python
def test_defaults():
    from config import Settings

    s = Settings(_env_file=None)
    assert s.llm_provider == "anthropic"
    assert s.anthropic_model == "claude-opus-4-8"
    assert s.embedding_model == "text-embedding-3-small"
    assert s.embedding_dim == 1536
    assert s.qdrant_url == "http://localhost:6333"
    assert s.qdrant_collection == "papers"
    assert s.chunk_max_tokens == 500
    assert s.chunk_overlap_tokens == 50
    assert s.retrieval_top_k == 5


def test_env_override(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("RETRIEVAL_TOP_K", "9")
    from config import Settings

    s = Settings(_env_file=None)
    assert s.llm_provider == "openai"
    assert s.retrieval_top_k == 9
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv sync && uv run pytest tests/test_config.py -v`
Expected: FAIL / collection error — `ModuleNotFoundError: No module named 'config'`

- [ ] **Step 4: Write config.py**

```python
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration. Values come from env vars / .env (12-factor style)."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # API keys (empty default so unit tests never need real keys)
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # LLM
    llm_provider: Literal["anthropic", "openai"] = "anthropic"
    anthropic_model: str = "claude-opus-4-8"
    openai_model: str = "gpt-5"
    llm_max_tokens: int = 4096
    llm_max_retries: int = 4  # SDK retries 429/5xx with exponential backoff

    # Embeddings
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # Vector store
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "papers"

    # RAG
    chunk_max_tokens: int = 500
    chunk_overlap_tokens: int = 50
    retrieval_top_k: int = 5
    pdf_dir: str = "data/pdfs"

    # Agent
    agent_max_steps: int = 8


settings = Settings()
```

Note: pydantic-settings maps env var `ANTHROPIC_API_KEY` → field `anthropic_api_key` automatically (case-insensitive).

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: 2 passed

- [ ] **Step 6: Verify Qdrant compose file**

Run: `docker compose config --quiet && echo OK`
Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock .gitignore .env.example docker-compose.yml conftest.py config.py llm rag agents eval api tests
git commit -m "feat: project scaffold with uv, config, and qdrant compose"
```

---

### Task 2: LLM abstraction — text generation + Anthropic client

**Files:**
- Create: `llm/base.py`, `llm/anthropic_client.py`
- Test: `tests/test_llm_base.py`

**Interfaces:**
- Consumes: `config.settings`.
- Produces:
  - `llm.base.ToolCall(id: str, name: str, input: dict)` (pydantic model)
  - `llm.base.LLMResponse(text: str, tool_calls: list[ToolCall], parsed: Any | None, stop_reason: str | None, usage: dict)` (pydantic model)
  - `llm.base.generate(messages: list[dict], *, system: str | list[dict] | None = None, tools: list[dict] | None = None, structured_schema: type[BaseModel] | None = None, provider: str | None = None, max_tokens: int | None = None) -> LLMResponse`
  - `llm.anthropic_client.generate_anthropic(...)` and `_get_client()` (tests monkeypatch `_get_client`).
- Message format (canonical, Anthropic-shaped — used by every later task):
  - `{"role": "user", "content": "plain text"}`
  - `{"role": "assistant", "content": [{"type": "text", "text": ...}, {"type": "tool_use", "id": ..., "name": ..., "input": {...}}]}`
  - `{"role": "user", "content": [{"type": "tool_result", "tool_use_id": ..., "content": "...", "is_error": false}]}`
- Tool format (canonical): `{"name": str, "description": str, "input_schema": {JSON Schema}}`.

- [ ] **Step 1: Write the failing tests**

`tests/test_llm_base.py`:

```python
from types import SimpleNamespace

import pytest


def _fake_response(blocks, stop="end_turn"):
    return SimpleNamespace(
        content=blocks,
        stop_reason=stop,
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )


def test_generate_text(monkeypatch):
    import llm.anthropic_client as ac
    from llm.base import generate

    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _fake_response([SimpleNamespace(type="text", text="hello")])

    monkeypatch.setattr(ac, "_get_client", lambda: SimpleNamespace(messages=FakeMessages()))

    resp = generate([{"role": "user", "content": "hi"}], provider="anthropic")

    assert resp.text == "hello"
    assert resp.tool_calls == []
    assert resp.stop_reason == "end_turn"
    assert resp.usage["input_tokens"] == 10
    assert captured["model"] == "claude-opus-4-8"
    assert captured["max_tokens"] > 0
    assert "system" not in captured  # omitted when None


def test_system_passed_through(monkeypatch):
    import llm.anthropic_client as ac
    from llm.base import generate

    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _fake_response([SimpleNamespace(type="text", text="ok")])

    monkeypatch.setattr(ac, "_get_client", lambda: SimpleNamespace(messages=FakeMessages()))

    generate([{"role": "user", "content": "hi"}], system="be terse", provider="anthropic")
    assert captured["system"] == "be terse"


def test_unknown_provider_raises():
    from llm.base import generate

    with pytest.raises(ValueError, match="Unknown provider"):
        generate([{"role": "user", "content": "hi"}], provider="grok")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_llm_base.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm.base'`

- [ ] **Step 3: Write llm/base.py**

```python
from typing import Any

from pydantic import BaseModel

from config import settings


class ToolCall(BaseModel):
    id: str
    name: str
    input: dict


class LLMResponse(BaseModel):
    text: str = ""
    tool_calls: list[ToolCall] = []
    parsed: Any | None = None  # populated when structured_schema was given
    stop_reason: str | None = None
    usage: dict = {}


def generate(
    messages: list[dict],
    *,
    system: str | list[dict] | None = None,
    tools: list[dict] | None = None,
    structured_schema: type[BaseModel] | None = None,
    provider: str | None = None,
    max_tokens: int | None = None,
) -> LLMResponse:
    """Provider-neutral chat entrypoint.

    `messages` and `tools` use the Anthropic shape everywhere in this codebase;
    the OpenAI client adapts them. `structured_schema` is a pydantic model class;
    the validated instance comes back on `LLMResponse.parsed`.
    """
    provider = provider or settings.llm_provider
    max_tokens = max_tokens or settings.llm_max_tokens
    if provider == "anthropic":
        from llm.anthropic_client import generate_anthropic

        return generate_anthropic(
            messages, system=system, tools=tools,
            structured_schema=structured_schema, max_tokens=max_tokens,
        )
    if provider == "openai":
        from llm.openai_client import generate_openai

        return generate_openai(
            messages, system=system, tools=tools,
            structured_schema=structured_schema, max_tokens=max_tokens,
        )
    raise ValueError(f"Unknown provider: {provider}")
```

(`llm/openai_client.py` arrives in Task 3; the import is lazy so nothing breaks now.)

- [ ] **Step 4: Write llm/anthropic_client.py**

```python
import anthropic
from pydantic import BaseModel

from config import settings
from llm.base import LLMResponse, ToolCall

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        # max_retries: SDK retries 429/5xx with exponential backoff.
        _client = anthropic.Anthropic(max_retries=settings.llm_max_retries)
    return _client


def generate_anthropic(
    messages: list[dict],
    *,
    system: str | list[dict] | None = None,
    tools: list[dict] | None = None,
    structured_schema: type[BaseModel] | None = None,
    max_tokens: int = 4096,
) -> LLMResponse:
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
    if structured_schema is not None:
        response = client.messages.parse(output_format=structured_schema, **kwargs)
    else:
        response = client.messages.create(**kwargs)
    return _to_llm_response(response, structured_schema)


def _to_llm_response(response, structured_schema=None) -> LLMResponse:
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(ToolCall(id=block.id, name=block.name, input=dict(block.input)))
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cache_read_input_tokens": getattr(response.usage, "cache_read_input_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
    }
    parsed = getattr(response, "parsed_output", None) if structured_schema else None
    return LLMResponse(
        text="\n".join(text_parts),
        tool_calls=tool_calls,
        parsed=parsed,
        stop_reason=response.stop_reason,
        usage=usage,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_llm_base.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add llm/base.py llm/anthropic_client.py tests/test_llm_base.py
git commit -m "feat: provider-neutral generate() with anthropic text generation"
```

---

### Task 3: LLM abstraction — tool calls, structured output, OpenAI adapter

**Files:**
- Create: `llm/openai_client.py`
- Modify: nothing (Anthropic tool/structured paths already exist from Task 2)
- Test: `tests/test_llm_tools.py`, `tests/test_openai_adapter.py`

**Interfaces:**
- Consumes: `llm.base.LLMResponse`, `ToolCall`, canonical message/tool formats (Task 2), `config.settings`.
- Produces:
  - `llm.openai_client.generate_openai(messages, *, system=None, tools=None, structured_schema=None, max_tokens=4096) -> LLMResponse`
  - `llm.openai_client.convert_messages(messages: list[dict], system=None) -> list[dict]` (Anthropic shape → OpenAI chat shape)
  - `llm.openai_client.convert_tools(tools: list[dict]) -> list[dict]` (Anthropic tool spec → OpenAI function spec)
  - `llm.openai_client._get_client()` (tests monkeypatch it)
- After this task `generate(..., tools=...)` returns `tool_calls`, and `generate(..., structured_schema=SomeModel)` returns `parsed` on **both** providers.

- [ ] **Step 1: Write failing tests for the Anthropic tool + structured paths**

`tests/test_llm_tools.py`:

```python
from types import SimpleNamespace

from pydantic import BaseModel


def _fake_response(blocks, stop="end_turn", parsed=None):
    resp = SimpleNamespace(
        content=blocks,
        stop_reason=stop,
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )
    if parsed is not None:
        resp.parsed_output = parsed
    return resp


def test_anthropic_tool_call(monkeypatch):
    import llm.anthropic_client as ac
    from llm.base import generate

    captured = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _fake_response(
                [
                    SimpleNamespace(type="text", text="Looking that up."),
                    SimpleNamespace(type="tool_use", id="tu_1", name="rag_query",
                                    input={"question": "what is attention?"}),
                ],
                stop="tool_use",
            )

    monkeypatch.setattr(ac, "_get_client", lambda: SimpleNamespace(messages=FakeMessages()))

    tools = [{"name": "rag_query", "description": "d",
              "input_schema": {"type": "object", "properties": {"question": {"type": "string"}},
                               "required": ["question"]}}]
    resp = generate([{"role": "user", "content": "hi"}], tools=tools, provider="anthropic")

    assert captured["tools"] == tools
    assert resp.stop_reason == "tool_use"
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "rag_query"
    assert resp.tool_calls[0].input == {"question": "what is attention?"}


def test_anthropic_structured_output(monkeypatch):
    import llm.anthropic_client as ac
    from llm.base import generate

    class Scores(BaseModel):
        faithfulness: int

    captured = {}

    class FakeMessages:
        def parse(self, **kwargs):
            captured.update(kwargs)
            return _fake_response(
                [SimpleNamespace(type="text", text='{"faithfulness": 5}')],
                parsed=Scores(faithfulness=5),
            )

    monkeypatch.setattr(ac, "_get_client", lambda: SimpleNamespace(messages=FakeMessages()))

    resp = generate([{"role": "user", "content": "judge"}],
                    structured_schema=Scores, provider="anthropic")

    assert captured["output_format"] is Scores
    assert resp.parsed == Scores(faithfulness=5)
```

- [ ] **Step 2: Write failing tests for the OpenAI adapter**

`tests/test_openai_adapter.py`:

```python
import json
from types import SimpleNamespace

from pydantic import BaseModel


def _fake_completion(message, finish="stop"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish)],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


def test_convert_messages_full_tool_round_trip():
    from llm.openai_client import convert_messages

    anthropic_msgs = [
        {"role": "user", "content": "find the paper"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Searching."},
            {"type": "tool_use", "id": "tu_1", "name": "arxiv_search",
             "input": {"query": "attention"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu_1",
             "content": "found 1706.03762", "is_error": False},
        ]},
    ]
    out = convert_messages(anthropic_msgs, system="be terse")

    assert out[0] == {"role": "system", "content": "be terse"}
    assert out[1] == {"role": "user", "content": "find the paper"}
    assert out[2]["role"] == "assistant"
    assert out[2]["content"] == "Searching."
    assert out[2]["tool_calls"][0]["id"] == "tu_1"
    assert out[2]["tool_calls"][0]["function"]["name"] == "arxiv_search"
    assert json.loads(out[2]["tool_calls"][0]["function"]["arguments"]) == {"query": "attention"}
    assert out[3] == {"role": "tool", "tool_call_id": "tu_1", "content": "found 1706.03762"}


def test_convert_messages_joins_system_blocks_and_drops_cache_control():
    from llm.openai_client import convert_messages

    system = [
        {"type": "text", "text": "instructions"},
        {"type": "text", "text": "context", "cache_control": {"type": "ephemeral"}},
    ]
    out = convert_messages([{"role": "user", "content": "q"}], system=system)
    assert out[0] == {"role": "system", "content": "instructions\n\ncontext"}


def test_convert_tools():
    from llm.openai_client import convert_tools

    schema = {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}
    out = convert_tools([{"name": "rag_query", "description": "d", "input_schema": schema}])
    assert out == [{"type": "function",
                    "function": {"name": "rag_query", "description": "d", "parameters": schema}}]


def test_generate_openai_tool_call(monkeypatch):
    import llm.openai_client as oc
    from llm.base import generate

    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="rag_query", arguments='{"question": "q"}'),
    )
    message = SimpleNamespace(content=None, tool_calls=[tool_call])

    class FakeCompletions:
        def create(self, **kwargs):
            return _fake_completion(message, finish="tool_calls")

    fake = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr(oc, "_get_client", lambda: fake)

    tools = [{"name": "rag_query", "description": "d",
              "input_schema": {"type": "object", "properties": {}}}]
    resp = generate([{"role": "user", "content": "hi"}], tools=tools, provider="openai")

    assert resp.tool_calls[0].id == "call_1"
    assert resp.tool_calls[0].input == {"question": "q"}


def test_generate_openai_structured(monkeypatch):
    import llm.openai_client as oc
    from llm.base import generate

    class Scores(BaseModel):
        relevance: int

    message = SimpleNamespace(content='{"relevance": 4}', tool_calls=None,
                              parsed=Scores(relevance=4))

    class FakeCompletions:
        def parse(self, **kwargs):
            return _fake_completion(message)

    fake = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr(oc, "_get_client", lambda: fake)

    resp = generate([{"role": "user", "content": "judge"}],
                    structured_schema=Scores, provider="openai")
    assert resp.parsed == Scores(relevance=4)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_llm_tools.py tests/test_openai_adapter.py -v`
Expected: `test_llm_tools.py` PASSES already (Task 2 implemented those paths — that's fine, they lock the behavior); `test_openai_adapter.py` FAILS with `ModuleNotFoundError: No module named 'llm.openai_client'`

- [ ] **Step 4: Write llm/openai_client.py**

```python
import json

from openai import OpenAI
from pydantic import BaseModel

from config import settings
from llm.base import LLMResponse, ToolCall

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(max_retries=settings.llm_max_retries)
    return _client


def _system_to_text(system: str | list[dict]) -> str:
    if isinstance(system, str):
        return system
    # Anthropic system blocks; cache_control has no OpenAI equivalent, drop it.
    return "\n\n".join(b["text"] for b in system if b.get("type") == "text")


def convert_messages(messages: list[dict], system: str | list[dict] | None = None) -> list[dict]:
    """Anthropic-shaped messages -> OpenAI chat.completions messages."""
    out: list[dict] = []
    if system is not None:
        out.append({"role": "system", "content": _system_to_text(system)})
    for msg in messages:
        content = msg["content"]
        if isinstance(content, str):
            out.append({"role": msg["role"], "content": content})
            continue
        if msg["role"] == "assistant":
            text = "".join(b["text"] for b in content if b["type"] == "text")
            tool_calls = [
                {"id": b["id"], "type": "function",
                 "function": {"name": b["name"], "arguments": json.dumps(b["input"])}}
                for b in content if b["type"] == "tool_use"
            ]
            entry: dict = {"role": "assistant", "content": text or None}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            out.append(entry)
        else:  # user message with content blocks
            for b in content:
                if b["type"] == "tool_result":
                    out.append({"role": "tool", "tool_call_id": b["tool_use_id"],
                                "content": b["content"]})
                elif b["type"] == "text":
                    out.append({"role": "user", "content": b["text"]})
    return out


def convert_tools(tools: list[dict]) -> list[dict]:
    """Anthropic tool spec -> OpenAI function-calling spec."""
    return [
        {"type": "function",
         "function": {"name": t["name"], "description": t.get("description", ""),
                      "parameters": t["input_schema"]}}
        for t in tools
    ]


def _usage(completion) -> dict:
    u = completion.usage
    return {"input_tokens": u.prompt_tokens, "output_tokens": u.completion_tokens}


def generate_openai(
    messages: list[dict],
    *,
    system: str | list[dict] | None = None,
    tools: list[dict] | None = None,
    structured_schema: type[BaseModel] | None = None,
    max_tokens: int = 4096,
) -> LLMResponse:
    client = _get_client()
    kwargs: dict = {
        "model": settings.openai_model,
        "messages": convert_messages(messages, system),
        "max_completion_tokens": max_tokens,
    }
    if tools:
        kwargs["tools"] = convert_tools(tools)
    if structured_schema is not None:
        completion = client.chat.completions.parse(response_format=structured_schema, **kwargs)
        choice = completion.choices[0]
        return LLMResponse(text=choice.message.content or "", parsed=choice.message.parsed,
                           stop_reason=choice.finish_reason, usage=_usage(completion))
    completion = client.chat.completions.create(**kwargs)
    choice = completion.choices[0]
    tool_calls = [
        ToolCall(id=tc.id, name=tc.function.name, input=json.loads(tc.function.arguments))
        for tc in (choice.message.tool_calls or [])
    ]
    return LLMResponse(text=choice.message.content or "", tool_calls=tool_calls,
                       stop_reason=choice.finish_reason, usage=_usage(completion))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_llm_tools.py tests/test_openai_adapter.py -v`
Expected: 7 passed

- [ ] **Step 6: Commit**

```bash
git add llm/openai_client.py tests/test_llm_tools.py tests/test_openai_adapter.py
git commit -m "feat: tool calls, structured output, and openai adapter for generate()"
```

---

### Task 4: Prompt templates + few-shot citations + caching breakpoint

**Files:**
- Create: `llm/prompts.py`
- Test: `tests/test_prompts.py`

**Interfaces:**
- Consumes: nothing from other packages (pure functions; contexts are plain dicts so `llm` never imports `rag`).
- Produces:
  - `llm.prompts.CITATION_SYSTEM_PROMPT: str`
  - `llm.prompts.AGENT_SYSTEM_PROMPT: str`
  - `llm.prompts.FEW_SHOT_MESSAGES: list[dict]`
  - `llm.prompts.format_context(contexts: list[dict]) -> str` — each context dict has keys `paper_id`, `title`, `text`
  - `llm.prompts.build_rag_prompt(question: str, contexts: list[dict]) -> tuple[list[dict], list[dict]]` returning `(system_blocks, messages)`; last system block carries `cache_control`

- [ ] **Step 1: Write the failing tests**

`tests/test_prompts.py`:

```python
def test_format_context_labels_papers():
    from llm.prompts import format_context

    out = format_context([
        {"paper_id": "1706.03762", "title": "Attention Is All You Need", "text": "Self-attention..."},
        {"paper_id": "1810.04805", "title": "BERT", "text": "Masked LM..."},
    ])
    assert "[paper 1706.03762 — Attention Is All You Need]" in out
    assert "Self-attention..." in out
    assert "[paper 1810.04805 — BERT]" in out


def test_build_rag_prompt_structure():
    from llm.prompts import CITATION_SYSTEM_PROMPT, FEW_SHOT_MESSAGES, build_rag_prompt

    contexts = [{"paper_id": "1706.03762", "title": "Attention", "text": "chunk text"}]
    system, messages = build_rag_prompt("What is attention?", contexts)

    # system: [instructions, cached context block]
    assert system[0]["text"] == CITATION_SYSTEM_PROMPT
    assert "chunk text" in system[1]["text"]
    assert system[1]["cache_control"] == {"type": "ephemeral"}

    # messages: few-shot pairs first, real question last
    assert messages[: len(FEW_SHOT_MESSAGES)] == FEW_SHOT_MESSAGES
    assert messages[-1]["role"] == "user"
    assert "What is attention?" in messages[-1]["content"]


def test_few_shot_demonstrates_citation_format():
    from llm.prompts import FEW_SHOT_MESSAGES

    assistant_turns = [m for m in FEW_SHOT_MESSAGES if m["role"] == "assistant"]
    assert assistant_turns, "few-shot must include an assistant example"
    assert any("[" in m["content"] and "]" in m["content"] for m in assistant_turns)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm.prompts'`

- [ ] **Step 3: Write llm/prompts.py**

```python
"""Prompt templates.

Demonstrates: system prompt design, few-shot examples (citation format), and an
Anthropic prompt-caching breakpoint (cache_control on the long paper context).
Contexts are plain dicts ({paper_id, title, text}) so llm/ never imports rag/.
"""

CITATION_SYSTEM_PROMPT = """You are a research assistant that answers questions using ONLY the provided paper excerpts.

Rules:
- Base every claim on the excerpts. If they are insufficient, say "I don't have enough information in the ingested papers" and suggest fetching more.
- Cite the source of every claim inline with the arXiv id in square brackets, e.g. [1706.03762].
- When several excerpts support a claim, stack citations: [1706.03762][1810.04805].
- Be concise and technical. Do not invent paper ids."""

AGENT_SYSTEM_PROMPT = """You are a research paper assistant with tools.

For each user message decide:
- If the question concerns papers likely already ingested, call rag_query first.
- If rag_query reports it doesn't have enough information, call arxiv_search to find the paper, then arxiv_fetch_paper to ingest it, then call rag_query again.
- If the user gives a URL, call fetch to read it.

Cite papers inline as [paper_id] whenever an answer comes from ingested papers.
If a tool call fails, decide whether to retry once with adjusted input or explain the failure to the user. Never fabricate tool output."""

# Few-shot pair demonstrating the citation format (uses a fake paper id on purpose).
FEW_SHOT_MESSAGES = [
    {
        "role": "user",
        "content": (
            "Paper excerpts:\n\n[paper 1234.56789 — Example Networks]\n"
            "Example Networks use gated residual connections to stabilize training.\n\n"
            "Question: How do Example Networks stabilize training?"
        ),
    },
    {
        "role": "assistant",
        "content": "Example Networks stabilize training with gated residual connections [1234.56789].",
    },
]


def format_context(contexts: list[dict]) -> str:
    parts = [f"[paper {c['paper_id']} — {c['title']}]\n{c['text']}" for c in contexts]
    return "\n\n---\n\n".join(parts)


def build_rag_prompt(question: str, contexts: list[dict]) -> tuple[list[dict], list[dict]]:
    """Returns (system_blocks, messages) for generate().

    The context block carries cache_control so Anthropic caches the paper
    excerpts: repeat/follow-up questions that retrieve the same top-k chunks
    get a cache hit (visible in usage["cache_read_input_tokens"]). The OpenAI
    adapter simply drops cache_control.
    """
    system_blocks = [
        {"type": "text", "text": CITATION_SYSTEM_PROMPT},
        {
            "type": "text",
            "text": "Paper excerpts:\n\n" + format_context(contexts),
            "cache_control": {"type": "ephemeral"},
        },
    ]
    messages = FEW_SHOT_MESSAGES + [{"role": "user", "content": f"Question: {question}"}]
    return system_blocks, messages
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add llm/prompts.py tests/test_prompts.py
git commit -m "feat: prompt templates with few-shot citations and cache_control"
```

---

### Task 5: Chunking (recursive splitter with overlap)

**Files:**
- Create: `rag/chunk.py`
- Test: `tests/test_chunk.py`

**Interfaces:**
- Consumes: `config.settings` (defaults only).
- Produces:
  - `rag.chunk.count_tokens(text: str) -> int` (tiktoken `cl100k_base`)
  - `rag.chunk.chunk_text(text: str, max_tokens: int | None = None, overlap_tokens: int | None = None) -> list[str]`

- [ ] **Step 1: Write the failing tests**

`tests/test_chunk.py`:

```python
from rag.chunk import chunk_text, count_tokens


def test_empty_text_gives_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_short_text_is_single_chunk():
    assert chunk_text("Hello world.", max_tokens=50) == ["Hello world."]


def test_chunks_respect_token_limit():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 40 for i in range(20))
    chunks = chunk_text(text, max_tokens=100, overlap_tokens=20)
    assert len(chunks) > 1
    assert all(count_tokens(c) <= 100 for c in chunks)


def test_consecutive_chunks_overlap():
    # Pieces of ~12 tokens; overlap of 20 tokens carries the previous tail piece.
    text = "\n\n".join(
        f"para {i} alpha beta gamma delta epsilon zeta eta theta" for i in range(12)
    )
    chunks = chunk_text(text, max_tokens=40, overlap_tokens=20)
    assert len(chunks) > 1
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.splitlines()[0] == prev.splitlines()[-1]


def test_oversized_single_piece_is_hard_split():
    text = "x" * 5000  # no separators at all
    chunks = chunk_text(text, max_tokens=100, overlap_tokens=0)
    assert len(chunks) > 1
    assert all(count_tokens(c) <= 100 for c in chunks)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_chunk.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.chunk'`

- [ ] **Step 3: Write rag/chunk.py**

```python
"""Recursive text splitter with token-based sizing and overlap.

Splits on progressively finer separators until every piece fits, then greedily
packs pieces into chunks of <= max_tokens, carrying the tail pieces of each
chunk into the next one as overlap.
"""

import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")  # tokenizer used by text-embedding-3-small

_SEPARATORS = ["\n\n", "\n", ". ", " "]


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _split(text: str, max_tokens: int, separators: list[str]) -> list[str]:
    if count_tokens(text) <= max_tokens:
        return [text]
    if not separators:
        # No separators left: hard-cut on token boundaries.
        toks = _enc.encode(text)
        return [_enc.decode(toks[i:i + max_tokens]) for i in range(0, len(toks), max_tokens)]
    sep, rest = separators[0], separators[1:]
    parts = [p for p in text.split(sep) if p.strip()]
    if len(parts) <= 1:
        return _split(text, max_tokens, rest)
    pieces: list[str] = []
    for part in parts:
        pieces.extend(_split(part, max_tokens, rest))
    return pieces


def chunk_text(text: str, max_tokens: int | None = None,
               overlap_tokens: int | None = None) -> list[str]:
    from config import settings

    max_tokens = max_tokens or settings.chunk_max_tokens
    if overlap_tokens is None:
        overlap_tokens = settings.chunk_overlap_tokens
    text = text.strip()
    if not text:
        return []

    pieces = _split(text, max_tokens, _SEPARATORS)

    chunks: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for piece in pieces:
        piece_tokens = count_tokens(piece) + 1  # +1 for the join newline
        if current and current_tokens + piece_tokens > max_tokens:
            chunks.append(current)
            # Carry trailing pieces of the finished chunk as overlap.
            tail: list[str] = []
            tail_tokens = 0
            for prev in reversed(current):
                t = count_tokens(prev) + 1
                if tail_tokens + t > overlap_tokens:
                    break
                tail.insert(0, prev)
                tail_tokens += t
            current, current_tokens = tail, tail_tokens
        current.append(piece)
        current_tokens += piece_tokens
    if current:
        chunks.append(current)
    return ["\n".join(c) for c in chunks]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_chunk.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add rag/chunk.py tests/test_chunk.py
git commit -m "feat: recursive token-aware chunker with overlap"
```

---

### Task 6: arXiv client (search, lookup, PDF download)

**Files:**
- Create: `rag/arxiv_client.py`
- Test: `tests/test_arxiv_client.py`, `tests/test_integration_arxiv.py`

**Interfaces:**
- Consumes: `config.settings.pdf_dir`, `arxiv` library.
- Produces:
  - `rag.arxiv_client.PaperMeta(paper_id: str, title: str, summary: str)` (pydantic model)
  - `rag.arxiv_client.search_papers(query: str, max_results: int = 5) -> list[PaperMeta]`
  - `rag.arxiv_client.get_paper(paper_id: str) -> PaperMeta | None`
  - `rag.arxiv_client.download_pdf(paper_id: str) -> str` (path; raises `ValueError` if id unknown)
  - `rag.arxiv_client._client()` (tests monkeypatch it)

- [ ] **Step 1: Write the failing unit tests**

`tests/test_arxiv_client.py`:

```python
from types import SimpleNamespace

import pytest


def _fake_result(short_id="2405.10098v2", title="T", summary="S"):
    r = SimpleNamespace(title=title, summary=summary)
    r.get_short_id = lambda: short_id
    r.download_pdf = lambda dirpath, filename: f"{dirpath}/{filename}"
    return r


class _FakeArxivClient:
    def __init__(self, results):
        self._results = results
        self.searches = []

    def results(self, search):
        self.searches.append(search)
        return iter(self._results)


def test_search_papers_strips_version(monkeypatch):
    import rag.arxiv_client as axc

    fake = _FakeArxivClient([_fake_result("2405.10098v2", "Paper A", "About A")])
    monkeypatch.setattr(axc, "_client", lambda: fake)

    papers = axc.search_papers("attention", max_results=1)
    assert len(papers) == 1
    assert papers[0].paper_id == "2405.10098"
    assert papers[0].title == "Paper A"
    assert fake.searches[0].query == "attention"
    assert fake.searches[0].max_results == 1


def test_get_paper_found_and_missing(monkeypatch):
    import rag.arxiv_client as axc

    fake = _FakeArxivClient([_fake_result("1706.03762v7", "Attention", "S")])
    monkeypatch.setattr(axc, "_client", lambda: fake)
    meta = axc.get_paper("1706.03762")
    assert meta is not None and meta.paper_id == "1706.03762"
    assert fake.searches[0].id_list == ["1706.03762"]

    monkeypatch.setattr(axc, "_client", lambda: _FakeArxivClient([]))
    assert axc.get_paper("0000.00000") is None


def test_download_pdf(monkeypatch, tmp_path):
    import rag.arxiv_client as axc
    from config import settings

    monkeypatch.setattr(settings, "pdf_dir", str(tmp_path / "pdfs"))
    fake = _FakeArxivClient([_fake_result("1706.03762v7")])
    monkeypatch.setattr(axc, "_client", lambda: fake)

    path = axc.download_pdf("1706.03762")
    assert path.endswith("1706.03762.pdf")
    assert (tmp_path / "pdfs").is_dir()  # created eagerly


def test_download_pdf_unknown_id(monkeypatch):
    import rag.arxiv_client as axc

    monkeypatch.setattr(axc, "_client", lambda: _FakeArxivClient([]))
    with pytest.raises(ValueError, match="No arXiv paper"):
        axc.download_pdf("0000.00000")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_arxiv_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.arxiv_client'`

- [ ] **Step 3: Write rag/arxiv_client.py**

```python
import re
from pathlib import Path

import arxiv
from pydantic import BaseModel

from config import settings


class PaperMeta(BaseModel):
    paper_id: str
    title: str
    summary: str


def _client() -> arxiv.Client:
    return arxiv.Client()


def _short_id(result) -> str:
    # "2405.10098v2" -> "2405.10098" (version-free ids keep dedup simple)
    return re.sub(r"v\d+$", "", result.get_short_id())


def _to_meta(result) -> PaperMeta:
    return PaperMeta(paper_id=_short_id(result), title=result.title, summary=result.summary)


def search_papers(query: str, max_results: int = 5) -> list[PaperMeta]:
    search = arxiv.Search(query=query, max_results=max_results)
    return [_to_meta(r) for r in _client().results(search)]


def get_paper(paper_id: str) -> PaperMeta | None:
    search = arxiv.Search(id_list=[paper_id])
    results = list(_client().results(search))
    return _to_meta(results[0]) if results else None


def download_pdf(paper_id: str) -> str:
    """Downloads the PDF to settings.pdf_dir and returns the file path."""
    search = arxiv.Search(id_list=[paper_id])
    results = list(_client().results(search))
    if not results:
        raise ValueError(f"No arXiv paper found for id {paper_id}")
    Path(settings.pdf_dir).mkdir(parents=True, exist_ok=True)
    return results[0].download_pdf(dirpath=settings.pdf_dir, filename=f"{paper_id}.pdf")
```

- [ ] **Step 4: Run unit tests to verify they pass**

Run: `uv run pytest tests/test_arxiv_client.py -v`
Expected: 4 passed

- [ ] **Step 5: Write the integration test (real network, opt-in)**

`tests/test_integration_arxiv.py`:

```python
import pytest

pytestmark = pytest.mark.integration


def test_real_search_and_lookup():
    from rag.arxiv_client import get_paper, search_papers

    papers = search_papers("attention is all you need", max_results=3)
    assert papers and all(p.paper_id and p.title for p in papers)

    meta = get_paper("1706.03762")
    assert meta is not None
    assert "Attention" in meta.title
```

Run: `uv run pytest tests/test_integration_arxiv.py -v`
Expected: `2 deselected` (integration excluded by default). Optionally verify for real: `uv run pytest tests/test_integration_arxiv.py -m integration -v` → 1 passed (needs network).

- [ ] **Step 6: Commit**

```bash
git add rag/arxiv_client.py tests/test_arxiv_client.py tests/test_integration_arxiv.py
git commit -m "feat: arxiv search, lookup, and pdf download"
```

---

### Task 7: PDF parsing (skip-on-failure)

**Files:**
- Create: `rag/parse.py`
- Test: `tests/test_parse.py`

**Interfaces:**
- Consumes: `pypdf`.
- Produces: `rag.parse.extract_text(pdf_path: str) -> str | None` — `None` means "unparseable, skip this paper" (per spec: skip, log, continue batch).

- [ ] **Step 1: Write the failing tests**

`tests/test_parse.py`:

```python
from types import SimpleNamespace


def _fake_page(text):
    return SimpleNamespace(extract_text=lambda: text)


def test_extracts_and_joins_pages(monkeypatch):
    import rag.parse as parse

    monkeypatch.setattr(
        parse, "PdfReader",
        lambda path: SimpleNamespace(pages=[_fake_page("page one"), _fake_page("page two")]),
    )
    assert parse.extract_text("x.pdf") == "page one\npage two"


def test_handles_none_page_text(monkeypatch):
    import rag.parse as parse

    monkeypatch.setattr(
        parse, "PdfReader",
        lambda path: SimpleNamespace(pages=[_fake_page(None), _fake_page("real text")]),
    )
    assert parse.extract_text("x.pdf") == "real text"


def test_parse_failure_returns_none(monkeypatch, caplog):
    import rag.parse as parse

    def _boom(path):
        raise ValueError("corrupt pdf")

    monkeypatch.setattr(parse, "PdfReader", _boom)
    assert parse.extract_text("bad.pdf") is None
    assert "bad.pdf" in caplog.text


def test_empty_pdf_returns_none(monkeypatch):
    import rag.parse as parse

    monkeypatch.setattr(parse, "PdfReader",
                        lambda path: SimpleNamespace(pages=[_fake_page("")]))
    assert parse.extract_text("empty.pdf") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_parse.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.parse'`

- [ ] **Step 3: Write rag/parse.py**

```python
import logging

from pypdf import PdfReader

logger = logging.getLogger(__name__)


def extract_text(pdf_path: str) -> str | None:
    """Extract plain text from a PDF.

    Returns None when the PDF can't be parsed or yields no text — callers skip
    the paper and continue the batch (spec: skip, log, continue).
    """
    try:
        reader = PdfReader(pdf_path)
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception:
        logger.exception("Failed to parse %s", pdf_path)
        return None
    text = "\n".join(line for line in text.splitlines() if line.strip())
    if not text.strip():
        logger.warning("No text extracted from %s", pdf_path)
        return None
    return text
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_parse.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add rag/parse.py tests/test_parse.py
git commit -m "feat: pdf text extraction with skip-on-failure"
```

---

### Task 8: Embeddings (OpenAI text-embedding-3-small)

**Files:**
- Create: `rag/embed.py`
- Test: `tests/test_embed.py`

**Interfaces:**
- Consumes: `config.settings.embedding_model`, `openai` SDK.
- Produces:
  - `rag.embed.embed_texts(texts: list[str]) -> list[list[float]]` (batched, order-preserving)
  - `rag.embed.embed_query(text: str) -> list[float]`
  - `rag.embed._get_client()` (tests monkeypatch it)

- [ ] **Step 1: Write the failing tests**

`tests/test_embed.py`:

```python
from types import SimpleNamespace


class FakeEmbeddings:
    def __init__(self):
        self.calls = []

    def create(self, model, input):
        self.calls.append({"model": model, "input": input})
        return SimpleNamespace(data=[SimpleNamespace(embedding=[float(len(t))]) for t in input])


def _patch(monkeypatch):
    import rag.embed as embed

    fake = FakeEmbeddings()
    monkeypatch.setattr(embed, "_get_client",
                        lambda: SimpleNamespace(embeddings=FakeEmbeddings.__call__ and fake))
    return fake


def test_embed_texts_preserves_order(monkeypatch):
    import rag.embed as embed

    fake = FakeEmbeddings()
    monkeypatch.setattr(embed, "_get_client", lambda: SimpleNamespace(embeddings=fake))

    vectors = embed.embed_texts(["a", "bb", "ccc"])
    assert vectors == [[1.0], [2.0], [3.0]]
    assert fake.calls[0]["model"] == "text-embedding-3-small"


def test_embed_texts_batches(monkeypatch):
    import rag.embed as embed

    fake = FakeEmbeddings()
    monkeypatch.setattr(embed, "_get_client", lambda: SimpleNamespace(embeddings=fake))
    monkeypatch.setattr(embed, "BATCH_SIZE", 2)

    vectors = embed.embed_texts(["a", "b", "c", "d", "e"])
    assert len(vectors) == 5
    assert len(fake.calls) == 3  # 2 + 2 + 1


def test_embed_texts_empty(monkeypatch):
    import rag.embed as embed

    monkeypatch.setattr(embed, "_get_client",
                        lambda: (_ for _ in ()).throw(AssertionError("must not call API")))
    assert embed.embed_texts([]) == []


def test_embed_query(monkeypatch):
    import rag.embed as embed

    fake = FakeEmbeddings()
    monkeypatch.setattr(embed, "_get_client", lambda: SimpleNamespace(embeddings=fake))
    assert embed.embed_query("hi") == [2.0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_embed.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.embed'`

- [ ] **Step 3: Write rag/embed.py**

```python
from openai import OpenAI

from config import settings

_client: OpenAI | None = None

BATCH_SIZE = 100  # texts per embeddings API request


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(max_retries=settings.llm_max_retries)
    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts in order, batching requests."""
    if not texts:
        return []
    client = _get_client()
    vectors: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        resp = client.embeddings.create(model=settings.embedding_model, input=batch)
        vectors.extend(d.embedding for d in resp.data)
    return vectors


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_embed.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add rag/embed.py tests/test_embed.py
git commit -m "feat: batched openai embeddings"
```

---

### Task 9: Qdrant vector store (fail-fast, upsert, search)

**Files:**
- Create: `rag/store.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Consumes: `config.settings` (qdrant_url, qdrant_collection, embedding_dim, retrieval_top_k), `qdrant_client`.
- Produces:
  - `rag.store.ChunkRecord(paper_id: str, title: str, chunk_index: int, text: str, section: str = "", vector: list[float])`
  - `rag.store.ScoredChunk(paper_id: str, title: str, text: str, score: float)`
  - `rag.store.VectorStore(url=None, collection=None, client=None)` with methods:
    - `ping() -> None` (raises `RuntimeError` with docker hint when Qdrant unreachable)
    - `ensure_collection() -> None`
    - `upsert_chunks(records: list[ChunkRecord]) -> None`
    - `search(vector: list[float], top_k: int | None = None) -> list[ScoredChunk]`
    - `has_paper(paper_id: str) -> bool`
- Payload schema: `{paper_id, title, chunk_index, chunk_text, section}` (spec metadata; `section` stored but always `""` — pypdf gives no reliable section boundaries, field kept for schema parity).

- [ ] **Step 1: Write the failing tests**

`tests/test_store.py`:

```python
from types import SimpleNamespace

import pytest


class FakeQdrant:
    def __init__(self, exists=False, hits=None, fail=False):
        self._exists = exists
        self._hits = hits or []
        self._fail = fail
        self.created = []
        self.upserted = []

    def get_collections(self):
        if self._fail:
            raise ConnectionError("refused")
        return []

    def collection_exists(self, name):
        return self._exists

    def create_collection(self, collection_name, vectors_config):
        self.created.append((collection_name, vectors_config))

    def upsert(self, collection_name, points):
        self.upserted.append((collection_name, points))

    def query_points(self, collection_name, query, limit):
        return SimpleNamespace(points=self._hits[:limit])

    def scroll(self, collection_name, scroll_filter, limit):
        return (self._hits[:limit], None)


def _store(fake):
    from rag.store import VectorStore

    return VectorStore(client=fake)


def test_ping_fail_fast():
    store = _store(FakeQdrant(fail=True))
    with pytest.raises(RuntimeError, match="docker compose up"):
        store.ping()


def test_ping_ok():
    _store(FakeQdrant()).ping()  # no raise


def test_ensure_collection_creates_once():
    fake = FakeQdrant(exists=False)
    _store(fake).ensure_collection()
    assert len(fake.created) == 1

    fake2 = FakeQdrant(exists=True)
    _store(fake2).ensure_collection()
    assert fake2.created == []


def test_upsert_builds_deterministic_points():
    from rag.store import ChunkRecord

    fake = FakeQdrant()
    store = _store(fake)
    rec = ChunkRecord(paper_id="1706.03762", title="Attention", chunk_index=0,
                      text="chunk", vector=[0.1, 0.2])
    store.upsert_chunks([rec])
    store.upsert_chunks([rec])

    (_, points1), (_, points2) = fake.upserted
    assert points1[0].id == points2[0].id  # uuid5 → idempotent re-ingest
    assert points1[0].payload == {"paper_id": "1706.03762", "title": "Attention",
                                  "chunk_index": 0, "chunk_text": "chunk", "section": ""}


def test_search_maps_hits():
    hit = SimpleNamespace(score=0.87, payload={"paper_id": "1706.03762", "title": "Attention",
                                               "chunk_index": 0, "chunk_text": "self-attention",
                                               "section": ""})
    store = _store(FakeQdrant(hits=[hit]))
    results = store.search([0.1, 0.2], top_k=3)
    assert len(results) == 1
    assert results[0].paper_id == "1706.03762"
    assert results[0].text == "self-attention"
    assert results[0].score == 0.87


def test_has_paper():
    hit = SimpleNamespace(score=1.0, payload={})
    assert _store(FakeQdrant(hits=[hit])).has_paper("1706.03762") is True
    assert _store(FakeQdrant(hits=[])).has_paper("1706.03762") is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.store'`

- [ ] **Step 3: Write rag/store.py**

```python
import uuid

from pydantic import BaseModel
from qdrant_client import QdrantClient, models

from config import settings


class ChunkRecord(BaseModel):
    paper_id: str
    title: str
    chunk_index: int
    text: str
    section: str = ""  # kept for schema parity; pypdf gives no reliable sections
    vector: list[float]


class ScoredChunk(BaseModel):
    paper_id: str
    title: str
    text: str
    score: float


class VectorStore:
    """Thin wrapper around Qdrant for paper chunks."""

    def __init__(self, url: str | None = None, collection: str | None = None, client=None):
        self.collection = collection or settings.qdrant_collection
        self.client = client or QdrantClient(url=url or settings.qdrant_url)

    def ping(self) -> None:
        """Fail fast with a clear message when Qdrant is down."""
        try:
            self.client.get_collections()
        except Exception as exc:
            raise RuntimeError(
                f"Qdrant is not reachable at {settings.qdrant_url}. "
                "Start it with: docker compose up -d"
            ) from exc

    def ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(
                    size=settings.embedding_dim, distance=models.Distance.COSINE
                ),
            )

    def upsert_chunks(self, records: list[ChunkRecord]) -> None:
        points = [
            models.PointStruct(
                # uuid5 of paper_id:chunk_index → re-ingesting a paper overwrites
                # its old points instead of duplicating them.
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{r.paper_id}:{r.chunk_index}")),
                vector=r.vector,
                payload={"paper_id": r.paper_id, "title": r.title,
                         "chunk_index": r.chunk_index, "chunk_text": r.text,
                         "section": r.section},
            )
            for r in records
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, vector: list[float], top_k: int | None = None) -> list[ScoredChunk]:
        top_k = top_k or settings.retrieval_top_k
        hits = self.client.query_points(
            collection_name=self.collection, query=vector, limit=top_k
        ).points
        return [
            ScoredChunk(paper_id=h.payload["paper_id"], title=h.payload["title"],
                        text=h.payload["chunk_text"], score=h.score)
            for h in hits
        ]

    def has_paper(self, paper_id: str) -> bool:
        hits, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=models.Filter(
                must=[models.FieldCondition(key="paper_id",
                                            match=models.MatchValue(value=paper_id))]
            ),
            limit=1,
        )
        return len(hits) > 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_store.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add rag/store.py tests/test_store.py
git commit -m "feat: qdrant vector store with fail-fast ping and idempotent upsert"
```

---

### Task 10: Ingest pipeline (search → download → parse → chunk → embed → upsert)

**Files:**
- Create: `rag/ingest.py`
- Test: `tests/test_ingest.py`

**Interfaces:**
- Consumes: `rag.arxiv_client.{PaperMeta, search_papers, download_pdf}`, `rag.parse.extract_text`, `rag.chunk.chunk_text`, `rag.embed.embed_texts`, `rag.store.{ChunkRecord, VectorStore}`.
- Produces:
  - `rag.ingest.IngestResult(ingested: list[str], skipped: list[str])` (pydantic model, both default `[]`)
  - `rag.ingest.ingest_paper(meta: PaperMeta, store: VectorStore) -> int | None` — chunk count, `0` if already ingested, `None` if skipped (download/parse failure)
  - `rag.ingest.ingest_query(query: str, max_results: int = 3, store: VectorStore | None = None) -> IngestResult`

- [ ] **Step 1: Write the failing tests**

`tests/test_ingest.py`:

```python
from types import SimpleNamespace


def _meta(pid="1706.03762", title="Attention"):
    from rag.arxiv_client import PaperMeta

    return PaperMeta(paper_id=pid, title=title, summary="s")


class FakeStore:
    def __init__(self, existing=()):
        self.existing = set(existing)
        self.upserts = []
        self.pinged = False
        self.ensured = False

    def ping(self):
        self.pinged = True

    def ensure_collection(self):
        self.ensured = True

    def has_paper(self, paper_id):
        return paper_id in self.existing

    def upsert_chunks(self, records):
        self.upserts.append(records)


def _patch_pipeline(monkeypatch, text="some paper text"):
    import rag.ingest as ingest

    monkeypatch.setattr(ingest, "download_pdf", lambda pid: f"/tmp/{pid}.pdf")
    monkeypatch.setattr(ingest, "extract_text", lambda path: text)
    monkeypatch.setattr(ingest, "chunk_text", lambda t: ["chunk a", "chunk b"])
    monkeypatch.setattr(ingest, "embed_texts", lambda chunks: [[0.1], [0.2]])


def test_ingest_paper_happy_path(monkeypatch):
    import rag.ingest as ingest

    _patch_pipeline(monkeypatch)
    store = FakeStore()
    n = ingest.ingest_paper(_meta(), store)

    assert n == 2
    records = store.upserts[0]
    assert [r.chunk_index for r in records] == [0, 1]
    assert records[0].paper_id == "1706.03762"
    assert records[0].vector == [0.1]


def test_ingest_paper_skips_already_ingested(monkeypatch):
    import rag.ingest as ingest

    _patch_pipeline(monkeypatch)
    store = FakeStore(existing={"1706.03762"})
    assert ingest.ingest_paper(_meta(), store) == 0
    assert store.upserts == []


def test_ingest_paper_parse_failure_returns_none(monkeypatch):
    import rag.ingest as ingest

    _patch_pipeline(monkeypatch)
    monkeypatch.setattr(ingest, "extract_text", lambda path: None)
    assert ingest.ingest_paper(_meta(), FakeStore()) is None


def test_ingest_paper_download_failure_returns_none(monkeypatch):
    import rag.ingest as ingest

    _patch_pipeline(monkeypatch)

    def _boom(pid):
        raise ConnectionError("network down")

    monkeypatch.setattr(ingest, "download_pdf", _boom)
    assert ingest.ingest_paper(_meta(), FakeStore()) is None


def test_ingest_query_continues_after_failures(monkeypatch):
    import rag.ingest as ingest

    _patch_pipeline(monkeypatch)
    metas = [_meta("1111.11111", "Good"), _meta("2222.22222", "Bad"), _meta("3333.33333", "Good2")]
    monkeypatch.setattr(ingest, "search_papers", lambda q, max_results: metas)
    # Second paper fails to parse.
    monkeypatch.setattr(ingest, "extract_text",
                        lambda path: None if "2222" in path else "text")

    store = FakeStore()
    result = ingest.ingest_query("test", max_results=3, store=store)

    assert store.pinged and store.ensured
    assert result.ingested == ["1111.11111", "3333.33333"]
    assert result.skipped == ["2222.22222"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.ingest'`

- [ ] **Step 3: Write rag/ingest.py**

```python
import logging

from pydantic import BaseModel

from rag.arxiv_client import PaperMeta, download_pdf, search_papers
from rag.chunk import chunk_text
from rag.embed import embed_texts
from rag.parse import extract_text
from rag.store import ChunkRecord, VectorStore

logger = logging.getLogger(__name__)


class IngestResult(BaseModel):
    ingested: list[str] = []
    skipped: list[str] = []


def ingest_paper(meta: PaperMeta, store: VectorStore) -> int | None:
    """Download, parse, chunk, embed, and upsert one paper.

    Returns the number of chunks upserted, 0 if the paper was already
    ingested, or None if it had to be skipped (download/parse failure).
    """
    if store.has_paper(meta.paper_id):
        logger.info("Already ingested %s", meta.paper_id)
        return 0
    try:
        pdf_path = download_pdf(meta.paper_id)
    except Exception:
        logger.exception("Download failed for %s, skipping", meta.paper_id)
        return None
    text = extract_text(pdf_path)
    if text is None:
        return None  # extract_text already logged the reason
    chunks = chunk_text(text)
    vectors = embed_texts(chunks)
    records = [
        ChunkRecord(paper_id=meta.paper_id, title=meta.title,
                    chunk_index=i, text=chunk, vector=vector)
        for i, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]
    store.upsert_chunks(records)
    logger.info("Ingested %s (%d chunks)", meta.paper_id, len(records))
    return len(records)


def ingest_query(query: str, max_results: int = 3,
                 store: VectorStore | None = None) -> IngestResult:
    """Search arXiv and ingest the results. Failures skip the paper, not the batch."""
    store = store or VectorStore()
    store.ping()  # fail fast before doing any network work
    store.ensure_collection()
    result = IngestResult()
    for meta in search_papers(query, max_results=max_results):
        n = ingest_paper(meta, store)
        if n is None:
            result.skipped.append(meta.paper_id)
        else:
            result.ingested.append(meta.paper_id)
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add rag/ingest.py tests/test_ingest.py
git commit -m "feat: ingest pipeline with per-paper skip on failure"
```

---

### Task 11: Retrieval + grounded answering (RAG query flow)

**Files:**
- Create: `rag/retrieve.py`, `rag/answer.py`
- Test: `tests/test_retrieve_answer.py`, `tests/test_integration_rag.py`

**Interfaces:**
- Consumes: `rag.embed.embed_query`, `rag.store.{VectorStore, ScoredChunk}`, `llm.base.generate`, `llm.prompts.build_rag_prompt`.
- Produces:
  - `rag.retrieve.retrieve(question: str, top_k: int | None = None, store: VectorStore | None = None) -> list[ScoredChunk]`
  - `rag.answer.RagAnswer(text: str, sources: list[str])` (pydantic model)
  - `rag.answer.answer_question(question: str, store: VectorStore | None = None) -> RagAnswer`

- [ ] **Step 1: Write the failing unit tests**

`tests/test_retrieve_answer.py`:

```python
from types import SimpleNamespace


def _chunk(pid="1706.03762", title="Attention", text="self-attention", score=0.9):
    from rag.store import ScoredChunk

    return ScoredChunk(paper_id=pid, title=title, text=text, score=score)


def test_retrieve_embeds_and_searches(monkeypatch):
    import rag.retrieve as retrieve_mod

    monkeypatch.setattr(retrieve_mod, "embed_query", lambda q: [0.5])
    fake_store = SimpleNamespace(search=lambda vector, top_k: [_chunk()] if vector == [0.5] else [])

    chunks = retrieve_mod.retrieve("what is attention?", top_k=3, store=fake_store)
    assert len(chunks) == 1
    assert chunks[0].paper_id == "1706.03762"


def test_answer_question_builds_grounded_prompt(monkeypatch):
    import rag.answer as answer_mod
    from llm.base import LLMResponse

    monkeypatch.setattr(answer_mod, "retrieve",
                        lambda q, store=None: [_chunk(), _chunk(pid="1810.04805", title="BERT")])
    captured = {}

    def fake_generate(messages, **kwargs):
        captured["messages"] = messages
        captured.update(kwargs)
        return LLMResponse(text="Self-attention is key [1706.03762].",
                           usage={"cache_read_input_tokens": 0})

    monkeypatch.setattr(answer_mod, "generate", fake_generate)

    result = answer_mod.answer_question("what is attention?")

    assert result.text == "Self-attention is key [1706.03762]."
    assert result.sources == ["1706.03762", "1810.04805"]
    # Grounded prompt: context in system with a cache breakpoint, question last.
    assert captured["system"][1]["cache_control"] == {"type": "ephemeral"}
    assert "self-attention" in captured["system"][1]["text"]
    assert "what is attention?" in captured["messages"][-1]["content"]


def test_answer_question_empty_store(monkeypatch):
    import rag.answer as answer_mod

    monkeypatch.setattr(answer_mod, "retrieve", lambda q, store=None: [])
    monkeypatch.setattr(answer_mod, "generate",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no LLM call")))

    result = answer_mod.answer_question("anything")
    assert result.sources == []
    assert "ingest" in result.text.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_retrieve_answer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.retrieve'`

- [ ] **Step 3: Write rag/retrieve.py**

```python
from config import settings
from rag.embed import embed_query
from rag.store import ScoredChunk, VectorStore


def retrieve(question: str, top_k: int | None = None,
             store: VectorStore | None = None) -> list[ScoredChunk]:
    """Embed the question and return the top-k chunks from Qdrant."""
    store = store or VectorStore()
    return store.search(embed_query(question), top_k=top_k or settings.retrieval_top_k)
```

- [ ] **Step 4: Write rag/answer.py**

```python
import logging

from pydantic import BaseModel

from llm.base import generate
from llm.prompts import build_rag_prompt
from rag.retrieve import retrieve
from rag.store import VectorStore

logger = logging.getLogger(__name__)


class RagAnswer(BaseModel):
    text: str
    sources: list[str]


def answer_question(question: str, store: VectorStore | None = None) -> RagAnswer:
    """RAG query flow: embed → retrieve → grounded prompt → generate."""
    chunks = retrieve(question, store=store)
    if not chunks:
        return RagAnswer(
            text="I don't have any ingested papers to answer from yet. "
                 "Ingest some papers first.",
            sources=[],
        )
    contexts = [{"paper_id": c.paper_id, "title": c.title, "text": c.text} for c in chunks]
    system, messages = build_rag_prompt(question, contexts)
    resp = generate(messages, system=system)
    logger.info(
        "answer usage: cache_read=%s cache_creation=%s",
        resp.usage.get("cache_read_input_tokens"),
        resp.usage.get("cache_creation_input_tokens"),
    )
    return RagAnswer(text=resp.text, sources=sorted({c.paper_id for c in chunks}))
```

- [ ] **Step 5: Run unit tests to verify they pass**

Run: `uv run pytest tests/test_retrieve_answer.py -v`
Expected: 3 passed

- [ ] **Step 6: Write the ingest→query round-trip integration test**

`tests/test_integration_rag.py`:

```python
"""Full round trip against real APIs. Requires: docker compose up -d, real keys in .env.

Run: uv run pytest tests/test_integration_rag.py -m integration -v
"""

import pytest

pytestmark = pytest.mark.integration


def test_ingest_then_query_round_trip():
    from rag.answer import answer_question
    from rag.arxiv_client import get_paper
    from rag.ingest import ingest_paper
    from rag.store import VectorStore

    store = VectorStore()
    store.ping()
    store.ensure_collection()

    meta = get_paper("1706.03762")
    assert meta is not None
    n = ingest_paper(meta, store)
    assert n is not None  # ingested now or already present

    result = answer_question("What attention mechanism does the Transformer use?", store=store)
    assert "1706.03762" in result.sources
    assert "[1706.03762]" in result.text  # inline citation present
```

Run: `uv run pytest tests/test_integration_rag.py -v`
Expected: `1 deselected`

- [ ] **Step 7: Commit**

```bash
git add rag/retrieve.py rag/answer.py tests/test_retrieve_answer.py tests/test_integration_rag.py
git commit -m "feat: retrieval and grounded answering with citations"
```

---

### Task 12: Custom MCP server (arxiv_search, arxiv_fetch_paper)

**Files:**
- Create: `agents/mcp_server.py`
- Test: `tests/test_mcp_server.py`

**Interfaces:**
- Consumes: `mcp.server.fastmcp.FastMCP`, `rag.arxiv_client.{search_papers, get_paper}`, `rag.ingest.ingest_paper`, `rag.store.VectorStore`.
- Produces:
  - `agents.mcp_server.mcp` — a `FastMCP("arxiv")` instance exposing tools `arxiv_search(query: str, max_results: int = 5) -> str` (JSON list of `{paper_id, title, summary}`) and `arxiv_fetch_paper(paper_id: str) -> str` (status string; errors returned as `"Error: ..."` strings, never raised)
  - Runnable as a stdio server: `python -m agents.mcp_server`
  - Module-level plain functions `arxiv_search` / `arxiv_fetch_paper` are also importable for direct unit testing.

- [ ] **Step 1: Write the failing tests**

`tests/test_mcp_server.py`:

```python
import json


async def test_tool_schemas_registered():
    from agents.mcp_server import mcp

    tools = await mcp.list_tools()
    names = {t.name for t in tools}
    assert names == {"arxiv_search", "arxiv_fetch_paper"}

    search = next(t for t in tools if t.name == "arxiv_search")
    assert "query" in search.inputSchema["properties"]
    assert "query" in search.inputSchema.get("required", [])

    fetch = next(t for t in tools if t.name == "arxiv_fetch_paper")
    assert "paper_id" in fetch.inputSchema["properties"]


def test_arxiv_search_returns_json(monkeypatch):
    import agents.mcp_server as srv
    from rag.arxiv_client import PaperMeta

    monkeypatch.setattr(
        srv, "search_papers",
        lambda query, max_results: [PaperMeta(paper_id="1706.03762", title="Attention",
                                              summary="s")],
    )
    out = json.loads(srv.arxiv_search("attention", max_results=1))
    assert out == [{"paper_id": "1706.03762", "title": "Attention", "summary": "s"}]


def test_fetch_paper_ingests(monkeypatch):
    import agents.mcp_server as srv
    from rag.arxiv_client import PaperMeta

    meta = PaperMeta(paper_id="1706.03762", title="Attention", summary="s")
    monkeypatch.setattr(srv, "get_paper", lambda pid: meta)
    monkeypatch.setattr(srv, "ingest_paper", lambda m, store: 42)

    class FakeStore:
        def ping(self):
            pass

        def ensure_collection(self):
            pass

    monkeypatch.setattr(srv, "VectorStore", FakeStore)

    out = srv.arxiv_fetch_paper("1706.03762")
    assert "1706.03762" in out and "42" in out


def test_fetch_paper_errors_are_strings(monkeypatch):
    import agents.mcp_server as srv

    monkeypatch.setattr(srv, "get_paper", lambda pid: None)
    assert srv.arxiv_fetch_paper("0000.00000").startswith("Error:")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_mcp_server.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agents.mcp_server'`

- [ ] **Step 3: Write agents/mcp_server.py**

```python
"""Custom MCP server exposing arXiv tools over stdio.

Run: python -m agents.mcp_server
Tool errors are returned as "Error: ..." strings (tool results), never raised —
the agent decides whether to retry or give up (spec: no silent crash).
"""

import json

from mcp.server.fastmcp import FastMCP

from rag.arxiv_client import get_paper, search_papers
from rag.ingest import ingest_paper
from rag.store import VectorStore

mcp = FastMCP("arxiv")


@mcp.tool()
def arxiv_search(query: str, max_results: int = 5) -> str:
    """Search arXiv for papers. Returns a JSON list of {paper_id, title, summary}."""
    papers = search_papers(query, max_results=max_results)
    return json.dumps([p.model_dump() for p in papers])


@mcp.tool()
def arxiv_fetch_paper(paper_id: str) -> str:
    """Download an arXiv paper by id and ingest it into the vector store so
    rag_query can answer questions about it."""
    meta = get_paper(paper_id)
    if meta is None:
        return f"Error: no arXiv paper found with id {paper_id}"
    store = VectorStore()
    try:
        store.ping()
    except RuntimeError as exc:
        return f"Error: {exc}"
    store.ensure_collection()
    n = ingest_paper(meta, store)
    if n is None:
        return f"Error: failed to download or parse {paper_id}"
    if n == 0:
        return f"{paper_id} was already ingested: {meta.title}"
    return f"Ingested {paper_id} ({n} chunks): {meta.title}"


if __name__ == "__main__":
    mcp.run()  # stdio transport
```

Note: `@mcp.tool()` wraps but does not replace the plain functions in this module's namespace — FastMCP registers them and they stay directly callable for tests.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_mcp_server.py -v`
Expected: 4 passed. If `test_arxiv_search_returns_json` fails because the decorator wrapped the function into a non-callable, register tools without sugar instead: define plain functions, then `mcp.tool()(arxiv_search)` and `mcp.tool()(arxiv_fetch_paper)` after the definitions.

- [ ] **Step 5: Commit**

```bash
git add agents/mcp_server.py tests/test_mcp_server.py
git commit -m "feat: custom mcp server with arxiv_search and arxiv_fetch_paper"
```

---

### Task 13: MCP client (custom server + external mcp-server-fetch)

**Files:**
- Create: `agents/mcp_client.py`
- Test: `tests/test_mcp_client.py`, `tests/test_integration_mcp.py`

**Interfaces:**
- Consumes: `mcp` SDK client (`ClientSession`, `StdioServerParameters`, `stdio_client`).
- Produces:
  - `agents.mcp_client.SERVERS: dict[str, StdioServerParameters]` — `"arxiv"` → `python -m agents.mcp_server`, `"fetch"` → `uvx mcp-server-fetch`
  - `agents.mcp_client.MCPToolbox(servers: dict | None = None)` — async context manager:
    - `list_tools() -> list[dict]` (Anthropic tool format: `name`, `description`, `input_schema`)
    - `async call_tool(name: str, arguments: dict) -> tuple[str, bool]` — `(content, is_error)`; connection/execution failures come back as `(message, True)`, never raised

- [ ] **Step 1: Write the failing unit tests**

`tests/test_mcp_client.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_mcp_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agents.mcp_client'`

- [ ] **Step 3: Write agents/mcp_client.py**

```python
"""MCP client aggregating tools from all configured servers.

Servers: our custom arxiv server (stdio subprocess) and the official
mcp-server-fetch (via uvx). Tool failures are returned as (message, True)
results so the agent can decide retry vs give up — never raised.
"""

import sys
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

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
        for server_name, params in self.servers.items():
            read, write = await self._stack.enter_async_context(stdio_client(params))
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            listing = await session.list_tools()
            for tool in listing.tools:
                self._sessions[tool.name] = session
                self._tools.append({
                    "name": tool.name,
                    "description": tool.description or "",
                    "input_schema": tool.inputSchema,
                })
        return self

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
```

- [ ] **Step 4: Run unit tests to verify they pass**

Run: `uv run pytest tests/test_mcp_client.py -v`
Expected: 5 passed

- [ ] **Step 5: Write the integration test (spawns real servers)**

`tests/test_integration_mcp.py`:

```python
"""Spawns the real MCP servers as subprocesses. Needs network + uvx on PATH.

Run: uv run pytest tests/test_integration_mcp.py -m integration -v
"""

import json

import pytest

pytestmark = pytest.mark.integration


async def test_toolbox_aggregates_both_servers_and_calls_search():
    from agents.mcp_client import MCPToolbox

    async with MCPToolbox() as box:
        names = {t["name"] for t in box.list_tools()}
        assert {"arxiv_search", "arxiv_fetch_paper"} <= names
        assert "fetch" in names  # from mcp-server-fetch

        content, is_error = await box.call_tool(
            "arxiv_search", {"query": "attention is all you need", "max_results": 2}
        )
        assert is_error is False
        assert json.loads(content)  # non-empty JSON list
```

Run: `uv run pytest tests/test_integration_mcp.py -v`
Expected: `1 deselected`

- [ ] **Step 6: Commit**

```bash
git add agents/mcp_client.py tests/test_mcp_client.py tests/test_integration_mcp.py
git commit -m "feat: mcp toolbox client over custom arxiv server and mcp-server-fetch"
```

---

### Task 14: LangGraph agent (decide → tools → loop → answer)

**Files:**
- Create: `agents/graph.py`
- Test: `tests/test_graph.py`, `tests/test_integration_agent.py`

**Interfaces:**
- Consumes: `langgraph.graph.{StateGraph, END}`, `llm.base.generate`, `llm.prompts.AGENT_SYSTEM_PROMPT`, `agents.mcp_client.MCPToolbox`, `rag.answer.answer_question`, `config.settings.agent_max_steps`.
- Produces:
  - `agents.graph.RAG_QUERY_TOOL: dict` (local tool spec)
  - `agents.graph.AgentState` — `TypedDict{messages: list[dict], steps: int}`
  - `agents.graph.build_graph(toolbox) -> CompiledStateGraph` (toolbox needs `.list_tools()` and `async .call_tool()` — duck-typed for tests)
  - `agents.graph.final_text(state: dict) -> str`
  - `agents.graph.run_agent(question: str) -> str` (async; opens `MCPToolbox`, invokes graph, returns final assistant text)

- [ ] **Step 1: Write the failing unit tests**

`tests/test_graph.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_graph.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agents.graph'`

- [ ] **Step 3: Write agents/graph.py**

```python
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
```

- [ ] **Step 4: Run unit tests to verify they pass**

Run: `uv run pytest tests/test_graph.py -v`
Expected: 4 passed

- [ ] **Step 5: Write the agent-loop integration test**

`tests/test_integration_agent.py`:

```python
"""End-to-end agent loop with real LLM, MCP servers, arXiv, and Qdrant.

Run: uv run pytest tests/test_integration_agent.py -m integration -v
"""

import pytest

pytestmark = pytest.mark.integration


async def test_agent_answers_with_tools():
    from agents.graph import run_agent

    reply = await run_agent(
        "Fetch the arXiv paper 1706.03762 if you don't have it, then tell me "
        "what attention mechanism it introduces."
    )
    assert reply
    assert "1706.03762" in reply  # cited
```

Run: `uv run pytest tests/test_integration_agent.py -v`
Expected: `1 deselected`

- [ ] **Step 6: Commit**

```bash
git add agents/graph.py tests/test_graph.py tests/test_integration_agent.py
git commit -m "feat: langgraph agent with rag_query and mcp tool loop"
```

---

### Task 15: Golden dataset + retrieval metrics

**Files:**
- Create: `eval/golden.json`, `eval/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `eval/golden.json` — list of `{"question": str, "expected_paper_ids": list[str], "expected_answer_gist": str}`
  - `eval.metrics.precision_recall(retrieved: list[str], expected: list[str]) -> tuple[float, float]`

- [ ] **Step 1: Write the failing tests**

`tests/test_metrics.py`:

```python
import json
from pathlib import Path

from eval.metrics import precision_recall


def test_perfect_retrieval():
    assert precision_recall(["a", "b"], ["a", "b"]) == (1.0, 1.0)


def test_partial_overlap():
    p, r = precision_recall(["a", "b", "c", "d"], ["a", "x"])
    assert p == 0.25  # 1 of 4 retrieved is relevant
    assert r == 0.5   # 1 of 2 expected was found


def test_no_retrieved():
    assert precision_recall([], ["a"]) == (0.0, 0.0)


def test_duplicates_do_not_inflate():
    p, r = precision_recall(["a", "a", "a"], ["a"])
    assert p == 1.0
    assert r == 1.0


def test_golden_dataset_shape():
    items = json.loads(Path("eval/golden.json").read_text())
    assert len(items) >= 3
    for item in items:
        assert item["question"]
        assert isinstance(item["expected_paper_ids"], list) and item["expected_paper_ids"]
        assert item["expected_answer_gist"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval.metrics'`

- [ ] **Step 3: Write eval/golden.json**

```json
[
  {
    "question": "What attention mechanism does the Transformer architecture rely on, and why does it help with long-range dependencies?",
    "expected_paper_ids": ["1706.03762"],
    "expected_answer_gist": "The Transformer relies on multi-head self-attention, which connects any two positions in a sequence in a constant number of sequential operations, making long-range dependencies easier to learn than with recurrence or convolution."
  },
  {
    "question": "What is the key idea behind BERT's pre-training?",
    "expected_paper_ids": ["1810.04805"],
    "expected_answer_gist": "BERT pre-trains deep bidirectional representations using a masked language model objective (plus next sentence prediction), then fine-tunes the same model on downstream tasks."
  },
  {
    "question": "How does retrieval-augmented generation combine parametric and non-parametric memory?",
    "expected_paper_ids": ["2005.11401"],
    "expected_answer_gist": "RAG combines a pre-trained seq2seq generator (parametric memory) with a dense vector index of Wikipedia accessed by a neural retriever (non-parametric memory), conditioning generation on retrieved passages."
  }
]
```

- [ ] **Step 4: Write eval/metrics.py**

```python
def precision_recall(retrieved: list[str], expected: list[str]) -> tuple[float, float]:
    """Set-based retrieval precision/recall over paper ids."""
    retrieved_set, expected_set = set(retrieved), set(expected)
    overlap = len(retrieved_set & expected_set)
    precision = overlap / len(retrieved_set) if retrieved_set else 0.0
    recall = overlap / len(expected_set) if expected_set else 0.0
    return precision, recall
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add eval/golden.json eval/metrics.py tests/test_metrics.py
git commit -m "feat: golden qa dataset and retrieval precision/recall"
```

---

### Task 16: LLM-as-judge scorer (structured output)

**Files:**
- Create: `eval/judge.py`
- Test: `tests/test_judge.py`

**Interfaces:**
- Consumes: `llm.base.generate` (with `structured_schema`).
- Produces:
  - `eval.judge.JudgeScores(faithfulness: int, relevance: int, citation_accuracy: int, reasoning: str)` — ints constrained 1–5
  - `eval.judge.judge_answer(question: str, answer: str, expected_gist: str, contexts: list[dict]) -> JudgeScores` — contexts are `{"paper_id", "text"}` dicts

- [ ] **Step 1: Write the failing tests**

`tests/test_judge.py`:

```python
import pytest
from pydantic import ValidationError


def test_scores_are_bounded():
    from eval.judge import JudgeScores

    JudgeScores(faithfulness=5, relevance=1, citation_accuracy=3, reasoning="ok")
    with pytest.raises(ValidationError):
        JudgeScores(faithfulness=6, relevance=1, citation_accuracy=3, reasoning="ok")
    with pytest.raises(ValidationError):
        JudgeScores(faithfulness=0, relevance=1, citation_accuracy=3, reasoning="ok")


def test_judge_answer_uses_structured_output(monkeypatch):
    import eval.judge as judge_mod
    from llm.base import LLMResponse

    expected = judge_mod.JudgeScores(faithfulness=4, relevance=5,
                                     citation_accuracy=4, reasoning="solid")
    captured = {}

    def fake_generate(messages, **kwargs):
        captured["messages"] = messages
        captured.update(kwargs)
        return LLMResponse(text="", parsed=expected)

    monkeypatch.setattr(judge_mod, "generate", fake_generate)

    scores = judge_mod.judge_answer(
        question="What is attention?",
        answer="Self-attention [1706.03762].",
        expected_gist="Transformers use self-attention.",
        contexts=[{"paper_id": "1706.03762", "text": "self-attention connects positions"}],
    )

    assert scores == expected
    assert captured["structured_schema"] is judge_mod.JudgeScores
    prompt = captured["messages"][0]["content"]
    assert "What is attention?" in prompt
    assert "Self-attention [1706.03762]." in prompt
    assert "Transformers use self-attention." in prompt
    assert "self-attention connects positions" in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_judge.py -v`
Expected: FAIL — `ModuleNotFoundError` (no `eval.judge`)

- [ ] **Step 3: Write eval/judge.py**

```python
"""LLM-as-judge scorer. No eval framework — a structured-output call and a rubric."""

from pydantic import BaseModel, Field

from llm.base import generate


class JudgeScores(BaseModel):
    faithfulness: int = Field(
        ge=1, le=5, description="Is every claim in the answer supported by the context? 5 = fully grounded, 1 = mostly fabricated."
    )
    relevance: int = Field(
        ge=1, le=5, description="Does the answer actually address the question? 5 = directly and completely."
    )
    citation_accuracy: int = Field(
        ge=1, le=5, description="Do the [paper_id] citations point at context excerpts that support the cited claims? 5 = all correct."
    )
    reasoning: str = Field(description="One short paragraph justifying the scores.")


JUDGE_SYSTEM_PROMPT = """You are a strict evaluator of a research assistant's answers.
Score each dimension 1-5 (5 = perfect). Judge faithfulness ONLY against the provided
context excerpts, and citation accuracy against the paper ids appearing in them.
The reference gist describes what a good answer should convey — use it for relevance."""


def judge_answer(question: str, answer: str, expected_gist: str,
                 contexts: list[dict]) -> JudgeScores:
    context_text = "\n\n".join(f"[{c['paper_id']}] {c['text']}" for c in contexts)
    user = f"""Question: {question}

Reference gist (what a good answer should convey):
{expected_gist}

Context excerpts the assistant had:
{context_text}

Assistant answer to evaluate:
{answer}"""
    resp = generate([{"role": "user", "content": user}],
                    system=JUDGE_SYSTEM_PROMPT, structured_schema=JudgeScores)
    return resp.parsed
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_judge.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add eval/judge.py tests/test_judge.py
git commit -m "feat: llm-as-judge scorer with structured output"
```

---

### Task 17: Eval runner (`python -m eval.run`) + smoke threshold test

**Files:**
- Create: `eval/run.py`
- Test: `tests/test_eval_run.py`, `tests/test_integration_eval.py`

**Interfaces:**
- Consumes: `rag.retrieve.retrieve`, `rag.answer.answer_question`, `eval.metrics.precision_recall`, `eval.judge.judge_answer`.
- Produces:
  - `eval.run.run_eval(dataset_path: str = "eval/golden.json", report_path: str = "report.json") -> dict` — report dict `{"summary": {...}, "rows": [...]}`; summary keys: `n`, `avg_precision`, `avg_recall`, `avg_faithfulness`, `avg_relevance`, `avg_citation_accuracy`
  - `python -m eval.run` writes `report.json` and prints the summary

- [ ] **Step 1: Write the failing unit test**

`tests/test_eval_run.py`:

```python
import json


def test_run_eval_writes_report(monkeypatch, tmp_path):
    import eval.run as run_mod
    from eval.judge import JudgeScores
    from rag.answer import RagAnswer
    from rag.store import ScoredChunk

    dataset = [
        {"question": "q1", "expected_paper_ids": ["1706.03762"], "expected_answer_gist": "g1"},
        {"question": "q2", "expected_paper_ids": ["1810.04805"], "expected_answer_gist": "g2"},
    ]
    dataset_path = tmp_path / "golden.json"
    dataset_path.write_text(json.dumps(dataset))
    report_path = tmp_path / "report.json"

    chunk = ScoredChunk(paper_id="1706.03762", title="Attention", text="ctx", score=0.9)
    monkeypatch.setattr(run_mod, "retrieve", lambda q: [chunk])
    monkeypatch.setattr(run_mod, "answer_question",
                        lambda q: RagAnswer(text="ans [1706.03762]", sources=["1706.03762"]))
    monkeypatch.setattr(
        run_mod, "judge_answer",
        lambda question, answer, expected_gist, contexts: JudgeScores(
            faithfulness=4, relevance=5, citation_accuracy=3, reasoning="r"),
    )

    report = run_mod.run_eval(dataset_path=str(dataset_path), report_path=str(report_path))

    assert report_path.exists()
    on_disk = json.loads(report_path.read_text())
    assert on_disk["summary"] == report["summary"]

    s = report["summary"]
    assert s["n"] == 2
    assert s["avg_precision"] == 1.0
    assert s["avg_recall"] == 0.5  # q2 expected 1810.04805, retrieved 1706.03762
    assert s["avg_faithfulness"] == 4.0
    assert s["avg_relevance"] == 5.0
    assert s["avg_citation_accuracy"] == 3.0

    row = report["rows"][0]
    assert row["question"] == "q1"
    assert row["answer"] == "ans [1706.03762]"
    assert row["reasoning"] == "r"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_eval_run.py -v`
Expected: FAIL — `ModuleNotFoundError` (no `eval.run`)

- [ ] **Step 3: Write eval/run.py**

```python
"""Offline eval harness. Standalone: uv run python -m eval.run

For each golden item: retrieve → precision/recall vs expected ids; answer via
the RAG pipeline; LLM-judge the answer. Writes report.json + prints a summary.
(answer_question retrieves internally too — the double retrieval is accepted
for simplicity; both calls hit the same store deterministically.)
"""

import json
from pathlib import Path
from statistics import mean

from eval.judge import judge_answer
from eval.metrics import precision_recall
from rag.answer import answer_question
from rag.retrieve import retrieve


def run_eval(dataset_path: str = "eval/golden.json",
             report_path: str = "report.json") -> dict:
    dataset = json.loads(Path(dataset_path).read_text())
    rows: list[dict] = []
    for item in dataset:
        question = item["question"]
        chunks = retrieve(question)
        precision, recall = precision_recall(
            [c.paper_id for c in chunks], item["expected_paper_ids"]
        )
        answer = answer_question(question)
        contexts = [{"paper_id": c.paper_id, "text": c.text} for c in chunks]
        scores = judge_answer(question, answer.text, item["expected_answer_gist"], contexts)
        rows.append({
            "question": question,
            "expected_paper_ids": item["expected_paper_ids"],
            "retrieved_paper_ids": sorted({c.paper_id for c in chunks}),
            "precision": precision,
            "recall": recall,
            **scores.model_dump(),
            "answer": answer.text,
        })

    summary = {
        "n": len(rows),
        "avg_precision": mean(r["precision"] for r in rows),
        "avg_recall": mean(r["recall"] for r in rows),
        "avg_faithfulness": mean(r["faithfulness"] for r in rows),
        "avg_relevance": mean(r["relevance"] for r in rows),
        "avg_citation_accuracy": mean(r["citation_accuracy"] for r in rows),
    }
    report = {"summary": summary, "rows": rows}
    Path(report_path).write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    report = run_eval()
    s = report["summary"]
    print(f"\nEvaluated {s['n']} questions -> report.json")
    print(f"  retrieval precision : {s['avg_precision']:.2f}")
    print(f"  retrieval recall    : {s['avg_recall']:.2f}")
    print(f"  faithfulness        : {s['avg_faithfulness']:.2f} / 5")
    print(f"  relevance           : {s['avg_relevance']:.2f} / 5")
    print(f"  citation accuracy   : {s['avg_citation_accuracy']:.2f} / 5")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run unit test to verify it passes**

Run: `uv run pytest tests/test_eval_run.py -v`
Expected: 1 passed

- [ ] **Step 5: Write the regression smoke test (threshold floor)**

`tests/test_integration_eval.py`:

```python
"""Eval smoke test: scores must stay above a floor to catch regressions.

Prereqs: docker compose up -d, real keys, and the golden papers ingested:
  uv run python -c "
from rag.arxiv_client import get_paper
from rag.ingest import ingest_paper
from rag.store import VectorStore
store = VectorStore(); store.ping(); store.ensure_collection()
for pid in ['1706.03762', '1810.04805', '2005.11401']:
    ingest_paper(get_paper(pid), store)
"

Run: uv run pytest tests/test_integration_eval.py -m integration -v
"""

import pytest

pytestmark = pytest.mark.integration


def test_eval_scores_stay_above_floor(tmp_path):
    from eval.run import run_eval

    report = run_eval(report_path=str(tmp_path / "report.json"))
    s = report["summary"]
    assert s["avg_recall"] >= 0.5, s
    assert s["avg_faithfulness"] >= 3.5, s
    assert s["avg_relevance"] >= 3.5, s
    assert s["avg_citation_accuracy"] >= 3.0, s
```

Run: `uv run pytest tests/test_integration_eval.py -v`
Expected: `1 deselected`

- [ ] **Step 6: Commit**

```bash
git add eval/run.py tests/test_eval_run.py tests/test_integration_eval.py
git commit -m "feat: eval runner with report.json and regression smoke test"
```

---

### Task 18: FastAPI app + static frontend + README

**Files:**
- Create: `api/main.py`, `api/static/index.html`, `api/static/app.js`, `README.md`
- Delete: `api/static/.gitkeep`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `agents.graph.run_agent`, `rag.ingest.{ingest_query, IngestResult}`, `rag.store.VectorStore`.
- Produces:
  - `api.main.app` — FastAPI app; `POST /api/chat {"message": str} → {"reply": str}`; `POST /api/ingest {"query": str, "max_results": int} → IngestResult`; `/` serves the static page. Startup pings Qdrant (fail fast).
  - Run: `uv run uvicorn api.main:app --reload`

- [ ] **Step 1: Write the failing tests**

`tests/test_api.py`:

```python
def _client(monkeypatch):
    import api.main as api_main
    from fastapi.testclient import TestClient

    class FakeStore:
        def ping(self):
            pass

    monkeypatch.setattr(api_main, "VectorStore", FakeStore)
    return TestClient(api_main.app)


def test_chat_calls_agent(monkeypatch):
    import api.main as api_main

    async def fake_run_agent(question):
        return f"echo: {question}"

    monkeypatch.setattr(api_main, "run_agent", fake_run_agent)
    with _client(monkeypatch) as client:
        resp = client.post("/api/chat", json={"message": "what is attention?"})
    assert resp.status_code == 200
    assert resp.json() == {"reply": "echo: what is attention?"}


def test_ingest_endpoint(monkeypatch):
    import api.main as api_main
    from rag.ingest import IngestResult

    captured = {}

    def fake_ingest(query, max_results):
        captured.update(query=query, max_results=max_results)
        return IngestResult(ingested=["1706.03762"], skipped=[])

    monkeypatch.setattr(api_main, "ingest_query", fake_ingest)
    with _client(monkeypatch) as client:
        resp = client.post("/api/ingest", json={"query": "attention", "max_results": 2})
    assert resp.status_code == 200
    assert resp.json() == {"ingested": ["1706.03762"], "skipped": []}
    assert captured == {"query": "attention", "max_results": 2}


def test_startup_fails_fast_when_qdrant_down(monkeypatch):
    import pytest

    import api.main as api_main
    from fastapi.testclient import TestClient

    class DownStore:
        def ping(self):
            raise RuntimeError("Qdrant is not reachable")

    monkeypatch.setattr(api_main, "VectorStore", DownStore)
    with pytest.raises(RuntimeError, match="not reachable"):
        with TestClient(api_main.app):
            pass


def test_index_served(monkeypatch):
    with _client(monkeypatch) as client:
        resp = client.get("/")
    assert resp.status_code == 200
    assert "Paper Research Assistant" in resp.text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_api.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'api.main'`

- [ ] **Step 3: Write api/main.py**

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agents.graph import run_agent
from rag.ingest import IngestResult, ingest_query
from rag.store import VectorStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    VectorStore().ping()  # fail fast at startup if Qdrant is down (docker compose up -d)
    yield


app = FastAPI(title="Paper Research Assistant", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


class IngestRequest(BaseModel):
    query: str
    max_results: int = 3


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    reply = await run_agent(req.message)
    return ChatResponse(reply=reply)


@app.post("/api/ingest", response_model=IngestResult)
async def ingest(req: IngestRequest) -> IngestResult:
    # ingest_query is blocking (network + embeddings); keep the event loop free.
    return await run_in_threadpool(ingest_query, req.query, req.max_results)


# Mounted last so /api/* wins routing; html=True serves index.html at /.
app.mount("/", StaticFiles(directory="api/static", html=True), name="static")
```

- [ ] **Step 4: Write the static frontend**

`api/static/index.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Paper Research Assistant</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 760px; margin: 2rem auto; padding: 0 1rem; }
    h1 { font-size: 1.4rem; }
    fieldset { margin-bottom: 1.5rem; border: 1px solid #ccc; border-radius: 6px; }
    input[type=text] { width: 70%; padding: .5rem; }
    button { padding: .5rem 1rem; }
    #log { border: 1px solid #ccc; border-radius: 6px; padding: 1rem; min-height: 200px; white-space: pre-wrap; }
    .user { color: #005; font-weight: 600; }
    .bot { color: #050; }
    .status { color: #777; font-style: italic; }
  </style>
</head>
<body>
  <h1>Paper Research Assistant</h1>

  <fieldset>
    <legend>Ingest papers</legend>
    <input id="ingest-query" type="text" placeholder="arXiv search, e.g. attention is all you need">
    <button id="ingest-btn">Ingest</button>
    <div id="ingest-status" class="status"></div>
  </fieldset>

  <fieldset>
    <legend>Chat</legend>
    <input id="chat-input" type="text" placeholder="Ask about the ingested papers…">
    <button id="chat-btn">Send</button>
  </fieldset>

  <div id="log"></div>

  <script src="/app.js"></script>
</body>
</html>
```

`api/static/app.js`:

```javascript
const log = document.getElementById("log");

function append(cls, text) {
  const div = document.createElement("div");
  div.className = cls;
  div.textContent = text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

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

document.getElementById("chat-btn").addEventListener("click", async () => {
  const input = document.getElementById("chat-input");
  const message = input.value.trim();
  if (!message) return;
  input.value = "";
  append("user", `You: ${message}`);
  append("status", "thinking…");
  try {
    const result = await post("/api/chat", { message });
    log.lastChild.remove();
    append("bot", `Assistant: ${result.reply}`);
  } catch (err) {
    log.lastChild.remove();
    append("status", `Chat failed: ${err.message}`);
  }
});
```

Delete `api/static/.gitkeep`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_api.py -v`
Expected: 4 passed

- [ ] **Step 6: Run the full unit suite**

Run: `uv run pytest -v`
Expected: all unit tests pass; integration tests deselected

- [ ] **Step 7: Write README.md**

```markdown
# Paper Research Assistant

Learning project covering LLM APIs + prompting, RAG, evaluation, and agents + MCP.
Ingests arXiv papers, answers questions grounded in them with [paper_id] citations,
and autonomously fetches papers it doesn't have yet.

## Setup

```bash
uv sync
cp .env.example .env   # add ANTHROPIC_API_KEY and OPENAI_API_KEY
docker compose up -d   # Qdrant on localhost:6333
```

## Use

```bash
# Web UI (chat + ingest) at http://localhost:8000
uv run uvicorn api.main:app --reload

# Ingest from the CLI
uv run python -c "from rag.ingest import ingest_query; print(ingest_query('attention is all you need'))"

# Offline eval -> report.json + printed summary
uv run python -m eval.run
```

## Tests

```bash
uv run pytest                  # unit tests (mocked, no keys needed)
uv run pytest -m integration   # real APIs; needs keys, Qdrant, network, uvx
```

## Layout

- `llm/` — provider abstraction (Anthropic + OpenAI), prompts, structured output, prompt caching
- `rag/` — arXiv fetch, PDF parse, chunk, embed, Qdrant store, retrieve, answer
- `agents/` — LangGraph agent; custom MCP server (`python -m agents.mcp_server`); MCP client (also consumes `mcp-server-fetch`)
- `eval/` — golden dataset, LLM judge, retrieval metrics, report generator
- `api/` — FastAPI routes + static frontend

Imports flow one way: `api → agents → rag/llm`; `eval → rag/agents/llm`.
```

- [ ] **Step 8: Commit**

```bash
git add api/main.py api/static/index.html api/static/app.js README.md
git rm api/static/.gitkeep
git commit -m "feat: fastapi app with chat/ingest endpoints and static frontend"
```

---

## Self-Review (done at plan-writing time)

**Spec coverage:** provider abstraction + prompting techniques (T2–T4: system prompt design, few-shot citations, structured output via pydantic, `cache_control` caching demo); RAG pipeline exactly as the spec's data flow (T5–T11); agent decides rag_query / custom MCP fetch-then-retry / external fetch (T12–T14); eval with golden dataset, judge, precision/recall, `report.json`, standalone `python -m eval.run`, threshold smoke test (T15–T17); FastAPI + vanilla frontend (T18). Error handling: SDK retry-with-backoff (T2), PDF skip-log-continue (T7, T10), MCP errors as tool results (T12–T14), Qdrant fail-fast (T9, T12, T18). All "Out of Scope" items excluded.

**Known simplifications (deliberate):** `section` metadata field stored but always empty (pypdf gives no section boundaries); eval runner retrieves twice per question (once for metrics, once inside `answer_question`); cache hits only occur when repeat questions retrieve identical top-k context.

**Type consistency:** `generate()` signature, `LLMResponse`/`ToolCall`, canonical message/tool formats, `ScoredChunk`/`ChunkRecord`/`PaperMeta`/`IngestResult`/`RagAnswer`/`JudgeScores` names verified consistent across all tasks.
