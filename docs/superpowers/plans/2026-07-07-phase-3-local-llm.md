# Phase 3: Local LLM Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fully local execution path — LLM provider "local" (Qwen 2.5 3B via Ollama's OpenAI-compatible endpoint) and local dense embeddings (fastembed bge-small) — so the whole system runs keyless on an 8 GB M1 MacBook Air.

**Architecture:** Ollama speaks the OpenAI protocol, so the existing `llm/openai_client.py` adapter is parameterized (optional `client`/`model`) and a thin `llm/local_client.py` points it at `http://localhost:11434/v1`. `rag/embed.py` gains an `embedding_provider` switch (local = fastembed `TextEmbedding`, same lazy-singleton pattern as sparse/rerank). `embedding_dim` becomes provider-derived; `VectorStore.check_schema` additionally compares the stored dense-vector size so an embedding-provider switch fails fast with the migrate message.

**Tech Stack:** openai SDK (existing) against Ollama `/v1`, fastembed `TextEmbedding` (existing dep), pydantic `model_validator`, new pytest marker `local`.

**Spec:** `docs/superpowers/specs/2026-07-07-phase-3-local-llm-design.md`

## Global Constraints

- Cloud providers stay untouched and remain defaults: `llm_provider="anthropic"`, `embedding_provider="openai"`. Local is opt-in via `.env`. No automatic fallback.
- Exact new defaults: `local_base_url="http://localhost:11434/v1"`, `local_model="qwen2.5:3b"`, `local_embedding_model="BAAI/bge-small-en-v1.5"`, derived `embedding_dim`: 1536 (openai) / 384 (local), explicit override honored.
- No new Python dependencies — openai SDK and fastembed are already installed.
- Unit tests keyless + offline (mock SDK classes / model singletons exactly like the existing suite). Real-model tests go behind a NEW pytest marker `local` (needs running Ollama, no keys), deselected by default alongside `integration`.
- Full unit suite (`uv run pytest`) 100% green before every commit.
- Commit style: `type: lowercase summary`.
- `uv` for everything.

---

### Task 1: Config — provider settings + derived embedding_dim

**Files:**
- Modify: `config.py`
- Test: `tests/test_config.py` (append)

**Interfaces:**
- Produces (later tasks consume by exact name): `llm_provider` accepts `"local"`; `local_base_url: str`; `local_model: str`; `embedding_provider: Literal["openai", "local"]`; `local_embedding_model: str`; `embedding_dim: int | None` derived after validation (1536 openai / 384 local, explicit value wins).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config.py`:

```python
def test_phase3_defaults():
    from config import Settings

    s = Settings(_env_file=None)
    assert s.llm_provider == "anthropic"  # cloud default unchanged
    assert s.local_base_url == "http://localhost:11434/v1"
    assert s.local_model == "qwen2.5:3b"
    assert s.embedding_provider == "openai"
    assert s.local_embedding_model == "BAAI/bge-small-en-v1.5"


def test_embedding_dim_derived_from_provider():
    from config import Settings

    assert Settings(_env_file=None).embedding_dim == 1536
    assert Settings(_env_file=None, embedding_provider="local").embedding_dim == 384
    # explicit override beats derivation
    assert Settings(_env_file=None, embedding_provider="local", embedding_dim=999).embedding_dim == 999


def test_local_is_valid_llm_provider():
    from config import Settings

    assert Settings(_env_file=None, llm_provider="local").llm_provider == "local"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: new tests FAIL (`AttributeError: 'Settings' object has no attribute 'local_base_url'`; `ValidationError` for `llm_provider="local"`).

- [ ] **Step 3: Implement in `config.py`**

Change the pydantic import line to include `model_validator`:

```python
from pydantic import model_validator
```

(add below the existing `from pydantic_settings import ...` line — `pydantic` is already an installed transitive dependency).

Update the LLM block:

```python
    # LLM
    llm_provider: Literal["anthropic", "openai", "local"] = "anthropic"
    anthropic_model: str = "claude-opus-4-8"
    openai_model: str = "gpt-5"
    llm_max_tokens: int = 4096
    llm_max_retries: int = 4  # SDK retries 429/5xx with exponential backoff

    # Local LLM (Ollama's OpenAI-compatible endpoint; qwen2.5:3b fits an 8GB M1)
    local_base_url: str = "http://localhost:11434/v1"
    local_model: str = "qwen2.5:3b"
```

Replace the Embeddings block:

```python
    # Embeddings
    embedding_provider: Literal["openai", "local"] = "openai"
    embedding_model: str = "text-embedding-3-small"
    local_embedding_model: str = "BAAI/bge-small-en-v1.5"  # fastembed ONNX, 384-dim
    embedding_dim: int | None = None  # derived from provider below; explicit value wins

    @model_validator(mode="after")
    def _derive_embedding_dim(self) -> "Settings":
        if self.embedding_dim is None:
            self.embedding_dim = 1536 if self.embedding_provider == "openai" else 384
        return self
```

(The validator lives inside the `Settings` class body, directly after the embeddings fields. Existing `test_defaults` keeps passing: derivation yields 1536 for the openai default, same as before.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (all).

Run: `uv run pytest`
Expected: PASS (full suite — `embedding_dim` still resolves to 1536 everywhere by default).

- [ ] **Step 5: Commit**

```bash
git add config.py tests/test_config.py
git commit -m "feat: add local provider settings and provider-derived embedding_dim"
```

---

### Task 2: LLM provider "local" — parameterized OpenAI adapter + thin local client

**Files:**
- Modify: `llm/openai_client.py` (function signature only)
- Create: `llm/local_client.py`
- Modify: `llm/base.py` (dispatch branch + docstring line)
- Test: `tests/test_openai_adapter.py` (append), `tests/test_client_config.py` (append), `tests/test_llm_base.py` (append)

**Interfaces:**
- Consumes: `settings.local_base_url`, `settings.local_model` (Task 1).
- Produces: `generate_openai(..., client=None, model=None)` — omitted args = current behavior exactly; `llm.local_client.generate_local(messages, *, system=None, tools=None, structured_schema=None, max_tokens=4096) -> LLMResponse`; `generate(..., provider="local")` routes to it.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_openai_adapter.py`:

```python
def test_generate_openai_accepts_client_and_model_overrides():
    from types import SimpleNamespace

    from llm.openai_client import generate_openai

    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="local hi", tool_calls=None),
                                         finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    resp = generate_openai([{"role": "user", "content": "hi"}],
                           client=fake_client, model="qwen2.5:3b")

    assert resp.text == "local hi"
    assert captured["model"] == "qwen2.5:3b"  # override, not settings.openai_model
```

Append to `tests/test_client_config.py`:

```python
def test_local_client_points_at_ollama(monkeypatch):
    import llm.local_client as lc
    from config import settings

    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(lc, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(lc, "_client", None)
    lc._get_client()
    assert captured["base_url"] == settings.local_base_url
    assert captured["api_key"] == "ollama"  # Ollama ignores it; SDK requires one
    assert captured["max_retries"] == settings.llm_max_retries
```

Append to `tests/test_llm_base.py`:

```python
def test_local_provider_routes_to_local_client(monkeypatch):
    import llm.local_client as lc
    from llm.base import LLMResponse, generate

    def fake_generate_local(messages, **kwargs):
        return LLMResponse(text="from local")

    monkeypatch.setattr(lc, "generate_local", fake_generate_local)

    resp = generate([{"role": "user", "content": "hi"}], provider="local")
    assert resp.text == "from local"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_openai_adapter.py tests/test_client_config.py tests/test_llm_base.py -v`
Expected: FAIL — `TypeError: generate_openai() got an unexpected keyword argument 'client'`; `ModuleNotFoundError: No module named 'llm.local_client'`; `ValueError: Unknown provider: local`.

- [ ] **Step 3: Parameterize `generate_openai`**

In `llm/openai_client.py`, replace the `generate_openai` signature and first kwargs lines:

```python
def generate_openai(
    messages: list[dict],
    *,
    system: str | list[dict] | None = None,
    tools: list[dict] | None = None,
    structured_schema: type[BaseModel] | None = None,
    max_tokens: int = 4096,
    client=None,
    model: str | None = None,
) -> LLMResponse:
    client = client or _get_client()
    kwargs: dict = {
        "model": model or settings.openai_model,
        "messages": convert_messages(messages, system),
        "max_completion_tokens": max_tokens,
    }
```

(The rest of the function body is unchanged — it already uses the `client` local variable.)

- [ ] **Step 4: Create `llm/local_client.py`**

```python
"""Local provider: Ollama's OpenAI-compatible /v1 endpoint, reusing the OpenAI adapter.

Ollama ignores the API key but the SDK requires one — "ollama" by convention.
A down Ollama server surfaces as the SDK's connection error naming the base_url;
no special handling (same fail-loud policy as the cloud clients).
"""

from openai import OpenAI
from pydantic import BaseModel

from config import settings
from llm.base import LLMResponse
from llm.openai_client import generate_openai

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=settings.local_base_url, api_key="ollama",
                         max_retries=settings.llm_max_retries)
    return _client


def generate_local(
    messages: list[dict],
    *,
    system: str | list[dict] | None = None,
    tools: list[dict] | None = None,
    structured_schema: type[BaseModel] | None = None,
    max_tokens: int = 4096,
) -> LLMResponse:
    return generate_openai(
        messages, system=system, tools=tools, structured_schema=structured_schema,
        max_tokens=max_tokens, client=_get_client(), model=settings.local_model,
    )
```

- [ ] **Step 5: Add the dispatch branch in `llm/base.py`**

After the `"openai"` branch, before the `raise ValueError` line:

```python
    if provider == "local":
        from llm.local_client import generate_local

        return generate_local(
            messages, system=system, tools=tools,
            structured_schema=structured_schema, max_tokens=max_tokens,
        )
```

Also update the docstring's provider note — replace the first docstring line of `generate`:

```python
    """Provider-neutral chat entrypoint (anthropic | openai | local/Ollama).
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_openai_adapter.py tests/test_client_config.py tests/test_llm_base.py -v`
Expected: PASS (all, old and new).

Run: `uv run pytest`
Expected: PASS (full suite).

- [ ] **Step 7: Commit**

```bash
git add llm/openai_client.py llm/local_client.py llm/base.py tests/test_openai_adapter.py tests/test_client_config.py tests/test_llm_base.py
git commit -m "feat: add local llm provider via ollama openai-compatible endpoint"
```

---

### Task 3: Local dense embeddings in `rag/embed.py`

**Files:**
- Modify: `rag/embed.py`
- Test: `tests/test_embed.py` (append)

**Interfaces:**
- Consumes: `settings.embedding_provider`, `settings.local_embedding_model` (Task 1).
- Produces: `embed_texts`/`embed_query` signatures unchanged; when `embedding_provider == "local"` they use a lazy-singleton fastembed `TextEmbedding` (`_get_local_model()`, `_local_model` module global — the names the tests patch).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_embed.py`:

```python
class FakeDenseModel:
    """Mimics fastembed.TextEmbedding: embed() yields numpy arrays."""

    def __init__(self, dim=4):
        self.dim = dim
        self.calls = []

    def embed(self, texts):
        import numpy as np

        self.calls.append(list(texts))
        for i, _ in enumerate(texts):
            yield np.array([float(i)] * self.dim)


def test_embed_texts_local_provider(monkeypatch):
    import rag.embed as embed
    from config import settings

    monkeypatch.setattr(settings, "embedding_provider", "local")
    fake = FakeDenseModel()
    monkeypatch.setattr(embed, "_local_model", fake)

    vectors = embed.embed_texts(["a", "b"])
    assert vectors == [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]  # arrays -> lists
    assert fake.calls == [["a", "b"]]


def test_embed_query_local_provider(monkeypatch):
    import rag.embed as embed
    from config import settings

    monkeypatch.setattr(settings, "embedding_provider", "local")
    monkeypatch.setattr(embed, "_local_model", FakeDenseModel(dim=3))

    assert embed.embed_query("q") == [0.0, 0.0, 0.0]


def test_embed_texts_openai_path_untouched_by_flag(monkeypatch):
    """Default provider still goes through the OpenAI client, never the local model."""
    import rag.embed as embed
    from config import settings

    monkeypatch.setattr(settings, "embedding_provider", "openai")

    class Boom:
        def embed(self, texts):
            raise AssertionError("local model must not be used for provider=openai")

    monkeypatch.setattr(embed, "_local_model", Boom())

    from types import SimpleNamespace

    class FakeEmbeddings:
        def create(self, model, input):
            return SimpleNamespace(data=[SimpleNamespace(embedding=[0.5]) for _ in input])

    monkeypatch.setattr(embed, "_client", SimpleNamespace(embeddings=FakeEmbeddings()))
    assert embed.embed_texts(["a"]) == [[0.5]]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_embed.py -v`
Expected: new tests FAIL with `AttributeError: <module 'rag.embed'> does not have the attribute '_local_model'`.

- [ ] **Step 3: Rewrite `rag/embed.py`**

```python
"""Dense embeddings: OpenAI (default) or local fastembed, per settings.embedding_provider.

Switching providers changes the vector dimension (1536 vs 384) — the Qdrant
collection must be recreated (python -m rag.migrate --yes) and papers re-ingested.
"""

from fastembed import TextEmbedding
from openai import OpenAI

from config import settings

_client: OpenAI | None = None
_local_model: TextEmbedding | None = None

BATCH_SIZE = 100  # texts per embeddings API request


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.openai_api_key or None, max_retries=settings.llm_max_retries)
    return _client


def _get_local_model() -> TextEmbedding:
    global _local_model
    if _local_model is None:
        _local_model = TextEmbedding(model_name=settings.local_embedding_model)
    return _local_model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts in order; provider chosen by settings.embedding_provider."""
    if not texts:
        return []
    if settings.embedding_provider == "local":
        return [vector.tolist() for vector in _get_local_model().embed(texts)]
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

Note: the local path in `embed_texts` calls `_get_local_model()`, but the tests patch `_local_model` directly — that works because `_get_local_model()` returns the patched instance when `_local_model is not None` (same convention as `rag/sparse.py` / `rag/rerank.py` tests).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_embed.py tests/test_client_config.py -v`
Expected: PASS (all — `test_embed_client_gets_key_from_settings` in test_client_config still passes since `_get_client` is unchanged).

Run: `uv run pytest`
Expected: PASS (full suite).

- [ ] **Step 5: Commit**

```bash
git add rag/embed.py tests/test_embed.py
git commit -m "feat: local dense embeddings via fastembed behind embedding_provider flag"
```

---

### Task 4: Dimension guard in `VectorStore.check_schema`

**Files:**
- Modify: `rag/store.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Consumes: `settings.embedding_dim` (derived, Task 1); existing `check_schema` / `LEGACY_SCHEMA_MESSAGE`.
- Produces: `check_schema()` additionally raises `RuntimeError` (message names `rag.migrate`, both dim values) when the stored dense-vector size differs from `settings.embedding_dim`. New module constant `DIM_MISMATCH_MESSAGE`.

- [ ] **Step 1: Update the fake + write the failing test**

In `tests/test_store.py`, make `FakeQdrant`'s dense size configurable — replace the `__init__` and `get_collection` methods:

```python
    def __init__(self, exists=False, hits=None, fail=False, legacy=False, dense_size=1536):
        self._exists = exists
        self._hits = hits or []
        self._fail = fail
        self._legacy = legacy
        self._dense_size = dense_size
        self.created = []
        self.upserted = []
        self.deleted = []
        self.queries = []
```

```python
    def get_collection(self, name):
        if self._legacy:
            # phase-1 schema: single unnamed dense vector, no sparse vectors
            params = SimpleNamespace(vectors=SimpleNamespace(size=1536), sparse_vectors=None)
        else:
            params = SimpleNamespace(vectors={"dense": SimpleNamespace(size=self._dense_size)},
                                     sparse_vectors={"bm25": SimpleNamespace()})
        return SimpleNamespace(config=SimpleNamespace(params=params))
```

Append the new test:

```python
def test_check_schema_rejects_dimension_mismatch(monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "embedding_dim", 384)  # e.g. switched to local embeddings
    fake = FakeQdrant(exists=True, dense_size=1536)      # collection built with openai dims
    with pytest.raises(RuntimeError, match="1536.*384|384.*1536"):
        _store(fake).check_schema()


def test_check_schema_accepts_matching_dimension(monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "embedding_dim", 384)
    fake = FakeQdrant(exists=True, dense_size=384)
    _store(fake).check_schema()  # no raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_store.py -v`
Expected: `test_check_schema_rejects_dimension_mismatch` FAILS (`DID NOT RAISE`); everything else passes.

- [ ] **Step 3: Implement the guard**

In `rag/store.py`, add below `LEGACY_SCHEMA_MESSAGE`:

```python
DIM_MISMATCH_MESSAGE = (
    "Collection '{collection}' stores {found}-dim dense vectors but the configured "
    "embedding provider expects {expected}-dim. Recreate it with: "
    "uv run python -m rag.migrate --yes (then re-ingest papers)"
)
```

Extend `check_schema` — after the existing legacy check, add:

```python
        dense = params.vectors[DENSE_VECTOR]
        if dense.size != settings.embedding_dim:
            raise RuntimeError(DIM_MISMATCH_MESSAGE.format(
                collection=self.collection, found=dense.size,
                expected=settings.embedding_dim,
            ))
```

(The legacy check already guarantees `params.vectors` is a dict containing `DENSE_VECTOR` before this line runs.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_store.py -v`
Expected: PASS (all).

Run: `uv run pytest`
Expected: PASS (full suite — default runs use matching 1536/1536).

- [ ] **Step 5: Commit**

```bash
git add rag/store.py tests/test_store.py
git commit -m "feat: fail fast on embedding dimension mismatch in check_schema"
```

---

### Task 5: `local` pytest marker + real-model test file

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/test_local_llm.py`

**Interfaces:**
- Consumes: `generate(..., provider="local")` (Task 2), local embeddings (Task 3).
- Produces: marker `local`; `uv run pytest -m local` runs real-model tests against Ollama; default runs deselect them.

- [ ] **Step 1: Register the marker and extend addopts**

In `pyproject.toml` `[tool.pytest.ini_options]`, replace the `markers` list and `addopts` line:

```toml
markers = [
    "integration: needs real API keys and a running Qdrant; run with `pytest -m integration`",
    "local: needs a running Ollama server (no API keys); run with `pytest -m local`",
]
addopts = "-m 'not integration and not local'"
```

- [ ] **Step 2: Create `tests/test_local_llm.py`**

```python
"""Real-model tests against a local Ollama server — no API keys.

Run: uv run pytest -m local
Needs: `ollama serve` with `ollama pull qwen2.5:3b` done (first pull ~1.9GB);
test_local_embeddings downloads the fastembed model (~130MB) on first run.
A 3B model is small: the tool-call and structured tests use deliberately
unambiguous prompts, but occasional flakes are expected — rerun before
treating a failure as a regression.
"""

import pytest
from pydantic import BaseModel

pytestmark = pytest.mark.local


def test_generate_round_trip():
    from llm.base import generate

    resp = generate([{"role": "user", "content": "Reply with exactly one word: hello"}],
                    provider="local", max_tokens=50)
    assert resp.text.strip()


def test_structured_output():
    from llm.base import generate

    class Verdict(BaseModel):
        answer: str
        confident: bool

    resp = generate(
        [{"role": "user", "content": "What is the capital of France? Answer confidently."}],
        provider="local", structured_schema=Verdict, max_tokens=100,
    )
    assert resp.parsed is not None
    assert "paris" in resp.parsed.answer.lower()


def test_tool_call():
    from llm.base import generate

    tools = [{
        "name": "get_current_time",
        "description": "Returns the current time. Use this whenever asked about the time.",
        "input_schema": {"type": "object", "properties": {}},
    }]
    resp = generate(
        [{"role": "user", "content": "What time is it right now? Use the tool."}],
        provider="local", tools=tools, max_tokens=100,
    )
    assert resp.tool_calls, f"expected a tool call, got text: {resp.text!r}"
    assert resp.tool_calls[0].name == "get_current_time"


def test_local_embeddings_shape(monkeypatch):
    import rag.embed as embed
    from config import settings

    monkeypatch.setattr(settings, "embedding_provider", "local")
    monkeypatch.setattr(embed, "_local_model", None)  # force real construction

    vectors = embed.embed_texts(["hybrid retrieval", "sparse vectors"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 384
    assert vectors[0] != vectors[1]
```

- [ ] **Step 3: Verify deselection + collection**

Run: `uv run pytest tests/test_local_llm.py`
Expected: `4 deselected` — no leak into unit runs.

Run: `uv run pytest tests/test_local_llm.py --collect-only -q -m local`
Expected: 4 tests collected, no errors.

Run: `uv run pytest`
Expected: full suite green, `14 deselected` total (10 integration + 4 local).

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml tests/test_local_llm.py
git commit -m "test: add local marker and real-model ollama tests"
```

---

### Task 6: README + .env.example

**Files:**
- Modify: `README.md`
- Modify: `.env.example`

**Interfaces:**
- Consumes: everything above; documents the keyless path.

- [ ] **Step 1: Update `.env.example`**

Replace the whole file with:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
LLM_PROVIDER=anthropic

# Fully local, no API keys (needs Ollama running — see README):
# LLM_PROVIDER=local
# EMBEDDING_PROVIDER=local
```

- [ ] **Step 2: Add the README section**

In `README.md`, insert a new section between `## Setup` and `## Use`:

```markdown
## Fully local (no API keys)

Runs the whole system — ingest, cited chat, memory, multi-agent, eval — on a
local model. Fits an 8GB M1 MacBook Air.

```bash
brew install ollama
ollama pull qwen2.5:3b                      # ~1.9GB
OLLAMA_CONTEXT_LENGTH=8192 ollama serve     # grounded prompts need >4k context
```

Set in `.env`: `LLM_PROVIDER=local` and `EMBEDDING_PROVIDER=local`.

Switching the embedding provider changes vector dimensions (1536 → 384), so
recreate the collection and re-ingest: `uv run python -m rag.migrate --yes`.

Notes: `qwen2.5:3b` is the safe default — `qwen2.5:7b` (~4.7GB) answers better
but only fits with Docker/browser mostly closed. A 3B model follows citation
and JSON-schema instructions less reliably than the cloud models; the eval
harness quantifies the gap. Real-model tests: `uv run pytest -m local`.
```

Also update the Tests section to mention the new marker — replace the `## Tests` code block with:

```markdown
```bash
uv run pytest                  # unit tests (mocked, no keys needed)
uv run pytest -m integration   # real cloud APIs; needs keys, Qdrant, network, uvx
uv run pytest -m local         # real local model; needs Ollama running, no keys
```
```

- [ ] **Step 3: Verify + commit**

Run: `uv run pytest`
Expected: full suite green (docs-only change).

```bash
git add README.md .env.example
git commit -m "docs: document fully local no-keys setup"
```

---

## Post-plan validation (on this machine, today — no keys needed)

1. `brew install ollama && ollama pull qwen2.5:3b && OLLAMA_CONTEXT_LENGTH=8192 ollama serve`
2. `uv run pytest -m local` — real-model round trip, structured output, tool call, embeddings
3. `.env`: `LLM_PROVIDER=local`, `EMBEDDING_PROVIDER=local`; `docker compose up -d`; `uv run python -m rag.migrate --yes`
4. Ingest a paper via the UI, chat with citations, verify multi-turn memory
5. `uv run python -m eval.run --ablation` — first fully keyless ablation table
