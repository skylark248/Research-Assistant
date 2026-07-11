import pytest


@pytest.fixture(autouse=True)
def _isolated_checkpoint_db(monkeypatch, tmp_path):
    """Chat endpoint upserts thread rows; keep unit runs out of data/checkpoints.db."""
    import api.threads as threads_mod

    monkeypatch.setattr(threads_mod.settings, "checkpoint_db", str(tmp_path / "cp.db"))


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


def test_chat_generates_thread_id(monkeypatch):
    import api.main as api_main
    from agents.graph import AgentResult

    async def fake_run_chat(question, thread_id=None, provider=None):
        return AgentResult(text=f"echo: {question} [{thread_id}]", citations=[])

    monkeypatch.setattr(api_main, "run_chat", fake_run_chat)
    with _client(monkeypatch) as client:
        resp = client.post("/api/chat", json={"message": "what is attention?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["thread_id"]  # server minted one
    assert body["reply"] == f"echo: what is attention? [{body['thread_id']}]"


def test_chat_reuses_given_thread_id(monkeypatch):
    import api.main as api_main
    from agents.graph import AgentResult

    async def fake_run_chat(question, thread_id=None, provider=None):
        return AgentResult(text=f"echo: {question} [{thread_id}]", citations=[])

    monkeypatch.setattr(api_main, "run_chat", fake_run_chat)
    with _client(monkeypatch) as client:
        resp = client.post("/api/chat", json={"message": "follow-up", "thread_id": "t-42"})
    assert resp.json() == {"reply": "echo: follow-up [t-42]", "thread_id": "t-42", "citations": [], "faithful": None}


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
    from agents.graph import AgentResult

    captured = {}

    async def fake_run_chat(question, thread_id=None, provider=None):
        captured["provider"] = provider
        return AgentResult(text="ok", citations=[])

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


def test_chat_returns_citations(monkeypatch):
    import api.main as api_main
    from agents.graph import AgentResult

    async def fake_run_chat(question, thread_id=None, provider=None):
        return AgentResult(text="grounded", citations=["1706.03762"])

    monkeypatch.setattr(api_main, "run_chat", fake_run_chat)
    with _client(monkeypatch) as client:
        resp = client.post("/api/chat", json={"message": "hi"})
    assert resp.json()["citations"] == ["1706.03762"]


def test_thread_endpoints(monkeypatch):
    import api.main as api_main
    from agents.graph import AgentResult

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


def test_chat_skips_thread_upsert_when_not_checkpointed(monkeypatch):
    import api.main as api_main
    from agents.graph import AgentResult

    async def fake_run_chat(question, thread_id=None, provider=None):
        return AgentResult(text="synth", citations=[], checkpointed=False)

    monkeypatch.setattr(api_main, "run_chat", fake_run_chat)
    with _client(monkeypatch) as client:
        resp = client.post("/api/chat", json={"message": "decomposed multi question"})
        assert resp.status_code == 200
        assert client.get("/api/threads").json() == []


def test_chat_returns_faithful(monkeypatch):
    import api.main as api_main
    from agents.graph import AgentResult

    async def fake_run_chat(question, thread_id=None, provider=None):
        return AgentResult(text="grounded", citations=["1706.03762"], faithful=False)

    monkeypatch.setattr(api_main, "run_chat", fake_run_chat)
    with _client(monkeypatch) as client:
        resp = client.post("/api/chat", json={"message": "hi"})
    assert resp.json()["faithful"] is False
