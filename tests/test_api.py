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

    async def fake_run_agent(question, thread_id=None):
        return f"echo: {question} [{thread_id}]"

    monkeypatch.setattr(api_main, "run_agent", fake_run_agent)
    with _client(monkeypatch) as client:
        resp = client.post("/api/chat", json={"message": "what is attention?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["thread_id"]  # server minted one
    assert body["reply"] == f"echo: what is attention? [{body['thread_id']}]"


def test_chat_reuses_given_thread_id(monkeypatch):
    import api.main as api_main

    async def fake_run_agent(question, thread_id=None):
        return f"echo: {question} [{thread_id}]"

    monkeypatch.setattr(api_main, "run_agent", fake_run_agent)
    with _client(monkeypatch) as client:
        resp = client.post("/api/chat", json={"message": "follow-up", "thread_id": "t-42"})
    assert resp.json() == {"reply": "echo: follow-up [t-42]", "thread_id": "t-42"}


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
