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
