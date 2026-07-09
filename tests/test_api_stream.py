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
