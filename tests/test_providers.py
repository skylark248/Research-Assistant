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
