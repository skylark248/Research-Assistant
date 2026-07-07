"""Pins the seam where `.env`-loaded settings must reach SDK clients.

These tests do NOT mock `_get_client` away — they patch the SDK class itself
so the real `_get_client()` body runs and we can assert on the kwargs it
actually passed to the SDK constructor.
"""


def test_anthropic_client_gets_key_from_settings(monkeypatch):
    import llm.anthropic_client as ac
    from config import settings

    captured = {}

    class FakeAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(ac.anthropic, "Anthropic", FakeAnthropic)
    monkeypatch.setattr(ac, "_client", None)  # reset singleton; auto-restored
    monkeypatch.setattr(settings, "anthropic_api_key", "sk-test-123")
    ac._get_client()
    assert captured["api_key"] == "sk-test-123"
    assert captured["max_retries"] == settings.llm_max_retries


def test_anthropic_client_empty_key_becomes_none(monkeypatch):
    import llm.anthropic_client as ac
    from config import settings

    captured = {}

    class FakeAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(ac.anthropic, "Anthropic", FakeAnthropic)
    monkeypatch.setattr(ac, "_client", None)
    monkeypatch.setattr(settings, "anthropic_api_key", "")
    ac._get_client()
    assert captured["api_key"] is None


def test_openai_client_gets_key_from_settings(monkeypatch):
    import llm.openai_client as oc
    from config import settings

    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(oc, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(oc, "_client", None)
    monkeypatch.setattr(settings, "openai_api_key", "sk-test-456")
    oc._get_client()
    assert captured["api_key"] == "sk-test-456"
    assert captured["max_retries"] == settings.llm_max_retries


def test_openai_client_empty_key_becomes_none(monkeypatch):
    import llm.openai_client as oc
    from config import settings

    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(oc, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(oc, "_client", None)
    monkeypatch.setattr(settings, "openai_api_key", "")
    oc._get_client()
    assert captured["api_key"] is None


def test_embed_client_gets_key_from_settings(monkeypatch):
    import rag.embed as embed
    from config import settings

    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(embed, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(embed, "_client", None)
    monkeypatch.setattr(settings, "openai_api_key", "sk-test-789")
    embed._get_client()
    assert captured["api_key"] == "sk-test-789"
    assert captured["max_retries"] == settings.llm_max_retries


def test_embed_client_empty_key_becomes_none(monkeypatch):
    import rag.embed as embed
    from config import settings

    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(embed, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(embed, "_client", None)
    monkeypatch.setattr(settings, "openai_api_key", "")
    embed._get_client()
    assert captured["api_key"] is None


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
