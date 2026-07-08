from types import SimpleNamespace


class FakeEmbeddings:
    def __init__(self):
        self.calls = []

    def create(self, model, input):
        self.calls.append({"model": model, "input": input})
        return SimpleNamespace(data=[SimpleNamespace(embedding=[float(len(t))]) for t in input])


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
