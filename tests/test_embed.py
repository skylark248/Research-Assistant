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
