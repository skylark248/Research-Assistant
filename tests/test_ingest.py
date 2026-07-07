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
    from rag.sparse import SparseVector

    monkeypatch.setattr(ingest, "download_pdf", lambda pid: f"/tmp/{pid}.pdf")
    monkeypatch.setattr(ingest, "extract_text", lambda path: text)
    monkeypatch.setattr(ingest, "chunk_text", lambda t: ["chunk a", "chunk b"])
    monkeypatch.setattr(ingest, "embed_texts", lambda chunks: [[0.1], [0.2]])
    monkeypatch.setattr(ingest, "sparse_embed_texts",
                        lambda chunks: [SparseVector(indices=[1], values=[1.0]),
                                        SparseVector(indices=[2], values=[1.0])])


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
    assert records[0].sparse.indices == [1]
    assert records[1].sparse.indices == [2]


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
