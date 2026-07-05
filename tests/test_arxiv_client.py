from types import SimpleNamespace

import pytest


def _fake_result(short_id="2405.10098v2", title="T", summary="S"):
    r = SimpleNamespace(title=title, summary=summary)
    r.get_short_id = lambda: short_id
    r.download_pdf = lambda dirpath, filename: f"{dirpath}/{filename}"
    return r


class _FakeArxivClient:
    def __init__(self, results):
        self._results = results
        self.searches = []

    def results(self, search):
        self.searches.append(search)
        return iter(self._results)


def test_search_papers_strips_version(monkeypatch):
    import rag.arxiv_client as axc

    fake = _FakeArxivClient([_fake_result("2405.10098v2", "Paper A", "About A")])
    monkeypatch.setattr(axc, "_client", lambda: fake)

    papers = axc.search_papers("attention", max_results=1)
    assert len(papers) == 1
    assert papers[0].paper_id == "2405.10098"
    assert papers[0].title == "Paper A"
    assert fake.searches[0].query == "attention"
    assert fake.searches[0].max_results == 1


def test_get_paper_found_and_missing(monkeypatch):
    import rag.arxiv_client as axc

    fake = _FakeArxivClient([_fake_result("1706.03762v7", "Attention", "S")])
    monkeypatch.setattr(axc, "_client", lambda: fake)
    meta = axc.get_paper("1706.03762")
    assert meta is not None and meta.paper_id == "1706.03762"
    assert fake.searches[0].id_list == ["1706.03762"]

    monkeypatch.setattr(axc, "_client", lambda: _FakeArxivClient([]))
    assert axc.get_paper("0000.00000") is None


def test_download_pdf(monkeypatch, tmp_path):
    import rag.arxiv_client as axc
    from config import settings

    monkeypatch.setattr(settings, "pdf_dir", str(tmp_path / "pdfs"))
    fake = _FakeArxivClient([_fake_result("1706.03762v7")])
    monkeypatch.setattr(axc, "_client", lambda: fake)

    path = axc.download_pdf("1706.03762")
    assert path.endswith("1706.03762.pdf")
    assert (tmp_path / "pdfs").is_dir()  # created eagerly


def test_download_pdf_unknown_id(monkeypatch):
    import rag.arxiv_client as axc

    monkeypatch.setattr(axc, "_client", lambda: _FakeArxivClient([]))
    with pytest.raises(ValueError, match="No arXiv paper"):
        axc.download_pdf("0000.00000")
