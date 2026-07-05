from types import SimpleNamespace


def _fake_page(text):
    return SimpleNamespace(extract_text=lambda: text)


def test_extracts_and_joins_pages(monkeypatch):
    import rag.parse as parse

    monkeypatch.setattr(
        parse, "PdfReader",
        lambda path: SimpleNamespace(pages=[_fake_page("page one"), _fake_page("page two")]),
    )
    assert parse.extract_text("x.pdf") == "page one\npage two"


def test_handles_none_page_text(monkeypatch):
    import rag.parse as parse

    monkeypatch.setattr(
        parse, "PdfReader",
        lambda path: SimpleNamespace(pages=[_fake_page(None), _fake_page("real text")]),
    )
    assert parse.extract_text("x.pdf") == "real text"


def test_parse_failure_returns_none(monkeypatch, caplog):
    import rag.parse as parse

    def _boom(path):
        raise ValueError("corrupt pdf")

    monkeypatch.setattr(parse, "PdfReader", _boom)
    assert parse.extract_text("bad.pdf") is None
    assert "bad.pdf" in caplog.text


def test_empty_pdf_returns_none(monkeypatch):
    import rag.parse as parse

    monkeypatch.setattr(parse, "PdfReader",
                        lambda path: SimpleNamespace(pages=[_fake_page("")]))
    assert parse.extract_text("empty.pdf") is None
