import json


async def test_tool_schemas_registered():
    from agents.mcp_server import mcp

    tools = await mcp.list_tools()
    names = {t.name for t in tools}
    assert names == {"arxiv_search", "arxiv_fetch_paper"}

    search = next(t for t in tools if t.name == "arxiv_search")
    assert "query" in search.inputSchema["properties"]
    assert "query" in search.inputSchema.get("required", [])

    fetch = next(t for t in tools if t.name == "arxiv_fetch_paper")
    assert "paper_id" in fetch.inputSchema["properties"]


def test_arxiv_search_returns_json(monkeypatch):
    import agents.mcp_server as srv
    from rag.arxiv_client import PaperMeta

    monkeypatch.setattr(
        srv, "search_papers",
        lambda query, max_results: [PaperMeta(paper_id="1706.03762", title="Attention",
                                              summary="s")],
    )
    out = json.loads(srv.arxiv_search("attention", max_results=1))
    assert out == [{"paper_id": "1706.03762", "title": "Attention", "summary": "s"}]


def test_fetch_paper_ingests(monkeypatch):
    import agents.mcp_server as srv
    from rag.arxiv_client import PaperMeta

    meta = PaperMeta(paper_id="1706.03762", title="Attention", summary="s")
    monkeypatch.setattr(srv, "get_paper", lambda pid: meta)
    monkeypatch.setattr(srv, "ingest_paper", lambda m, store: 42)

    class FakeStore:
        def ping(self):
            pass

        def ensure_collection(self):
            pass

    monkeypatch.setattr(srv, "VectorStore", FakeStore)

    out = srv.arxiv_fetch_paper("1706.03762")
    assert "1706.03762" in out and "42" in out


def test_fetch_paper_errors_are_strings(monkeypatch):
    import agents.mcp_server as srv

    monkeypatch.setattr(srv, "get_paper", lambda pid: None)
    assert srv.arxiv_fetch_paper("0000.00000").startswith("Error:")
