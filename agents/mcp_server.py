"""Custom MCP server exposing arXiv tools over stdio.

Run: python -m agents.mcp_server
Tool errors are returned as "Error: ..." strings (tool results), never raised —
the agent decides whether to retry or give up (spec: no silent crash).
"""

import json

from mcp.server.fastmcp import FastMCP

from rag.arxiv_client import get_paper, search_papers
from rag.ingest import ingest_paper
from rag.store import VectorStore

mcp = FastMCP("arxiv")


@mcp.tool()
def arxiv_search(query: str, max_results: int = 5) -> str:
    """Search arXiv for papers. Returns a JSON list of {paper_id, title, summary}."""
    papers = search_papers(query, max_results=max_results)
    return json.dumps([p.model_dump() for p in papers])


@mcp.tool()
def arxiv_fetch_paper(paper_id: str) -> str:
    """Download an arXiv paper by id and ingest it into the vector store so
    rag_query can answer questions about it."""
    meta = get_paper(paper_id)
    if meta is None:
        return f"Error: no arXiv paper found with id {paper_id}"
    store = VectorStore()
    try:
        store.ping()
    except RuntimeError as exc:
        return f"Error: {exc}"
    store.ensure_collection()
    n = ingest_paper(meta, store)
    if n is None:
        return f"Error: failed to download or parse {paper_id}"
    if n == 0:
        return f"{paper_id} was already ingested: {meta.title}"
    return f"Ingested {paper_id} ({n} chunks): {meta.title}"


if __name__ == "__main__":
    mcp.run()  # stdio transport
