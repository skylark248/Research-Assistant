import re
from pathlib import Path

import arxiv
import requests
from pydantic import BaseModel

from config import settings


class PaperMeta(BaseModel):
    paper_id: str
    title: str
    summary: str


def _client() -> arxiv.Client:
    return arxiv.Client()


def _short_id(result) -> str:
    # "2405.10098v2" -> "2405.10098" (version-free ids keep dedup simple)
    return re.sub(r"v\d+$", "", result.get_short_id())


def _to_meta(result) -> PaperMeta:
    return PaperMeta(paper_id=_short_id(result), title=result.title, summary=result.summary)


def search_papers(query: str, max_results: int = 5) -> list[PaperMeta]:
    search = arxiv.Search(query=query, max_results=max_results)
    return [_to_meta(r) for r in _client().results(search)]


def get_paper(paper_id: str) -> PaperMeta | None:
    search = arxiv.Search(id_list=[paper_id])
    results = list(_client().results(search))
    return _to_meta(results[0]) if results else None


def download_pdf(paper_id: str) -> str:
    """Downloads the PDF to settings.pdf_dir and returns the file path."""
    search = arxiv.Search(id_list=[paper_id])
    results = list(_client().results(search))
    if not results:
        raise ValueError(f"No arXiv paper found for id {paper_id}")
    Path(settings.pdf_dir).mkdir(parents=True, exist_ok=True)
    # arxiv 4.0 removed Result.download_pdf; fetch pdf_url ourselves.
    path = Path(settings.pdf_dir) / f"{paper_id}.pdf"
    resp = requests.get(results[0].pdf_url, timeout=60)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return str(path)
