import logging

from pydantic import BaseModel

from rag.arxiv_client import PaperMeta, download_pdf, search_papers
from rag.chunk import chunk_text
from rag.embed import embed_texts
from rag.parse import extract_text
from rag.store import ChunkRecord, VectorStore

logger = logging.getLogger(__name__)


class IngestResult(BaseModel):
    ingested: list[str] = []
    skipped: list[str] = []


def ingest_paper(meta: PaperMeta, store: VectorStore) -> int | None:
    """Download, parse, chunk, embed, and upsert one paper.

    Returns the number of chunks upserted, 0 if the paper was already
    ingested, or None if it had to be skipped (download/parse failure).
    """
    if store.has_paper(meta.paper_id):
        logger.info("Already ingested %s", meta.paper_id)
        return 0
    try:
        pdf_path = download_pdf(meta.paper_id)
    except Exception:
        logger.exception("Download failed for %s, skipping", meta.paper_id)
        return None
    text = extract_text(pdf_path)
    if text is None:
        return None  # extract_text already logged the reason
    chunks = chunk_text(text)
    vectors = embed_texts(chunks)
    records = [
        ChunkRecord(paper_id=meta.paper_id, title=meta.title,
                    chunk_index=i, text=chunk, vector=vector)
        for i, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]
    store.upsert_chunks(records)
    logger.info("Ingested %s (%d chunks)", meta.paper_id, len(records))
    return len(records)


def ingest_query(query: str, max_results: int = 3,
                 store: VectorStore | None = None) -> IngestResult:
    """Search arXiv and ingest the results. Failures skip the paper, not the batch."""
    store = store or VectorStore()
    store.ping()  # fail fast before doing any network work
    store.ensure_collection()
    result = IngestResult()
    for meta in search_papers(query, max_results=max_results):
        n = ingest_paper(meta, store)
        if n is None:
            result.skipped.append(meta.paper_id)
        else:
            result.ingested.append(meta.paper_id)
    return result
