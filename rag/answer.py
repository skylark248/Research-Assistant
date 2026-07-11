import logging
from typing import Callable

from pydantic import BaseModel

from config import settings
from llm.base import generate
from llm.prompts import build_rag_prompt
from rag.faithfulness import check_faithfulness
from rag.grade import grade_chunks
from rag.retrieve import retrieve
from rag.rewrite import retry_rewrite_query
from rag.store import VectorStore

logger = logging.getLogger(__name__)

EMPTY_STORE_ANSWER = ("I don't have any ingested papers to answer from yet. "
                      "Ingest some papers first.")
NO_INFO_ANSWER = ("I don't have enough information in the ingested papers to "
                  "answer this. Try ingesting more papers on the topic.")


class RagAnswer(BaseModel):
    text: str
    sources: list[str]
    faithful: bool | None = None  # None = check disabled, skipped, or failed


def answer_question(
    question: str, store: VectorStore | None = None, provider: str | None = None,
    on_status: Callable[[str], None] | None = None,
) -> RagAnswer:
    """RAG query flow: retrieve → [grade → retry once] → grounded prompt →
    generate → [faithfulness].

    `provider` threads through the phase-5 LLM calls (grading, retry rewrite,
    faithfulness) and the final generate; retrieval's rewrite stage stays on
    the global setting. `on_status` (optional) receives human-readable progress
    lines for the UI activity feed. Guardrails fail open — see rag/grade.py
    and rag/faithfulness.py.
    """
    notify = on_status or (lambda text: None)
    chunks = retrieve(question, store=store)
    if not chunks:
        return RagAnswer(text=EMPTY_STORE_ANSWER, sources=[])

    if settings.grading_enabled:
        notify(f"grading {len(chunks)} chunks…")
        graded = grade_chunks(question, chunks, provider=provider)
        notify(f"{len(graded)} of {len(chunks)} chunks relevant")
        if not graded:
            retry_query = retry_rewrite_query(question, provider=provider)
            if retry_query != question:  # rewrite failed open → retry would repeat
                notify("retrying with rewritten query…")
                retried = retrieve(retry_query, store=store)
                # graded against the ORIGINAL question, like reranking
                graded = grade_chunks(question, retried, provider=provider)
                notify(f"{len(graded)} of {len(retried)} chunks relevant")
        if not graded:
            return RagAnswer(text=NO_INFO_ANSWER, sources=[])
        chunks = graded

    contexts = [{"paper_id": c.paper_id, "title": c.title, "text": c.text}
                for c in chunks]
    system, messages = build_rag_prompt(question, contexts)
    resp = generate(messages, system=system, provider=provider)
    logger.info(
        "answer usage: cache_read=%s cache_creation=%s",
        resp.usage.get("cache_read_input_tokens"),
        resp.usage.get("cache_creation_input_tokens"),
    )
    faithful = None
    if settings.faithfulness_enabled:
        notify("verifying citations…")
        faithful = check_faithfulness(question, resp.text, contexts,
                                      provider=provider)
    return RagAnswer(text=resp.text, sources=sorted({c.paper_id for c in chunks}),
                     faithful=faithful)
