import logging

from pydantic import BaseModel

from llm.base import generate
from llm.prompts import build_rag_prompt
from rag.retrieve import retrieve
from rag.store import VectorStore

logger = logging.getLogger(__name__)


class RagAnswer(BaseModel):
    text: str
    sources: list[str]


def answer_question(question: str, store: VectorStore | None = None) -> RagAnswer:
    """RAG query flow: embed → retrieve → grounded prompt → generate."""
    chunks = retrieve(question, store=store)
    if not chunks:
        return RagAnswer(
            text="I don't have any ingested papers to answer from yet. "
                 "Ingest some papers first.",
            sources=[],
        )
    contexts = [{"paper_id": c.paper_id, "title": c.title, "text": c.text} for c in chunks]
    system, messages = build_rag_prompt(question, contexts)
    resp = generate(messages, system=system)
    logger.info(
        "answer usage: cache_read=%s cache_creation=%s",
        resp.usage.get("cache_read_input_tokens"),
        resp.usage.get("cache_creation_input_tokens"),
    )
    return RagAnswer(text=resp.text, sources=sorted({c.paper_id for c in chunks}))
