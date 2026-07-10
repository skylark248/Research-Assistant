"""LLM query rewriting: question → focused search query (used for retrieval only;
answer generation still sees the original question).

Fails open: any LLM error logs a warning and returns the original question —
retrieval must never break because rewriting broke.
"""

import logging

from pydantic import BaseModel

from llm.base import generate
from llm.prompts import RETRY_REWRITE_SYSTEM_PROMPT, REWRITE_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class RewrittenQuery(BaseModel):
    query: str


def rewrite_query(question: str) -> str:
    try:
        resp = generate(
            [{"role": "user", "content": question}],
            system=REWRITE_SYSTEM_PROMPT,
            structured_schema=RewrittenQuery,
        )
        rewritten = resp.parsed.query.strip()
        return rewritten or question
    except Exception:
        logger.warning("Query rewrite failed; using the original question", exc_info=True)
        return question


def retry_rewrite_query(question: str, provider: str | None = None) -> str:
    """Alternative query after grading rejected every retrieved chunk (phase 5).

    Fails open to the original question — the caller detects that (retry_query
    == question) and skips a retry that would just repeat itself.
    """
    try:
        resp = generate(
            [{"role": "user", "content": question}],
            system=RETRY_REWRITE_SYSTEM_PROMPT,
            structured_schema=RewrittenQuery,
            provider=provider,
        )
        rewritten = resp.parsed.query.strip()
        return rewritten or question
    except Exception:
        logger.warning("Retry rewrite failed; using the original question",
                       exc_info=True)
        return question
