"""LLM query rewriting: question → focused search query (used for retrieval only;
answer generation still sees the original question).

Fails open: any LLM error logs a warning and returns the original question —
retrieval must never break because rewriting broke.
"""

import logging

from pydantic import BaseModel

from llm.base import generate
from llm.prompts import REWRITE_SYSTEM_PROMPT

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
