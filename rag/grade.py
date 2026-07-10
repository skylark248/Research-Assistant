"""LLM relevance grading of retrieved chunks (corrective RAG, phase 5).

One batched LLM call grades every chunk. Line-format output ("1: yes") rather
than JSON — small local models follow it far more reliably. Fails open: any
LLM or parse failure keeps all chunks — a broken grader must never make
retrieval worse than no grader.
"""

import logging
import re

from llm.base import generate
from llm.prompts import GRADE_SYSTEM_PROMPT
from rag.store import ScoredChunk

logger = logging.getLogger(__name__)

_VERDICT_LINE = re.compile(r"^\s*(\d+)\s*[:.)\-]\s*(yes|no)\b",
                           re.IGNORECASE | re.MULTILINE)


def _build_prompt(question: str, chunks: list[ScoredChunk]) -> str:
    parts = [f"{i}. [paper {c.paper_id} — {c.title}]\n{c.text}"
             for i, c in enumerate(chunks, start=1)]
    return f"Question: {question}\n\nExcerpts:\n\n" + "\n\n---\n\n".join(parts)


def grade_chunks(question: str, chunks: list[ScoredChunk],
                 provider: str | None = None) -> list[ScoredChunk]:
    """Keep only the chunks the grader marks relevant, original order preserved.

    Fail-open: a chunk with a missing/unparseable verdict passes; a grader
    exception or fully unparseable output returns all chunks unchanged.
    """
    if not chunks:
        return []
    try:
        resp = generate(
            [{"role": "user", "content": _build_prompt(question, chunks)}],
            system=GRADE_SYSTEM_PROMPT, provider=provider,
        )
    except Exception:
        logger.warning("Chunk grading failed; keeping all chunks", exc_info=True)
        return chunks
    verdicts = {int(n): v.lower() == "yes" for n, v in _VERDICT_LINE.findall(resp.text)}
    if not verdicts:
        logger.warning("Grader output unparseable; keeping all chunks: %r",
                       resp.text[:200])
        return chunks
    return [c for i, c in enumerate(chunks, start=1) if verdicts.get(i, True)]
