"""Post-answer citation-faithfulness guardrail (phase 5).

One LLM call asks whether the cited excerpts support the answer's claims.
Verdicts: True (supported), False (unsupported), None (check errored or
output unparseable). Advisory only — a failed check never fails the request.
"""

import logging

from llm.base import generate
from llm.prompts import FAITHFULNESS_SYSTEM_PROMPT, format_context

logger = logging.getLogger(__name__)


def check_faithfulness(question: str, answer: str, contexts: list[dict],
                       provider: str | None = None) -> bool | None:
    prompt = (f"Paper excerpts:\n\n{format_context(contexts)}\n\n"
              f"Question: {question}\n\nAnswer:\n{answer}")
    try:
        resp = generate([{"role": "user", "content": prompt}],
                        system=FAITHFULNESS_SYSTEM_PROMPT, provider=provider)
    except Exception:
        logger.warning("Faithfulness check failed", exc_info=True)
        return None
    words = resp.text.strip().lower().split()
    token = words[0].strip(".,!—:;") if words else ""
    if token == "yes":
        return True
    if token == "no":
        return False
    logger.warning("Faithfulness verdict unparseable: %r", resp.text[:100])
    return None
