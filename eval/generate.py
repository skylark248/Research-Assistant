"""Synthetic eval-item generator: uv run python -m eval.generate --count 50

Samples ingested chunks, has the LLM write one exam question + expected gist
per chunk, then self-checks each candidate against its source chunk. The
filter is FAIL-CLOSED — a candidate is dropped on a "no" verdict, a parse
failure, or any LLM error. That is the inverse of the phase-5 runtime
guardrails (fail-open): a dropped candidate costs one retry, but a bad item
that slips through poisons every future metric run.

Output: eval/golden-synthetic.json (full overwrite — regeneration is
stateless). eval.run picks the file up automatically when it exists.
"""

import argparse
import json
import logging
import re
from pathlib import Path

from pydantic import BaseModel

from llm.base import generate
from llm.prompts import SYNTH_CHECK_SYSTEM_PROMPT, SYNTH_QUESTION_SYSTEM_PROMPT
from rag.store import VectorStore

logger = logging.getLogger(__name__)

DEFAULT_OUT_PATH = "eval/golden-synthetic.json"

_CHECK_LINE = re.compile(r"^\s*(answerable|faithful)\s*[:=]\s*(yes|no)\b",
                         re.IGNORECASE | re.MULTILINE)


class Candidate(BaseModel):
    question: str
    expected_answer_gist: str


def _generate_candidate(chunk: dict, provider: str | None) -> Candidate | None:
    prompt = f"[paper {chunk['paper_id']} — {chunk['title']}]\n{chunk['text']}"
    try:
        resp = generate([{"role": "user", "content": prompt}],
                        system=SYNTH_QUESTION_SYSTEM_PROMPT,
                        structured_schema=Candidate, provider=provider)
        candidate = resp.parsed
        if not candidate.question.strip() or not candidate.expected_answer_gist.strip():
            return None
        return candidate
    except Exception:
        logger.warning("Candidate generation failed for %s; dropping",
                       chunk["paper_id"], exc_info=True)
        return None


def _self_check(chunk: dict, candidate: Candidate, provider: str | None) -> bool:
    """Fail-closed: only an explicit yes on BOTH verdicts keeps the item."""
    user = (f"Excerpt:\n{chunk['text']}\n\n"
            f"Question: {candidate.question}\n\n"
            f"Expected gist: {candidate.expected_answer_gist}")
    try:
        resp = generate([{"role": "user", "content": user}],
                        system=SYNTH_CHECK_SYSTEM_PROMPT, provider=provider)
    except Exception:
        logger.warning("Self-check failed; dropping candidate", exc_info=True)
        return False
    verdicts = {k.lower(): v.lower() == "yes"
                for k, v in _CHECK_LINE.findall(resp.text)}
    return verdicts.get("answerable") is True and verdicts.get("faithful") is True


def generate_dataset(count: int, provider: str | None = None, seed: int = 0,
                     out_path: str = DEFAULT_OUT_PATH, store=None) -> dict:
    store = store or VectorStore()
    store.ping()  # fail fast with a clear message when Qdrant is down
    store.check_schema()
    # Over-sample 3x: rejections are expected, exhaustion is handled below.
    chunks = store.sample_chunks(count * 3, seed=seed)
    if not chunks:
        raise RuntimeError("No ingested chunks to generate from — "
                           "ingest some papers first.")
    kept: list[dict] = []
    rejected = 0
    for chunk in chunks:
        if len(kept) >= count:
            break
        candidate = _generate_candidate(chunk, provider)
        if candidate is None or not _self_check(chunk, candidate, provider):
            rejected += 1
            continue
        kept.append({
            "question": candidate.question.strip(),
            "expected_paper_ids": [chunk["paper_id"]],
            "expected_answer_gist": candidate.expected_answer_gist.strip(),
            "synthetic": True,
        })
    Path(out_path).write_text(json.dumps(kept, indent=2))
    return {"kept": len(kept), "rejected": rejected, "requested": count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=50,
                        help="target number of kept synthetic items")
    parser.add_argument("--provider", choices=["anthropic", "openai", "local"],
                        default=None, help="LLM provider (default: configured)")
    parser.add_argument("--seed", type=int, default=0,
                        help="sampling seed (reproducible chunk draw)")
    args = parser.parse_args()
    stats = generate_dataset(count=args.count, provider=args.provider,
                             seed=args.seed)
    print(f"kept {stats['kept']} / requested {stats['requested']} "
          f"({stats['rejected']} rejected) -> {DEFAULT_OUT_PATH}")
    if stats["kept"] < stats["requested"]:
        print("Chunk supply exhausted before reaching the target — "
              "ingest more papers or lower --count.")


if __name__ == "__main__":
    main()
