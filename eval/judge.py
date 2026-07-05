"""LLM-as-judge scorer. No eval framework — a structured-output call and a rubric."""

from pydantic import BaseModel, Field

from llm.base import generate


class JudgeScores(BaseModel):
    faithfulness: int = Field(
        ge=1, le=5, description="Is every claim in the answer supported by the context? 5 = fully grounded, 1 = mostly fabricated."
    )
    relevance: int = Field(
        ge=1, le=5, description="Does the answer actually address the question? 5 = directly and completely."
    )
    citation_accuracy: int = Field(
        ge=1, le=5, description="Do the [paper_id] citations point at context excerpts that support the cited claims? 5 = all correct."
    )
    reasoning: str = Field(description="One short paragraph justifying the scores.")


JUDGE_SYSTEM_PROMPT = """You are a strict evaluator of a research assistant's answers.
Score each dimension 1-5 (5 = perfect). Judge faithfulness ONLY against the provided
context excerpts, and citation accuracy against the paper ids appearing in them.
The reference gist describes what a good answer should convey — use it for relevance."""


def judge_answer(question: str, answer: str, expected_gist: str,
                 contexts: list[dict]) -> JudgeScores:
    context_text = "\n\n".join(f"[{c['paper_id']}] {c['text']}" for c in contexts)
    user = f"""Question: {question}

Reference gist (what a good answer should convey):
{expected_gist}

Context excerpts the assistant had:
{context_text}

Assistant answer to evaluate:
{answer}"""
    resp = generate([{"role": "user", "content": user}],
                    system=JUDGE_SYSTEM_PROMPT, structured_schema=JudgeScores)
    return resp.parsed
