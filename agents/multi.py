"""Multi-agent supervisor: planner → researcher (per sub-question) → synthesizer.

Enabled with agent_mode=multi (config.py); run_chat is the dispatcher the API
calls. The planner may return simple=True and fall through to the single-agent
loop (which keeps thread memory). Researchers run the existing agent loop
sequentially, each on a fresh single-shot thread — multi mode itself keeps no
conversation memory. A failed sub-question is reported to the synthesizer,
which answers from what remains.
"""

import asyncio
import logging

from pydantic import BaseModel, Field

from agents.graph import AgentResult, _dedupe, run_agent
from config import settings
from llm.base import generate, generate_stream
from llm.prompts import PLANNER_SYSTEM_PROMPT, SYNTHESIZER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class Plan(BaseModel):
    simple: bool = Field(description="True when the question needs no decomposition")
    sub_questions: list[str] = Field(default_factory=list, max_length=4)


def _plan(question: str, provider: str | None = None) -> Plan:
    resp = generate([{"role": "user", "content": question}],
                    system=PLANNER_SYSTEM_PROMPT, structured_schema=Plan,
                    provider=provider)
    return resp.parsed


def _synthesize(question: str, findings: list[tuple[str, str]],
                provider: str | None = None, on_delta=None) -> str:
    parts = [f"Sub-question: {sq}\nFinding: {answer}" for sq, answer in findings]
    content = f"Question: {question}\n\n" + "\n\n---\n\n".join(parts)
    messages = [{"role": "user", "content": content}]
    if on_delta is None:
        resp = generate(messages, system=SYNTHESIZER_SYSTEM_PROMPT, provider=provider)
    else:
        resp = generate_stream(messages, system=SYNTHESIZER_SYSTEM_PROMPT,
                               provider=provider, on_delta=on_delta)
    return resp.text


async def run_multi_agent(question: str, thread_id: str | None = None,
                          provider: str | None = None, on_event=None) -> AgentResult:
    if on_event is not None:
        on_event({"event": "status", "text": "planning…"})
    plan = await asyncio.to_thread(_plan, question, provider)
    if plan.simple or not plan.sub_questions:
        return await run_agent(question, thread_id, provider=provider,
                               on_event=on_event)
    findings: list[tuple[str, str]] = []
    citations: list[str] = []
    for sub_question in plan.sub_questions[:4]:
        if on_event is not None:
            on_event({"event": "status", "text": f"researching: {sub_question}"})
        try:
            # researchers run silently — only the synthesizer token-streams
            result = await run_agent(sub_question, provider=provider)
            findings.append((sub_question, result.text))
            citations.extend(result.citations)
        except Exception as exc:
            logger.exception("Researcher failed for %r", sub_question)
            findings.append((sub_question, f"FAILED: {exc}"))
    on_delta = (lambda t: on_event({"event": "delta", "text": t})) if on_event else None
    text = await asyncio.to_thread(_synthesize, question, findings, provider, on_delta)
    if on_event is not None:
        on_event({"event": "turn_end", "has_tools": False})
    # Researchers ran on fresh single-shot threads; nothing was checkpointed
    # under the caller's thread_id, so there is no transcript to restore.
    return AgentResult(text=text, citations=_dedupe(citations), checkpointed=False)


async def run_chat(message: str, thread_id: str | None = None,
                   provider: str | None = None, on_event=None) -> AgentResult:
    """Dispatch on agent_mode: the single loop (default) or the supervisor."""
    if settings.agent_mode == "multi":
        return await run_multi_agent(message, thread_id, provider=provider,
                                     on_event=on_event)
    return await run_agent(message, thread_id, provider=provider, on_event=on_event)
