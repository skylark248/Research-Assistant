import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agents.multi import run_chat
from api.providers import ProviderStatus, check_provider, check_providers
from api.threads import (ThreadInfo, TranscriptTurn, delete_thread, get_transcript,
                         list_threads, upsert_thread)
from rag.ingest import IngestResult, ingest_query
from rag.store import VectorStore

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = VectorStore()
    store.ping()  # fail fast at startup if Qdrant is down (docker compose up -d)
    store.check_schema()  # fail fast on a phase-1 collection or an embedding-dim mismatch (rag.migrate)
    yield


app = FastAPI(title="Paper Research Assistant", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None  # omit to start a new conversation
    provider: Literal["anthropic", "openai", "local"] | None = None  # None → configured default


class ChatResponse(BaseModel):
    reply: str
    thread_id: str
    citations: list[str] = []


class IngestRequest(BaseModel):
    query: str
    max_results: int = 3


def _root_error(exc: BaseException) -> BaseException:
    """Innermost real exception — agent errors arrive wrapped in TaskGroup
    ExceptionGroups whose str() ("unhandled errors in a TaskGroup") hides the cause."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return exc


async def _require_available(provider: str | None) -> None:
    if provider is None:
        return
    status = await run_in_threadpool(check_provider, provider)
    if not status.available:
        raise HTTPException(status_code=400, detail=status.detail)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    await _require_available(req.provider)
    thread_id = req.thread_id or str(uuid.uuid4())
    result = await run_chat(req.message, thread_id, provider=req.provider)
    await run_in_threadpool(upsert_thread, thread_id, req.message)
    return ChatResponse(reply=result.text, thread_id=thread_id,
                        citations=result.citations)


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """SSE: status/delta/turn_end events while the agent works, then done|error."""
    await _require_available(req.provider)
    thread_id = req.thread_id or str(uuid.uuid4())
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def on_event(event: dict) -> None:
        # Called from worker threads (generate_stream runs in to_thread).
        loop.call_soon_threadsafe(queue.put_nowait, event)

    async def worker() -> None:
        try:
            result = await run_chat(req.message, thread_id,
                                    provider=req.provider, on_event=on_event)
            await run_in_threadpool(upsert_thread, thread_id, req.message)
            await queue.put({"event": "done", "reply": result.text,
                             "thread_id": thread_id, "citations": result.citations})
        except Exception as exc:
            logger.exception("chat stream failed")
            await queue.put({"event": "error", "message": str(_root_error(exc))})
        await queue.put(None)  # sentinel: stream complete

    async def sse():
        task = asyncio.create_task(worker())
        try:
            while (event := await queue.get()) is not None:
                name = event.pop("event")
                yield f"event: {name}\ndata: {json.dumps(event)}\n\n"
        finally:
            await task

    return StreamingResponse(sse(), media_type="text/event-stream")


@app.post("/api/ingest", response_model=IngestResult)
async def ingest(req: IngestRequest) -> IngestResult:
    # ingest_query is blocking (network + embeddings); keep the event loop free.
    return await run_in_threadpool(ingest_query, req.query, req.max_results)


@app.get("/api/providers", response_model=list[ProviderStatus])
async def providers() -> list[ProviderStatus]:
    # check_providers may block up to 1.5s probing Ollama; keep the loop free.
    return await run_in_threadpool(check_providers)


@app.get("/api/threads", response_model=list[ThreadInfo])
async def threads() -> list[ThreadInfo]:
    return await run_in_threadpool(list_threads)


@app.get("/api/threads/{thread_id}", response_model=list[TranscriptTurn])
async def thread_transcript(thread_id: str) -> list[TranscriptTurn]:
    turns = await get_transcript(thread_id)
    if turns is None:
        raise HTTPException(status_code=404, detail="unknown thread")
    return turns


@app.delete("/api/threads/{thread_id}")
async def remove_thread(thread_id: str) -> dict:
    await run_in_threadpool(delete_thread, thread_id)
    return {"deleted": thread_id}


# Mounted last so /api/* wins routing; html=True serves index.html at /.
app.mount("/", StaticFiles(directory="api/static", html=True), name="static")
