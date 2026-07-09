import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agents.multi import run_chat
from api.providers import ProviderStatus, check_provider, check_providers
from api.threads import (ThreadInfo, TranscriptTurn, delete_thread, get_transcript,
                         list_threads, upsert_thread)
from rag.ingest import IngestResult, ingest_query
from rag.store import VectorStore


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
