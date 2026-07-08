import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agents.multi import run_chat
from api.providers import ProviderStatus, check_provider, check_providers
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


# Mounted last so /api/* wins routing; html=True serves index.html at /.
app.mount("/", StaticFiles(directory="api/static", html=True), name="static")
