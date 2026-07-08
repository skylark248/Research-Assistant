import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agents.multi import run_chat
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


class ChatResponse(BaseModel):
    reply: str
    thread_id: str


class IngestRequest(BaseModel):
    query: str
    max_results: int = 3


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    thread_id = req.thread_id or str(uuid.uuid4())
    reply = await run_chat(req.message, thread_id)
    return ChatResponse(reply=reply, thread_id=thread_id)


@app.post("/api/ingest", response_model=IngestResult)
async def ingest(req: IngestRequest) -> IngestResult:
    # ingest_query is blocking (network + embeddings); keep the event loop free.
    return await run_in_threadpool(ingest_query, req.query, req.max_results)


# Mounted last so /api/* wins routing; html=True serves index.html at /.
app.mount("/", StaticFiles(directory="api/static", html=True), name="static")
