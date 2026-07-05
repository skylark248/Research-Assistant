from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agents.graph import run_agent
from rag.ingest import IngestResult, ingest_query
from rag.store import VectorStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    VectorStore().ping()  # fail fast at startup if Qdrant is down (docker compose up -d)
    yield


app = FastAPI(title="Paper Research Assistant", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


class IngestRequest(BaseModel):
    query: str
    max_results: int = 3


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    reply = await run_agent(req.message)
    return ChatResponse(reply=reply)


@app.post("/api/ingest", response_model=IngestResult)
async def ingest(req: IngestRequest) -> IngestResult:
    # ingest_query is blocking (network + embeddings); keep the event loop free.
    return await run_in_threadpool(ingest_query, req.query, req.max_results)


# Mounted last so /api/* wins routing; html=True serves index.html at /.
app.mount("/", StaticFiles(directory="api/static", html=True), name="static")
