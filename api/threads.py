"""Thread bookkeeping for the UI.

A small `threads` table lives in the same SQLite file as LangGraph's
checkpoint tables (one DB to manage); LangGraph's own tables are never
written here. Transcripts are read from the latest checkpoint, so there is
no duplicate message storage.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel

from config import settings


class ThreadInfo(BaseModel):
    thread_id: str
    title: str
    created_at: str
    updated_at: str


class TranscriptTurn(BaseModel):
    role: str  # "user" | "assistant"
    text: str
    citations: list[str] = []  # populated on the last assistant turn only


def _connect() -> sqlite3.Connection:
    Path(settings.checkpoint_db).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.checkpoint_db)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS threads (
               thread_id  TEXT PRIMARY KEY,
               title      TEXT NOT NULL,
               created_at TEXT NOT NULL,
               updated_at TEXT NOT NULL)"""
    )
    return conn


def upsert_thread(thread_id: str, first_message: str) -> None:
    """Insert with title on first sight; later calls only bump updated_at."""
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            """INSERT INTO threads (thread_id, title, created_at, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(thread_id) DO UPDATE SET updated_at = excluded.updated_at""",
            (thread_id, first_message[:80], now, now),
        )


def list_threads() -> list[ThreadInfo]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT thread_id, title, created_at, updated_at FROM threads "
            "ORDER BY updated_at DESC"
        ).fetchall()
    return [ThreadInfo(thread_id=r[0], title=r[1], created_at=r[2], updated_at=r[3])
            for r in rows]


def delete_thread(thread_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM threads WHERE thread_id = ?", (thread_id,))
        for table in ("checkpoints", "writes"):  # LangGraph's tables
            try:
                conn.execute(f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))
            except sqlite3.OperationalError:
                pass  # table not created yet (no real chat has run)


def _turns_from_messages(messages: list[dict],
                         citations: list[str] | None = None) -> list[TranscriptTurn]:
    """Plain-text turns only; tool_use/tool_result traffic is omitted.

    The thread's accumulated citations attach to the last assistant turn —
    same semantics as the live `done` event (whole-conversation sources).
    """
    turns: list[TranscriptTurn] = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            if message["role"] == "user" and content:
                turns.append(TranscriptTurn(role="user", text=content))
        elif message["role"] == "assistant":
            texts = [b["text"] for b in content if b["type"] == "text"]
            if texts:
                turns.append(TranscriptTurn(role="assistant", text="\n".join(texts)))
    if citations:
        for turn in reversed(turns):
            if turn.role == "assistant":
                turn.citations = list(dict.fromkeys(citations))
                break
    return turns


async def get_transcript(thread_id: str) -> list[TranscriptTurn] | None:
    """Turns from the latest checkpoint; None when the thread has none."""
    async with AsyncSqliteSaver.from_conn_string(settings.checkpoint_db) as saver:
        checkpoint = await saver.aget({"configurable": {"thread_id": thread_id}})
    if checkpoint is None:
        return None
    channels = checkpoint["channel_values"]
    return _turns_from_messages(channels.get("messages", []),
                                citations=channels.get("citations", []))
