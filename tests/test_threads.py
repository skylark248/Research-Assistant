def _use_tmp_db(monkeypatch, tmp_path):
    from api import threads as threads_mod

    monkeypatch.setattr(threads_mod.settings, "checkpoint_db", str(tmp_path / "cp.db"))
    return threads_mod


def test_upsert_and_list(monkeypatch, tmp_path):
    tm = _use_tmp_db(monkeypatch, tmp_path)
    tm.upsert_thread("t1", "x" * 100)
    tm.upsert_thread("t2", "second thread")
    tm.upsert_thread("t1", "ignored on update")
    rows = tm.list_threads()
    assert [r.thread_id for r in rows][0] == "t1"  # most recently updated first
    t1 = next(r for r in rows if r.thread_id == "t1")
    assert t1.title == "x" * 80  # truncated, set on first insert only
    assert t1.updated_at >= t1.created_at


def test_delete_thread(monkeypatch, tmp_path):
    tm = _use_tmp_db(monkeypatch, tmp_path)
    tm.upsert_thread("t1", "hello")
    tm.delete_thread("t1")
    assert tm.list_threads() == []


async def test_get_transcript_unknown_thread(monkeypatch, tmp_path):
    tm = _use_tmp_db(monkeypatch, tmp_path)
    assert await tm.get_transcript("nope") is None


def test_turns_from_messages():
    from api.threads import _turns_from_messages

    messages = [
        {"role": "user", "content": "what is attention?"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Let me check."},
            {"type": "tool_use", "id": "tu_1", "name": "rag_query", "input": {}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu_1", "content": "stuff"},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": "It is attention."}]},
    ]
    turns = _turns_from_messages(messages)
    assert [(t.role, t.text) for t in turns] == [
        ("user", "what is attention?"),
        ("assistant", "Let me check."),
        ("assistant", "It is attention."),
    ]
