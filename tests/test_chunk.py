import pytest

from rag.chunk import chunk_text, count_tokens


def test_empty_text_gives_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_short_text_is_single_chunk():
    assert chunk_text("Hello world.", max_tokens=50) == ["Hello world."]


def test_chunks_respect_token_limit():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 40 for i in range(20))
    chunks = chunk_text(text, max_tokens=100, overlap_tokens=20)
    assert len(chunks) > 1
    assert all(count_tokens(c) <= 100 for c in chunks)


def test_consecutive_chunks_overlap():
    # Pieces of ~12 tokens; overlap of 20 tokens carries the previous tail piece.
    text = "\n\n".join(
        f"para {i} alpha beta gamma delta epsilon zeta eta theta" for i in range(12)
    )
    chunks = chunk_text(text, max_tokens=40, overlap_tokens=20)
    assert len(chunks) > 1
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.splitlines()[0] == prev.splitlines()[-1]


def test_oversized_single_piece_is_hard_split():
    text = "x" * 5000  # no separators at all
    chunks = chunk_text(text, max_tokens=100, overlap_tokens=0)
    assert len(chunks) > 1
    assert all(count_tokens(c) <= 100 for c in chunks)


def test_overlap_must_be_smaller_than_max():
    with pytest.raises(ValueError):
        chunk_text("some text", max_tokens=20, overlap_tokens=20)


def test_chunks_never_exceed_limit_with_large_overlap():
    # Valid-but-aggressive overlap: tail carry must still respect max_tokens.
    text = "\n\n".join(
        f"para {i} alpha beta gamma delta epsilon zeta eta theta iota kappa" for i in range(12)
    )
    chunks = chunk_text(text, max_tokens=40, overlap_tokens=30)
    assert len(chunks) > 1
    assert all(count_tokens(c) <= 40 for c in chunks)
