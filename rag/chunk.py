"""Recursive text splitter with token-based sizing and overlap.

Splits on progressively finer separators until every piece fits, then greedily
packs pieces into chunks of <= max_tokens, carrying the tail pieces of each
chunk into the next one as overlap.
"""

import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")  # tokenizer used by text-embedding-3-small

_SEPARATORS = ["\n\n", "\n", ". ", " "]


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _split(text: str, max_tokens: int, separators: list[str]) -> list[str]:
    if count_tokens(text) <= max_tokens:
        return [text]
    if not separators:
        # No separators left: hard-cut on token boundaries.
        toks = _enc.encode(text)
        return [_enc.decode(toks[i:i + max_tokens]) for i in range(0, len(toks), max_tokens)]
    sep, rest = separators[0], separators[1:]
    parts = [p for p in text.split(sep) if p.strip()]
    if len(parts) <= 1:
        return _split(text, max_tokens, rest)
    pieces: list[str] = []
    for part in parts:
        pieces.extend(_split(part, max_tokens, rest))
    return pieces


def chunk_text(text: str, max_tokens: int | None = None,
               overlap_tokens: int | None = None) -> list[str]:
    from config import settings

    max_tokens = max_tokens or settings.chunk_max_tokens
    if overlap_tokens is None:
        overlap_tokens = settings.chunk_overlap_tokens
    text = text.strip()
    if not text:
        return []

    pieces = _split(text, max_tokens, _SEPARATORS)

    chunks: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for piece in pieces:
        piece_tokens = count_tokens(piece) + 1  # +1 for the join newline
        if current and current_tokens + piece_tokens > max_tokens:
            chunks.append(current)
            # Carry trailing pieces of the finished chunk as overlap.
            tail: list[str] = []
            tail_tokens = 0
            for prev in reversed(current):
                t = count_tokens(prev) + 1
                if tail_tokens + t > overlap_tokens:
                    break
                tail.insert(0, prev)
                tail_tokens += t
            current, current_tokens = tail, tail_tokens
        current.append(piece)
        current_tokens += piece_tokens
    if current:
        chunks.append(current)
    return ["\n".join(c) for c in chunks]
