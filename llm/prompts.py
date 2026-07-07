"""Prompt templates.

Demonstrates: system prompt design, few-shot examples (citation format), and an
Anthropic prompt-caching breakpoint (cache_control on the long paper context).
Contexts are plain dicts ({paper_id, title, text}) so llm/ never imports rag/.
"""

CITATION_SYSTEM_PROMPT = """You are a research assistant that answers questions using ONLY the provided paper excerpts.

Rules:
- Base every claim on the excerpts. If they are insufficient, say "I don't have enough information in the ingested papers" and suggest fetching more.
- Cite the source of every claim inline with the arXiv id in square brackets, e.g. [1706.03762].
- When several excerpts support a claim, stack citations: [1706.03762][1810.04805].
- Be concise and technical. Do not invent paper ids."""

AGENT_SYSTEM_PROMPT = """You are a research paper assistant with tools.

For each user message decide:
- If the question concerns papers likely already ingested, call rag_query first.
- If rag_query reports it doesn't have enough information, call arxiv_search to find the paper, then arxiv_fetch_paper to ingest it, then call rag_query again.
- If the user gives a URL, call fetch to read it.

Cite papers inline as [paper_id] whenever an answer comes from ingested papers.
If a tool call fails, decide whether to retry once with adjusted input or explain the failure to the user. Never fabricate tool output."""

# Few-shot pair demonstrating the citation format (uses a fake paper id on purpose).
FEW_SHOT_MESSAGES = [
    {
        "role": "user",
        "content": (
            "Paper excerpts:\n\n[paper 1234.56789 — Example Networks]\n"
            "Example Networks use gated residual connections to stabilize training.\n\n"
            "Question: How do Example Networks stabilize training?"
        ),
    },
    {
        "role": "assistant",
        "content": "Example Networks stabilize training with gated residual connections [1234.56789].",
    },
]


def format_context(contexts: list[dict]) -> str:
    parts = [f"[paper {c['paper_id']} — {c['title']}]\n{c['text']}" for c in contexts]
    return "\n\n---\n\n".join(parts)


REWRITE_SYSTEM_PROMPT = """You rewrite user questions into search queries for a research-paper vector database.

Return one focused query: expand acronyms, drop conversational filler, keep every technical term. Do not answer the question."""


def build_rag_prompt(question: str, contexts: list[dict]) -> tuple[list[dict], list[dict]]:
    """Returns (system_blocks, messages) for generate().

    The context block carries cache_control so Anthropic caches the paper
    excerpts: repeat/follow-up questions that retrieve the same top-k chunks
    get a cache hit (visible in usage["cache_read_input_tokens"]). The OpenAI
    adapter simply drops cache_control.
    """
    system_blocks = [
        {"type": "text", "text": CITATION_SYSTEM_PROMPT},
        {
            "type": "text",
            "text": "Paper excerpts:\n\n" + format_context(contexts),
            "cache_control": {"type": "ephemeral"},
        },
    ]
    messages = FEW_SHOT_MESSAGES + [{"role": "user", "content": f"Question: {question}"}]
    return system_blocks, messages
