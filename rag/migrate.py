"""Recreate the Qdrant collection with the phase-2 named dense+sparse schema.

Destructive: drops all ingested chunks. PDFs stay cached in data/pdfs, so
re-ingesting re-parses and re-embeds them (dense re-embedding costs OpenAI
tokens unless EMBEDDING_PROVIDER=local).

Usage: uv run python -m rag.migrate --yes
"""

import argparse

from rag.store import VectorStore


def migrate(store: VectorStore | None = None) -> None:
    store = store or VectorStore()
    store.ping()
    if store.client.collection_exists(store.collection):
        store.client.delete_collection(store.collection)
    store.ensure_collection()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yes", action="store_true",
                        help="confirm dropping the existing collection")
    args = parser.parse_args()
    if not args.yes:
        parser.error("refusing to drop the collection without --yes")
    migrate()
    print("Collection recreated with the dense+sparse schema. Re-ingest papers now.")


if __name__ == "__main__":
    main()
