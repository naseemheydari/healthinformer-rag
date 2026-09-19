"""Entry point: load chunks.json → embed with PubMedBERT → store in ChromaDB.

Usage:
    python build_vectorstore.py              # Embed and store all chunks
    python build_vectorstore.py --test-query "What are the symptoms of diabetes?"
"""

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

from config.settings import DATA_RAW_DIR
from vectorstore.embedder import Embedder
from vectorstore.store import VectorStore


def build(chunks_path: Path, batch_size: int = 32) -> tuple[Embedder, VectorStore]:
    """Load chunks, embed them, and store in ChromaDB.

    Returns the embedder and store for optional follow-up queries.
    """
    # ── Load chunks ─────────────────────────────────────────────────────
    print(f"Loading chunks from {chunks_path} ...")
    with open(chunks_path, encoding="utf-8") as f:
        chunks: list[dict] = json.load(f)
    print(f"Loaded {len(chunks)} chunks")

    if not chunks:
        print("No chunks to embed. Run ingest_data.py first.")
        sys.exit(1)

    # ── Embed ───────────────────────────────────────────────────────────
    embedder = Embedder()
    texts = [c["chunk_text"] for c in chunks]

    print(f"Embedding {len(texts)} chunks (batch_size={batch_size}) ...")
    # Use tqdm wrapper for a nicer progress bar than sentence-transformers default
    embeddings = embedder.embed_batch(texts, batch_size=batch_size, show_progress=True)
    print(f"Embeddings shape: {embeddings.shape}")

    # ── Store ───────────────────────────────────────────────────────────
    store = VectorStore()
    store.add_chunks(chunks, embeddings)

    # ── Stats ───────────────────────────────────────────────────────────
    categories = {}
    for c in chunks:
        cat = c.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    print(f"\n{'='*60}")
    print(f"Vectorstore build complete!")
    print(f"  Total chunks embedded:  {len(chunks)}")
    print(f"  Embedding dimensions:   {embeddings.shape[1]}")
    print(f"  ChromaDB documents:     {store.collection.count()}")
    print(f"  Categories:")
    for cat, count in sorted(categories.items()):
        print(f"    {cat}: {count}")
    print(f"{'='*60}")

    return embedder, store


def test_query(embedder: Embedder, store: VectorStore, query: str, top_k: int = 5) -> None:
    """Run a test query and print results."""
    print(f"\n--- Test Query: {query!r} ---")
    query_vec = embedder.embed(query)
    results = store.query(query_vec, top_k=top_k)

    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        print(f"\n  [{i}] Distance: {r['distance']:.4f}")
        print(f"      PMID: {meta['pmid']}  |  Year: {meta['year']}  |  Category: {meta['category']}")
        print(f"      Title: {meta['title'][:80]}")
        print(f"      Journal: {meta['journal']}")
        print(f"      URL: {meta['url']}")
        print(f"      Text: {r['document'][:150]}...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build HealthInformer vectorstore")
    parser.add_argument(
        "--test-query",
        type=str,
        default=None,
        help="Run a test retrieval query after building",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size (default: 32)",
    )
    args = parser.parse_args()

    chunks_path = DATA_RAW_DIR / "chunks.json"
    if not chunks_path.exists():
        print(f"ERROR: {chunks_path} not found. Run ingest_data.py first.")
        sys.exit(1)

    embedder, store = build(chunks_path, batch_size=args.batch_size)

    if args.test_query:
        test_query(embedder, store, args.test_query)


if __name__ == "__main__":
    main()
