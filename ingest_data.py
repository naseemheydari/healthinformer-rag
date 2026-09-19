"""Entry point: fetch PubMed abstracts → chunk → save to data/raw/.

Usage:
    python ingest_data.py                    # Fetch all queries from constants.py
    python ingest_data.py --test             # Fetch only the first query (quick test)
    python ingest_data.py --max-results 50   # Override max results per query
"""

import argparse
import json
import sys
from pathlib import Path

from config.constants import PUBMED_QUERIES
from config.settings import DATA_RAW_DIR, PUBMED_EMAIL, PUBMED_MAX_RESULTS_PER_QUERY
from data.fetch_pubmed import fetch_abstracts, search_pubmed
from data.preprocess import chunk_articles


def ingest(
    queries: dict[str, str],
    max_results: int = PUBMED_MAX_RESULTS_PER_QUERY,
) -> None:
    """Run the full ingestion pipeline: search → fetch → chunk → save."""
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    all_articles: list[dict] = []
    all_chunks: list[dict] = []
    seen_pmids: set[str] = set()

    total_queries = len(queries)
    for i, (query, category) in enumerate(queries.items(), 1):
        print(f"\n[{i}/{total_queries}] Query: {query!r}  (category: {category})")

        # Search
        pmids = search_pubmed(query, max_results=max_results)
        # Deduplicate across queries
        new_pmids = [p for p in pmids if p not in seen_pmids]
        seen_pmids.update(new_pmids)
        print(f"  {len(new_pmids)} new PMIDs (skipped {len(pmids) - len(new_pmids)} duplicates)")

        if not new_pmids:
            continue

        # Fetch
        articles = fetch_abstracts(new_pmids)
        all_articles.extend(articles)
        print(f"  Fetched {len(articles)} articles with abstracts")

        # Chunk
        chunks = chunk_articles(articles, category)
        all_chunks.extend(chunks)
        print(f"  Produced {len(chunks)} chunks")

    # ── Save ────────────────────────────────────────────────────────────
    articles_path = DATA_RAW_DIR / "articles.json"
    chunks_path = DATA_RAW_DIR / "chunks.json"

    with open(articles_path, "w", encoding="utf-8") as f:
        json.dump(all_articles, f, indent=2, ensure_ascii=False)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Ingestion complete!")
    print(f"  Total unique articles: {len(all_articles)}")
    print(f"  Total chunks:          {len(all_chunks)}")
    print(f"  Articles saved to:     {articles_path}")
    print(f"  Chunks saved to:       {chunks_path}")
    print(f"{'='*60}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PubMed data for HealthInformer")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run with only the first query (quick test)",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=PUBMED_MAX_RESULTS_PER_QUERY,
        help=f"Max PubMed results per query (default: {PUBMED_MAX_RESULTS_PER_QUERY})",
    )
    args = parser.parse_args()

    if not PUBMED_EMAIL:
        print("ERROR: PUBMED_EMAIL not set. Add it to your .env file.")
        print("  PubMed requires an email for API usage tracking.")
        sys.exit(1)

    queries = PUBMED_QUERIES
    if args.test:
        first_key = next(iter(queries))
        queries = {first_key: queries[first_key]}
        print("TEST MODE: Running with a single query only")

    ingest(queries, max_results=args.max_results)


if __name__ == "__main__":
    main()
