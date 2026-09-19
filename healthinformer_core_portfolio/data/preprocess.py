"""Text chunking for PubMed abstracts.

Splits abstracts at sentence boundaries into chunks that fit within the
PubMedBERT 512-token window, with configurable sentence overlap between
consecutive chunks. Each chunk preserves the source article's metadata.
"""

import re
from typing import Any

from config.settings import CHUNK_MAX_TOKENS, CHUNK_OVERLAP_SENTENCES

# Rough token estimate: 1 token ≈ 3 characters (conservative for WordPiece
# tokenization on medical text where long terms get split into subwords).
_CHARS_PER_TOKEN: int = 3


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using a regex heuristic.

    Splits on period/question/exclamation followed by whitespace and an
    uppercase letter, which handles most biomedical abstract text well.
    """
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    return [s.strip() for s in parts if s.strip()]


def _estimate_tokens(text: str) -> int:
    """Estimate token count from character length."""
    return len(text) // _CHARS_PER_TOKEN


def chunk_article(
    article: dict[str, Any],
    category: str,
    max_tokens: int = CHUNK_MAX_TOKENS,
    overlap_sentences: int = CHUNK_OVERLAP_SENTENCES,
) -> list[dict[str, Any]]:
    """Split one article's abstract into overlapping chunks with metadata.

    Each returned chunk dict contains:
        - chunk_text: the text content
        - chunk_index: position within this article's chunks (0-indexed)
        - pmid, title, authors, journal, year, url, category: source metadata
    """
    abstract = article.get("abstract", "")
    if not abstract.strip():
        return []

    sentences = _split_sentences(abstract)
    if not sentences:
        return []

    max_chars = max_tokens * _CHARS_PER_TOKEN
    chunks: list[dict[str, Any]] = []
    start = 0

    while start < len(sentences):
        # Greedily add sentences until we exceed the character budget
        end = start
        current_text = sentences[end]

        while end + 1 < len(sentences):
            candidate = current_text + " " + sentences[end + 1]
            if len(candidate) > max_chars and end > start:
                break
            current_text = candidate
            end += 1

        chunk_meta = {
            "chunk_text": current_text,
            "chunk_index": len(chunks),
            "pmid": article["pmid"],
            "title": article["title"],
            "authors": article["authors"],
            "journal": article["journal"],
            "year": article["year"],
            "url": article["url"],
            "category": category,
        }
        chunks.append(chunk_meta)

        # Advance: step forward, then back by overlap amount
        next_start = end + 1
        if overlap_sentences > 0 and next_start < len(sentences):
            next_start = max(start + 1, end + 1 - overlap_sentences)
        start = next_start

    return chunks


def chunk_articles(
    articles: list[dict[str, Any]],
    category: str,
) -> list[dict[str, Any]]:
    """Chunk a list of articles, returning all chunks in a flat list."""
    all_chunks: list[dict[str, Any]] = []
    for article in articles:
        all_chunks.extend(chunk_article(article, category))
    return all_chunks


if __name__ == "__main__":
    # Quick test with a synthetic article
    test_article = {
        "pmid": "12345678",
        "title": "Test Article",
        "abstract": (
            "BACKGROUND: This is the first sentence about diabetes. "
            "It affects millions worldwide. "
            "METHODS: We conducted a randomized controlled trial. "
            "Participants were enrolled from three clinical sites. "
            "RESULTS: The intervention group showed significant improvement. "
            "HbA1c levels decreased by 1.2 percent on average. "
            "CONCLUSIONS: Early intervention is effective for management."
        ),
        "authors": ["Smith John", "Doe Jane"],
        "journal": "Test Journal",
        "year": "2024",
        "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
    }
    chunks = chunk_article(test_article, category="Diabetes")
    print(f"Produced {len(chunks)} chunk(s) from test article:")
    for c in chunks:
        tokens_est = _estimate_tokens(c["chunk_text"])
        print(f"  Chunk {c['chunk_index']}: ~{tokens_est} tokens, "
              f"{len(c['chunk_text'])} chars")
        print(f"    {c['chunk_text'][:120]}...")
