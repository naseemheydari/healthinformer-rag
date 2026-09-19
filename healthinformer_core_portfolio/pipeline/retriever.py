"""Retriever: query string → top-k relevant chunks from ChromaDB.

Loads the PubMedBERT embedder and ChromaDB store, embeds the user
query, and returns the most relevant chunks with metadata.

Optionally expands lay-language questions into a short hypothetical
PubMed-abstract excerpt (HyDE — Hypothetical Document Embeddings)
before embedding. This bridges the lay↔clinical vocabulary gap that
causes direct embedding of casual questions to retrieve poorly: the
embedding space of PubMedBERT is populated by abstract-shaped text,
so a query that also looks like an abstract snippet lands near real
abstracts on the same topic rather than random dense regions of the
space.

Empirical comparison on 4 diverse test questions (top-8 relevance):
    PLAIN query:         9/32  (28%)
    Keyword-bag rewrite: 2/32  (6%)   — actively harmful
    Prose single-sentence: 7/32 (22%) — roughly neutral
    HyDE fake abstract: 21/32 (66%)  — clear winner

Usage:
    from pipeline.retriever import Retriever

    retriever = Retriever()
    results = retriever.retrieve("What causes diabetes?", top_k=5)
"""

from typing import Any

from config.settings import RETRIEVAL_TOP_K
from llm.base import BaseLLM
from vectorstore.embedder import Embedder
from vectorstore.store import VectorStore

# HyDE system prompt. The "Avoid fabricating specific statistics" clause
# is load-bearing — without it the LLM invents percentages and numbers
# that pull the resulting embedding toward "papers with statistical
# result vibes" instead of the actual topic. With it, the output reads
# like real abstract prose and retrieves ~2.3x more relevant results.
_REWRITE_SYSTEM_PROMPT = (
    "Write a 2-3 sentence excerpt from a hypothetical PubMed abstract "
    "that would answer this health question. Use clinical language and "
    "the style of a real medical abstract (background or results "
    "section). Do not answer the question conversationally — write as "
    "if you are quoting an abstract. Avoid fabricating specific "
    "statistics. Return only the excerpt text, no preamble, no quotes."
)

# Max tokens for the HyDE generation. 2-3 clinical sentences fit well
# under 220 tokens; we also stay within PubMedBERT's 512-token input
# limit when the excerpt is later embedded.
_REWRITE_MAX_TOKENS = 220


class Retriever:
    """Embeds a query and retrieves relevant chunks from the vector store."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        rewriter: BaseLLM | None = None,
        rewrite: bool = True,
    ) -> None:
        """Initialize with an embedder and vector store.

        Args:
            embedder: Pre-loaded Embedder instance. Created if None.
            store: Pre-loaded VectorStore instance. Created if None.
            rewriter: Optional LLM for query rewriting. Lazily constructed
                as GroqLLM on first use if None and rewrite=True.
            rewrite: If True, rewrite lay-language queries into clinical
                terminology before embedding. Improves retrieval quality
                on general-public questions.
        """
        self.embedder = embedder or Embedder()
        self.store = store or VectorStore()
        self.rewrite = rewrite
        self._rewriter = rewriter

    def _get_rewriter(self) -> BaseLLM:
        """Lazily construct a rewriter client so the import cost
        and API-key requirement only apply when rewriting is enabled."""
        if self._rewriter is None:
            from llm.bedrock_client import BedrockLLM
            self._rewriter = BedrockLLM()
        return self._rewriter

    def rewrite_query(self, query: str) -> str:
        """Expand a plain-language question into a HyDE abstract excerpt.

        Generates a 2-3 sentence hypothetical PubMed-abstract-style
        passage that would answer the question, then returns it as the
        search query. This is the HyDE technique (Hypothetical Document
        Embeddings, Gao et al. 2022): embedding a query that already
        looks like the target documents lands much closer to on-topic
        abstracts in PubMedBERT's space than embedding the raw question.

        Temperature 0 for determinism. Falls back to the original query
        on any error so retrieval never hard-fails on a rewriter hiccup.

        Args:
            query: Original user question in plain language.

        Returns:
            A short hypothetical-abstract string, or the original query
            if the rewriter call fails.
        """
        try:
            llm = self._get_rewriter()
            rewritten = llm.generate(
                prompt=query,
                max_tokens=_REWRITE_MAX_TOKENS,
                temperature=0.0,
                system_prompt=_REWRITE_SYSTEM_PROMPT,
            )
            # Strip surrounding quotes/whitespace the LLM sometimes adds.
            rewritten = rewritten.strip().strip('"').strip("'").strip()
            return rewritten or query
        except Exception as e:
            print(f"  [rewrite fallback: {type(e).__name__}: {str(e)[:120]}]")
            return query

    def retrieve(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> list[dict[str, Any]]:
        """Retrieve the top-k most relevant chunks for a query.

        If rewrite is enabled, the original query is rewritten into
        clinical terminology before embedding. The rewritten query is
        used only for retrieval — callers still see the original query.

        Args:
            query: Natural language question from the user.
            top_k: Number of chunks to retrieve.

        Returns:
            List of result dicts, each with keys:
                id, document, metadata, distance.
        """
        search_query = self.rewrite_query(query) if self.rewrite else query
        query_embedding = self.embedder.embed(search_query)
        results = self.store.query(query_embedding, top_k=top_k)
        return results


if __name__ == "__main__":
    retriever = Retriever()
    query = "What are the warning signs of type 2 diabetes?"
    results = retriever.retrieve(query, top_k=5)
    print(f"Query: {query!r}\n")
    for i, r in enumerate(results, 1):
        m = r["metadata"]
        print(f"[{i}] dist={r['distance']:.4f} | {m['title'][:70]}")
