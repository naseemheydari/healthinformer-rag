"""ChromaDB vector store interface.

Provides init/load, add, and query operations for the HealthInformer
chunk collection. Embeddings are supplied externally (from embedder.py)
so ChromaDB stores them as-is without its own embedding function.

Usage:
    from vectorstore.store import VectorStore

    store = VectorStore()                     # init or load existing
    store.add_chunks(chunks, embeddings)      # add data
    results = store.query(query_embedding, top_k=5)  # retrieve
"""

from typing import Any

import chromadb
import numpy as np

from config.settings import CHROMA_COLLECTION_NAME, CHROMA_DB_DIR, RETRIEVAL_TOP_K


class VectorStore:
    """Persistent ChromaDB collection for embedded PubMed chunks."""

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str = CHROMA_COLLECTION_NAME,
    ) -> None:
        """Initialize or load an existing ChromaDB collection.

        Args:
            persist_dir: Directory for persistent storage. Defaults to
                         settings.CHROMA_DB_DIR.
            collection_name: Name of the ChromaDB collection.
        """
        persist_dir = persist_dir or str(CHROMA_DB_DIR)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"ChromaDB collection '{collection_name}' ready "
              f"({self.collection.count()} existing documents)")

    def add_chunks(
        self,
        chunks: list[dict[str, Any]],
        embeddings: np.ndarray,
    ) -> None:
        """Add chunks with pre-computed embeddings to the collection.

        Args:
            chunks: List of chunk dicts (must have 'pmid', 'chunk_index',
                    'chunk_text', and metadata fields).
            embeddings: 2-D array of shape (len(chunks), dim).
        """
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, str]] = []

        for chunk in chunks:
            # Unique ID: pmid + chunk index
            chunk_id = f"{chunk['pmid']}_{chunk['chunk_index']}"
            ids.append(chunk_id)
            documents.append(chunk["chunk_text"])
            metadatas.append({
                "pmid": chunk["pmid"],
                "title": chunk["title"],
                "authors": ", ".join(chunk["authors"][:3]),
                "journal": chunk["journal"],
                "year": chunk["year"],
                "url": chunk["url"],
                "category": chunk["category"],
                "chunk_index": str(chunk["chunk_index"]),
            })

        # ChromaDB expects list-of-lists for embeddings
        embedding_lists = embeddings.tolist()

        # Upsert so re-running is idempotent
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            self.collection.upsert(
                ids=ids[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
                embeddings=embedding_lists[i:end],
            )

        print(f"Upserted {len(ids)} chunks. Collection now has "
              f"{self.collection.count()} documents.")

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = RETRIEVAL_TOP_K,
    ) -> list[dict[str, Any]]:
        """Query the collection by embedding vector.

        Args:
            query_embedding: 1-D array of shape (dim,).
            top_k: Number of results to return.

        Returns:
            List of dicts with keys: id, document, metadata, distance.
            Sorted by relevance (lowest distance first for cosine).
        """
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        output: list[dict[str, Any]] = []
        for i in range(len(results["ids"][0])):
            output.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return output


if __name__ == "__main__":
    store = VectorStore()
    print(f"Collection count: {store.collection.count()}")
