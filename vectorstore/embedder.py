"""PubMedBERT embedding wrapper using sentence-transformers.

Loads the PubMedBERT model once and provides methods to embed single
texts or batches, returning 768-dimensional vectors.

Usage:
    from vectorstore.embedder import Embedder

    embedder = Embedder()
    vec = embedder.embed("What causes type 2 diabetes?")       # (768,)
    vecs = embedder.embed_batch(["text1", "text2"], batch_size=32)  # (2, 768)
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import EMBEDDING_DIM, EMBEDDING_MODEL_NAME


class Embedder:
    """Thin wrapper around sentence-transformers for PubMedBERT embeddings."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        """Load the embedding model.

        Args:
            model_name: HuggingFace model identifier. Defaults to PubMedBERT.
        """
        print(f"Loading embedding model: {model_name} ...")
        self.model = SentenceTransformer(model_name)
        self.dim = EMBEDDING_DIM
        print(f"Model loaded. Embedding dimension: {self.dim}")

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string.

        Args:
            text: Input text to embed.

        Returns:
            1-D numpy array of shape (768,).
        """
        vector = self.model.encode(text, show_progress_bar=False)
        return np.asarray(vector, dtype=np.float32)

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Embed a batch of texts.

        Args:
            texts: List of input strings.
            batch_size: Texts per forward pass (tune for GPU/RAM).
            show_progress: Whether to show a progress bar.

        Returns:
            2-D numpy array of shape (len(texts), 768).
        """
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
        )
        return np.asarray(vectors, dtype=np.float32)


if __name__ == "__main__":
    embedder = Embedder()
    test_text = "What are the symptoms of type 2 diabetes?"
    vec = embedder.embed(test_text)
    print(f"Input: {test_text!r}")
    print(f"Vector shape: {vec.shape}, dtype: {vec.dtype}")
    print(f"First 5 values: {vec[:5]}")
