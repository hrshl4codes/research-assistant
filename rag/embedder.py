"""
Embedder. Wraps sentence-transformers for local, free embedding generation.

Model : all-MiniLM-L6-v2  (384 dimensions, ~80 MB download on first use)
Cache : ~/.cache/huggingface/hub/  (managed by Hugging Face hub)

Embeddings are L2-normalised so cosine similarity reduces to a dot product on
unit vectors. DuckDB's array_cosine_distance assumes this. Without normalisation
the distances are still numerically valid but unstable when vectors have very
different magnitudes.
"""

from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class Embedder:
    """
    Thin wrapper around SentenceTransformer for batched text encoding.
    Model loading takes about 2 seconds on CPU, so instantiate once and reuse.
    """

    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        """
        Load the embedding model.

        Args:
            model_name: Hugging Face model identifier. Defaults to
                        all-MiniLM-L6-v2, a 384-dim model that runs
                        well on CPU without a GPU.
        """
        logger.info("Loading embedding model '%s' …", model_name)
        self._model = SentenceTransformer(model_name)
        logger.info("Embedding model ready.")

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode a list of strings into normalised float32 embeddings.

        Args:
            texts:         Strings to embed.  Must be non-empty.
            batch_size:    Texts per forward pass.  Larger batches are faster
                           on GPU; 32–64 is a safe default on CPU.
            show_progress: Show a tqdm progress bar (useful for >100 chunks).

        Returns:
            Float32 ndarray of shape (len(texts), 384), each row L2-normalised.

        Raises:
            ValueError: If texts is empty.
        """
        if not texts:
            raise ValueError("encode() received an empty list of texts.")

        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        """
        Convenience wrapper for a single string (e.g. a user query).

        Returns:
            1-D float32 array of length 384.
        """
        return self.encode([text], batch_size=1)[0]
