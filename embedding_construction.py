from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=None)
def _load_model(model_id: str) -> SentenceTransformer:
    return SentenceTransformer(model_id)


class Embedder:
    def __init__(self, model_id: str) -> None:
        self.sentenceTransformer = _load_model(model_id)

    def embed(self, text: str | list[str]) -> np.ndarray:
        """Return L2-normalised embedding(s) for text."""
        return self.sentenceTransformer.encode(text, normalize_embeddings=True)
