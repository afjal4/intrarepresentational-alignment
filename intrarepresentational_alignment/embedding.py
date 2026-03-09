from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from .models import EmbeddingModel


@lru_cache(maxsize=None)
def _load_model(model_id: EmbeddingModel) -> SentenceTransformer:
    return SentenceTransformer(model_id)


class Embedder:
    def __init__(self, model_id: EmbeddingModel) -> None:
        self._sentence_transformer = _load_model(model_id)

    def embed(self, text: str | list[str]) -> np.ndarray:
        """Return normalised embedding(s) for text."""
        return self._sentence_transformer.encode(text, normalize_embeddings=True)
