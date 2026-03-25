from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from .models import EmbeddingModel

_SINGLE_EMBED_CACHE_SIZE = 4096


@lru_cache(maxsize=None)
def _load_model(model_id: EmbeddingModel) -> SentenceTransformer:
    return SentenceTransformer(model_id)


@lru_cache(maxsize=_SINGLE_EMBED_CACHE_SIZE)
def _embed_single(model_id: EmbeddingModel, text: str) -> np.ndarray:
    return _load_model(model_id).encode(text, normalize_embeddings=True)


class Embedder:
    def __init__(self, model_id: EmbeddingModel) -> None:
        self._model_id = model_id
        self._sentence_transformer = _load_model(model_id)

    def embed(self, text: str | list[str]) -> np.ndarray:
        """Return normalised embedding(s) for text."""
        if isinstance(text, str):
            if not text.strip():
                raise ValueError("text must not be empty")
            return _embed_single(self._model_id, text).copy()

        if not text:
            raise ValueError("text list must not be empty")
        if any(not t.strip() for t in text):
            raise ValueError("text list must not contain empty strings")

        return self._sentence_transformer.encode(text, normalize_embeddings=True)
