from functools import lru_cache

import gensim.downloader as api
import numpy as np

from .embedding_models import EmbeddingModel


@lru_cache(maxsize=None)
def _load_model(model_id: EmbeddingModel):
    return api.load(str(model_id))


def _embed_text(model, text: str) -> np.ndarray:
    tokens = text.lower().split()
    vecs = [model[t] for t in tokens if t in model]
    if not vecs:
        return np.zeros(model.vector_size, dtype=np.float32)
    v = np.mean(vecs, axis=0).astype(np.float32)
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-9 else v


class Embedder:
    def __init__(self, model_id: EmbeddingModel) -> None:
        self._model_id = model_id
        self._model = _load_model(model_id)

    def embed(self, text: str | list[str]) -> np.ndarray:
        """Return normalised embedding(s) for text."""
        if isinstance(text, str):
            if not text.strip():
                raise ValueError("text must not be empty")
            return _embed_text(self._model, text)

        if not text:
            raise ValueError("text list must not be empty")
        if any(not t.strip() for t in text):
            raise ValueError("text list must not contain empty strings")

        return np.vstack([_embed_text(self._model, t) for t in text])
