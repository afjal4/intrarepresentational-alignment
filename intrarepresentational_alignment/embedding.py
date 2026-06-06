from __future__ import annotations

import urllib.request
from functools import lru_cache
from pathlib import Path

import gensim.downloader as api
import numpy as np
from gensim.models import KeyedVectors

from .embedding_models import EmbeddingModel

_NUMBERBATCH_URL = (
    "https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/"
    "numberbatch-en-19.08.txt.gz"
)
_NUMBERBATCH_PATH = Path("data/numberbatch-en-19.08.txt.gz")


class _NumberbatchWrapper:
    """Thin wrapper around Numberbatch KeyedVectors.

    Numberbatch stores words as ConceptNet URIs (/c/en/word).  This wrapper
    tries both the bare token and the /c/en/<token> URI so that the rest of
    the embedding code can work with plain English words.
    """

    def __init__(self, kv: KeyedVectors) -> None:
        self._kv = kv
        self.vector_size = kv.vector_size

    def _resolve(self, word: str) -> str | None:
        if word in self._kv:
            return word
        uri = f"/c/en/{word}"
        if uri in self._kv:
            return uri
        return None

    def __contains__(self, word: str) -> bool:
        return self._resolve(word) is not None

    def __getitem__(self, word: str) -> np.ndarray:
        key = self._resolve(word)
        if key is None:
            raise KeyError(word)
        return self._kv[key]


def _load_numberbatch(path: Path = _NUMBERBATCH_PATH) -> _NumberbatchWrapper:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading Numberbatch (~186 MB) to {path} ...")
        urllib.request.urlretrieve(_NUMBERBATCH_URL, path)
        print("Download complete.")
    kv = KeyedVectors.load_word2vec_format(str(path), binary=False)
    return _NumberbatchWrapper(kv)


@lru_cache(maxsize=None)
def _load_model(model_id: EmbeddingModel):
    if model_id == EmbeddingModel.CONCEPTNET_NUMBERBATCH_300:
        return _load_numberbatch()
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
