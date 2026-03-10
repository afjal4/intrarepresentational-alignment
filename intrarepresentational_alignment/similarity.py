from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import combinations_with_replacement

import numpy as np


class SimilarityMetric(ABC):
    """
    Abstract base for pairwise similarity functions over embedding vectors.

    Subclasses must implement `__call__` for a single pair. The default
    `matrix` implementation calls `__call__` in an O(N²) loop; subclasses
    should override it with a vectorised version where possible.
    """

    @abstractmethod
    def __call__(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute similarity between two embedding vectors."""
        ...

    def matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Compute an NxN pairwise similarity matrix.
        Default implementation is O(N^2) via `__call__`.
        Override it for efficiency.
        """
        n = len(embeddings)
        out = np.zeros((n, n))
        for i, j in combinations_with_replacement(range(n), 2):
            out[i, j] = out[j, i] = self(embeddings[i], embeddings[j])
        return out


class CosineSimilarity(SimilarityMetric):
    """Cosine similarity: dot(a, b) / (|a|||b|). Range: [-1, 1]."""

    def __call__(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def matrix(self, embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalised = embeddings / norms
        return normalised @ normalised.T


class RBFKernel(SimilarityMetric):
    """Radial Basis Function (RBF) kernel: exp(-gamma|a-b|^2). Range: (0, 1]."""

    def __init__(self, gamma: float = 1.0) -> None:
        self.gamma = gamma

    def __call__(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.exp(-self.gamma * np.sum((a - b) ** 2)))

    def matrix(self, embeddings: np.ndarray) -> np.ndarray:
        # |a − b|^2 = |a|^2 + |b|^2 − 2 a^Tb
        sq_norms = np.sum(embeddings ** 2, axis=1)
        sq_dists = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (embeddings @ embeddings.T)
        return np.exp(-self.gamma * sq_dists)
