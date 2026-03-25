from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .graph import SparsificationStrategy


def _validate_embeddings(embeddings: np.ndarray) -> None:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    if embeddings.shape[0] == 0:
        raise ValueError("embeddings must have at least one row")


class MatrixComputationStrategy(ABC):
    @abstractmethod
    def compute(self, metric: "SimilarityMetric", embeddings: np.ndarray) -> np.ndarray:
        ...


class DenseComputationStrategy(MatrixComputationStrategy):
    def compute(self, metric: "SimilarityMetric", embeddings: np.ndarray) -> np.ndarray:
        return metric.matrix(embeddings)


class ChunkedComputationStrategy(MatrixComputationStrategy):
    def __init__(self, chunk_size: int = 256) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        self.chunk_size = chunk_size

    def compute(self, metric: "SimilarityMetric", embeddings: np.ndarray) -> np.ndarray:
        _validate_embeddings(embeddings)
        n = embeddings.shape[0]
        out = np.zeros((n, n), dtype=float)
        for i in range(0, n, self.chunk_size):
            j = min(i + self.chunk_size, n)
            out[i:j, :] = metric.cross_matrix(embeddings[i:j], embeddings)
        return out


def compute_similarity_matrix(
    metric: "SimilarityMetric",
    embeddings: np.ndarray,
    strategy: MatrixComputationStrategy | None = None,
) -> np.ndarray:
    if strategy is None:
        strategy = DenseComputationStrategy()
    return strategy.compute(metric, embeddings)


class SimilarityMetric(ABC):
    """
    Abstract base for pairwise similarity functions over embedding vectors.

    Subclasses must implement `__call__` (single pair) and should override
    `cross_matrix` with a vectorised MxN implementation. `matrix` is
    defined as `cross_matrix(E, E)` and need not be overridden unless
    sparsification or other post-processing is required.
    """

    @abstractmethod
    def __call__(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute similarity between two embedding vectors."""
        ...

    def matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute an NxN pairwise similarity matrix."""
        return self.cross_matrix(embeddings, embeddings)

    def cross_matrix(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Compute an MxN pairwise similarity matrix between rows of A and rows of B.
        Default implementation is O(M*N) via `__call__`.
        Override it for efficiency.
        """
        _validate_embeddings(A)
        _validate_embeddings(B)
        m, n = A.shape[0], B.shape[0]
        out = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                out[i, j] = self(A[i], B[j])
        return out


class CosineSimilarity(SimilarityMetric):
    """Cosine similarity: a.b / (|a|||b|). Range: [-1, 1]."""

    def __call__(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            raise ValueError("cosine similarity is undefined for zero vectors")
        return float(np.dot(a, b) / denom)

    def cross_matrix(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        _validate_embeddings(A)
        _validate_embeddings(B)
        norms_A = np.linalg.norm(A, axis=1, keepdims=True)
        norms_B = np.linalg.norm(B, axis=1, keepdims=True)
        if np.any(norms_A == 0) or np.any(norms_B == 0):
            raise ValueError("cosine similarity is undefined for zero vectors")
        return (A / norms_A) @ (B / norms_B).T


class RBFKernel(SimilarityMetric):
    """Radial Basis Function (RBF) kernel: exp(-gamma|a-b|^2). Range: (0, 1]."""

    def __init__(self, gamma: float = 1.0) -> None:
        if gamma <= 0:
            raise ValueError("gamma must be > 0")
        self.gamma = gamma

    def __call__(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.exp(-self.gamma * np.sum((a - b) ** 2)))

    def cross_matrix(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        _validate_embeddings(A)
        _validate_embeddings(B)
        sq_A = np.sum(A**2, axis=1, keepdims=True)
        sq_B = np.sum(B**2, axis=1, keepdims=True)
        sq_dists = sq_A + sq_B.T - 2.0 * (A @ B.T)
        return np.exp(-self.gamma * sq_dists)


class SparseKernel(SimilarityMetric):
    """Sparsified kernel: dense metric followed by a sparsification strategy.

    Computes the full pairwise kernel with `base`, then zeros out edges that
    the `strategy` discards.  The resulting matrix retains only the retained
    entries (mutual-KNN neighbours, or values above an epsilon threshold).

    Pair-wise `__call__` and `cross_matrix` fall through to the base metric
    unchanged; sparsification requires the full matrix and is only applied in
    `matrix`.
    """

    def __init__(
        self,
        base: SimilarityMetric,
        strategy: SparsificationStrategy,
    ) -> None:
        self._base = base
        self._strategy = strategy

    def __call__(self, a: np.ndarray, b: np.ndarray) -> float:
        return self._base(a, b)

    def matrix(self, embeddings: np.ndarray) -> np.ndarray:
        dense = self._base.matrix(embeddings)
        return self._strategy(dense)

    def cross_matrix(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return self._base.cross_matrix(A, B)
