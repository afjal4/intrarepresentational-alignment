from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property

import numpy as np


class SparsificationStrategy(ABC):
    """
    Abstract base for kernel-matrix sparsification strategies.

    Subclasses must implement `__call__`, which maps a dense NxN kernel
    matrix to a sparse NxN weighted adjacency matrix (zeros for absent edges).
    The returned matrix must be symmetric with a zero diagonal.
    """

    @abstractmethod
    def __call__(self, kernel: np.ndarray) -> np.ndarray:
        """Convert a dense kernel matrix to a sparse weighted adjacency matrix."""
        ...


class KNN(SparsificationStrategy):
    """
    Mutual k-nearest-neighbour sparsification.

    An edge (i, j) is retained only when j is among the top-k neighbours
    of i, and i is among the top-k neighbours of j (mutual requirement).
    This guarantees a symmetric adjacency matrix and reflects the intuition
    that semantic relationships should be bidirectional.
    """

    def __init__(self, k: int) -> None:
        self.k = k

    def __call__(self, kernel: np.ndarray) -> np.ndarray:
        n = kernel.shape[0]
        K = kernel.copy()
        np.fill_diagonal(K, -np.inf)
        idx = np.argsort(K, axis=1)[:, -self.k:]
        mask = np.zeros((n, n), dtype=bool)
        np.put_along_axis(mask, idx, True, axis=1)
        mutual = mask & mask.T # mutual requirement
        return np.where(mutual, kernel, 0.0)


class EpsilonThreshold(SparsificationStrategy):
    """
    Epsilon-threshold sparsification.

    Retains all edges where the kernel value is at or above `epsilon`,
    zeroing out the rest.
    """

    def __init__(self, epsilon: float) -> None:
        self.epsilon = epsilon

    def __call__(self, kernel: np.ndarray) -> np.ndarray:
        adjacency = np.where(kernel >= self.epsilon, kernel, 0.0)
        np.fill_diagonal(adjacency, 0.0)
        return adjacency


class TopFraction(SparsificationStrategy):
    """
    Top-fraction sparsification.

    Keeps exactly ``ceil(fraction * n*(n-1)/2)`` edges — the strongest ones
    by weight — so every matrix of the same size yields the same edge count.
    This makes graphs directly comparable regardless of their weight scale.

    Example: ``TopFraction(0.20)`` keeps the top 20 % of possible edges.
    """

    def __init__(self, fraction: float) -> None:
        if not 0 < fraction <= 1:
            raise ValueError("fraction must be in (0, 1]")
        self.fraction = fraction

    def __call__(self, kernel: np.ndarray) -> np.ndarray:
        n = kernel.shape[0]
        rows, cols = np.triu_indices(n, k=1)
        upper = kernel[rows, cols]
        k = max(1, int(np.ceil(self.fraction * len(upper))))
        # select exactly k edges by rank to avoid tie ambiguity
        top_k_idx = np.argpartition(upper, -k)[-k:]
        mask = np.zeros((n, n), dtype=bool)
        mask[rows[top_k_idx], cols[top_k_idx]] = True
        mask = mask | mask.T          # symmetrise
        adjacency = np.where(mask, kernel, 0.0)
        np.fill_diagonal(adjacency, 0.0)
        return adjacency


class SparseGraph:
    """A sparse weighted graph derived from a dense kernel matrix."""

    def __init__(self, kernel: np.ndarray, strategy: SparsificationStrategy) -> None:
        self._kernel = kernel
        self._strategy = strategy

    @cached_property
    def adjacency(self) -> np.ndarray:
        """NxN sparse weighted adjacency matrix."""
        return self._strategy(self._kernel)

    @property
    def n_edges(self) -> int:
        """Number of undirected edges."""
        return int(np.count_nonzero(np.triu(self.adjacency, k=1)))
