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
    of i *and* i is among the top-k neighbours of j (mutual requirement).
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
