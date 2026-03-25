from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

DEFAULT_EPSILON = 0.0
DEFAULT_TOP_FRACTION = 0.2


def _validate_square_kernel(kernel: np.ndarray) -> None:
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("kernel must be a square matrix")


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
        if k <= 0:
            raise ValueError("k must be > 0")
        self.k = k

    def __call__(self, kernel: np.ndarray) -> np.ndarray:
        _validate_square_kernel(kernel)
        n = kernel.shape[0]
        if self.k >= n:
            raise ValueError("k must be smaller than the number of nodes")
        K = kernel.copy()
        np.fill_diagonal(K, -np.inf)
        idx = np.argpartition(K, kth=n - self.k, axis=1)[:, -self.k:]
        mask = np.zeros((n, n), dtype=bool)
        np.put_along_axis(mask, idx, True, axis=1)
        mutual = mask & mask.T
        return np.where(mutual, kernel, 0.0)


class EpsilonThreshold(SparsificationStrategy):
    """
    Epsilon-threshold sparsification.

    Retains all edges where the kernel value is at or above `epsilon`,
    zeroing out the rest.
    """

    def __init__(self, epsilon: float = DEFAULT_EPSILON) -> None:
        self.epsilon = epsilon

    def __call__(self, kernel: np.ndarray) -> np.ndarray:
        _validate_square_kernel(kernel)
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

    def __init__(self, fraction: float = DEFAULT_TOP_FRACTION) -> None:
        if not 0 < fraction <= 1:
            raise ValueError("fraction must be in (0, 1]")
        self.fraction = fraction

    def __call__(self, kernel: np.ndarray) -> np.ndarray:
        _validate_square_kernel(kernel)
        n = kernel.shape[0]
        rows, cols = np.triu_indices(n, k=1)
        upper = kernel[rows, cols]
        k = max(1, int(np.ceil(self.fraction * len(upper))))
        top_k_idx = np.argpartition(upper, -k)[-k:]
        mask = np.zeros((n, n), dtype=bool)
        mask[rows[top_k_idx], cols[top_k_idx]] = True
        mask = mask | mask.T
        adjacency = np.where(mask, kernel, 0.0)
        np.fill_diagonal(adjacency, 0.0)
        return adjacency
