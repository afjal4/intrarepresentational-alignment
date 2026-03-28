from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

DEFAULT_PERMUTATIONS = 1000
DEFAULT_GED_THRESHOLD = 0.0
DEFAULT_WL_ITERATIONS = 3


def _validate_square_same_shape(K: np.ndarray, L: np.ndarray) -> None:
    if K.ndim != 2 or L.ndim != 2:
        raise ValueError("K and L must be 2D matrices")
    if K.shape[0] != K.shape[1] or L.shape[0] != L.shape[1]:
        raise ValueError("K and L must be square matrices")
    if K.shape != L.shape:
        raise ValueError("K and L must have the same shape")


def _validate_n_permutations(n_permutations: int) -> None:
    if n_permutations <= 0:
        raise ValueError("n_permutations must be > 0")


def _center_kernel(kernel: np.ndarray) -> np.ndarray:
    return kernel - kernel.mean(1, keepdims=True) - kernel.mean(0, keepdims=True) + kernel.mean()


def _cka_from_centered(K_c: np.ndarray, L_c: np.ndarray) -> float:
    hsic_kl = np.sum(K_c * L_c)
    hsic_kk = np.sum(K_c * K_c)
    hsic_ll = np.sum(L_c * L_c)
    denom = np.sqrt(hsic_kk * hsic_ll)
    if denom == 0:
        return 0.0
    return float(hsic_kl / denom)


def cka(K: np.ndarray, L: np.ndarray) -> float:
    """Centered Kernel Alignment between two kernel matrices."""
    _validate_square_same_shape(K, L)
    return _cka_from_centered(_center_kernel(K), _center_kernel(L))


@dataclass
class PermutationTestResult:
    observed: float
    null: np.ndarray
    p_value: float


class _PermutationStatistic(ABC):
    @abstractmethod
    def setup(self, K: np.ndarray, L: np.ndarray) -> None:
        ...

    @abstractmethod
    def observed(self) -> float:
        ...

    @abstractmethod
    def score_permutation(self, pi: np.ndarray) -> float:
        ...

    @abstractmethod
    def p_value(self, observed: float, null: np.ndarray) -> float:
        ...


def _run_permutation_test(
    K: np.ndarray,
    L: np.ndarray,
    statistic: _PermutationStatistic,
    n_permutations: int,
    rng: np.random.Generator,
) -> PermutationTestResult:
    _validate_square_same_shape(K, L)
    _validate_n_permutations(n_permutations)

    statistic.setup(K, L)
    observed = statistic.observed()
    null = np.empty(n_permutations, dtype=float)
    n = K.shape[0]
    for i in range(n_permutations):
        null[i] = statistic.score_permutation(rng.permutation(n))

    return PermutationTestResult(
        observed=observed,
        null=null,
        p_value=statistic.p_value(observed, null),
    )


class _CKAStatistic(_PermutationStatistic):
    def __init__(self) -> None:
        self._K_c: np.ndarray | None = None
        self._L_c: np.ndarray | None = None
        self._denom: float | None = None
        self._degenerate = False

    def setup(self, K: np.ndarray, L: np.ndarray) -> None:
        K_c = _center_kernel(K)
        L_c = _center_kernel(L)
        hsic_kk = float(np.sum(K_c * K_c))
        hsic_ll = float(np.sum(L_c * L_c))
        denom = float(np.sqrt(hsic_kk * hsic_ll))
        if denom == 0:
            self._degenerate = True
            self._K_c = K_c
            self._L_c = L_c
            self._denom = 1.0
            return
        self._degenerate = False
        self._K_c = K_c
        self._L_c = L_c
        self._denom = denom

    def observed(self) -> float:
        if self._K_c is None or self._L_c is None or self._denom is None:
            raise RuntimeError("setup() must be called before observed()")
        if self._degenerate:
            return 0.0
        return float(np.sum(self._K_c * self._L_c) / self._denom)

    def score_permutation(self, pi: np.ndarray) -> float:
        if self._K_c is None or self._L_c is None or self._denom is None:
            raise RuntimeError("setup() must be called before score_permutation()")
        if self._degenerate:
            return 0.0
        permuted_L_c = self._L_c[np.ix_(pi, pi)]
        return float(np.sum(self._K_c * permuted_L_c) / self._denom)

    def p_value(self, observed: float, null: np.ndarray) -> float:
        return float((null >= observed).mean())


class _GEDStatistic(_PermutationStatistic):
    def __init__(self, threshold: float) -> None:
        self._threshold = threshold
        self._K: np.ndarray | None = None
        self._L: np.ndarray | None = None

    def setup(self, K: np.ndarray, L: np.ndarray) -> None:
        self._K = K
        self._L = L

    def observed(self) -> float:
        if self._K is None or self._L is None:
            raise RuntimeError("setup() must be called before observed()")
        return ged(self._K, self._L, self._threshold)

    def score_permutation(self, pi: np.ndarray) -> float:
        if self._K is None or self._L is None:
            raise RuntimeError("setup() must be called before score_permutation()")
        return ged(self._K, self._L[np.ix_(pi, pi)], self._threshold)

    def p_value(self, observed: float, null: np.ndarray) -> float:
        return float((null <= observed).mean())


def cka_permutation_test(
    K: np.ndarray,
    L: np.ndarray,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    rng: np.random.Generator | None = None,
) -> PermutationTestResult:
    """Test whether CKA(K, L) is significantly above chance."""
    if rng is None:
        rng = np.random.default_rng()
    return _run_permutation_test(
        K=K,
        L=L,
        statistic=_CKAStatistic(),
        n_permutations=n_permutations,
        rng=rng,
    )


def ged(K: np.ndarray, L: np.ndarray, threshold: float = DEFAULT_GED_THRESHOLD) -> float:
    """Graph Edit Distance between two weighted kernel matrices."""
    _validate_square_same_shape(K, L)
    idx = np.triu_indices(K.shape[0], k=1)
    k, l = K[idx], L[idx]

    k_present = k > threshold
    l_present = l > threshold

    both = k_present & l_present
    cost = np.sum(np.abs(k[both] - l[both]))
    cost += np.sum(k[k_present & ~l_present])
    cost += np.sum(l[~k_present & l_present])

    total = np.sum(k[k_present]) + np.sum(l[l_present])
    return float(cost / total) if total > 0 else 0.0


def ged_permutation_test(
    K: np.ndarray,
    L: np.ndarray,
    threshold: float = DEFAULT_GED_THRESHOLD,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    rng: np.random.Generator | None = None,
) -> PermutationTestResult:
    """Test whether GED(K, L) is significantly below chance."""
    if rng is None:
        rng = np.random.default_rng()
    return _run_permutation_test(
        K=K,
        L=L,
        statistic=_GEDStatistic(threshold=threshold),
        n_permutations=n_permutations,
        rng=rng,
    )


# ---------------------------------------------------------------------------
# Weisfeiler–Lehman structural distance
# ---------------------------------------------------------------------------

def _wl_norm_adj(A: np.ndarray) -> np.ndarray:
    """Row-normalise adjacency so rows sum to 1 (isolated nodes keep their row as-is)."""
    d = A.sum(axis=1, keepdims=True)
    d[d == 0.0] = 1.0
    return A / d


def _wl_sorted_l1(x_K: np.ndarray, x_L: np.ndarray) -> float:
    """Normalised L1 distance between two sorted feature vectors; range [0, 1]."""
    a = np.sort(x_K)
    b = np.sort(x_L)
    total = float(np.sum(np.abs(a)) + np.sum(np.abs(b)))
    if total == 0.0:
        return 0.0
    return float(np.sum(np.abs(a - b)) / total)


def wl_distance(
    K: np.ndarray,
    L: np.ndarray,
    n_iter: int = DEFAULT_WL_ITERATIONS,
) -> float:
    """Weisfeiler–Lehman structural distance between two weighted graphs.

    Initialises each node's feature as its weighted degree, then iteratively
    propagates features through the degree-normalised adjacency.  At every
    WL iteration the *sorted* node-feature distributions of the two graphs
    are compared with a normalised L1 distance.  The final score is the mean
    over all iterations (including iteration 0).

    Returns a value in [0, 1] where 0 = structurally identical.
    """
    _validate_square_same_shape(K, L)

    K_adj = K.copy()
    np.fill_diagonal(K_adj, 0.0)
    L_adj = L.copy()
    np.fill_diagonal(L_adj, 0.0)

    A_K = _wl_norm_adj(K_adj)
    A_L = _wl_norm_adj(L_adj)

    # Initial features: weighted degree
    x_K = K_adj.sum(axis=1)
    x_L = L_adj.sum(axis=1)

    total_dist = _wl_sorted_l1(x_K, x_L)
    for _ in range(n_iter):
        x_K = x_K + A_K @ x_K
        x_L = x_L + A_L @ x_L
        total_dist += _wl_sorted_l1(x_K, x_L)

    return total_dist / (n_iter + 1)


class _WLStatistic(_PermutationStatistic):
    def __init__(self, n_iter: int) -> None:
        self._n_iter = n_iter
        self._K: np.ndarray | None = None
        self._L: np.ndarray | None = None

    def setup(self, K: np.ndarray, L: np.ndarray) -> None:
        self._K = K
        self._L = L

    def observed(self) -> float:
        if self._K is None or self._L is None:
            raise RuntimeError("setup() must be called before observed()")
        return wl_distance(self._K, self._L, self._n_iter)

    def score_permutation(self, pi: np.ndarray) -> float:
        if self._K is None or self._L is None:
            raise RuntimeError("setup() must be called before score_permutation()")
        return wl_distance(self._K, self._L[np.ix_(pi, pi)], self._n_iter)

    def p_value(self, observed: float, null: np.ndarray) -> float:
        return float((null <= observed).mean())


def wl_permutation_test(
    K: np.ndarray,
    L: np.ndarray,
    n_iter: int = DEFAULT_WL_ITERATIONS,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    rng: np.random.Generator | None = None,
) -> PermutationTestResult:
    """Test whether WL distance(K, L) is significantly below chance."""
    if rng is None:
        rng = np.random.default_rng()
    return _run_permutation_test(
        K=K,
        L=L,
        statistic=_WLStatistic(n_iter=n_iter),
        n_permutations=n_permutations,
        rng=rng,
    )


# ---------------------------------------------------------------------------
# Maximum Common Subgraph (MCS) distance
# ---------------------------------------------------------------------------

def mcs_distance(K: np.ndarray, L: np.ndarray) -> float:
    """Maximum Common Subgraph distance between two weighted graphs.

    For weighted graphs with a fixed node set, the maximum common weighted
    subgraph has edge weight ``min(K[i,j], L[i,j])`` on each edge.  The
    distance is a Dice-style dissimilarity:

        MCS_distance = 1 - 2 * sum(min(K,L)) / (sum(K) + sum(L))

    where sums run over the upper triangle (self-loops excluded).
    Returns a value in [0, 1] where 0 = identical edge-weight structure.
    """
    _validate_square_same_shape(K, L)
    idx = np.triu_indices(K.shape[0], k=1)
    k = K[idx]
    l = L[idx]

    common = np.sum(np.minimum(k, l))
    total = float(np.sum(k) + np.sum(l))
    if total == 0.0:
        return 0.0
    return float(1.0 - 2.0 * common / total)


class _MCSStatistic(_PermutationStatistic):
    def __init__(self) -> None:
        self._K: np.ndarray | None = None
        self._L: np.ndarray | None = None

    def setup(self, K: np.ndarray, L: np.ndarray) -> None:
        self._K = K
        self._L = L

    def observed(self) -> float:
        if self._K is None or self._L is None:
            raise RuntimeError("setup() must be called before observed()")
        return mcs_distance(self._K, self._L)

    def score_permutation(self, pi: np.ndarray) -> float:
        if self._K is None or self._L is None:
            raise RuntimeError("setup() must be called before score_permutation()")
        return mcs_distance(self._K, self._L[np.ix_(pi, pi)])

    def p_value(self, observed: float, null: np.ndarray) -> float:
        return float((null <= observed).mean())


def mcs_permutation_test(
    K: np.ndarray,
    L: np.ndarray,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    rng: np.random.Generator | None = None,
) -> PermutationTestResult:
    """Test whether MCS distance(K, L) is significantly below chance."""
    if rng is None:
        rng = np.random.default_rng()
    return _run_permutation_test(
        K=K,
        L=L,
        statistic=_MCSStatistic(),
        n_permutations=n_permutations,
        rng=rng,
    )
