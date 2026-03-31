from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import networkx as nx
import numpy as np

DEFAULT_PERMUTATIONS = 1000
DEFAULT_GED_THRESHOLD = 0.4
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


class _KLStatistic(_PermutationStatistic, ABC):
    """Base for permutation statistics that store K and L."""

    def __init__(self) -> None:
        self._K: np.ndarray | None = None
        self._L: np.ndarray | None = None

    def setup(self, K: np.ndarray, L: np.ndarray) -> None:
        self._K = K
        self._L = L

    def _require_setup(self) -> tuple[np.ndarray, np.ndarray]:
        if self._K is None or self._L is None:
            raise RuntimeError("setup() must be called before use")
        return self._K, self._L


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


class _GEDStatistic(_KLStatistic):
    def __init__(self, threshold: float) -> None:
        super().__init__()
        self._threshold = threshold

    def observed(self) -> float:
        K, L = self._require_setup()
        return ged(K, L, self._threshold)

    def score_permutation(self, pi: np.ndarray) -> float:
        K, L = self._require_setup()
        return ged(K, L[np.ix_(pi, pi)], self._threshold)

    def p_value(self, observed: float, null: np.ndarray) -> float:
        return float((null <= observed).mean())


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


def _wl_norm_adj(A: np.ndarray) -> np.ndarray:
    """Row-normalise adjacency so rows sum to 1 (isolated nodes keep their row as-is)."""
    d = A.sum(axis=1, keepdims=True)
    d[d == 0.0] = 1.0
    return A / d


def _wl_sorted_l1(x_K: np.ndarray, x_L: np.ndarray) -> float:
    """Normalised L1 distance between two sorted feature vectors; range [0, 1]."""
    total = float(np.sum(np.abs(x_K)) + np.sum(np.abs(x_L)))
    if total == 0.0:
        return 0.0
    return float(np.sum(np.abs(x_K - x_L)) / total)


def _wl_from_precomputed(
    A_K: np.ndarray,
    x_K_init: np.ndarray,
    L: np.ndarray,
    n_iter: int,
) -> float:
    """WL distance using a precomputed K-side normalised adjacency and initial features."""
    L_adj = L.copy()
    np.fill_diagonal(L_adj, 0.0)
    A_L = _wl_norm_adj(L_adj)

    x_K = x_K_init.copy()
    x_L = L_adj.sum(axis=1)

    total_dist = _wl_sorted_l1(x_K, x_L)
    for _ in range(n_iter):
        x_K = x_K + A_K @ x_K
        x_L = x_L + A_L @ x_L
        total_dist += _wl_sorted_l1(x_K, x_L)

    return total_dist / (n_iter + 1)


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
    return _wl_from_precomputed(_wl_norm_adj(K_adj), K_adj.sum(axis=1), L, n_iter)


class _WLStatistic(_KLStatistic):
    def __init__(self, n_iter: int) -> None:
        super().__init__()
        self._n_iter = n_iter
        self._A_K: np.ndarray | None = None
        self._x_K_init: np.ndarray | None = None

    def setup(self, K: np.ndarray, L: np.ndarray) -> None:
        super().setup(K, L)
        K_adj = K.copy()
        np.fill_diagonal(K_adj, 0.0)
        self._A_K = _wl_norm_adj(K_adj)
        self._x_K_init = K_adj.sum(axis=1)

    def observed(self) -> float:
        _, L = self._require_setup()
        return _wl_from_precomputed(self._A_K, self._x_K_init, L, self._n_iter)

    def score_permutation(self, pi: np.ndarray) -> float:
        _, L = self._require_setup()
        return _wl_from_precomputed(self._A_K, self._x_K_init, L[np.ix_(pi, pi)], self._n_iter)

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


def _mcis_size(K_bin: np.ndarray, L_bin: np.ndarray) -> int:
    """Node count of the maximum common induced subgraph of two binary adjacency matrices.

    Builds the conflict graph C where {i,j} is an edge iff the two graphs disagree
    on the edge between i and j.  The MCIS node count equals the maximum independent
    set of C, which equals the maximum clique of complement(C).  Uses Bron-Kerbosch
    via networkx.  Tractable for n <= ~35.
    """
    n = K_bin.shape[0]
    rows, cols = np.triu_indices(n, k=1)
    disagree = K_bin[rows, cols] != L_bin[rows, cols]

    conflict = nx.Graph()
    conflict.add_nodes_from(range(n))
    for r, c in zip(rows[disagree].tolist(), cols[disagree].tolist()):
        conflict.add_edge(r, c)

    complement = nx.complement(conflict)
    return max(len(clique) for clique in nx.find_cliques(complement))


def mcs_distance(K: np.ndarray, L: np.ndarray, threshold: float = DEFAULT_GED_THRESHOLD) -> float:
    """Maximum Common Induced Subgraph distance between two binary graphs (Bunke 1998).

    Edges are present if their weight exceeds *threshold*.  The MCIS of two
    labeled graphs on the same node set is the largest U such that G_K[U] = G_L[U]
    (identical induced subgraphs).  Distance is:

        MCSD(K, L) = 1 - m(K, L) / n

    where m is the node count of the MCIS and n is the total node count.
    Returns a value in [0, 1] where 0 = identical graphs.
    """
    _validate_square_same_shape(K, L)
    n = K.shape[0]
    if n == 0:
        return 0.0

    idx = np.triu_indices(n, k=1)
    K_bin = (K[idx] > threshold).astype(np.uint8)
    L_bin = (L[idx] > threshold).astype(np.uint8)

    K_adj = np.zeros((n, n), dtype=np.uint8)
    L_adj = np.zeros((n, n), dtype=np.uint8)
    rows, cols = idx
    K_adj[rows, cols] = K_bin; K_adj[cols, rows] = K_bin
    L_adj[rows, cols] = L_bin; L_adj[cols, rows] = L_bin

    m = _mcis_size(K_adj, L_adj)
    return float(1.0 - m / n)


class _MCSStatistic(_KLStatistic):
    def __init__(self, threshold: float) -> None:
        super().__init__()
        self._threshold = threshold

    def observed(self) -> float:
        K, L = self._require_setup()
        return mcs_distance(K, L, self._threshold)

    def score_permutation(self, pi: np.ndarray) -> float:
        K, L = self._require_setup()
        return mcs_distance(K, L[np.ix_(pi, pi)], self._threshold)

    def p_value(self, observed: float, null: np.ndarray) -> float:
        return float((null <= observed).mean())


def mcs_permutation_test(
    K: np.ndarray,
    L: np.ndarray,
    threshold: float = DEFAULT_GED_THRESHOLD,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    rng: np.random.Generator | None = None,
) -> PermutationTestResult:
    """Test whether MCS distance(K, L) is significantly below chance."""
    if rng is None:
        rng = np.random.default_rng()
    return _run_permutation_test(
        K=K,
        L=L,
        statistic=_MCSStatistic(threshold=threshold),
        n_permutations=n_permutations,
        rng=rng,
    )
