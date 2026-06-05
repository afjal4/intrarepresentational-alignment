from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

DEFAULT_PERMUTATIONS = 1000
DEFAULT_GED_THRESHOLD = 0.4


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


def ged(
    K: np.ndarray,
    L: np.ndarray,
    threshold: float = DEFAULT_GED_THRESHOLD,
    source_threshold: float | None = None,
    target_threshold: float | None = None,
) -> float:
    """Weighted GED between two weighted kernel matrices.

    Applies threshold sparsification first, then compares surviving edge weights:

        K'_ij = K_ij if K_ij > tau_S else 0
        L'_ij = L_ij if L_ij > tau_T else 0
        d(K, L) = sum_{i<j} |K'_ij - L'_ij| / sum_{i<j} (K'_ij + L'_ij)

    If source_threshold/target_threshold are not provided, both default to
    the shared *threshold* value.
    """
    _validate_square_same_shape(K, L)
    idx = np.triu_indices(K.shape[0], k=1)
    k, l = K[idx], L[idx]

    tau_s = threshold if source_threshold is None else source_threshold
    tau_t = threshold if target_threshold is None else target_threshold

    k_sparse = np.where(k > tau_s, k, 0.0)
    l_sparse = np.where(l > tau_t, l, 0.0)

    cost = np.sum(np.abs(k_sparse - l_sparse))
    total = np.sum(k_sparse) + np.sum(l_sparse)
    return float(cost / total) if total > 0 else 0.0


class _GEDStatistic(_KLStatistic):
    def __init__(
        self,
        threshold: float,
        source_threshold: float | None = None,
        target_threshold: float | None = None,
    ) -> None:
        super().__init__()
        self._threshold = threshold
        self._source_threshold = source_threshold
        self._target_threshold = target_threshold

    def observed(self) -> float:
        K, L = self._require_setup()
        return ged(
            K,
            L,
            self._threshold,
            source_threshold=self._source_threshold,
            target_threshold=self._target_threshold,
        )

    def score_permutation(self, pi: np.ndarray) -> float:
        K, L = self._require_setup()
        return ged(
            K,
            L[np.ix_(pi, pi)],
            self._threshold,
            source_threshold=self._source_threshold,
            target_threshold=self._target_threshold,
        )

    def p_value(self, observed: float, null: np.ndarray) -> float:
        return float((null <= observed).mean())


def ged_permutation_test(
    K: np.ndarray,
    L: np.ndarray,
    threshold: float = DEFAULT_GED_THRESHOLD,
    source_threshold: float | None = None,
    target_threshold: float | None = None,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    rng: np.random.Generator | None = None,
) -> PermutationTestResult:
    """Test whether thresholded weighted GED(K, L) is significantly below chance.

    If source_threshold/target_threshold are omitted, both use *threshold*.
    """
    if rng is None:
        rng = np.random.default_rng()
    return _run_permutation_test(
        K=K,
        L=L,
        statistic=_GEDStatistic(
            threshold=threshold,
            source_threshold=source_threshold,
            target_threshold=target_threshold,
        ),
        n_permutations=n_permutations,
        rng=rng,
    )
