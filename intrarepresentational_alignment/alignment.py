from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def cka(K: np.ndarray, L: np.ndarray) -> float:
    """Centered Kernel Alignment between two kernel matrices.

    CKA(K, L) = HSIC(K, L) / sqrt(HSIC(K, K) * HSIC(L, L))

    where HSIC is estimated as tr(K_c L_c) with K_c = HKH the doubly-centred
    kernel (H = I - 11^T/n).  Both K and L must be square and the same size.
    Range: [0, 1].  A value of 1 means the two representations are identical
    up to an orthogonal transformation and scaling.
    """
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    K_c = H @ K @ H
    L_c = H @ L @ H
    hsic_kl = np.sum(K_c * L_c)
    hsic_kk = np.sum(K_c * K_c)
    hsic_ll = np.sum(L_c * L_c)
    return float(hsic_kl / np.sqrt(hsic_kk * hsic_ll))


@dataclass
class PermutationTestResult:
    observed: float
    null: np.ndarray   # CKA scores under the null
    p_value: float


def cka_permutation_test(
    K: np.ndarray,
    L: np.ndarray,
    n_permutations: int = 1000,
    rng: np.random.Generator | None = None,
) -> PermutationTestResult:
    """Test whether CKA(K, L) is significantly above chance.

    Under the null hypothesis the pairing between the N items in K and the N
    items in L is arbitrary.  Each permutation shuffles the row/column order of
    L (equivalently, breaks the S→T correspondence) and recomputes CKA.  The
    p-value is the fraction of null scores >= the observed score.
    """
    if rng is None:
        rng = np.random.default_rng()

    observed = cka(K, L)
    null = np.empty(n_permutations)
    n = K.shape[0]

    for i in range(n_permutations):
        pi = rng.permutation(n)
        null[i] = cka(K, L[np.ix_(pi, pi)])

    p_value = float((null >= observed).mean())
    return PermutationTestResult(observed=observed, null=null, p_value=p_value)
