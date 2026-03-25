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


def ged(K: np.ndarray, L: np.ndarray, threshold: float = 0.0) -> float:
    """Graph Edit Distance between two weighted kernel matrices.

    Node correspondences are assumed fixed: both matrices are over the same
    N items in the same order (the paired S→T expressions), so GED reduces
    to a sum of per-edge edit costs over the upper triangle:

      - Deletion  (edge in K, absent in L):  cost = K[i,j]
      - Insertion (edge in L, absent in K):  cost = L[i,j]
      - Substitution (edge in both):         cost = |K[i,j] - L[i,j]|
      - Absent in both:                      cost = 0

    Entries at or below `threshold` are treated as absent edges.
    The result is normalised by the total edge weight of both graphs so
    it lies in [0, 1] regardless of matrix size or density.
    """
    idx = np.triu_indices(K.shape[0], k=1)
    k, l = K[idx], L[idx]

    k_present = k > threshold
    l_present = l > threshold

    cost  = np.sum(np.abs(k[k_present & l_present] - l[k_present & l_present]))
    cost += np.sum(k[k_present & ~l_present])
    cost += np.sum(l[~k_present & l_present])

    total = np.sum(k[k_present]) + np.sum(l[l_present])
    return float(cost / total) if total > 0 else 0.0


def ged_permutation_test(
    K: np.ndarray,
    L: np.ndarray,
    threshold: float = 0.0,
    n_permutations: int = 1000,
    rng: np.random.Generator | None = None,
) -> PermutationTestResult:
    """Test whether GED(K, L) is significantly below chance (lower = more similar).

    Each permutation shuffles the row/column order of L, breaking the S→T
    correspondence.  The p-value is the fraction of null GEDs <= observed GED.
    """
    if rng is None:
        rng = np.random.default_rng()

    observed = ged(K, L, threshold)
    null = np.empty(n_permutations)
    n = K.shape[0]

    for i in range(n_permutations):
        pi = rng.permutation(n)
        null[i] = ged(K, L[np.ix_(pi, pi)], threshold)

    p_value = float((null <= observed).mean())
    return PermutationTestResult(observed=observed, null=null, p_value=p_value)
