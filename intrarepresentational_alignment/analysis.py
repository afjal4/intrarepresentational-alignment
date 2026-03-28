from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .alignment import (
    DEFAULT_GED_THRESHOLD,
    DEFAULT_PERMUTATIONS,
    DEFAULT_WL_ITERATIONS,
    PermutationTestResult,
    cka_permutation_test,
    ged_permutation_test,
    mcs_permutation_test,
    wl_permutation_test,
)
from .data import LccInstance
from .embedding import Embedder
from .similarity import (
    CosineSimilarity,
    MatrixComputationStrategy,
    SimilarityMetric,
    compute_similarity_matrix,
)

_DEFAULT_EXCLUDE: frozenset[str] = frozenset({"OTHER"})


@dataclass
class DomainAlignmentResult:
    n_pairs: int
    cka: PermutationTestResult
    ged: PermutationTestResult
    wl: PermutationTestResult
    mcs: PermutationTestResult
    K_source: np.ndarray | None = None
    K_target: np.ndarray | None = None


def group_domain_pairs(
    instances: list[LccInstance],
    exclude_source: frozenset[str] = _DEFAULT_EXCLUDE,
) -> dict[tuple[str, str], list[tuple[str, str]]]:
    """Group (source_expr, target_expr) pairs by (source_concept, target_concept).

    Instances without a source concept or with a concept in *exclude_source* are skipped.
    """
    pairs: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for inst in instances:
        if not inst.source_concept or inst.source_concept in exclude_source:
            continue
        for s in inst.source_expressions:
            for t in inst.target_expressions:
                pairs[(inst.source_concept, inst.target_concept)].append((s, t))
    return dict(pairs)


def analyse_domain_pairs(
    domain_pairs: dict[tuple[str, str], list[tuple[str, str]]],
    embedder: Embedder,
    metric: SimilarityMetric | None = None,
    matrix_strategy: MatrixComputationStrategy | None = None,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    ged_threshold: float = DEFAULT_GED_THRESHOLD,
    wl_n_iter: int = DEFAULT_WL_ITERATIONS,
    rng: np.random.Generator | None = None,
    min_pairs: int = 1,
    max_pairs: int | None = None,
    return_kernels: bool = False,
) -> dict[tuple[str, str], DomainAlignmentResult]:
    """Compute CKA, GED, WL and MCS alignment for each (source_concept, target_concept) group.

    Returns a result for every group in *domain_pairs* that meets the
    *min_pairs* / *max_pairs* size constraints.
    """
    if metric is None:
        metric = CosineSimilarity()
    if rng is None:
        rng = np.random.default_rng()

    results: dict[tuple[str, str], DomainAlignmentResult] = {}
    for key, pairs in domain_pairs.items():
        n = len(pairs)
        if n < min_pairs:
            continue
        if max_pairs is not None and n > max_pairs:
            continue

        src_texts = [s for s, _ in pairs]
        tgt_texts = [t for _, t in pairs]

        S = embedder.embed(src_texts)
        T = embedder.embed(tgt_texts)
        K_S = compute_similarity_matrix(metric, S, strategy=matrix_strategy)
        K_T = compute_similarity_matrix(metric, T, strategy=matrix_strategy)

        results[key] = DomainAlignmentResult(
            n_pairs=n,
            cka=cka_permutation_test(K_S, K_T, n_permutations=n_permutations, rng=rng),
            ged=ged_permutation_test(K_S, K_T, threshold=ged_threshold, n_permutations=n_permutations, rng=rng),
            wl=wl_permutation_test(K_S, K_T, n_iter=wl_n_iter, n_permutations=n_permutations, rng=rng),
            mcs=mcs_permutation_test(K_S, K_T, n_permutations=n_permutations, rng=rng),
            K_source=K_S if return_kernels else None,
            K_target=K_T if return_kernels else None,
        )

    return results
