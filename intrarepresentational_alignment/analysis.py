from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .alignment import (
    DEFAULT_GED_THRESHOLD,
    DEFAULT_PERMUTATIONS,
    PermutationTestResult,
    cka_permutation_test,
    ged_permutation_test,
)
from .data import LccInstance, load_lcc
from .embedding import Embedder
from .embedding_models import EmbeddingModel
from .graph import TopFraction
from .results import DomainAlignmentResult
from .similarity import (
    CosineSimilarity,
    MatrixComputationStrategy,
    SimilarityMetric,
    compute_similarity_matrix,
)

_DEFAULT_EXCLUDE: frozenset[str] = frozenset({"OTHER"})


@dataclass
class SetupResult:
    domain_result: DomainAlignmentResult
    K_S_sparse: np.ndarray
    K_T_sparse: np.ndarray
    src_texts: list[str]
    tgt_texts: list[str]


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


DEFAULT_SPARSE_CKA_FRACTION: float = 0.15


def analyse_domain_pairs(
    domain_pairs: dict[tuple[str, str], list[tuple[str, str]]],
    embedder: Embedder,
    metric: SimilarityMetric | None = None,
    matrix_strategy: MatrixComputationStrategy | None = None,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    ged_threshold: float = DEFAULT_GED_THRESHOLD,
    ged_source_threshold: float | None = None,
    ged_target_threshold: float | None = None,
    rng: np.random.Generator | None = None,
    min_pairs: int = 1,
    max_pairs: int | None = None,
    return_kernels: bool = False,
) -> dict[tuple[str, str], DomainAlignmentResult]:
    """Compute CKA and GED alignment for each (source_concept, target_concept) group.

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

        valid = (np.linalg.norm(S, axis=1) > 0) & (np.linalg.norm(T, axis=1) > 0)
        if valid.sum() < min_pairs:
            continue
        if valid.sum() < n:
            S, T = S[valid], T[valid]
            n = int(valid.sum())

        K_S = compute_similarity_matrix(metric, S, strategy=matrix_strategy)
        K_T = compute_similarity_matrix(metric, T, strategy=matrix_strategy)

        results[key] = DomainAlignmentResult(
            n_pairs=n,
            cka=cka_permutation_test(K_S, K_T, n_permutations=n_permutations, rng=rng),
            ged=ged_permutation_test(
                K_S,
                K_T,
                threshold=ged_threshold,
                source_threshold=ged_source_threshold,
                target_threshold=ged_target_threshold,
                n_permutations=n_permutations,
                rng=rng,
            ),
            K_source=K_S if return_kernels else None,
            K_target=K_T if return_kernels else None,
        )

    return results


_DEFAULT_GED_THRESHOLDS: list[float]  = [0.0, 0.1, 0.2, 0.3, 0.4]
_DEFAULT_SCKA_FRACTIONS: list[float]  = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def compute_kernel_pairs(
    domain_pairs: dict[tuple[str, str], list[tuple[str, str]]],
    embedder: Embedder,
    metric: SimilarityMetric | None = None,
    strategy: MatrixComputationStrategy | None = None,
    min_pairs: int = 1,
    max_pairs: int | None = None,
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    """Pre-compute (K_source, K_target) kernel pairs for each domain pair.

    Call this once before a parameter sweep to avoid re-embedding on every
    parameter value.
    """
    if metric is None:
        metric = CosineSimilarity()
    kernels: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for key, pairs in domain_pairs.items():
        n = len(pairs)
        if n < min_pairs or (max_pairs is not None and n > max_pairs):
            continue
        src_texts = [s for s, _ in pairs]
        tgt_texts = [t for _, t in pairs]
        S = embedder.embed(src_texts)
        T = embedder.embed(tgt_texts)
        kernels[key] = (
            compute_similarity_matrix(metric, S, strategy=strategy),
            compute_similarity_matrix(metric, T, strategy=strategy),
        )
    return kernels


def sweep_parameters(
    kernels: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    *,
    ged_thresholds: list[float] | None = None,
    sparse_cka_fractions: list[float] | None = None,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    rng_seed: int = 0,
) -> dict[str, dict]:
    """Sweep alignment metric parameters on pre-computed kernel pairs.

    Returns a dict mapping sweep name to {param_value: list[PermutationTestResult]},
    one result per domain pair in *kernels* order.  Pass the output directly to
    viz.plot_parameter_sweep().
    """
    _ged_threshs = ged_thresholds        or _DEFAULT_GED_THRESHOLDS
    _scka_fracs  = sparse_cka_fractions  or _DEFAULT_SCKA_FRACTIONS

    sweep: dict[str, dict] = {
        "ged_threshold":      {},
        "sparse_cka_fraction": {},
    }

    for thresh in _ged_threshs:
        rng = np.random.default_rng(rng_seed)
        sweep["ged_threshold"][thresh] = [
            ged_permutation_test(K_S, K_T, threshold=thresh,
                                 n_permutations=n_permutations, rng=rng)
            for K_S, K_T in kernels.values()
        ]

    for frac in _scka_fracs:
        sparsify = TopFraction(frac)
        rng = np.random.default_rng(rng_seed)
        sweep["sparse_cka_fraction"][frac] = [
            cka_permutation_test(sparsify(K_S), sparsify(K_T),
                                 n_permutations=n_permutations, rng=rng)
            for K_S, K_T in kernels.values()
        ]

    return sweep


def setup_domain_analysis(
    lcc_path: Path,
    source_concept: str,
    target_concept: str,
    *,
    n_perms: int,
    ged_threshold: float,
    seed: int,
    edge_fraction: float = DEFAULT_SPARSE_CKA_FRACTION,
    model: EmbeddingModel = EmbeddingModel.GLOVE_WIKI_GIGAWORD_300,
) -> SetupResult:
    """Load corpus, embed, and run alignment tests for a single domain pair.

    Returns a SetupResult with the full DomainAlignmentResult, visualisation-ready
    sparse kernel matrices, and the source/target expression lists.
    """
    instances = load_lcc(lcc_path)
    all_pairs = group_domain_pairs(instances)
    key = (source_concept, target_concept)
    if key not in all_pairs:
        available = sorted(all_pairs, key=lambda k: len(all_pairs[k]), reverse=True)
        raise ValueError(
            f"No pairs found for {source_concept!r} -> {target_concept!r}.\n"
            f"Top available pairs: {available[:10]}"
        )
    pairs = all_pairs[key]
    src_texts = [s for s, _ in pairs]
    tgt_texts = [t for _, t in pairs]

    embedder = Embedder(model)
    domain_result = analyse_domain_pairs(
        {key: pairs}, embedder,
        n_permutations=n_perms,
        ged_threshold=ged_threshold,
        rng=np.random.default_rng(seed),
        return_kernels=True,
    )[key]

    K_S_sparse = TopFraction(edge_fraction)(domain_result.K_source)
    K_T_sparse = TopFraction(edge_fraction)(domain_result.K_target)

    return SetupResult(
        domain_result=domain_result,
        K_S_sparse=K_S_sparse,
        K_T_sparse=K_T_sparse,
        src_texts=src_texts,
        tgt_texts=tgt_texts,
    )
