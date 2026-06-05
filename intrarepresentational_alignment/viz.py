from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .alignment import PermutationTestResult
from .results import DomainAlignmentResult

_GRAPH_NODE_SIZE = 4000
_GRAPH_LABEL_FONT_SIZE = 22.5
_GRAPH_EDGE_WIDTH_MAX = 3.5
_GRAPH_EDGE_EPSILON = 1e-9
_GRAPH_PAIR_LINE_WIDTH = 0.7

_SIG_COLOR = "#e74c3c"
_INSIG_COLOR = "#95a5a6"
_SIG_THRESHOLD = 0.05


def _is_significant(result: DomainAlignmentResult) -> bool:
    return any(
        getattr(result, attr).p_value < _SIG_THRESHOLD
        for attr in ("cka", "ged", "wl", "mcs")
    )


def plot_cka_bar(
    domain_results: dict[tuple[str, str], DomainAlignmentResult],
    save_path: Path | None = None,
) -> None:
    """Horizontal bar chart of CKA scores, coloured by significance."""
    keys   = sorted(domain_results, key=lambda k: domain_results[k].cka.observed, reverse=True)
    scores = [domain_results[k].cka.observed for k in keys]
    p_vals = [domain_results[k].cka.p_value  for k in keys]
    colors = [_SIG_COLOR if p < _SIG_THRESHOLD else _INSIG_COLOR for p in p_vals]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(range(len(keys)), scores, color=colors, edgecolor="white")
    for i, (score, p) in enumerate(zip(scores, p_vals)):
        ax.text(score + 0.002, i, f"p={p:.3f}", va="center", fontsize=8)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([f"{s}  ->  {t}" for s, t in keys], fontsize=8)
    ax.set_xlabel("CKA(K_source, K_target)", fontsize=10)
    ax.set_title(
        "Source-target kernel alignment per conceptual metaphor\n(red = significant at alpha=0.05)",
        fontsize=11,
    )
    ax.set_xlim(left=0)
    ax.legend(
        handles=[
            mpatches.Patch(color=_SIG_COLOR, label="p < 0.05"),
            mpatches.Patch(color=_INSIG_COLOR, label="p >= 0.05"),
        ],
        loc="lower right",
        fontsize=9,
    )
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def plot_ged_vs_cka(
    domain_results: dict[tuple[str, str], DomainAlignmentResult],
    save_path: Path | None = None,
) -> None:
    """Side-by-side horizontal bar charts comparing GED and CKA scores."""
    keys       = sorted(domain_results, key=lambda k: domain_results[k].ged.observed)
    ged_scores = [domain_results[k].ged.observed for k in keys]
    cka_scores = [domain_results[k].cka.observed for k in keys]
    ged_p      = [domain_results[k].ged.p_value  for k in keys]
    cka_p      = [domain_results[k].cka.p_value  for k in keys]
    labels     = [f"{s}  \u2192  {t}" for s, t in keys]

    y = np.arange(len(keys))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].barh(y, ged_scores, color=[_SIG_COLOR if p < _SIG_THRESHOLD else _INSIG_COLOR for p in ged_p], edgecolor="white")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_xlabel("Normalised GED  (lower = more similar)", fontsize=9)
    axes[0].set_title("Graph Edit Distance\n(red = p < 0.05)", fontsize=10)

    axes[1].barh(y, cka_scores, color=[_SIG_COLOR if p < _SIG_THRESHOLD else _INSIG_COLOR for p in cka_p], edgecolor="white")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(labels, fontsize=8)
    axes[1].set_xlabel("CKA  (higher = more similar)", fontsize=9)
    axes[1].set_title("CKA\n(red = p < 0.05)", fontsize=10)

    fig.suptitle("GED vs CKA \u2014 source\u2013target domain alignment", fontsize=11)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def plot_dense_vs_sparse(
    dense_results: dict[tuple[str, str], DomainAlignmentResult],
    sparse_results: dict[tuple[str, str], DomainAlignmentResult],
    edge_fraction: float,
    save_path: Path | None = None,
) -> None:
    """Paired horizontal bars comparing dense and sparse CKA scores."""
    keys          = sorted(dense_results, key=lambda k: dense_results[k].cka.observed, reverse=True)
    dense_scores  = [dense_results[k].cka.observed  for k in keys]
    sparse_scores = [sparse_results[k].cka.observed for k in keys]
    dense_p       = [dense_results[k].cka.p_value   for k in keys]
    sparse_p      = [sparse_results[k].cka.p_value  for k in keys]

    y, h = np.arange(len(keys)), 0.38
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(y + h / 2, dense_scores,  h, color="#3498db", alpha=0.85, label="Dense cosine")
    ax.barh(y - h / 2, sparse_scores, h, color="#e67e22", alpha=0.85,
            label=f"Sparse cosine (TopFraction {int(edge_fraction * 100)}%)")

    for i, (ds, ss, dp, sp) in enumerate(zip(dense_scores, sparse_scores, dense_p, sparse_p)):
        if dp < _SIG_THRESHOLD:
            ax.text(ds + 0.001, i + h / 2, "*", va="center", color="#3498db", fontweight="bold")
        if sp < _SIG_THRESHOLD:
            ax.text(ss + 0.001, i - h / 2, "*", va="center", color="#e67e22", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([f"{s}  \u2192  {t}" for s, t in keys], fontsize=9)
    ax.set_xlabel("CKA(K_source, K_target)", fontsize=10)
    ax.set_title("Dense vs sparse kernel \u2014 source-target alignment\n(* = p < 0.05)", fontsize=11)
    ax.set_xlim(left=0)
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def plot_kernel_heatmaps(
    K_S_dense: np.ndarray,
    K_T_dense: np.ndarray,
    K_S_sparse: np.ndarray,
    K_T_sparse: np.ndarray,
    source_concept: str,
    target_concept: str,
    edge_fraction: float,
    cka_res: PermutationTestResult,
    ged_res: PermutationTestResult,
) -> None:
    """2\u00d72 grid of kernel heatmaps: dense and sparse, source and target."""
    n = K_S_dense.shape[0]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    def _heat(ax, M, title, vmin=None, vmax=None):
        im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11, pad=6)
        ax.set_xlabel("item index", fontsize=9)
        ax.set_ylabel("item index", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    pct = int(edge_fraction * 100)
    _heat(axes[0, 0], K_S_dense,  f"K_source - dense\n{source_concept}  [n={n}]")
    _heat(axes[0, 1], K_T_dense,  f"K_target - dense\n{target_concept}  [n={n}]")
    _heat(axes[1, 0], K_S_sparse, f"K_source - top {pct}% of edges\n{source_concept}", vmin=0)
    _heat(axes[1, 1], K_T_sparse, f"K_target - top {pct}% of edges\n{target_concept}", vmin=0)

    fig.suptitle(
        f"{source_concept} -> {target_concept}  |  "
        f"CKA = {cka_res.observed:.3f} (p={cka_res.p_value:.3f})"
        f"  GED = {ged_res.observed:.3f} (p={ged_res.p_value:.3f})",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


def plot_semantic_graphs(
    K_S_sparse: np.ndarray,
    K_T_sparse: np.ndarray,
    src_texts: list[str],
    tgt_texts: list[str],
    source_concept: str,
    target_concept: str,
    edge_fraction: float,
) -> None:
    """Spring-layout graphs of the sparse source and target kernels."""

    def _draw(ax, G: nx.Graph, labels: list[str], title: str, color: str) -> dict:
        pos = nx.spring_layout(G, seed=42, weight="weight")
        weights = np.array([G[u][v]["weight"] for u, v in G.edges()]) if G.edges() else np.array([])
        w_scaled = (
            (0.5 + _GRAPH_EDGE_WIDTH_MAX * (weights - weights.min()) / (weights.max() - weights.min() + _GRAPH_EDGE_EPSILON)).tolist()
            if len(weights) else []
        )
        nx.draw_networkx_edges(ax=ax, G=G, pos=pos, width=w_scaled, alpha=0.55, edge_color="#888888")
        nx.draw_networkx_nodes(ax=ax, G=G, pos=pos, node_size=_GRAPH_NODE_SIZE, node_color=color, alpha=0.9)
        nx.draw_networkx_labels(
            ax=ax, G=G, pos=pos,
            labels={i: labels[i][:22] for i in G.nodes()},
            font_size=_GRAPH_LABEL_FONT_SIZE, font_color="#111111",
        )
        ax.set_title(f"{title}\n{G.number_of_nodes()} nodes  |  {G.number_of_edges()} edges", fontsize=11)
        ax.axis("off")
        return pos

    np.fill_diagonal(K_S_sparse, 0.0)
    np.fill_diagonal(K_T_sparse, 0.0)
    G_S = nx.from_numpy_array(K_S_sparse)
    G_T = nx.from_numpy_array(K_T_sparse)

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    pos_S = _draw(axes[0], G_S, src_texts, f"Source graph  ({source_concept} expressions)", "#3498db")
    pos_T = _draw(axes[1], G_T, tgt_texts, f"Target graph  ({target_concept} expressions)", "#e67e22")

    for i in range(len(src_texts)):
        if i not in pos_S or i not in pos_T:
            continue
        fig.add_artist(mpatches.ConnectionPatch(
            xyA=pos_S[i], xyB=pos_T[i],
            coordsA=axes[0].transData, coordsB=axes[1].transData,
            linestyle="dotted", linewidth=_GRAPH_PAIR_LINE_WIDTH, color="#999999", alpha=0.5, clip_on=False,
        ))

    fig.suptitle(
        f"{source_concept} -> {target_concept}  \u2014  "
        f"sparse semantic graphs (top {int(edge_fraction * 100)}% of edges)  |  dotted = matched pairs",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


def plot_null_distributions(
    cka_res: PermutationTestResult,
    ged_res: PermutationTestResult,
    source_concept: str,
    target_concept: str,
    wl_res: PermutationTestResult | None = None,
    mcs_res: PermutationTestResult | None = None,
) -> None:
    """Permutation null distribution histograms for CKA, GED, and optionally WL and MCS."""
    metrics = [
        (cka_res, "#3498db", "CKA  (higher = more similar)"),
        (ged_res, "#e67e22", "GED  (lower = more similar)"),
    ]
    if wl_res is not None:
        metrics.append((wl_res, "#2ecc71", "WL distance  (lower = more similar)"))
    if mcs_res is not None:
        metrics.append((mcs_res, "#9b59b6", "MCS distance  (lower = more similar)"))

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (res, color, xlabel) in zip(axes, metrics):
        ax.hist(res.null, bins=40, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(res.observed, color=_SIG_COLOR, linewidth=2.5,
                   label=f"observed = {res.observed:.3f}\np = {res.p_value:.3f}")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("count", fontsize=10)
        metric_name = xlabel.split()[0]
        ax.set_title(f"{metric_name} permutation null  \u2014  {source_concept} -> {target_concept}", fontsize=11)
        ax.legend(fontsize=10)

    plt.tight_layout()
    plt.show()


def print_domain_summary(
    result: DomainAlignmentResult,
    source_concept: str,
    target_concept: str,
) -> None:
    """Print CKA/GED/WL/MCS observed values and p-values for a single domain pair."""
    print(f"{source_concept} -> {target_concept}: {result.n_pairs} expression pairs")
    print(f"CKA = {result.cka.observed:.3f}  (p = {result.cka.p_value:.3f})")
    print(f"GED = {result.ged.observed:.3f}  (p = {result.ged.p_value:.3f})")
    print(f"WL  = {result.wl.observed:.3f}  (p = {result.wl.p_value:.3f})")
    print(f"MCS = {result.mcs.observed:.3f}  (p = {result.mcs.p_value:.3f})")


def print_cka_table(
    domain_results: dict[tuple[str, str], DomainAlignmentResult],
    keys: list[tuple[str, str]] | None = None,
) -> None:
    """Print a compact CKA score and p-value table."""
    if keys is None:
        keys = list(domain_results)
    header = f"{'Source domain':<28} {'Target domain':<22} {'n':>4}  {'CKA':>6}  {'p':>6}"
    print(header)
    print("-" * len(header))
    for key in keys:
        r = domain_results[key]
        src, tgt = key
        sig = " *" if r.cka.p_value < _SIG_THRESHOLD else ""
        print(f"{src:<28} {tgt:<22} {r.n_pairs:>4}  {r.cka.observed:>6.3f}  {r.cka.p_value:>6.3f}{sig}")


def print_alignment_table(
    domain_results: dict[tuple[str, str], DomainAlignmentResult],
    keys: list[tuple[str, str]] | None = None,
) -> None:
    """Print a CKA/GED/WL/MCS comparison table for the given domain pairs."""
    if keys is None:
        keys = list(domain_results)
    header = (
        f"{'Source domain':<28} {'Target domain':<22} "
        f"{'CKA':>6}  {'GED':>6}  {'WL':>6}  {'MCS':>6}  "
        f"{'p(CKA)':>7}  {'p(GED)':>7}  {'p(WL)':>6}  {'p(MCS)':>7}"
    )
    print(header)
    print("-" * len(header))
    for key in keys:
        r = domain_results[key]
        src, tgt = key
        print(
            f"{src:<28} {tgt:<22} "
            f"{r.cka.observed:>6.3f}  {r.ged.observed:>6.3f}  "
            f"{r.wl.observed:>6.3f}  {r.mcs.observed:>6.3f}  "
            f"{r.cka.p_value:>6.3f}{'*' if r.cka.p_value < _SIG_THRESHOLD else ' '}  "
            f"{r.ged.p_value:>6.3f}{'*' if r.ged.p_value < _SIG_THRESHOLD else ' '}  "
            f"{r.wl.p_value:>5.3f}{'*' if r.wl.p_value < _SIG_THRESHOLD else ' '}  "
            f"{r.mcs.p_value:>6.3f}{'*' if r.mcs.p_value < _SIG_THRESHOLD else ' '}"
        )


def print_summary_table(
    domain_results: dict[tuple[str, str], DomainAlignmentResult],
    keys: list[tuple[str, str]] | None = None,
) -> None:
    """Print significant results only, sorted by CKA descending."""
    if keys is None:
        keys = list(domain_results)

    interesting = sorted(
        [k for k in keys if _is_significant(domain_results[k])],
        key=lambda k: domain_results[k].cka.observed,
        reverse=True,
    )
    header = (
        f"{'Source domain':<22}  {'Target domain':<22}  {'n':>4}  "
        f"{'CKA':>6}  {'p(CKA)':>7}  {'GED':>6}  {'p(GED)':>7}  "
        f"{'WL':>6}  {'p(WL)':>6}  {'MCS':>6}  {'p(MCS)':>7}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)
    for key in interesting:
        src, tgt = key
        r = domain_results[key]
        print(
            f"{src:<22}  {tgt:<22}  {r.n_pairs:>4}  "
            f"{r.cka.observed:>6.3f}  {r.cka.p_value:>6.3f}{'*' if r.cka.p_value < _SIG_THRESHOLD else ' '}  "
            f"{r.ged.observed:>6.3f}  {r.ged.p_value:>6.3f}{'*' if r.ged.p_value < _SIG_THRESHOLD else ' '}  "
            f"{r.wl.observed:>6.3f}  {r.wl.p_value:>5.3f}{'*' if r.wl.p_value < _SIG_THRESHOLD else ' '}  "
            f"{r.mcs.observed:>6.3f}  {r.mcs.p_value:>6.3f}{'*' if r.mcs.p_value < _SIG_THRESHOLD else ' '}"
        )
    print(sep)
    print(f"  {len(interesting)} of {len(keys)} domain pairs significant at p < 0.05")


def print_significance_summary(
    dense_results: dict[tuple[str, str], DomainAlignmentResult],
    sparse_results: dict[tuple[str, str], DomainAlignmentResult],
    keys: list[tuple[str, str]] | None = None,
) -> None:
    """Print significance counts and CKA-vs-GED overlap for dense and sparse kernels."""
    if keys is None:
        keys = list(dense_results)
    n = len(keys)

    cka_sig = {k for k in keys if dense_results[k].cka.p_value < _SIG_THRESHOLD}
    ged_sig = {k for k in keys if sparse_results[k].ged.p_value < _SIG_THRESHOLD}
    overlap = cka_sig & ged_sig
    ged_new = ged_sig - cka_sig
    cka_only = cka_sig - ged_sig
    either = cka_sig | ged_sig

    print(
        f"CKA identifies significant alignment between {len(cka_sig)} metaphor-domain pairs."
    )
    print(
        f"Sparse GED identifies {len(ged_sig)} significant pairs, "
        f"{len(overlap)} of which were previously identified by CKA."
    )
    print(
        f"Sparse GED contributes {len(ged_new)} new pairs beyond dense CKA "
        f"({len(cka_only)} dense-CKA-only; {len(either)} pairs significant by either metric in total)."
    )
    if ged_new:
        for src, tgt in sorted(ged_new):
            print(f"  GED-only: {src} -> {tgt}")

    metrics = ("cka", "ged", "wl", "mcs")
    labels  = ("CKA", "GED", "WL", "MCS")

    def _sig_count(results: dict, metric: str) -> int:
        return sum(1 for k in keys if getattr(results[k], metric).p_value < _SIG_THRESHOLD)

    dense_counts  = [_sig_count(dense_results,  m) for m in metrics]
    sparse_counts = [_sig_count(sparse_results, m) for m in metrics]

    dense_any  = sum(1 for k in keys if _is_significant(dense_results[k]))
    sparse_any = sum(1 for k in keys if _is_significant(sparse_results[k]))

    col = 10
    header = f"{'Metric':<8}  {'Dense':>{col}}  {'Sparse':>{col}}"
    sep    = "-" * len(header)
    print(header)
    print(sep)
    for label, dc, sc in zip(labels, dense_counts, sparse_counts):
        print(f"{label:<8}  {dc:>{col}} / {n}  {sc:>{col}} / {n}")
    print(sep)
    print(f"{'Any':<8}  {dense_any:>{col}} / {n}  {sparse_any:>{col}} / {n}")


def plot_model_comparison_heatmap(
    model_results: dict[str, dict[tuple[str, str], DomainAlignmentResult]],
    metric: str = "cka",
    save_path: Path | None = None,
) -> None:
    """Heatmap of alignment scores per domain pair (rows) × model (columns).

    Cells are hatched where the result is significant (p < 0.05).
    Use metric='cka' for similarity metrics (higher = more similar) or
    'ged'/'wl'/'mcs' for distance metrics (lower = more similar).
    """
    model_labels = list(model_results.keys())
    all_keys: set[tuple[str, str]] = set()
    for res in model_results.values():
        all_keys.update(res.keys())

    # Sort rows by mean observed value across models (desc for similarity, asc for distance)
    distance_metrics = {"ged", "wl", "mcs"}
    def _row_mean(key: tuple[str, str]) -> float:
        vals = [
            getattr(model_results[m][key], metric).observed
            for m in model_labels
            if key in model_results[m]
        ]
        return float(np.mean(vals)) if vals else 0.0

    reverse = metric not in distance_metrics
    row_keys = sorted(all_keys, key=_row_mean, reverse=reverse)

    n_rows = len(row_keys)
    n_cols = len(model_labels)
    data = np.full((n_rows, n_cols), np.nan)
    sig = np.zeros((n_rows, n_cols), dtype=bool)

    for j, label in enumerate(model_labels):
        res = model_results[label]
        for i, key in enumerate(row_keys):
            if key in res:
                r = getattr(res[key], metric)
                data[i, j] = r.observed
                sig[i, j] = r.p_value < _SIG_THRESHOLD

    cmap = "viridis_r" if metric in distance_metrics else "viridis"
    fig, ax = plt.subplots(figsize=(max(8, 2.5 * n_cols), max(6, 0.35 * n_rows + 2)))
    im = ax.imshow(data, aspect="auto", cmap=cmap, interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label=metric.upper())

    # Hatch significant cells
    for i in range(n_rows):
        for j in range(n_cols):
            if sig[i, j]:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, hatch="//", edgecolor="white", linewidth=0,
                ))
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if data[i, j] < np.nanmax(data) * 0.6 else "#111")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(model_labels, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([f"{s}  →  {t}" for s, t in row_keys], fontsize=7)
    direction = "lower = more similar" if metric in distance_metrics else "higher = more similar"
    ax.set_title(
        f"{metric.upper()} per domain pair × embedding model  ({direction})\n"
        f"hatched = p < 0.05",
        fontsize=11,
    )
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def plot_model_comparison_significance(
    model_results: dict[str, dict[tuple[str, str], DomainAlignmentResult]],
    save_path: Path | None = None,
) -> None:
    """Grouped bar chart: # significant domain pairs per model, broken down by metric."""
    model_labels = list(model_results.keys())
    metrics  = ("cka", "ged", "wl", "mcs")
    labels   = ("CKA", "GED", "WL", "MCS")
    colors   = ("#3498db", "#e67e22", "#2ecc71", "#9b59b6")

    counts = np.zeros((len(model_labels), len(metrics)), dtype=int)
    for j, label in enumerate(model_labels):
        res = model_results[label]
        for i, m in enumerate(metrics):
            counts[j, i] = sum(
                1 for r in res.values()
                if getattr(r, m).p_value < _SIG_THRESHOLD
            )

    x = np.arange(len(model_labels))
    n_metrics = len(metrics)
    w = 0.18
    offsets = np.linspace(-(n_metrics - 1) / 2, (n_metrics - 1) / 2, n_metrics) * w

    fig, ax = plt.subplots(figsize=(max(8, 1.8 * len(model_labels)), 5))
    for i, (label, color, offset) in enumerate(zip(labels, colors, offsets)):
        bars = ax.bar(x + offset, counts[:, i], w, label=label, color=color, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, counts[:, i]):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                        str(val), ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=9)
    ax.set_ylabel("# significant domain pairs (p < 0.05)", fontsize=10)
    ax.set_title("Significant alignment detections per embedding model and metric", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()


def print_dense_vs_sparse_table(
    dense_results: dict[tuple[str, str], DomainAlignmentResult],
    sparse_results: dict[tuple[str, str], DomainAlignmentResult],
    keys: list[tuple[str, str]] | None = None,
) -> None:
    """Print a comparison of dense vs. sparse CKA scores."""
    if keys is None:
        keys = list(dense_results)
    header = (
        f"{'Source domain':<28} {'Target domain':<22} "
        f"{'CKA(dense)':>10}  {'CKA(sparse)':>11}  {'p(dense)':>8}  {'p(sparse)':>9}"
    )
    print(header)
    print("-" * len(header))
    for key in keys:
        src, tgt = key
        dense    = dense_results[key]
        sparse_r = sparse_results[key]
        sig_d = " *" if dense.cka.p_value    < _SIG_THRESHOLD else "  "
        sig_s = " *" if sparse_r.cka.p_value < _SIG_THRESHOLD else "  "
        print(
            f"{src:<28} {tgt:<22} "
            f"{dense.cka.observed:>10.3f}  {sparse_r.cka.observed:>11.3f}  "
            f"{dense.cka.p_value:>8.3f}{sig_d}  {sparse_r.cka.p_value:>9.3f}{sig_s}"
        )
