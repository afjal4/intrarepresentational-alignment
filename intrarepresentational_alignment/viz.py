from __future__ import annotations

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .alignment import PermutationTestResult
from .analysis import DomainAlignmentResult


def plot_cka_bar(
    domain_results: dict[tuple[str, str], DomainAlignmentResult],
) -> None:
    """Horizontal bar chart of CKA scores, coloured by significance."""
    keys   = sorted(domain_results, key=lambda k: domain_results[k].cka.observed, reverse=True)
    scores = [domain_results[k].cka.observed for k in keys]
    p_vals = [domain_results[k].cka.p_value  for k in keys]
    colors = ["#e74c3c" if p < 0.05 else "#95a5a6" for p in p_vals]

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
            mpatches.Patch(color="#e74c3c", label="p < 0.05"),
            mpatches.Patch(color="#95a5a6", label="p >= 0.05"),
        ],
        loc="lower right",
        fontsize=9,
    )
    plt.tight_layout()
    plt.show()


def plot_ged_vs_cka(
    domain_results: dict[tuple[str, str], DomainAlignmentResult],
) -> None:
    """Side-by-side horizontal bar charts comparing GED and CKA scores."""
    keys       = sorted(domain_results, key=lambda k: domain_results[k].ged.observed)
    ged_scores = [domain_results[k].ged.observed for k in keys]
    cka_scores = [domain_results[k].cka.observed for k in keys]
    ged_p      = [domain_results[k].ged.p_value  for k in keys]
    labels     = [f"{s}  \u2192  {t}" for s, t in keys]

    y = np.arange(len(keys))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].barh(y, ged_scores, color=["#e74c3c" if p < 0.05 else "#95a5a6" for p in ged_p], edgecolor="white")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_xlabel("Normalised GED  (lower = more similar)", fontsize=9)
    axes[0].set_title("Graph Edit Distance\n(red = p < 0.05)", fontsize=10)

    axes[1].barh(y, cka_scores,
                 color=["#e74c3c" if domain_results[k].cka.p_value < 0.05 else "#95a5a6" for k in keys],
                 edgecolor="white")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(labels, fontsize=8)
    axes[1].set_xlabel("CKA  (higher = more similar)", fontsize=9)
    axes[1].set_title("CKA\n(red = p < 0.05)", fontsize=10)

    fig.suptitle("GED vs CKA \u2014 source\u2013target domain alignment", fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_dense_vs_sparse(
    dense_results: dict[tuple[str, str], DomainAlignmentResult],
    sparse_results: dict[tuple[str, str], DomainAlignmentResult],
    edge_fraction: float,
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
        if dp < 0.05:
            ax.text(ds + 0.001, i + h / 2, "*", va="center", color="#3498db", fontweight="bold")
        if sp < 0.05:
            ax.text(ss + 0.001, i - h / 2, "*", va="center", color="#e67e22", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([f"{s}  \u2192  {t}" for s, t in keys], fontsize=9)
    ax.set_xlabel("CKA(K_source, K_target)", fontsize=10)
    ax.set_title("Dense vs sparse kernel \u2014 source-target alignment\n(* = p < 0.05)", fontsize=11)
    ax.set_xlim(left=0)
    ax.legend(fontsize=9)
    plt.tight_layout()
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

    def _adj_to_graph(adj: np.ndarray) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(range(adj.shape[0]))
        for i in range(adj.shape[0]):
            for j in range(i + 1, adj.shape[0]):
                if adj[i, j] > 0:
                    G.add_edge(i, j, weight=float(adj[i, j]))
        return G

    def _draw(ax, G: nx.Graph, labels: list[str], title: str, color: str) -> dict:
        pos = nx.spring_layout(G, seed=42, weight="weight")
        weights = np.array([G[u][v]["weight"] for u, v in G.edges()]) if G.edges() else np.array([])
        w_scaled = (
            (0.5 + 3.5 * (weights - weights.min()) / (weights.max() - weights.min() + 1e-9)).tolist()
            if len(weights) else []
        )
        nx.draw_networkx_edges(ax=ax, G=G, pos=pos, width=w_scaled, alpha=0.55, edge_color="#888888")
        nx.draw_networkx_nodes(ax=ax, G=G, pos=pos, node_size=4000, node_color=color, alpha=0.9)
        nx.draw_networkx_labels(
            ax=ax, G=G, pos=pos,
            labels={i: labels[i][:22] for i in G.nodes()},
            font_size=22.5, font_color="#111111",
        )
        ax.set_title(f"{title}\n{G.number_of_nodes()} nodes  |  {G.number_of_edges()} edges", fontsize=11)
        ax.axis("off")
        return pos

    G_S = _adj_to_graph(K_S_sparse)
    G_T = _adj_to_graph(K_T_sparse)

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    pos_S = _draw(axes[0], G_S, src_texts, f"Source graph  ({source_concept} expressions)", "#3498db")
    pos_T = _draw(axes[1], G_T, tgt_texts, f"Target graph  ({target_concept} expressions)", "#e67e22")

    for i in range(len(src_texts)):
        if i not in pos_S or i not in pos_T:
            continue
        fig.add_artist(mpatches.ConnectionPatch(
            xyA=pos_S[i], xyB=pos_T[i],
            coordsA=axes[0].transData, coordsB=axes[1].transData,
            linestyle="dotted", linewidth=0.7, color="#999999", alpha=0.5, clip_on=False,
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
) -> None:
    """Permutation null distribution histograms for CKA and GED."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(cka_res.null, bins=40, color="#3498db", edgecolor="white", alpha=0.85)
    axes[0].axvline(cka_res.observed, color="#e74c3c", linewidth=2.5,
                    label=f"observed = {cka_res.observed:.3f}\np = {cka_res.p_value:.3f}")
    axes[0].set_xlabel("CKA  (higher = more similar)", fontsize=10)
    axes[0].set_ylabel("count", fontsize=10)
    axes[0].set_title(f"CKA permutation null  \u2014  {source_concept} -> {target_concept}", fontsize=11)
    axes[0].legend(fontsize=10)

    axes[1].hist(ged_res.null, bins=40, color="#e67e22", edgecolor="white", alpha=0.85)
    axes[1].axvline(ged_res.observed, color="#e74c3c", linewidth=2.5,
                    label=f"observed = {ged_res.observed:.3f}\np = {ged_res.p_value:.3f}")
    axes[1].set_xlabel("GED  (lower = more similar)", fontsize=10)
    axes[1].set_ylabel("count", fontsize=10)
    axes[1].set_title(f"GED permutation null  \u2014  {source_concept} -> {target_concept}", fontsize=11)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    plt.show()


def print_alignment_table(
    domain_results: dict[tuple[str, str], DomainAlignmentResult],
    keys: list[tuple[str, str]] | None = None,
) -> None:
    """Print a CKA/GED comparison table for the given domain pairs."""
    if keys is None:
        keys = list(domain_results)
    print(f"{'Source domain':<28} {'Target domain':<22} {'CKA':>6}  {'GED':>6}  {'p(CKA)':>7}  {'p(GED)':>7}")
    print("-" * 82)
    for key in keys:
        r = domain_results[key]
        src, tgt = key
        print(f"{src:<28} {tgt:<22} "
              f"{r.cka.observed:>6.3f}  {r.ged.observed:>6.3f}  "
              f"{r.cka.p_value:>6.3f}{'*' if r.cka.p_value < 0.05 else ' '}  "
              f"{r.ged.p_value:>6.3f}{'*' if r.ged.p_value < 0.05 else ' '}")


def print_summary_table(
    domain_results: dict[tuple[str, str], DomainAlignmentResult],
    keys: list[tuple[str, str]] | None = None,
) -> None:
    """Print significant results only, sorted by CKA descending."""
    if keys is None:
        keys = list(domain_results)
    interesting = sorted(
        [k for k in keys if domain_results[k].cka.p_value < 0.05 or domain_results[k].ged.p_value < 0.05],
        key=lambda k: domain_results[k].cka.observed,
        reverse=True,
    )
    header = f"{'Source domain':<22}  {'Target domain':<22}  {'n':>4}  {'CKA':>6}  {'p(CKA)':>7}  {'GED':>6}  {'p(GED)':>7}"
    sep = "-" * len(header)
    print(header)
    print(sep)
    for key in interesting:
        src, tgt = key
        r = domain_results[key]
        print(f"{src:<22}  {tgt:<22}  {r.n_pairs:>4}  "
              f"{r.cka.observed:>6.3f}  {r.cka.p_value:>6.3f}{'*' if r.cka.p_value < 0.05 else ' '}  "
              f"{r.ged.observed:>6.3f}  {r.ged.p_value:>6.3f}{'*' if r.ged.p_value < 0.05 else ' '}")
    print(sep)
    print(f"  {len(interesting)} of {len(keys)} domain pairs significant at p < 0.05")
