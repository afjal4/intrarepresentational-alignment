# intrarepresentational-alignment

An experiment to identify Conceptual Metaphor structure in a distributional semantic embedding space. This is done by modelling "conceptual domains" as representations that can be compared using Representational Alignment Metrics from computational neuroscience, as well as a new metric that uses Graph Distance, which is a natural way to represent metaphor as in Gentner's "Metaphor as a Structure-Mapping".

Dataset:
https://github.com/lcc-api/metaphor
Mohler et al. 2016 -- Introducing the LCC Metaphor Datasets
https://metanet.arts.ubc.ca/metaphor-databases/
MetaNet Group. (2023-). English MetaNet Metaphor Wiki Database.

# Poster (copy-pasted LaTeX)

## Abstract

**The shared structure of neural representations is commonly quantified using representational alignment metrics like Centered Kernel Alignment (CKA). A similar notion of shared structure appears in Conceptual Metaphor Theory, where one group of concepts (a “conceptual domain”) is understood in terms of another through shared relational structure. We propose using representational alignment metrics to compare conceptual-domain word embeddings to detect metaphor and, more generally, to measure structural similarity *within* a representation space. We further explore graph-based metrics such as Graph Edit Distance (GED) to extend methods like CKA by comparing the topology of sparse graphs instead of dense similarity matrices. We demonstrate and compare the effectiveness of CKA and GED in detecting alignment internal to a representation.**

---

## Introduction

Conceptual Metaphor is one of several approaches to understanding how humans use metaphorical language \citep{Steen2007}, focusing on the cognitive mechanisms underlying metaphor and analogy processing \citep{Gentner1983}. In this framework, a metaphor is understood as a mapping $\varphi:S \rightarrow T$ between source and target conceptual domains. Concepts and relations from one domain are used to structure understanding of another. For example, LOVE ($S$) can be understood in terms of a JOURNEY ($T$) \citep{LakoffJohnson1980}.

**Figure 1.** Adapted from *Metaphor as Structure-Mapping* \citep{GentnerBowdle2008}. Metaphor is modeled as an isomorphism $\varphi$ of graphs constructed from conceptual domains $S, T$. Nodes represent terms in the metaphor, and edges represent relations between them.

We investigate the extent to which these cognitive structures are detectable in LLM embedding space. This involves measuring the "representational alignment" between the embedded conceptual domains.

**Figure 2.** Example embeddings of the source and target domains $S, T$ in vocabulary $V$. We measure how the metaphorical structure of $\varphi$ (e.g., in Figure 1) is reflected in this embedding geometry.

We calculate kernel matrices $K_S, K_T$, with word correspondences derived from the mapping $\varphi$, then enforce sparsity to prune weak or spurious word relations to create $\widetilde{K}_S, \widetilde{K}_T$. This provides an estimate of the underlying adjacency structure in Figure 1. Sparse kernel matrices have been used in representational alignment metrics such as kNN-CKA \citep{Huh2024PRH}. In our setting, the small number of words allows us to tractably compare graphs directly using graph distance metrics.

---

## Table 1

Top metaphor-domain pairs with CKA and sparse GED significance tests ($\alpha = 5%$).

| Source domain   | Target domain | $n$ | CKA   | $p(\mathrm{CKA})$ | GED   | $p(\mathrm{GED})$ |
| --------------- | ------------- | --- | ----- | ----------------- | ----- | ----------------- |
| PHYSICAL_HARM   | BUREAUCRACY   | 8   | 0.825 | $0.018^{*}$       | 0.493 | $0.018^{*}$       |
| MOVEMENT        | ELECTIONS     | 9   | 0.699 | $<0.001^{*}$      | 0.553 | $0.006^{*}$       |
| A_GOD           | GUNS          | 11  | 0.542 | $0.002^{*}$       | 0.567 | $0.012^{*}$       |
| UPWARD_MOVEMENT | WEALTH        | 13  | 0.435 | $0.062$           | 0.597 | $0.016^{*}$       |
| THEFT           | ELECTIONS     | 12  | 0.280 | $0.144$           | 0.600 | $0.026^{*}$       |

---

## Methodology

Let $S, T$ denote sets of $n$ words from the source and target conceptual domains, where each word is paired by the (bijective) metaphorical mapping $\varphi : S \to T$. For an embedding model $\Phi : V \rightarrow \mathbb{R}^D$, we embed the words in $S, T$ as vectors, and construct kernel matrices $K_S, K_T$ using the cosine-similarity kernel.

### Sparse graph and thresholding

One method of enforcing sparsity on $K_S, K_T$, used in kNN-CKA \citep{Huh2024PRH}, is to retain only edges between a word and its $k$-nearest neighbors.

$$
[\widetilde{K}*{S}]*{ij} = [K_S]*{ij},\mathbf{1}{j \in \mathcal{N}^{S}*{k}(i)},\mathbf{1}{i \in \mathcal{N}^{S}_{k}(j)}
$$

where $\mathcal{N}^{S}_{k}(i)$ denotes the indices of the $k$-nearest neighbors from $S$ of word $i$ in embedding space.

$K_S, K_T$ can also be sparsified by simply thresholding entries:

$$
[\widetilde{K}*S]*{ij} =
\begin{cases}
[K_S]*{ij} & \text{if } [K_S]*{ij} \ge \tau_S \
0 & \text{otherwise}
\end{cases}
$$

where thresholds $\tau_S, \tau_T$ are chosen to produce comparable graph densities across domains.

We investigate graph-distance metrics on the unweighted graphs $G_S, G_T$, derived from the estimated adjacency matrices $\widetilde{K}_S, \widetilde{K}_T$.

Since $|G_S| = |G_T|$ ($=n$), we use simplified definitions of graph distance that depend only on edge discrepancies, rather than more general operations involving node insertion and deletion.

---

### Graph Edit Distance

We define a metric to measure discrete structural mismatch between $G_S$ and $G_T$. Since they are the same size, this simply equates to the number of edge insertions and deletions needed to transform $G_S$ into $G_T$. We normalize this by $\frac{n(n-1)}{2} = \binom{n}{2}$, which is the largest possible edge discrepancy between two graphs with $n$ nodes:

$$
\mathrm{GED}(S, T) = \frac{2\lvert E(G_S) \triangle E(G_T)\rvert}{n(n - 1)}
$$

Without normalization, this is equivalent to the Hamming distance $d_H$ of the adjacency matrices of $G_S, G_T$, and thus defines a valid metric.

CKA (continuous geometry) and GED (discrete topology) provide complementary views of domain alignment.

Future work may benefit from exploring alternative graph distance measures, including Weighted GED and Maximum Common Subgraph Distance (MCSD) \citep{Bunke1998MCS}.

---

### Randomized Relabeling Test

We use a relabeling test from \citet{Kriegeskorte2008RSA} to assess whether the observed metaphorical alignment is coincidental. Intuitively, the mapping $\varphi$ is repeatedly shuffled, and the alignment is recomputed.

The p-value is the proportion of permutations that achieve an alignment score at least as large as the observed score.

---

## Results

We use the LCC Metaphor Dataset \citep{Mohler2016LCC} to source the metaphor domain data. To embed the words, we use the Sentence-Transformer model $\Phi = \texttt{all-MiniLM-L6-v2}$ \citep{allMiniLML6v2}.

Our experiments threshold the kernel matrices as above, with $\tau_S = \tau_T = 0.4$.

**Figure 3.** Calculated graph of the metaphor $\varphi : \text{PHYSICAL HARM} \rightarrow \text{BUREAUCRACY}$.

We find that CKA identifies significant alignment between 8 metaphor-domain pairs. GED identifies 11 significant pairs, 6 of which were previously unidentified by CKA, including $\text{MOVEMENT}\rightarrow\text{DEMOCRACY}$ and $\text{UPWARD MOVEMENT}\rightarrow\text{WEALTH}$.

---

## Discussion

Our results show that Conceptual Metaphor structure can be detected in some cases using both traditional and graph-based representational alignment metrics. One reason several metaphors were not detected is that metaphorical relations often occur between dissimilar words, and thus were removed during sparsification: for example, "relationship" and "problems" are dissimilar within the metaphor "LOVE is a JOURNEY":

$$
(\text{relationship} \xrightarrow{\text{faces}} \text{problems})
;\stackrel{\varphi}{\Rrightarrow};
(\text{road} \xrightarrow{\text{has}} \text{obstacles})
$$

This work represents an initial exploration of "intra-representational alignment"—the representational alignment of structured subsets within a single representation—and a first step toward investigating such Conceptual Metaphor structure in LLM embedding spaces.
