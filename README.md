# DL--final--proj

# Anisotropic Edge-to-Edge Attention for Link Prediction via Line Graphs

This repository contains the implementation, benchmarking framework, and experimental results for augmenting Graph Isomorphism Networks (GIN) with residual anisotropic edge attention on line graphs.

---

## Overview

Standard GNN baselines (such as SLRGNN) leverage Line Graph transformations to reformulate link prediction as node classification. However, this conversion substantially increases graph density and introduces topological noise. Because standard GIN relies on isotropic sum aggregation, it treats all adjacent connections uniformly.

To mitigate this bottleneck while maintaining theoretical expressive bounds, this project proposes an **anisotropic edge-to-edge attention layer** integrated via a residual connection:

$$h_i^{\text{final}} = h_i^{\text{GIN}} + \sum_{j \in \mathcal{N}(i)} \alpha_{ij} h_j$$

where attention coefficients are computed using a learnable projection vector $\mathbf{a}$ and LeakyReLU activations:

$$e_{ij} = \text{LeakyReLU}\left(\mathbf{a}^T [h_i \,\Vert{}\, h_j]\right), \quad \alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}$$

---

## Key Results

* **Significant Error Reduction:** Achieves a **38.35% relative error reduction** on the primary BUP benchmark.
* **Noise Suppression:** Directly eliminates 19 baseline false alarms caused by isotropic over-aggregation in dense clusters.
* **Empirical Anisotropy:** Mechanistic analysis demonstrates active edge differentiation ($\sigma = 0.0306$) rather than uniform weight collapse.
* **Multi-Seed Robustness:** Evaluated under a deterministic paired 5-seed framework across extensive hyperparameter sweeps (learning rates, layer depths, hidden dimensions, and dropout rates).

---

## Repository Structure

```text
├── data/                   # Benchmark datasets (BUP, NSC)
├── notebooks/              # Main experiment notebook (.ipynb)
├── figures/                # Attention distributions and ego-network plots
├── requirements.txt        # Environment dependencies
├── .gitignore
└── README.md
