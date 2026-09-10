# DL--final--proj

# Anisotropic Edge-to-Edge Attention for Link Prediction via Line Graphs

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<YOUR-USERNAME>/<YOUR-REPO-NAME>/blob/main/SLRGNN_Att-2.ipynb)

Empirical evaluation and benchmarking framework for augmenting Graph Isomorphism Networks (GIN) with residual anisotropic edge-to-edge structural attention over line graphs.

---

## Overview

Baseline link-representation methods (such as SLRGNN) reformulate link prediction as node classification via line graphs[cite: 1]. However, line graph conversion substantially increases edge density and structural noise. Because traditional GIN relies on isotropic sum aggregation, every incident connection receives identical weighting regardless of structural relevance.

To overcome this bottleneck while strictly preserving GIN's 1-WL theoretical upper bound, we incorporate a residual anisotropic edge-to-edge attention layer[cite: 1]:

$$h_i^{\text{final}} = h_i^{\text{GIN}} + \sum_{j \in \mathcal{N}(i)} \alpha_{ij} h_j$$

where structural coefficients are computed using a learnable parameter vector $\mathbf{a}$ and LeakyReLU activations[cite: 1]:

$$e_{ij} = \text{LeakyReLU}\left(\mathbf{a}^T [h_i \,\|\, h_j]\right), \quad \alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}$$

---

## Pre-computed Results & Execution Logs

The repository notebook (`SLRGNN_Att-2.ipynb`) is committed with all cell outputs, evaluation tables, and training logs fully preserved. You can inspect the entire end-to-end execution workflow directly within GitHub's viewer—including the full 5-configuration $\times$ 5-seed grid search, the strict test set comparison, false alarm suppression statistics, and the exported ego-network attention plots—without needing to re-run the pipeline.

## Key Results

* **38.35% Test Error Reduction:** Achieves a +38.35% error reduction over the baseline GIN on the benchmark dataset[cite: 1].
* **Noise & False Alarm Suppression:** Mitigates isotropic over-aggregation by directly suppressing 19 baseline false positive edge predictions[cite: 1].
* **Empirical Anisotropy:** Mechanistic analysis confirms dynamic edge differentiation ($\sigma = 0.0306$) rather than uniform weight collapse[cite: 1].
* **5-Seed Deterministic Framework:** Validated across 5 fixed seeds under identical topological splits to eliminate stochastic initialization bias[cite: 1].

---

## Recommended: Run Directly in Google Colab

The pipeline is fully self-contained and pre-configured for **Google Colab (GPU Runtime - Tesla T4)**.  
Click the badge above or upload `SLRGNN_Att-2.ipynb` to Colab and select **Runtime → Run all**[cite: 1].

The notebook automatically handles:
1. Cloning the baseline benchmark repository and setting up dataset graphs[cite: 1].
2. Installing and patching environment dependencies (`torch-geometric`, `torch-scatter`, `torch-sparse`)[cite: 1].
3. Injecting deterministic seed controls and defining model architectures[cite: 1].
4. Executing the 5-configuration $\times$ 5-seed grid search benchmark[cite: 1].
5. Computing strict test metrics and exporting explainability visualizations[cite: 1].

> **Alternative: One-Click Launch from a Fresh / Blank Colab Session**  
> If you prefer opening a clean Colab notebook instead of loading the `.ipynb` file manually, paste and run the snippet below in an empty cell. It clones the entire repository into your session workspace and triggers the complete pipeline:

```python
# Paste this into a blank Colab cell to clone and run automatically
import os

REPO_NAME = "<YOUR-REPO-NAME>"
REPO_URL = f"[https://github.com/](https://github.com/)<YOUR-USERNAME>/{REPO_NAME}.git"

if not os.path.exists(REPO_NAME):
    !git clone {REPO_URL}
%cd {REPO_NAME}

# Run the complete experiment pipeline
%run SLRGNN_Att-2.ipynb
