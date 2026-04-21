# EXDIG: Explainable Dynamic Incomplete Graph Anomaly Detection

> A focused implementation of the interpretability-enhanced component for dynamic incomplete graph anomaly detection.

## Overview

**EXDIG** is a research repository for **Explainable Dynamic Incomplete Graph Anomaly Detection**, developed from our Journal of Software paper:

**Explainable Dynamic Incomplete Graph Anomaly Detection Based on Masked Learning with Strong-weak Mutual Information**.

This repository is **not intended to be the full project release**. Instead, it concentrates on the **upgraded interpretability-oriented part** of the method, making it easier to inspect, reuse, and extend the explanation pipeline in isolation. The current repository is lightweight and currently contains a compact implementation centered around `pipeline.py` and the dependency file `requirements.txt`.

For the **more detailed and comprehensive implementation** of the overall framework, please refer to:

**Full repository:** `https://github.com/sethGu/GMAE-SWMI`

---

## What problem does EXDIG target?

Most graph anomaly detection methods assume relatively clean and static graph data. Real-world graph data, however, are often far messier: labels may be missing, graph structure may be incomplete, and the graph itself may evolve over time. In the paper, these challenges are summarized as **dynamic incomplete graphs (DIGs)**.

The method combines:

- **masked graph learning** to simulate realistic incomplete and dynamic graph conditions,
- **strong-weak mutual information optimization** to preserve both structural consistency and local relational signals,
- and an **interpretability-oriented perturbation mechanism** over nodes, edges, and features, so that anomaly predictions can be explained more transparently.

---

## Why this repository exists

This repository is built for users who are primarily interested in the **interpretability extension** of the method rather than the full training and evaluation stack.

In other words:

- If you want the **focused explainability-oriented code path**, use this repository.
- If you want the **full project implementation**, use `GMAE-SWMI`.

This separation keeps the core explanation pipeline easier to read, modify, and integrate into related graph learning research.

This repository is a minimal implementation. If you need more comprehensive training and evaluation code, please contact gujunquan@shu.edu.cn.

---

## Method highlights

EXDIG is built around three ideas:

1. **Masking as a simulation tool**  
   Nodes, edges, and features are masked to mimic real dynamic incomplete graph scenarios.

2. **Strong-Weak Mutual Information (SWMI)**  
   Strong mutual information helps preserve global structural integrity, while weak mutual information captures more local relationships and improves generalization.

3. **Interpretability through perturbation**  
   By introducing masking perturbations over nodes, edges, and features, EXDIG identifies which components matter most to anomaly detection decisions, providing more transparent explanations.

---

## Repository structure

```text
EXDIG/
├── pipeline.py
└── requirements.txt
```

At this stage, the repository is intentionally compact and focuses on the explanation-oriented pipeline.

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/sethGu/EXDIG.git
cd EXDIG
pip install -r requirements.txt
```

---

## Usage

This repository currently centers on the main pipeline implementation in `pipeline.py`.

A typical workflow is:

1. prepare your graph data,
2. configure the explanation / perturbation pipeline,
3. run the core script,
4. inspect the resulting interpretability outputs.

Since this repository is a focused research release rather than a large packaged toolkit, you may want to read and adapt `pipeline.py` directly for your own datasets and experimental settings.

---

## Paper

If you use this repository in your research, please cite the paper below:

```bibtex
@Article{20261492,
title = {Explainable Dynamic Incomplete Graph Anomaly Detection Based on Masked Learning with Strong-weak Mutual Information},
author = {Luo, Xiangfeng and Gu, Junquan and Yu, Hang},
 journal = {Journal of Software},
 volume = {37},
 number = {4},
 pages = {1492-1510},
 numpages = {21.0000},
 year = {2026},
 month = {04},
 doi = {10.13328/j.cnki.jos.007523},
 note = {(in Chinese)}
}
```

Paper page: `https://jos.org.cn/html/7523.htm`

---

## Related repository

For researchers who want the **complete implementation** beyond the interpretability-focused upgrade in this repository, please visit:

**GMAE-SWMI:** `https://github.com/sethGu/GMAE-SWMI`

---

## Research keywords

`graph anomaly detection` · `dynamic incomplete graph` · `masked autoencoder` · `mutual information` · `explainable graph learning`

---

## Contact

If you find this repository useful, feel free to open an issue or reach out through GitHub.

---

## Acknowledgment

This repository is part of our ongoing research on robust and explainable graph anomaly detection under incomplete and dynamic graph settings. We hope it helps others build graph learning systems that are not only accurate, but also interpretable.
