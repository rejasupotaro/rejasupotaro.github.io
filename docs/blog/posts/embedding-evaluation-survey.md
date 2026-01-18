---
date: 2026-01-18
authors:
  - rejasupotaro
tags:
  - embeddings
  - evaluation
  - vector-search
  - search
---

# Embedding & Retrieval: A Comprehensive Evaluation Survey

This survey provides a theoretical and practical framework for evaluating modern vector retrieval systems, bridging the gap between intrinsic embedding geometry and end-to-end e-commerce search quality.

## 🏗️ The Layered Evaluation Philosophy

A single metric like nDCG is often a "black box"—it tells you *that* the model is failing, but not *why*. Our 4-layer approach is designed to diagnose the entire lifecycle of a vector search system:

1.  **Intrinsic (The Engine)**: If the geometric distribution is collapsed, the model is fundamentally limited.
2.  **Extrinsic (The Goal)**: Standard benchmarks for relevance.
3.  **Behavioral (The Experience)**: Catches "semantic hallucination" (e.g., brand/color mismatch) which hurts user trust.
4.  **Safety (The Reliability)**: Provides a "Trust Score" for production monitoring.

### 🇯🇵 Relevance to the Amazon JP Project
- **Model Efficiency**: Smaller Japanese models (like `ruri-small`) require strict **Layer 1** checks to ensure they haven't "collapsed" during fine-tuning.
- **E-commerce Precision**: In the Amazon catalog, **Layer 3** is critical. A model that retrieves a "substitute" (e.g., a different brand of batteries) is acceptable, but a model that retrieves a "mismatch" (e.g., a phone case for the wrong model) is a failure.
- **Scale (1.8M Products)**: With millions of items, we cannot manually verify every query. **Layer 4** acts as our automated "smoke test" for thousands of high-scale benchmarks.

<!-- more -->

## 🧱 Layer 1: Representational Geometry (Intrinsic)
Assess the mathematical integrity of the embedding space.

*   **Alignment**:
    *   **The Intuition**: Measures if related items (e.g., a query and its "Exact" product) are mapped to nearby points on the hypersphere.
    *   **ESCI Implementation**: We extract all pairs where `relevance_label = 'Exact'`. We encode both the query and the product title, normalize them to unit length, and calculate the **Mean Squared Error (MSE)** between them. A lower alignment score indicates high similarity between relevant items.
*   **Uniformity**:
    *   **The Intuition**: Measures how well the embeddings "spread out" across the entire hypersphere. If uniformity is poor (feature collapse), the model maps everything to a narrow cone, losing the ability to distinguish between different concepts.
    *   **ESCI Implementation**: We take a random sample of 2,000+ product titles and calculate the **Log-Sum-Exp of Gaussian Kernels** (RBF kernel) between all pairs (Wang & Isola, 2020). Lower "energy" indicates a more uniform distribution.
*   **Intrinsic Dimension (ID)**:
    *   **The Intuition**: While a model outputs vectors of a certain size (e.g., 384 for `ruri-small`, 768 for `base` models), the "real" information often lives on a much lower-dimensional manifold.
    *   **ESCI Implementation**: We use the **Two-Nearest Neighbors (Two-NN)** estimator. If the ID is significantly lower than the model's native output dimension (e.g., ID < 10% of $d_{model}$), it suggests the model has high redundancy or that the "semantic space" of the dataset is bottlenecked.

## 🎯 Layer 2: Domain-Specific Retrieval Performance (Extrinsic)
Assess the utility of embeddings in our specific ranking task.

*   **MTEB / JMTEB (Foundational Baseline)**:
    *   **The Role**: Acts as a sanity check. We use a targeted subset of Japanese tasks (JSTS, JaQuAD) to verify the model maintains general language understanding post-fine-tuning.
*   **nDCG@10 (Initial Page Satisfaction)**: 
    *   **Purpose**: Measures overall quality of the first result page. Weights "Exact" matches higher than "Substitutes."
*   **MRR@10 (Buy Box Precision)**: 
    - **Purpose**: Measures the rank of the *first* "Exact" match. Critical for high-intent model searches.
*   **Recall@5 (Subset Precision)**:
    *   **Purpose**: In our **Subset Evaluation** (20-50 products), `Recall@100` would be nonsensical (always 1.0). We use `@5` to measure if the target is in the absolute top tier.
*   **Recall@100 (Catalog Ceiling - Phase 3)**:
    *   **Purpose**: Reserved for **Full-Catalog Retrieval** (1.8M products). Measures the system's ability to "find the needle" before reranking.

## 🧠 Layer 3: Behavioral & Semantic Diagnostics
Analyze *why* a model fails on specific domains or archetypes.

*   **Behavioral Testing (CheckList for Search)**:
    *   Inspired by **Ribeiro et al. (2020)**. We adapt this for search via **Attribute Mismatch Detection** (Color, Brand, Size).
    *   **Semantic Gap**: The delta between character-level overlap (Lexical) and embedding proximity (Semantic). Helps detect "over-generalization" where a model ignores specific keyword constraints.

## 🛡️ Layer 4: Semantic Certainty (Individual Query Quality)
The frontier of real-time search reliability.

*   **Framework**: Based on **arXiv:2407.15814**.
*   **Quantization Robustness**: High-certainty embeddings are stable under numerical precision changes (e.g., Float32 ↔ Int8).
*   **Neighborhood Coherence**: Measures density of the retrieved Top-K cluster. Low coherence = High ambiguity or Out-of-Catalog query.

---

## 🏛️ Theoretical Synthesis

| Domain | References | Key Metric |
| :--- | :--- | :--- |
| **Geometry** | Wang & Isola (2020) | `Alignment`, `Uniformity` |
| **Ranking** | Reddy et al. (2022) | `nDCG@10`, `MRR@10`, `Recall@5` |
| **Reliability**| arXiv:2407.15814 | `Robustness`, `Coherence` |
| **Behavioral** | Ribeiro et al. (2020) | `Attribute Mismatch` |
