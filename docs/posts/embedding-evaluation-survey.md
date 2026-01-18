---
title: Embedding & Retrieval
summary: A comprehensive evaluation survey of embedding and retrieval systems.
---

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

*   **Alignment & Uniformity**:
    *   **The Intuition**: (Wang & Isola, 2020) Related items should be nearby (Alignment), while the overall distribution should spread across the hypersphere to prevent feature collapse (Uniformity).
*   **The Anisotropy Problem (The Cone Effect)**:
    *   **The Problem**: **Ethayarajh (2019)** observed that contextual embeddings often occupy a narrow cone in the vector space. High anisotropy leads to "artificial" high cosine similarity even between unrelated items.
    *   **Isotropization**: Techniques like **"All-but-the-Top" (Mu et al., 2018)** improve downstream performance by removing the dominant principal components that drive this directional bias.
*   **Intrinsic Dimension (ID)**:
    *   **Implementation**: We use the **Two-Nearest Neighbors (Two-NN)** estimator. If $ID \ll d_{model}$, the model may be over-parameterized or the dataset's semantic richness is bottlenecked.

## 🎯 Layer 2: Domain-Specific Retrieval Performance (Extrinsic)
Assess the utility of embeddings in our specific ranking task.

*   **JMTEB (Foundational Baseline)**:
    *   **The Role**: Acts as a linguistic sanity check. We use a targeted subset of Japanese tasks (JSTS, JaQuAD) to verify the model maintains general language understanding post-fine-tuning.
*   **Stage-Aware Metrics**:
    *   **Stage 1 (Retrieval)**: Prioritizes **Recall@100** or **Recall@200**. The goal is a broad "net" to catch all potential matches.
    *   **Stage 2 (Ranking)**: Prioritizes **nDCG@10** and **MRR@10**. Weights "Exact" matches higher and values the absolute order of the top result page.
*   **Ranking Robustness & Stability**:
    *   **Stability**: Measures the Jaccard similarity between Top-K lists across minor system perturbations (e.g., changes in HNSW parameters).
    *   **Robustness**: Consistency of results for semantically identical but lexically different queries (e.g., "Apple iPhone 15" vs "iPhone 15 Apple").
*   **Tail Performance**: Evaluate metrics specifically on **Long-Tail Queries** where interaction data is sparse, forcing the model to rely purely on semantic understanding rather than historical popularity.

## 🧠 Layer 3: Behavioral & Semantic Diagnostics
Analyze *why* a model fails on specific domains or archetypes.

*   **Behavioral Testing (CheckList for Search)**:
    *   Inspired by **Ribeiro et al. (2020)**. We use **Minimum Functionality Tests (MFTs)** to catch specific failure modes like **Attribute Mismatch**.
    *   **Counterfactual Testing**: "If I change the brand from A to B, does the ranking change correctly?"
*   **ESCI Implementation (The Metadata Bridge)**:
    *   **Why Metadata?**: We leverage product columns like `product_brand` and `product_color` to programmatically identify "mismatch" candidates.
    *   **The Model Requirement**: implementing this requires a **Query NER Model** (or a catalog-based entity extractor) to parse the search string into structured constraints. Without extracting `color:red` from the query, we cannot verify if a `color:blue` battery in the Top-10 is a performance failure or a valid result.
*   **Semantic Gap**:
    *   **Metric**: $Gap = |Sim_{lexical} - Sim_{embedding}|$
    *   **Diagnostics**: Identifies **"Keyword Blindness"** (ignoring exact matches) vs. **"Semantic Hallucination"** (unrelated concept mapping).

## 🛡️ Layer 4: Semantic Certainty (Individual Query Quality)
The frontier of real-time search reliability. Framework based on **arXiv:2407.15814**.

*   **The Unified Semantic Reliability Score ($R_q$)**:
    *   **The Framework**: Instead of a single metric, we calculate a composite score using the **Harmonic Mean** of geometric and semantic properties. This ensures that a query is only considered "certain" if it is both stable and coherent.
    *   **Formula**: $R_q = \frac{2 \cdot G_q \cdot I_q}{G_q + I_q}$
*   **Component 1: Geometric Stability ($G_q$)**:
    *   **The Intuition**: Measures robustness to transformations (e.g., Quantization). High-quality embeddings representing clear concepts remain stable even under numerical precision loss.
*   **Component 2: Information Density ($I_q$)**:
    *   **The Intuition**: Measures the density of the retrieved Top-K cluster (Neighborhood Coherence).
    *   **Implementation**:
        ```python
        def neighborhood_coherence(top_k_embeddings):
            # Mean pairwise similarity of the retrieved set
            sim_matrix = cosine_similarity(top_k_embeddings)
            return np.mean(sim_matrix[np.triu_indices(len(sim_matrix), k=1)])
        ```
*   **ESCI Implementation**:
    *   **Approach**: Retrieve the top 50 products for a query. Calculate $I_q$ (Coherence). Queries with low $I_q$ are flagged as **"Out-of-Catalog"** or **"Ambiguous"**, allowing the system to trigger a "Refinement" UI or fallback to lexical search.

---

## 🏛️ Theoretical Synthesis

| Layer | Domain | References | Key Metric / Concept |
| :--- | :--- | :--- | :--- |
| **1** | **Geometry** | Wang (2020), Ethayarajh (2019) | Alignment, Uniformity, Anisotropy |
| **1** | **Post-processing**| Mu et al. (2018) | Isotropization (All-but-the-Top) |
| **2** | **Ranking** | Reddy (2022), Jaccard et al. | nDCG@10, Ranking Stability |
| **3** | **Behavioral** | Ribeiro et al. (2020) | MFTs, Counterfactuals |
| **4** | **Reliability**| arXiv:2407.15814 | Neighborhood Coherence |
