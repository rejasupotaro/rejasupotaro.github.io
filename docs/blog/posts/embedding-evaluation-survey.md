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

This survey explores state-of-the-art evaluation methodologies for product search systems, bridging the gap between raw embedding quality and end-to-end search performance.

<!-- more -->

## 🧱 Layer 1: Embedding Health (Intrinsic)
Assess the mathematical and semantic integrity of high-dimensional vectors in isolation.

| Metric | Description | Suitability |
| :--- | :--- | :--- |
| **Alignment** | Measures the closeness of "positive" pairs (Query ↔ Relevant Product). | **Essential** for measuring fine-tuning success. |
| **Uniformity** | Measures how evenly embeddings are distributed. Low uniformity signals feature collapse. | **Critical** for retrieval diversity. |
| **Intrinsic Dimension**| Measures the "effective" dimensionality. High redundancy suggests model over-parameterization. | Experimental (Advanced). |
| **DVO (Fragmentation)**| Ratio of subwords per domain term. Detects "semantic shattering" of brands/entities. | **High Value** for Japanese E-commerce. |

## 🎯 Layer 2: Retrieval Performance (Extrinsic)
Assess the utility of embeddings in the context of a search ranking task.

| Metric | Description | Suitability |
| :--- | :--- | :--- |
| **nDCG@K** | Normalized Discounted Cumulative Gain. Weights results by 4-level relevance. | **Standard** (Official ESCI Metric). |
| **Recall@K** | Percentage of relevant items found in top K. | Essential for 1st-stage retrieval. |
| **MRR** | Reciprocal of the rank of the first relevant result. | Useful for "Top 1" precision. |

## 🧠 Layer 3: Behavioral & Semantic Diagnostics
Analyze *why* the model fails on specific queries or domains.

| Metric | Description | Suitability |
| :--- | :--- | :--- |
| **Attribute Mismatch**| Checks for Color, Brand, or Size conflicts between query and result. | **High** (Commercial Precision). |
| **Lexical Overlap** | Jaccard similarity between query/product text. Monitors "keyword bias." | Medium (System Tuning). |
| **Semantic Certainty** | Measures stability under quantization and neighborhood density. | **New (Instance-Level Trust).** |

---

## 🗺️ Coverage & Implementation Roadmap

### Phase A: Core Metric Unification
Maintain nDCG, Alignment, and Uniformity as baseline health checks.

### Phase B: E-commerce Sensitivity
Improve Japanese attribute extraction (Color/Brand) to increase the accuracy of the Mismatch Evaluator.

### Phase C: Semantic Certainty
Implement "Trust Scores" based on **Quantization Robustness** (stability under perturbation) and **Neighborhood Coherence** (top-K cluster density).

### Phase D: Automated Diagnostic Reporting
Update the HTML reporter to automatically highlight "low-certainty" queries, allowing developers to debug edge cases instantly.
