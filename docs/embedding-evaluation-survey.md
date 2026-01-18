---
title: Embedding Evaluation Survey
summary: A survey of embedding and retrieval systems.
---

## 🏗️ The Layered Evaluation Philosophy

A single metric like nDCG is often a "black box"—it tells you *that* the model is failing, but not *why*. Our 4-layer approach is designed to diagnose the entire lifecycle of a vector search system:

1.  **Intrinsic (The Engine)**: If the geometric distribution is collapsed, the model is fundamentally limited.
2.  **Extrinsic (The Goal)**: Standard benchmarks for relevance.
3.  **Behavioral (The Experience)**: Catches "semantic hallucination" (e.g., brand/color mismatch) which hurts user trust.
4.  **Safety (The Reliability)**: Provides a "Trust Score" for production monitoring.

## 🧱 Layer 1: Representational Geometry (Intrinsic)
Assess the mathematical integrity of the embedding space.

### 1.1. Alignment & Uniformity
**(Wang & Isola, 2020)**

Related items should be nearby (Alignment), while the overall distribution should spread across the hypersphere to prevent feature collapse (Uniformity).

#### Formulas

**Alignment Loss**:
$$
L_{align}(f; \alpha) = E_{(x,y) \sim p_{pos}} [ ||f(x) - f(y)||_2^\alpha ]
$$

**Uniformity Loss**:
$$
L_{uniform}(f; t) = \log E_{x,y \sim p_{data}} [ e^{-t ||f(x) - f(y)||_2^2} ]
$$

#### Implementation

```python
def alignment_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniformity_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()
```

#### Failure Modes and E-commerce Impact

Errors in alignment mean the model fails to map semantically identical but lexically different items together. In e-commerce, this causes "wireless headphones" and "Bluetooth headsets" to be far apart, leading to **Recall** loss.

Conversely, bad uniformity leads to **Feature Collapse**, where all representations cluster into a small region. This makes distinct items indistinguishable (e.g., all "Electronics" items clustering together). A query for "Sony TV" might retrieve "Samsung cables" because the entire category has collapsed into a single dense cluster.

### 1.2. The Anisotropy Problem (The Cone Effect)
**(Ethayarajh, 2019)**

Contextual embeddings often occupy a narrow cone in the vector space. High anisotropy leads to "artificial" high cosine similarity even between unrelated items.

#### Formula

$$
A(f) = \frac{1}{N(N-1)} \sum_{i \neq j} \cos(f(x_i), f(x_j))
$$

#### Failure Modes and E-commerce Impact

**High Anisotropy** results in the "Cone Effect," where all embeddings share a dominant direction. This often manifests as **Hubness**, where certain "hub" points become nearest neighbors to everything.

In e-commerce, high-frequency tokens (e.g., brand names like "Nike" or generic terms like "Case") dominate the vector direction. This causes **Popularity Bias**, where unrelated items with these common terms achieve artificially high cosine similarity, crowding out relevant but less popular items.

To mitigate this, techniques like **"All-but-the-Top" (Mu et al., 2018)** can be used to improve downstream performance by removing the dominant principal components.

### 1.3. Intrinsic Dimension (Two-NN Estimator)
Metric acts as an estimator for the true dimension of the data manifold. We use the **Two-Nearest Neighbors (Two-NN)** estimator for this purpose.

#### Formula

$$
ID = \frac{N}{\sum_{i=1}^N \ln(\frac{r_{i,2}}{r_{i,1}})}
$$

#### Failure Modes and E-commerce Impact

A **Low ID** indicates that the manifold has collapsed ($\text{ID} \ll d_{model}$). In e-commerce, this means the model cannot distinguish nuanced differences (e.g., failing to differentiate "iPhone 12" from "iPhone 12 Pro").

## 🎯 Layer 2: Domain-Specific Retrieval Performance (Extrinsic)
Assess the utility of embeddings in our specific ranking task.

### 2.1. JMTEB (Foundational Baseline)
Evaluation starts with a linguistic sanity check. We use a targeted subset of Japanese tasks (JSTS, JaQuAD) to verify the model maintains general language understanding post-fine-tuning.

### 2.2. Stage-Aware Metrics (Retrieval & Ranking)
We evaluate performance across two distinct stages: Retrieval and Ranking.

#### Formulas

**Retrieval (Recall@K)**:
$$ Recall@K = \frac{relevant\_items \cap retrieved\_top\_K}{relevant\_items} $$

**Ranking (MRR)**:
$$ MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{rank_i} $$

**Ranking (nDCG)**:
$$ nDCG_p = \frac{DCG_p}{IDCG_p} $$

#### Failure Modes and E-commerce Impact

**Zero Recall** occurs when the correct item is missing from the top K results. **Rank Reversal** happens when an accessory (e.g., a phone case) is ranked higher than the main product (e.g., the phone itself). Both failures lead to lost sales due to non-discoverability.

### 2.3. Ranking Robustness & Stability
We measure system reliability by analyzing stability and robustness.

#### Formula (Jaccard Similarity)

$$ J(A, B) = \frac{A \cap B}{A \cup B} $$

#### Failure Modes

**High Churn** occurs when a small model change alters more than 50% of the top results.

We measure **Stability** by calculating Jaccard similarity across minor system perturbations (e.g., changes in HNSW parameters). **Robustness** is measured by checking the consistency of results for semantically identical but lexically different queries.

Finally, we evaluate **Tail Performance** specifically on **Long-Tail Queries** (sparse data) to test pure semantic understanding.

## 🧠 Layer 3: Behavioral & Semantic Diagnostics
Analyze *why* a model fails on specific domains or archetypes.

### 3.1. Behavioral Testing (CheckList for Search)
Inspired by **Ribeiro et al. (2020)**, we use **Minimum Functionality Tests (MFTs)** to catch specific failure modes.

#### Implementation

```python
def test_brand_invariance(model, query="Nike running shoes", brand_a="Nike", brand_b="Adidas"):
    q1 = query.replace("Nike", brand_a)
    q2 = query.replace("Nike", brand_b)
    # Expect similar ranking distribution (high correlation)
    assert correlation(model.search(q1), model.search(q2)) > 0.9
```

#### Failure Modes and E-commerce Impact

Failures include **Attribute Hallucination**, where the model ignores a brand constraint, and **Negation Failure**, where a query like "No sugar" retrieves sugary products. In e-commerce, this manifests as a user searching for "Nike Shoes" but getting "Adidas Shoes" (Brand Mismatch).

### 3.2. ESCI Implementation (The Metadata Bridge)
We leverage product columns like `product_brand` for programmatic verification.

#### The Model Requirement

This approach requires a **Query NER Model** to extract constraints. Without extraction, we cannot verify if `color:blue` in results is a failure for a `color:red` query.

### 3.3. Semantic Gap
#### Formula

$$ Gap = |Sim_{lexical} - Sim_{embedding}| $$

#### Failure Modes

A high gap indicates **Keyword Blindness**, where the model ignores exact matches. Conversely, **Semantic Hallucination** occurs when the model maps unrelated concepts together.

## 🛡️ Layer 4: Semantic Certainty (Individual Query Quality)
The frontier of real-time search reliability. Framework based on **arXiv:2407.15814**.

### 4.1. Unified Semantic Reliability Score
We calculate a composite score (Harmonic Mean) to ensure both stability and coherence.

#### Formula

$$ R_q = \frac{2 \cdot G_q \cdot I_q}{G_q + I_q} $$

### 4.2. Component 1: Geometric Stability ($G_q$)
This component measures robustness to transformations (e.g., Quantization).

#### Formula

$$ G_q = \cos(f(x), Q(f(x))) $$

#### Failure Modes

**Quantization Collapse** occurs when $G_q$ is small, indicating the model loses meaning when compressed to `int8`.

### 4.3. Component 2: Information Density ($I_q$)
This component measures local neighborhood density.

#### Formula

$$ I_q = \frac{2}{K(K-1)} \sum_{i<j} \cos(v_i, v_j) $$

#### Implementation

```python
def neighborhood_coherence(top_k_embeddings):
    sim_matrix = cosine_similarity(top_k_embeddings)
    # Mean of upper triangle (pairwise similarities)
    return np.mean(sim_matrix[np.triu_indices(len(sim_matrix), k=1)])
```

#### Failure Modes and ESCI Implementation

**Vague Queries** (e.g., "gifts") result in low $I_q$ and retrieve random items. Similarly, **Out of Domain** queries (e.g., "dsfjsdklf") retrieve random nearest neighbors.

In ESCI, queries with low $I_q$ trigger a "Refinement UI" instead of showing poor results.

---

## 🏛️ Theoretical Synthesis

| Layer | Domain | References | Key Metric / Concept |
| :--- | :--- | :--- | :--- |
| **1** | **Geometry** | Wang (2020), Ethayarajh (2019) | Alignment, Uniformity, Anisotropy |
| **1** | **Post-processing**| Mu et al. (2018) | Isotropization (All-but-the-Top) |
| **2** | **Ranking** | Reddy (2022), Jaccard et al. | nDCG@10, Ranking Stability |
| **3** | **Behavioral** | Ribeiro et al. (2020) | MFTs, Counterfactuals |
| **4** | **Reliability**| arXiv:2407.15814 | Neighborhood Coherence |
