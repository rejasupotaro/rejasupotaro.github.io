---
title: Embedding Evaluation Survey
summary: A survey of embedding and retrieval systems.
---

## 🎭 The Adaptation Dichotomy: MLM vs. CL

Evaluation metrics should be interpreted differently depending on the fine-tuning strategy employed:

### 1. Masked Language Modeling (MLM) = Domain Adaptation
*   **Goal**: Pre-condition the encoder to understand the "language of the domain" (e.g., e-commerce jargon, brand relationships).
*   **Focus**: **Layer 1 (Intrinsic)** and **Layer 3 (Behavioral)**.
*   **Key Indicator**: Improved **Alignment** and reduced **Anisotropy** without necessarily boosting nDCG immediately. It mitigates the effects of **Tokenizer Fragmentation** by teaching the model to reconstruct domain terms.

### 2. Contrastive Learning (CL) = Task Adaptation
*   **Goal**: Optimize the vector space for the specific "Search Task" (Query → Product).
*   **Focus**: **Layer 2 (Extrinsic)**.
*   **Key Indicator**: Significant uplift in **nDCG@10** and **Recall@K**. However, over-focusing on CL can lead to **Catastrophic Forgetting**, where general language capability (measured by **JMTEB**) degrades.

---

### Comparison Matrix

| Feature | MLM (Domain Adaptation) | CL (Task Adaptation) |
| :--- | :--- | :--- |
| **Primary Data** | Unlabeled Product Titles/Descriptions | Labeled Query-Product Pairs |
| **Learns** | "E-commerce grammar" & Jargon | "Search Intent" & Ranking Logic |
| **Key Metric** | Alignment, Anisotropy, Coherence | nDCG, Recall, MRR |
| **Failure Mode** | Model remains a bad retriever | Overfitting to head queries (Poor Zero-Shot) |

---

## 🏗️ The Layered Evaluation Philosophy

A single metric like nDCG is often a "black box"—it tells you *that* the model is failing, but not *why*. Our 4-layer approach is designed to diagnose the entire lifecycle of a vector search system:

1.  **Intrinsic (The Engine)**: Measures **Domain Adaptation (MLM)** success. If the geometric distribution is collapsed, the model is fundamentally limited.
2.  **Extrinsic (The Goal)**: Measures **Task Adaptation (CL)** success. Standard benchmarks for relevance.
3.  **Behavioral (The Experience)**: Catches "semantic hallucination" (brand/color mismatch) and measures **Semantic Gap**.
4.  **Safety (The Reliability)**: Provides a "Trust Score" for production monitoring.

## 🧱 Layer 1: Representational Geometry (Intrinsic)
Assess the mathematical integrity of the embedding space.

### 1.1. Alignment & Uniformity
**(Wang & Isola, 2020)**

Related items should be nearby (Alignment), while the overall distribution should spread across the hypersphere to prevent feature collapse (Uniformity). **MLM** typically improves alignment by teaching the model functional synonyms in the domain.

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

Conversely, bad uniformity leads to **Feature Collapse**, where all representations cluster into a small region. This makes distinct items indistinguishable.

### 1.2. The Anisotropy Problem (The Cone Effect)
**(Ethayarajh, 2019)**

Contextual embeddings often occupy a narrow cone. **MLM** helps "center" the model on your data, while CL with negative sampling (In-batch/Hard negatives) is the primary tool for reducing anisotropy.

#### Formula

$$
A(f) = \frac{1}{N(N-1)} \sum_{i \neq j} \cos(f(x_i), f(x_j))
$$

#### Failure Modes and E-commerce Impact

**High Anisotropy** results in "artificial" high cosine similarity. In e-commerce, this causes **Popularity Bias**, where high-frequency tokens dominate the vector direction, crowding out relevant but less popular items.

### 1.3. Intrinsic Dimension (Two-NN Estimator)
Metric acts as an estimator for the true dimension of the data manifold. We use the **Two-Nearest Neighbors (Two-NN)** estimator.

#### Formula

$$
ID = \frac{N}{\sum_{i=1}^N \ln(\frac{r_{i,2}}{r_{i,1}})}
$$

#### Failure Modes and E-commerce Impact

A **Low ID** indicates that the manifold has collapsed ($\text{ID} \ll d_{model}$). This means the model cannot distinguish nuanced differences (e.g., failing to differentiate "iPhone 12" from "iPhone 12 Pro").

## 🎯 Layer 2: Domain-Specific Retrieval Performance (Extrinsic)
Assess the utility of embeddings in our specific ranking task.

### 2.1. JMTEB (Foundational Baseline)
Evaluation starts with a linguistic sanity check. We use **JMTEB** to verify the model maintains general language understanding post-adaptation. A sharp drop here indicates **Catastrophic Forgetting**.

### 2.2. Stage-Aware Metrics (Retrieval & Ranking)
We evaluate performance across two distinct stages: Retrieval and Ranking. This is the primary scorecard for **Task Adaptation (CL)**.

#### Formulas

**Retrieval (Recall@K)**:
$$ Recall@K = \frac{relevant\_items \cap retrieved\_top\_K}{relevant\_items} $$

**Ranking (MRR)**:
$$ MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{rank_i} $$

**Ranking (nDCG)**:
$$ nDCG_p = \frac{DCG_p}{IDCG_p} $$

#### Failure Modes and E-commerce Impact

**Zero Recall** occurs when the correct item is missing from the top K results. **Rank Reversal** happens when an accessory is ranked higher than the main product. Both lead to lost sales.

### 2.3. Ranking Robustness & Stability
We measure system reliability by analyzing stability and robustness.

#### Formula (Jaccard Similarity)

$$ J(A, B) = \frac{A \cap B}{A \cup B} $$

#### Failure Modes

**High Churn** occurs when a small model change alters more than 50% of the top results. We measure **Stability** across system perturbations and **Robustness** across semantically identical queries.

## 🧠 Layer 3: Behavioral & Semantic Diagnostics
Analyze *why* a model fails on specific domains or archetypes.

### 3.1. Behavioral Testing (CheckList for Search)
**(Ribeiro et al., 2020)**

We use **Minimum Functionality Tests (MFTs)** to catch specific failure modes like **Attribute Hallucination** or **Negation Failure**.

#### Implementation

```python
def test_brand_invariance(model, query="Nike running shoes", brand_a="Nike", brand_b="Adidas"):
    q1 = query.replace("Nike", brand_a)
    q2 = query.replace("Nike", brand_b)
    # Expect similar ranking distribution (high correlation)
    assert correlation(model.search(q1), model.search(q2)) > 0.9
```

### 3.2. ESCI Implementation & Semantic Gap
#### Formula

$$ Gap = |Sim_{lexical} - Sim_{embedding}| $$

#### Failure Modes

A high gap indicates **Keyword Blindness** (ignoring exact matches) or **Semantic Hallucination**. **MLM** adaptation should ideally reduce the semantic gap for domain-specific terms by aligning them in the vector space.

## 🛡️ Layer 4: Semantic Certainty (Individual Query Quality)
The frontier of real-time search reliability. Framework based on **arXiv:2407.15814**.

### 4.1. Unified Semantic Reliability Score
We calculate a composite score (Harmonic Mean) to ensure both stability and coherence.

#### Formula

$$ R_q = \frac{2 \cdot G_q \cdot I_q}{G_q + I_q} $$

### 4.2. Component 1: Geometric Stability ($G_q$)
Measures robustness to transformations (e.g., Quantization).

#### Formula

$$ G_q = \cos(f(x), Q(f(x))) $$

### 4.3. Component 2: Information Density ($I_q$)
Measures local neighborhood density (Coherence).

#### Formula

$$ I_q = \frac{2}{K(K-1)} \sum_{i<j} \cos(v_i, v_j) $$

---

## 🏛️ Theoretical Synthesis

| Layer | Domain | Adaptation Strategy | Key Metric |
| :--- | :--- | :--- | :--- |
| **1** | **Geometry** | **MLM** | Alignment, Anisotropy, ID |
| **2** | **Ranking** | **CL** | nDCG@10, Recall@K, JMTEB |
| **3** | **Behavioral**| **MLM + CL** | Attribute Match, Semantic Gap |
| **4** | **Reliability**| **Model Design** | $G_q$ (Stability), $I_q$ (Coherence) |
