---
title: Embedding Evaluation
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
3.  **Behavioral (The Experience)**: Catches "semantic hallucination" (**Attribute Integrity** errors like brand or color mismatch) and measures **Semantic Gap**.
4.  **Safety (The Reliability)**: Provides a "Trust Score" for production monitoring.

## 🧱 Layer 1: Representational Geometry (Intrinsic)
Assess the mathematical integrity of the embedding space. This layer diagnostic centers on how effectively the model uses its available vector capacity.

### 1.1. Alignment & Uniformity
**(Wang & Isola, 2020)**

These two metrics formally define what makes a "good" representation on a hypersphere. They are often in tension: the model must pull similar items together without collapsing the entire space.

![Alignment and Uniformity Visualization](images/embedding-evaluation-survey/alignment_uniformity_visualization.png)

#### Theory
*   **Alignment**: Measures the expected distance between embeddings of positive pairs. For a fixed set of "ground truth" related items (e.g., a product and its updated version, or a query and its purchase), they should map to nearly identical coordinates.
*   **Uniformity**: Measures how well the embeddings are distributed across the unit hypersphere. High uniformity preserves maximal information. If all embeddings cluster in one spot, the model loses the ability to distinguish between different categories.

#### Formulas
**Alignment Loss**:
$$
L_{align}(f; \alpha) = E_{(x,y) \sim p_{pos}} [ ||f(x) - f(y)||_2^\alpha ]
$$
*Typically $\alpha=2$. Lower is better.*

**Uniformity Loss**:
$$
L_{uniform}(f; t) = \log E_{x,y \sim p_{data}} [ e^{-t ||f(x) - f(y)||_2^2} ]
$$
*Typically $t=2$. Lower is better (more uniform).*

#### E-commerce Example
*   **Good Alignment**: A search for "Noise Cancelling Headphones" and "Bose QC45" produces vectors that are almost touching.
*   **Poor Uniformity**: The model places "Laptops," "Apples," and "Shampoo" all within the same 5-degree arc of the hypersphere because they all happen to be "Amazon Choice" items.

### 1.2. The Anisotropy Problem (The Cone Effect)
**(Ethayarajh, 2019)**

Anisotropy is the tendency of embeddings to occupy a narrow, directional cone rather than utilizing the full space.

![Anisotropy Cone Effect](images/embedding-evaluation-survey/anisotropy_cone_effect.png)

#### Theory
High anisotropy leads to the **Hubness Problem**, where a few points (hubs) become the nearest neighbors to many unrelated queries. It also results in "artificial" high cosine similarity (e.g., every pair has >0.8 similarity), making the retrieval system overly sensitive to noise.

#### Formula
$$
A(f) = \frac{1}{N(N-1)} \sum_{i \neq j} \cos(f(x_i), f(x_j))
$$
*Lower is better. A value near 0 indicates an isotropic (well-spread) space.*

#### E-commerce Example
In a highly anisotropic space, generic terms like "Packaging" or "Official" might dominate the vector direction. Consequently, a search for "Nike Shoes" might retrieve a "Plastic Bottle" simply because both product descriptions contain the word "Official," which is a high-frequency "hub" token.

### 1.3. Intrinsic Dimension (Two-NN Estimator)
This metric estimates the true dimensionality of the data manifold.

![Intrinsic Dimension Manifold](images/embedding-evaluation-survey/intrinsic_dimension_manifold.png)

#### Theory
Even if your model outputs 768-dimensional vectors, the data often lives on a much lower-dimensional "manifold" (e.g., 20-30 dimensions). If the **Intrinsic Dimension (ID)** is too low, the model lacks the "degrees of freedom" to represent complex relationships.

#### Formula (Two-NN)
We calculate the ratio of the distance to the 2nd nearest neighbor ($r_2$) and the 1st nearest neighbor ($r_1$):
$$
ID = \frac{N}{\sum_{i=1}^N \ln(\frac{r_{i,2}}{r_{i,1}})}
$$

#### E-commerce Example
*   **High ID**: The model can distinguish between "iPhone 13 Red 128GB" and "iPhone 13 Blue 128GB" because it has enough dimensions to represent Color, Model, and Capacity independently.
*   **Low ID (Collapse)**: The model treats all "iPhones" as a single point. It can tell an iPhone from a Toaster, but it cannot differentiate between specific variants because the manifold has collapsed.

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

### 3.1. Attribute Integrity Benchmarking
**The "Business Logic" layer.**

Standard nDCG treats all retrieved items as equally "unlabeled" if they aren't in the qrels. However, in e-commerce, some retrievals are objectively wrong based on hard constraints (e.g., retrieving a *Red* phone for a *Blue* query).

We define two levels of failure:
*   **Global Failure Rate (GFR)**: Average mismatch rate across the entire dataset. This is a "North Star" metric for overall system health.
*   **Explicit Failure Rate (EFR)**: Mismatch rate calculated only on the subset of queries where the user expressed a clear attribute intent (e.g., "Sony battery", "Red dress"). This is the true diagnostic of the model's sensitivity to keywords.

#### Dimensional Integrity Metrics
We decompose EFR into three primary dimensions:

1.  **Color Integrity**: Maps keywords and their Japanese/English synonyms (e.g., `赤`, `レッド`, `red`) into a unified color bucket. A mismatch is triggered if the query contains a color specified *not* found in the product metadata or title.
2.  **Bilingual Brand Integrity**: Handles compound brand names (e.g., `エーワン(a-one)`). The system canonicalizes either `エーワン` or `a-one` in the query to the same entity, preventing false-positive mismatches due to language variations.
3.  **Dimension Integrity**: Detects physical measurements (e.g., `128GB`, `500ml`). It uses a value-unit aware regex to ensure that a search for `500ml` doesn't retrieve a `1L` bottle, even if they are semantically similar "containers".

---

### 3.2. ESCI Implementation & Semantic Gap
#### Formula

$$ Gap = |Sim_{lexical} - Sim_{embedding}| $$

#### Failure Modes

A high gap indicates **Keyword Blindness** (ignoring exact matches) or **Semantic Hallucination**. **MLM** adaptation should ideally reduce the semantic gap for domain-specific terms by aligning them in the vector space.

## 🛡️ Layer 4: Semantic Certainty (Individual Query Quality)
The frontier of real-time search reliability. Framework based on **arXiv:2407.15814**. This layer provides a "Trust Score" for every individual query, allowing systems to flag unreliable results before they reach the user.

### 4.1. Unified Semantic Reliability Score ($R_q$)

![Unified Semantic Reliability Score](images/embedding-evaluation-survey/semantic_reliability_composite.png)

#### Theory
A model's output is only useful if it is both **stable** (mathematically robust) and **coherent** (semantically logical). $R_q$ is the composite metric that balances these two dimensions. If $R_q$ is low, the search engine should consider fallback strategies (e.g., keyword search or a "did you mean" prompt).

#### Formula (Harmonic Mean)
$$ R_q = \frac{2 \cdot G_q \cdot I_q}{G_q + I_q} $$
*A high score ($>0.8$) indicates a "high-confidence" search result.*

### 4.2. Component 1: Geometric Stability ($G_q$)

![Geometric Stability Visualization](images/embedding-evaluation-survey/geometric_stability_quantization.png)

#### Theory
$G_q$ measures how "brittle" an embedding is to mathematical transformations. In production, we often compress vectors (Quantization) or use different hardware. A stable embedding retains its semantic "meaning" (its angle in vector space) even when data is lost during compression.

#### Formula
$$ G_q = \cos(f(x), Q(f(x))) $$
where $Q(f(x))$ is the quantized (e.g., `int8`) version of the vector.

#### Implementation
```python
def geometric_stability(original_vec, quantized_vec):
    # Higher cosine similarity = High Stability
    return cosine_similarity(original_vec, quantized_vec)
```

#### E-commerce Failure Mode: Quantization Collapse
If $G_q$ is small, the model "hallucinates" after compression. A search for a specific "Sony Camera Model" might retrieve a "Generic Tripod" because the low-bit representation lost the specific model-number information.

### 4.3. Component 2: Information Density ($I_q$)

![Neighborhood Coherence Visualization](images/embedding-evaluation-survey/neighborhood_coherence_density.png)

#### Theory
$I_q$ measures **Neighborhood Coherence**. It looks at the top-K retrieved items and asks: *"Are these items actually related to each other, or are they just random nearest neighbors in an empty region of the space?"*

#### Formula (Pairwise Average Similarity)
$$ I_q = \frac{2}{K(K-1)} \sum_{i<j} \cos(v_i, v_j) $$

#### Implementation
```python
def neighborhood_coherence(top_k_embeddings):
    # sim_matrix = pairwise_cosine_similarity(top_k_embeddings)
    # Return mean of upper triangle (excluding diagonal)
    mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool()
    return sim_matrix[mask].mean().item()
```

#### E-commerce Failure Modes
1.  **Vague Queries**: A user searches for "gifts." The model retrieves a random mix of socks, books, and mugs. $I_q$ will be low because the items share little semantic overlap, flagging the result as "Low Confidence."
2.  **Out-of-Domain (OOD) Noise**: A user types gibberish like "asdfghjkl." The vector space returns the closest random items it has. $I_q$ will be extremely low, alerting the system to show a "No results found" page instead of garbage.

---

## 🏛️ Theoretical Synthesis

| Layer | Domain | Adaptation Strategy | Key Metric |
| :--- | :--- | :--- | :--- |
| **1** | **Geometry** | **MLM** | Alignment, Anisotropy, ID |
| **2** | **Ranking** | **CL** | nDCG@10, Recall@K, JMTEB |
| **3** | **Behavioral**| **MLM + CL** | Attribute Integrity, Semantic Gap |
| **4** | **Reliability**| **Score Calibration** | $R_q$ (Trust), $G_q$ (Stability), $I_q$ (Density) |
