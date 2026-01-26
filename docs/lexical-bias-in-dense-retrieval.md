---
title: Lexical Bias
summary: Biases in dense retrieval systems.
---

It is widely believed that vector search (Dense Retrieval) is highly effective for ambiguous queries (sensory queries) that do not contain specific terms, such as "fluffy" or "sophisticated."
However, it is known that simply fine-tuning a model using existing keyword search (Lexical Search) logs can cause the model to lose its inherent semantic search capabilities and degenerate into a mere "keyword matching model."

In the field of information retrieval research, this phenomenon is discussed as **Selection Bias** and **Lexical Overlap Bias**.

## Why Does Lexical Bias Occur?

In many real-world systems, "click logs of users on results from previous keyword searches (e.g., BM25)" are used as training data. This contains a significant pitfall:

1. **Restriction of Candidates**: The logs only contain items that were originally displayed in the top results because they "contained the keywords" by BM25 or similar systems.
2. **Model Overfitting**: If you train on these logs as "ground truth," the model learns that "the presence of query words in the document is the strongest signal of relevance."
3. **Collapse of Semantic Capability**: As a result, the scores for items that are semantically related but do not contain the specific keywords (which are the original targets for vector search) do not rise.

## Concrete Impacts of the Bias

When this bias occurs, the following "silent failures" are typically observed:

- **Failure to Improve Zero-hit Queries**: One of the main reasons to introduce vector search is to handle queries that return zero results in keyword search. A biased model, however, may still prioritize lexical matching and fail to surface semantically relevant items if they don't share exact words.
- **Vulnerability to Synonyms**: A biased model might fail to rank "blouson" highly for a query "jacket" if it has learned that literal word matching is the primary indicator of success.
- **Sensitivity to Lexical Noise**: The model may overreact to non-informative terms like "[Free Shipping]" if they frequently appear in clicked items in the training logs.

## Deep Diagnostic Techniques

To extract insights from your models and detect bias beyond simple recall metrics, several qualitative techniques are effective:

### 1. Vocabulary Projection
This technique visualizes what the model "sees" by projecting its query embeddings back into the vocabulary space.
Normally, a query like "sophisticated" should project to tokens like "elegant" or "refined." If a model has collapsed into lexical matching, it will instead project to subword fragments or exact keywords of the input.

### 2. Triplet Diagnostic (Contrastive Analysis)
A direct way to measure bias is to provide the model with a "Conflict Triplet":
- **Query**: "Sophisticated dress"
- **Positive (Semantic)**: "Elegant evening gown" (Meaning match, no word overlap)
- **Negative (Lexical)**: "Adult-like toy" (Word overlap, wrong intent)

A healthy model must yield a higher similarity score for the Positive than the Negative. A positive **Bias Score** ($Sim_{Neg} - Sim_{Pos}$) indicates a failure in semantic understanding.

### 3. Query Ablation (Perturbation Analysis)
By breaking down the query and projecting individual tokens, we can see which parts are driving the vector. If the vector for "Sophisticated dress" is almost identical to the vector for "dress" alone, the model is effectively ignoring the sensory modifier, likely because it hasn't seen enough diverse positives for it.

### 4. Feature Attribution (Gradient Analysis)
Using gradients (e.g., Integrated Gradients), we can identify which input tokens have the most influence on the final embedding. In a biased model, gradients are heavily concentrated on exact matching words. In a semantic model, they are distributed across intent-bearing words.

## Approaches to Mitigate Bias

Several techniques have been proposed to address this issue:

- **Generative Pseudo-Labeling (GPL)**:
  [GPL](https://arxiv.org/abs/2112.05604) generates synthetic queries from documents within the domain using LLMs and uses them for training. This allows the model to learn "semantic connections" that may not exist in existing logs.
- **Distillation from Cross-Encoders**:
  By training a Bi-Encoder (vector search model) using the scores from a Cross-Encoder (which is less susceptible to keyword overlap bias) as the "teacher," the model can learn to evaluate items with high relevance even when there is no lexical overlap.

## Conclusion

When introducing vector search, **"what data you train on"** is more important than the model architecture. To unlock the true potential of dense retrieval, especially for "fluffy" or sensory queries, you must proactively combat the bias inherent in traditional search logs using these deep diagnostic techniques.
