# Agent Guidelines

This repository follows a strict **Research-to-Documentation** workflow to maintain high technical standards and ensure clear separation between exploratory work and public documentation.

## 1. Overview: The Two-Tier System
Every task must adhere to the following directory separation:

- **`/research/` (Workspace)**: A private area for raw data, logs, messy drafts, and exploratory notes. This is the "thinking place" and is not published.
- **`/docs/` (Published)**: The public-facing technical blog. Only polished, professional content resides here.

## 2. Fundamental Workflow
Agents must move through these phases systematically:

1.  **Discovery (Think)**: Perform investigations and store raw findings in thematic subdirectories within `/research/`. Use the `research` skill for best practices.
2.  **Synthesis (Refine)**: Extract key insights and refine the technical narrative.
3.  **Publication (Publish)**: Use the `draft` skill to transform research into a professional post within `/docs/` and register it in `mkdocs.yml`.
4.  **Audit (Review)**: Perform a formal audit using the `review` skill to confirm adherence to the quality standards below.


## 3. Technical Standards (The "Docs" Bar)
Content promoted to `/docs/` must meet the following rigorous quality standards:

### Precision and Tone
- **Technical Accuracy**: Use industry-standard terms (e.g., "Anisotropy", "Embeddings"). Priority is technical correctness over readability or engagement.
- **Neutrality**: Maintain a sober, objective, and analytical perspective.
- **No Fluff**: Avoid rhetorical slogans, catchy metaphors, or marketing-like descriptions (e.g., "The Negative Sampling Trap").

### Structure and Formatting
- **Neutral Headers**: Use descriptive headers that accurately reflect technical content.
- **Visual Evidence**: Use Mermaid diagrams for architecture and Markdown tables for data comparison.
- **No Emojis**: Emojis are prohibited in all headers and body text.

### Evidence-Based Writing
- **Citations**: Provide clickable links to the original paper (e.g., arXiv) or source repository for every technical claim.
- **Objectivity**: Do not personify models (e.g., avoid "The model thinks..."). Focus on architectural constraints or performance metrics.

---
*For specific procedural instructions, consult the active skills in `.agent/skills/`.*
