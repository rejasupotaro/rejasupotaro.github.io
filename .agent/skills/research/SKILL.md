---
name: research
description: Systematically gather, analyze, and synthesize information into your research/ and docs/ directories.
---

# Research Skill

Use this skill to perform technical investigations, analyze research papers, and organize raw data into structured knowledge.

## 1. Discovery Phase
- **Search Web**: Use the `search_web` tool with targeted queries (e.g., "site:arxiv.org [topic]", "GitHub repository [topic]").
- **Content Extraction**: Use `read_url_content` to fetch full Markdown or text from relevant pages.
- **Deep Dives**: For PDFs or complex documentation, use the `browser_subagent` to navigate and extract specific tables, figures, or implementation details.

## 2. Organization in /research/
- Always start by creating a new topic subdirectory: `/research/<topic-name>/`.
- Store raw data in specialized files:
    - `sources.md`: A bibliography of URLs and key quotes.
    - `logs.txt`: Command outputs or raw extraction data.
    - `draft.md`: Initial synthesis and thoughts.

## 3. Analysis and Synthesis
- **Comparison**: Use tables to compare different methods (e.g., retrieval algorithms, embedding models).
- **Visualization**: Use Mermaid diagrams to map out system architectures or decision trees.

## 4. Publication Phase
- **Handover to `draft` skill**: Once research is stable, transition to the `draft` skill to convert your raw findings into a professional blog post in `/docs/`.





