---
name: draft
description: Converts structured knowledge from the research/ directory into professional, technical blog posts for the docs/ directory.
---

# Publication (Drafting) Skill

Use this skill to transform raw investigations (Discovery) and refined observations (Synthesis) from the `/research/` directory into polished technical blog posts.


## 1. Source Identification
- **Select Research**: Identify a specific topic directory in `/research/<topic-name>/` that is ready for promotion.
- **Review Content**: Read through `sources.md`, `logs.txt`, and `draft.md` within the topic directory to understand the key findings.

## 2. Content Synthesis
- **Distill Key Insights**: Identify the most valuable findings, architectural patterns, or technical performance metrics.
- **Structural Planning**: Organize the content using descriptive, neutral headers. Ensure the flow is logical and analytical.
- **Evidence Formatting**: Convert data from research logs into tables, and system designs into Mermaid diagrams. Ensure all citations are properly hyperlinked.

## 3. Writing the Post
- **Technical Excellence**: Create a new Markdown file in the `/docs/` directory (e.g., `/docs/<post-basename>.md`).
- **Follow AGENTS.md**: Adhere strictly to the professional tone, technical integrity, and accuracy guidelines.
    - **No Rhetorical Fluff**: Avoid "catchy" slogans or metaphorical summaries.
    - **Neutral Voice**: Maintain a sober and analytical perspective.
    - **No Emojis**: Do not use emojis in titles or body text.

## 4. Integration and Review
- **Metadata**: Add necessary title tags and meta descriptions for SEO.
- **MkDocs Registration**: Register the new file in the `nav` section of `mkdocs.yml`.
- **Handover to `review` skill**: Once the draft is written, perform a formal audit using the `review` skill to ensure 100% adherence to `AGENTS.md`.

