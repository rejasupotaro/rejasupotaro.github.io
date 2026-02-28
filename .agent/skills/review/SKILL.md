---
name: review
description: Audits content in the docs/ directory for adherence to the technical standards and professional tone defined in AGENTS.md.
---

# Review (Audit) Skill

Use this skill to perform a rigorous quality audit on any technical content within the `/docs/` directory before it is considered finalized.

## 1. Tone and Quality Audit
- **Strip Fluff**: Identify and remove "catchy" slogans, marketing-like language, and metaphorical descriptions.
- **Check Precision**: Ensure academic or industry-standard terms (e.g., "Selection Bias", "Anisotropy") are used correctly.
- **Remove Personification**: Ensure models or architectures are not personified (e.g., replace "The model thinks" with "The architecture limits").
- **Neutrality Check**: Verify that the text maintains a sober, objective, and analytical perspective throughout.

## 2. Structural and Visual Audit
- **Header Hierarchy**: Ensure headers are descriptive, neutral, and follow a logical nesting order (H1 -> H2 -> H3).
- **Data Visualization**: Confirm that complex data comparisons use Markdown tables and system architectures are represented with Mermaid diagrams.
- **Formatting**: Verify that no emojis are present in titles or body text.
- **SEO Check**: Ensure the document has a clear title and appropriate meta descriptions for MkDocs.

## 3. Evidence and Accuracy Audit
- **Citation Verification**: Every technical claim or optimization strategy must have a clickable citation (e.g., arXiv link, DOI, or source repository).
- **Consistency Check**: Ensure the technical detail and voice are consistent across the entire document.
- **Fact-Check**: Prioritize technical correctness over engagement; flag any simplified explanations that trade off accuracy.

## 4. Final Audit Decision
After performing the audit, provide a structured feedback report:
- **Status**: [PASS/FAIL]
- **Violations**: List any specific breaches of `AGENTS.md`.
- **Action items**: Clear instructions on how to correct any issues.
