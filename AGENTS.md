# Agent Instructions & Discoveries

This file contains critical discoveries and design conventions for future agentic coding iterations on this codebase.

## Hierarchical RAG & Clause-Level Fragmentation (June 16, 2026)

### The Problem
During evaluation, queries asking for general section requirements (e.g. *"What are the fit and proper requirements for appointing any person as a principal officer?"*) failed to retrieve the detailed clauses (e.g. fairness, financial integrity, or specific disqualifications). 
- **Cause 1**: The visual parser splits regulatory texts into granular Level 5 `CLAUSE` and Level 6 `DEFINITION` nodes.
- **Cause 2**: Individual clauses (like `(c) honesty.`) do not contain context keywords like *"principal officer"* or *"fit and proper requirements."* Consequently, they get very low semantic alignment scores by the cross-encoder reranker (e.g., `-9.37`) and are discarded from the generator context.
- **Cause 3**: The parent `SECTION` node matches the query well (score `+5.02`), but only contains the title (`9. Fit and Proper Requirements`) with zero actual clauses.

### The Solution: Hierarchical Section Rollup
To resolve this detail-retrieval gap:
1. **Ancestral Section Resolution**: For each candidate node retrieved, resolve its parent `SECTION` (Level 3) or `SUBSECTION` (Level 4).
2. **Descendant Collection**: Query all descendant nodes (clauses and sub-clauses) belonging to those parent sections.
3. **ctid-Preserved Sort**: Regulatory clauses lack sequence keys, and unnamed clauses share duplicate breadcrumb strings. To guarantee the text block preserves the original visual order of the regulation, sort the retrieved descendants using PostgreSQL's physical tuple location column `ctid` (e.g. `ctid::text` or `ctid` sorted natively in Python).
4. **Indented Merging**: Join the text contents of the nodes, indenting by `  ` spaces depending on their AST hierarchy depth, and use this unified block as the node's context.

## Spacing Mismatches in FTS
The visual parser extracts PDF text with double spaces between words (e.g., `fairness  and  integrity`). Standard FTS keyword searches (`plainto_tsquery('english', 'fairness and integrity')`) with single spaces fail to match these blocks. Text pre-processing should normalize whitespace, or query expanders must account for double-space variation.
