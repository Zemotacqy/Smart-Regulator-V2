# Agent Notes — Future Improvements & Known Limitations

## Ingestion Pipeline (Phase 1)
- **Context Overflow in LLM Calls**: When feeding lists of clauses or layout blocks to local LLMs (e.g. Ollama), relying purely on the model's physical context window (e.g. `num_ctx`) is not scalable and can cause infinite generation loops or silent truncation. Future implementations MUST use dynamic, token/character-bounded chunking and batching BEFORE sending prompts to the LLM. When expecting structured JSON array responses (e.g. relation extraction), batching MUST constrain both the input character size (e.g., max 20,000 characters) and the total number of items (e.g., max 10 nodes) to ensure the generated output does not exceed the model's max generation token limit and get cut off mid-JSON. Furthermore, `num_ctx` should be parameterized explicitly by callers based on their specific hardware/model constraints rather than hardcoded globally, to avoid out-of-memory errors on larger models.
- **Cross-document Reference Parsing**: The current matching in `corpus_resolver.py` scopes reference resolution to the same document. In later phases, we should parse references referencing other documents (e.g. "under the Act" or "under the SandBox framework") and resolve them globally.
- **Embedding Model MLEB Caveat**: We are using `nomic-embed-text:v1.5` as a zero-overhead local baseline. In Phase 3, we should evaluate upgrading to `BAAI/bge-m3` which supports dense sparse hybrid search to improve Recall@10.
- **Visual Table Structure Parsing**: Docling parses tables into layout blocks. Currently, table content is indexed as raw text. In later phases, we can keep the markdown representation of tables separate to improve structural Q&A reasoning.

## Database & Schema (Phase 2)
- **Index on glossary(source_node_id)**: Add an index on `glossary(source_node_id)` to optimize glossary term queries by source node and support fast cascade deletions.

## Issues Deferred During Pre-Phase-3 Audit (June 15, 2026)
- **Subsection Title Length Audit**: The title length check in `auditor.py` is restricted to `node_type == "SECTION"`. It should be extended to cover `SUBSECTION` title checks.
- **Startup Model Verification**: Lifespan startup code preloads the reranker model but does not verify if LLM and embedding models exist in the local Ollama registry, which can lead to silent errors on first query.
- **Breadcrumb Uniqueness**: Breadcrumbs for subclauses and clauses without explicit headings are not unique within a section. We resolved the immediate citation mapping issue using node UUIDs, but breadcrumb uniqueness remains a nice-to-have visual enhancement.


