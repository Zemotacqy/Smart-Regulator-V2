# Agent Notes — Future Improvements & Known Limitations

## Ingestion Pipeline (Phase 1)

- **Context Overflow in LLM Calls**: When feeding lists of clauses or layout blocks to local LLMs (e.g. Ollama), relying purely on the model's physical context window (e.g. `num_ctx`) is not scalable and can cause infinite generation loops or silent truncation. Future implementations MUST use dynamic, token/character-bounded chunking and batching BEFORE sending prompts to the LLM. When expecting structured JSON array responses (e.g. relation extraction), batching MUST constrain both the input character size (e.g., max 20,000 characters) and the total number of items (e.g., max 10 nodes) to ensure the generated output does not exceed the model's max generation token limit and get cut off mid-JSON. Furthermore, `num_ctx` should be parameterized explicitly by callers based on their specific hardware/model constraints rather than hardcoded globally, to avoid out-of-memory errors on larger models.
- **Cross-document Reference Parsing**: The current matching in `corpus_resolver.py` scopes reference resolution to the same document. In later phases, we should parse references referencing other documents (e.g. "under the Act" or "under the SandBox framework") and resolve them globally.
- **Embedding Model MLEB Caveat**: We are using `nomic-embed-text:v1.5` as a zero-overhead local baseline. In Phase 3, we should evaluate upgrading to `BAAI/bge-m3` which supports dense sparse hybrid search to improve Recall@10.
- **Visual Table Structure Parsing**: Docling parses tables into layout blocks. Currently, table content is indexed as raw text. In later phases, we can keep the markdown representation of tables separate to improve structural Q&A reasoning.

## Retrieval Pipeline (Phase 2) & Hierarchical RAG findings (June 16, 2026)

- **Clause-Level Fragmentation & Reranker Dismissals**: When a regulation's sections are split into granular level 5/6 clause nodes, queries about the section general topics (e.g. "fit and proper requirements") fail to retrieve the details. Reranker models score individual clause nodes extremely low (e.g. -9.37) because they lack context, while the parent SECTION node scores high (+5.02) but contains no text.
- **Hierarchical Section Rollup Solution**: To solve this, when a section or subsection is matched, a recursive query should fetch all descendant nodes (subsections, clauses, subclauses).
- **ctid sorting for visual order**: PostgreSQL `ctid` (physical block location) must be used to sort the descendants to guarantee the visual sequence order matches the original regulation structure, as unnamed clauses share identical breadcrumbs and lack sequential primary keys.

## Database & Schema (Phase 2)

- **Index on glossary(source_node_id)**: Add an index on `glossary(source_node_id)` to optimize glossary term queries by source node and support fast cascade deletions.

## Issues Deferred During Pre-Phase-3 Audit (June 15, 2026)

- **Subsection Title Length Audit**: The title length check in `auditor.py` is restricted to `node_type == "SECTION"`. It should be extended to cover `SUBSECTION` title checks.

## Fine-Tuning & Evaluation (Phase 3)

- **Low Retrieval Recall (Recall@10 = 73.63%)**: The baseline combination of `nomic-embed-text:v1.5` dense vector search + PostgreSQL FTS does not meet the Recall@10 target of 92%. Cross-reference and glossary lookup queries require higher precision. We recommend upgrading the embedding model to `BAAI/bge-m3` (which supports dense + sparse retrieval, reject as it doesn't meet compliance requirements) or implementing a custom legal-fine-tuned bi-encoder.
- **Faithfulness (83.08%) & Compressor Timeouts**: RESOLVED. Fixed the compressor stage (`ifsca-extractor-3b`) timeouts by running layout node compression sequentially (semaphore = 1) and increasing the timeout to 25s. Added a Pydantic `@model_validator` to `CompressorOutput` to map the LLM's alternative key outputs (like `sentences`) into `relevant_sentences`, resolving the silent dropped contexts that impacted faithfulness and citations.
- **Citation Precision (95.71%)**: RESOLVED. Handled by fixing the context compression dropping issues.
