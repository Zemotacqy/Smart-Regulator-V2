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
- **NDCG@5**: To verify if re-ranker is genuinely the issue, evaluate `recall@5` and `NDCG@5` numbers. First implement these metrics as part of eval pipeline and then measure them on the test documents.
- **Reranker Batching & Truncation (June 19, 2026)**: Reranking processes candidates in parallel batches of size 5 with a 2,000-character preview window. Keyword/stopword filters are avoided in favor of raw text prefixes to avoid context loss. Flat JSON key extraction combined with list/array normalization handles 3B model syntax anomalies. Batch exceptions are isolated, preserving final ranking by putting unscored candidates at the tail.

## Database & Schema (Phase 2)

- **Index on glossary(source_node_id)**: Add an index on `glossary(source_node_id)` to optimize glossary term queries by source node and support fast cascade deletions.

## Issues Deferred During Pre-Phase-3 Audit (June 15, 2026)

- **Subsection Title Length Audit**: The title length check in `auditor.py` is restricted to `node_type == "SECTION"`. It should be extended to cover `SUBSECTION` title checks.

## Fine-Tuning & Evaluation (Phase 3)

- **Low Retrieval Recall (Recall@10 = 73.63%)**: The baseline combination of `nomic-embed-text:v1.5` dense vector search + PostgreSQL FTS does not meet the Recall@10 target of 92%. Cross-reference and glossary lookup queries require higher precision. We recommend upgrading the embedding model to `BAAI/bge-m3` (which supports dense + sparse retrieval, reject as it doesn't meet compliance requirements) or implementing a custom legal-fine-tuned bi-encoder.
- **Faithfulness (83.08%) & Compressor Timeouts**: RESOLVED. Fixed the compressor stage (`ifsca-extractor-3b`) timeouts by running layout node compression sequentially (semaphore = 1) and increasing the timeout to 25s. Added a Pydantic `@model_validator` to `CompressorOutput` to map the LLM's alternative key outputs (like `sentences`) into `relevant_sentences`, resolving the silent dropped contexts that impacted faithfulness and citations.
- **Citation Precision (95.71%)**: RESOLVED. Handled by fixing the context compression dropping issues.
- **Model Baked Formatting (June 20, 2026)**: RESOLVED. Replaced duplicate system prompts in `generator.py` with model-level instructions in `Modelfile.saullm` (single source of truth). Shifted from JSON-SFT to Markdown-SFT, utilizing the 3-section structure (Executive Summary, Key Requirements, Verbatim Quote) directly as a legal reasoning chain-of-thought training scaffold.
- **RAFT Dataset Sequence Truncation Filtering**: Resolved model truncation warnings during fine-tuning. Filtered out all training examples with total length exceeding 4,500 characters (~1,000 tokens) to ensure the assistant responses (verbatim quotes at the end of the file) are never cut off during training. This reduced the raw dataset from 2,000 to 1,835 clean training pairs.

# Agent Instructions & Discoveries

This file contains critical discoveries and design conventions for future agentic coding iterations on this codebase.

## Hierarchical RAG & Clause-Level Fragmentation (June 16, 2026)

### The Problem

During evaluation, queries asking for general section requirements (e.g. _"What are the fit and proper requirements for appointing any person as a principal officer?"_) failed to retrieve the detailed clauses (e.g. fairness, financial integrity, or specific disqualifications).

- **Cause 1**: The visual parser splits regulatory texts into granular Level 5 `CLAUSE` and Level 6 `DEFINITION` nodes.
- **Cause 2**: Individual clauses (like `(c) honesty.`) do not contain context keywords like _"principal officer"_ or _"fit and proper requirements."_ Consequently, they get very low semantic alignment scores by the cross-encoder reranker (e.g., `-9.37`) and are discarded from the generator context.
- **Cause 3**: The parent `SECTION` node matches the query well (score `+5.02`), but only contains the title (`9. Fit and Proper Requirements`) with zero actual clauses.

### The Solution: Hierarchical Section Rollup

To resolve this detail-retrieval gap:

1. **Ancestral Section Resolution**: For each candidate node retrieved, resolve its parent `SECTION` (Level 3) or `SUBSECTION` (Level 4).
2. **Descendant Collection**: Query all descendant nodes (clauses and sub-clauses) belonging to those parent sections.
3. **ctid-Preserved Sort**: Regulatory clauses lack sequence keys, and unnamed clauses share duplicate breadcrumb strings. To guarantee the text block preserves the original visual order of the regulation, sort the retrieved descendants using PostgreSQL's physical tuple location column `ctid` (e.g. `ctid::text` or `ctid` sorted natively in Python).
4. **Indented Merging**: Join the text contents of the nodes, indenting by `  ` spaces depending on their AST hierarchy depth, and use this unified block as the node's context.

## Spacing Mismatches in FTS

The visual parser extracts PDF text with double spaces between words (e.g., `fairness  and  integrity`). Standard FTS keyword searches (`plainto_tsquery('english', 'fairness and integrity')`) with single spaces fail to match these blocks. Text pre-processing should normalize whitespace, or query expanders must account for double-space variation.

## RAG v3.0 Ingestion & Context Upgrades (June 18, 2026)

### 1. Heterogeneous Dump Schema Normalization

Ingested all 6 documents from `Dumps/` (ACT, CMI, Banking, ESR, FME, TechFin) into the `smart_regulator_v3` database. Handled 3 distinct schema structures (chapter-level span acts, section/chunk split regulations, and section-only employees service files) by converting to a common intermediate representation before building AST nodes.

### 2. Context Truncation for Embedding Generation

Ollama's `nomic-embed-text:v1.5` returns a HTTP 500 error when receiving excessively long text inputs (>20k characters, e.g., Consolidated Banking Section 3 Definitions). Limiting embedding inputs to 6,000 characters (~1,500 tokens) maintains robust ingestion throughput while preserving all title and beginning section semantics.

### 3. Prompt-Baked SLM Reranking

Replaced MiniLM cross-encoders with zero-shot, prompt-baked relevance scoring using `ifsca-reranker-3b`. Previews are capped at a maximum of `12,000 / N` characters per section payload to fit the 4k context limit, preventing reranker VRAM exhaustion.

### 4. Map-Reduce Context Synthesis

Implemented map-reduce for generator context overflows. When candidate sections exceed 12,000 characters, they are grouped into subsets. The generator runs mapping passes on each, streaming the first pass live to the client while refining subsequent batches, then synthesizes them using a reduce step on `ifsca-extractor-3b`.
TODO: Implement a timeout at each stage of answer generation. Goal is to ensure the answer generation doesn't ends up in an infinte loop

## RAG v3.0 Post-Implementation Audit Findings (June 18, 2026)

### 1. Reranker dict-vs-list Normalization Bug

The `ifsca-reranker-3b` SLM returns a single JSON object `{...}` when it judges only one section as highly relevant. The original code checked `isinstance(scores_data, list)` and fell to the fallback path (all scores `None`) when a dict was returned — making the reranker a complete no-op in practice. Fix: normalize with `if isinstance(scores_data, dict): scores_data = [scores_data]` before the list check. Also baked a stronger array-only output instruction into `Modelfile.reranker`.

### 2. Hop Expander ORDER BY Direction Bug

The ancestor resolution SQL used `ORDER BY original_id, level DESC`. For a CLAUSE at level 4, the recursive CTE terminated at level 4 (since `WHERE a.level > 3` is False for level=4), so `DISTINCT ON` returned the clause itself as anchor — no consolidation occurred. Confirmed in debug logs: `anchors_count=20, rolled_up_count=20` (no reduction from 20 input candidates). Fix: change to `ORDER BY original_id, level ASC` so the shallowest ancestor (SECTION at level 3) is selected. Post-fix: 20 candidates collapsed to 9 anchors.

### 3. Map-Reduce Merge Prompt Label Leakage

The original merge prompt labelled partial answers as `--- PARTIAL ANSWER N ---` and instructed the model to "flag conflicts". The 3b extractor parroted these labels verbatim in its output, producing prose like "According to Partial Answer 2..." and a "Conflict Resolution" section. Fix: renamed labels to `--- CONTENT SEGMENT ---` (no numbers), removed the conflict-flag instruction, added "Do not comment on the process of synthesis" to the system prompt.

### 4. Verified Pipeline Metrics Post-Fix

Query: "Fit and proper requirements for a principal officer"

- Stage C: 20 candidates → 9 rolled-up sections (correct section-level collapsing)
- Stage D: CMI Section 9 scored 8.0 (previously None due to reranker bug)
- Stage E: Detected 10k and 22k char sections, split into 4 batches (overflow triggered correctly)
- Stage F: Batch 1 streamed live, 3 map passes executed, reduce phase produced coherent final answer
- Total execution: 225s. Citations: CMI, ESR, TechFin, FME (all correct and relevant)

## Reranker Parallel Batching & Formatted JSON Normalization (June 19, 2026)

### 1. Parallel Batching & Expanded Previews

Reranking is split into parallel batches of 5 candidates via `asyncio.gather`. Limiting batch size to 5 enables expanding each candidate's content preview to 2,000 characters (up from 800-1,200), ensuring critical trailing sections (such as Section 13(5) and 13(6) of the IFSCA Act detailing penalty deposits and INR payments) are not truncated, while keeping the batch prompt within the 4,096 token limit.

### 2. Flat String-Key Parser & Normalization

Replaced recursive JSON score parsing with a flat, direct lookup matching the Modelfile output format (`{"0": {"relevance_score": X}, ...}`). Added normalisation to convert array outputs (`[{"relevance_score": X}, ...]`) into string-keyed dicts to handle model format variation.

### 3. Domain Scoring Cues

Baked legal domain rules into `Modelfile.reranker` to weight topics highly when scoring:

- "penalty", "deposit", "Consolidated Fund of India": 8.0–10.0 range.
- "currency", "INR", "Indian Rupee", "foreign currency": 8.0–10.0 range.

### 4. Isolated Batch Fault Tolerance

Individual batch exceptions are isolated in `_score_batch`. When a batch fails or outputs invalid JSON, the corresponding nodes receive `score = None` and are sorted to the tail of the final list, preventing total pipeline failure.

## Embedding Model & Local Cross-Encoder Reranker (June 19, 2026)

### 1. Vector Dimension Upgrade to 1024

- Upgraded the database schema and index configuration in `smart_regulator_v4` to store 1024-dimensional vectors generated by `snowflake-arctic-embed2`.
- Ingestion works seamlessly via local Ollama services, but chunk text inputs must still be kept under 6,000 characters to prevent Ollama VRAM/HTTP timeouts.

### 2. PyTorch CPU Cross-Encoder Integration

- Loaded `mixedbread-ai/mxbai-rerank-large-v1` directly inside the Python runtime using `transformers` to avoid Ollama cross-encoder limitations.
- Lazy-loading prevents cold-start overhead at module import, delaying initialization until the first query triggers the reranker phase.
- Added a `threading.Lock()` wrapping `_get_mxbai_model()` with a double-checked locking pattern to ensure thread safety and prevent duplicate model loading when multiple concurrent cold queries arrive.

### 3. Context Length Mitigation (MapReduce Scoring)

- To bypass the cross-encoder's 512-token context limit without discarding text:
  1. **Map:** Split large section texts into 380-token chunks with 40-token overlap using the actual model tokenizer to prevent character-to-token approximation drift.
  2. **Metadata Injection:** Prepended the doc title and AST breadcrumbs (`Document: {title} > {breadcrumb}\nContent: {text}`) to preserve context inside each sub-chunk.
  3. **Parallel Scoring:** Tokenized all sub-chunks in a single vectorized batch, passing it to the cross-encoder in one forward pass on the CPU.
  4. **Reduce (Weighted Peak-Density Aggregator):** Combined scores using:
     $$Score = 0.7 \cdot \max(S_i) + 0.3 \cdot \text{avg}(\text{Top-}K(S_i)) \quad \text{where } K = \min(2, n)$$
     This formula successfully balances peak relevance against overall match density.

### 4. Post-Reranking Rollup Isolation

- Run ancestral section rollups (Stage C) _before_ or _after_ the reranking phase? The plan isolates generator-level rollups to occur _after_ the top 5 nodes are selected by the reranker. This ensures the heavy cross-encoder scoring stage is kept completely isolated from the generator's large context blocks, saving significant CPU latency.

### 5. Future Hardening / To-Dos

- **GPU Accelerator Support:** If MPS (Apple Silicon) or CUDA is available in production environments, initialize the model with `.to("mps")` or `.to("cuda")` to speed up sequence classification batches. Currently, CPU batch latency is acceptable (~40-90ms) but will scale with concurrent requests.
- **Warmup Query:** Run a dummy query at server startup to trigger the lazy-loading of the cross-encoder so the very first user query does not experience the model-loading latency spike (~2-3 seconds).
