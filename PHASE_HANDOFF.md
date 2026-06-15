# PHASE HANDOFF: PHASE 2 Retrieval Pipeline & API

This document outlines the completion status of Phase 2 (Retrieval Pipeline & API) of the Smart Regulator RAG System, detailing the components built, environment configurations, and setup state for the upcoming Phase 3 (Fine-Tuning & Evaluation).

---

## 1. PHASE 2 COMPLETION STATUS

- **FastAPI Routing Layer**: Implemented a complete asynchronous FastAPI server routing system:
  - `/api/qa` (GET): Server-Sent Events (SSE) streaming answer generation, timing statistics, and early citation binding.
  - `/api/compliance` (POST): Extracting text from uploaded PDFs, chunking, retrieving relevant regulation contexts, performing compliance audits, and streaming results as SSE.
  - `/api/admin/ingest` (POST): Triggering background document ingestion tasks.
  - `/api/admin/ingest/logs` (GET): SSE logs streaming (supports `follow=false` query parameter to fetch the historical log buffer).
  - `/api/admin/documents` (GET): Listing all documents.
  - `/api/admin/stats` (GET): Database corpus stats.
- **Observability Tracing CLI**: Developed `scripts/debug_pipeline.py` which executes the retrieval pipeline and outputs tracing timing/quality statistics per retrieval stage.
- **Verification & Testing**:
  - Created [tests/test_api.py](file:///Users/manish/Downloads/repos/smart-regulator-v2/tests/test_api.py) containing integration tests for all FastAPI endpoints.
  - Resolved event loop mismatch issues and stream buffering/iterator hangs in test clients.
  - Verified that all **9 tests** (Ingestion, Retrieval, and API routes) pass successfully together.

---

## 2. FILES CREATED OR MODIFIED

- `backend/main.py` — FastAPI entry point, lifespan, CORS, and reranker pre-loading.
- `backend/api/__init__.py` — API package indicator.
- `backend/api/qa.py` — Q&A retrieval-generation streaming SSE endpoint.
- `backend/api/compliance.py` — Compliance check streaming SSE endpoint.
- `backend/api/admin.py` — Document management, stats, and log streaming endpoints.
- `backend/database/queries.py` — Added `get_all_documents` and `get_corpus_stats` helper queries.
- `backend/rag/extraction/schemas.py` — Added `ComplianceAuditResult` validation schema.
- `scripts/debug_pipeline.py` — CLI tracing tool.
- `tests/test_api.py` — API integration test suite.
- `PHASE_HANDOFF.md` — Complete handoff summary documentation for Phase 2.

---

## 3. ENVIRONMENT STATE

- **PostgreSQL**: Active local server listening on port `5432` with database `smart_regulator_v2` seeded with `IFSCA ACT.pdf`.
- **Ollama Service**: Background server listening on port `11434` with registered models:
  - `nomic-embed-text:v1.5` (Embedding layer)
  - `ifsca-classifier-3b` (Classifier SLM)
  - `ifsca-boundary-3b` (Boundary Detector SLM)
  - `ifsca-extractor-3b` (Relational Extractor SLM)
  - `ifsca-expander-3b` (Query Expander SLM)
  - `llama3.2:3b` (Generator fallback)
- **Environment Variables (`.env` in repo root)**:
  - `DATABASE_URL=postgresql://manish@localhost/smart_regulator_v2`
  - `OLLAMA_HOST=http://localhost:11434`

---

## 4. KNOWN ISSUES OR DEFERRED ITEMS

The following items are logged in `AGENT_NOTES.md`:

- **Cross-document Reference Parsing**: Reference resolution is currently scoped strictly within the same document context. Global/cross-document link resolution is deferred.
- **Dense/Sparse Embedding Upgrades**: Currently using `nomic-embed-text:v1.5` as a zero-overhead local baseline. Evaluating upgrading to `BAAI/bge-m3` in Phase 3.
- **Visual Table Structure Parsing**: Table content is indexed as raw text. Table structure markdown representation parsing is deferred.
- **Index on glossary(source_node_id)**: Add an index on `glossary(source_node_id)` to optimize glossary term queries by source node and support fast cascade deletions (deferred to future schema migration).

---

## 5. PHASE 3 ENTRY CONDITIONS

Before Phase 3 can begin:

1. Phase 2 Human Gate must be officially approved.
2. Ollama and MLX environment must be configured and running locally on Apple Silicon.

---

## 6. PRE-PHASE-3 AUDIT COMPLETION (JUNE 15, 2026)

A comprehensive Pre-Phase-3 codebase and database audit was completed. The following 4 approved non-destructive fixes were successfully applied:

- **`[MAJOR-01]` (Character Loss baseline)**: Updated ingestion orchestrator to compute character loss strictly using English blocks, preventing false positive `needs_repair = TRUE` flags in bilingual documents.
- **`[MAJOR-02]` (Scoping relative references)**: Upgraded the reference resolver to scope relative section/clause matching to the source node's parent section descendants tree before falling back document-wide.
- **`[MAJOR-03]` (Compliance query size)**: Fixed the `/api/compliance` routing to pass the short chunk prefix to query expansion and hybrid search, only setting `original_query` to the full text block before reranking and compression.
- **`[MINOR-02]` (Citation matching precision)**: Embedded the exact node UUID into compressed context metadata blocks, ensuring the generator maps citations precisely rather than matching non-unique breadcrumbs.
