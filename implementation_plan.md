# Tech Spec: IFSCA Regulatory Assistant v3.1 — Production Architecture

> **Document Type:** Definitive Production Architecture, Operations Runbook & Implementation Specification
> **Version:** 3.1 — All Review Comments Addressed
> **Date:** June 2026
> **Status:** Final — Ready for Developer Handoff

---

## Table of Contents

1. [Design Philosophy & 1000× Analysis](#1-design-philosophy--1000-analysis)
2. [Storage Layer: PostgreSQL + pgvector](#2-storage-layer-postgresql--pgvector)
3. [Model Registry, Selection & Legal Pretraining](#3-model-registry-selection--legal-pretraining)
4. [Ingestion Pipeline — Full SLM-Powered Architecture](#4-ingestion-pipeline--full-slm-powered-architecture)
5. [Retrieval Pipeline — Contextual Compression Chain](#5-retrieval-pipeline--contextual-compression-chain)
6. [Inter-Stage Data Contracts](#6-inter-stage-data-contracts)
7. [Model Baking Strategy](#7-model-baking-strategy)
8. [Evaluation Metrics & Benchmarking](#8-evaluation-metrics--benchmarking)
9. [Dependency Checklist & Idempotency Rules](#9-dependency-checklist--idempotency-rules)
10. [UI/UX Architecture](#10-uiux-architecture)
11. [Repository Structure](#11-repository-structure)
12. [Developer Setup Guide](#12-developer-setup-guide)
13. [Operations Runbook](#13-operations-runbook)
14. [Debugging & Pipeline Tracing Playbook](#14-debugging--pipeline-tracing-playbook)
15. [Phased Delivery Plan](#15-phased-delivery-plan)

---

## 1. Design Philosophy & 1000× Analysis

The baseline architecture (reference image) is built on:
1. `RecursiveCharacterTextSplitter(2000/300)` — blindly splits on character count; guaranteed to bisect regulation clauses.
2. Flat ChromaDB cosine similarity — no hierarchy, no amendment awareness, no cross-references.
3. Generic `intfloat/e5-large` embeddings — zero sensitivity to regulatory/legal syntax.
4. A heuristic confidence score: `0.5 × retrieval + 0.3 × faithfulness + 0.2 × diversity` — opaque, cannot decompose failures.
5. A single Modelfile-baked Llama 8B — the **one correct idea** we inherit and radically expand.

| Dimension | Baseline | Our Architecture |
|---|---|---|
| **Chunking** | Naive character split (2000/300) | Visual Layout AST — SLM-classified at section/clause boundary |
| **Storage** | Flat vector store | PostgreSQL + pgvector — hierarchy + relations + vectors in one ACID store |
| **Reference Resolution** | None | Programmatic SQL 1-hop joins — intra- and inter-doc in <10ms |
| **Amendment Handling** | None | `SUBSTITUTES`, `INSERTED_BY`, `OMITTED_BY` temporal edges |
| **Glossary & Definitions** | None | `DEFINES_TERM` edges — auto-expanded into query context |
| **Model** | Generic Llama 8B Modelfile | SaulLM-7B-Instruct (30B+ legal token pretraining) + RAFT QLoRA + Modelfile |
| **Evaluation** | Composite heuristic | Orthogonal RAG Triad + 5 additional retrieval/operational metrics |
| **User Output** | Dense technical text | Plain-English narrative with collapsible citation panel |

---

## 2. Storage Layer: PostgreSQL + pgvector

### 2.1 — Schema Design Decisions

#### Q: Why is `level` constrained to 1–6?
Indian regulatory documents have a finite structural hierarchy. We assign levels as follows:
- Level 1: Document root (e.g. "IFSCA Act 2019")
- Level 2: Chapter / Schedule / Part
- Level 3: Section
- Level 4: Sub-section
- Level 5: Clause
- Level 6: Sub-clause

Six levels cover 100% of observed IFSCA document structures. If a future document exceeds this (e.g. a 7-layer nested statutory instrument), the constraint is raised in a migration: `ALTER TABLE ast_nodes DROP CONSTRAINT ... ADD CHECK (level BETWEEN 1 AND 7)`. This is a **one-line migration**, not an architectural change.

#### Q: What are the two vector columns — `embedding` vs `ts_vector`?
These serve completely different retrieval mechanisms:

| Column | Type | Purpose | Query Method |
|---|---|---|---|
| `embedding VECTOR(768)` | Dense float32 array | Semantic similarity — captures *meaning*. "Capital requirements" will match "minimum financial threshold" | HNSW approximate nearest-neighbor via pgvector |
| `ts_vector TSVECTOR` | Pre-tokenised lexeme list | Exact keyword retrieval — critical for statutory references like "Section 4(2)(b)". Semantic embeddings miss exact clause citations. | GIN index full-text search with `ts_rank` |

Both are required because regulatory queries alternate between semantic questions ("what governs GIC registration?") and exact citation lookups ("what does Section 12(3)(b) say?"). Neither method alone achieves acceptable recall.

#### Q: Explain the HNSW index parameters `m=16, ef_construction=64`
HNSW (Hierarchical Navigable Small World) is a graph-based ANN index.

- **`m=16`**: Each vector node maintains up to 16 bidirectional edges in the graph. Higher `m` → denser graph → better recall → more memory and slower insert. The pgvector default of `m=16` is the correct starting point for a 50K–200K node corpus. Tune to `m=32` only if post-deployment recall audits show <80% Recall@5.
- **`ef_construction=64`**: The candidate list size during index *build*. `64` is the pgvector default and appropriate for our corpus size. Increasing to `128` improves index quality at build-time cost. This is a build-time-only setting.
- **`ef_search`** (query-time): This is tunable per session without rebuilding the index: `SET hnsw.ef_search = 100;`. Start at the default (`40`) and increase during evaluation if recall is insufficient.

> [!TIP]
> For tuning: adjust `ef_search` first (no index rebuild needed). Only rebuild with higher `m`/`ef_construction` if `ef_search` tuning has been exhausted.

#### Q: What happens if the same term has multiple definitions?
The `glossary` table uses `PRIMARY KEY (term, doc_id)`. This handles the common case where the same term is defined differently in different regulations (e.g. "Specified Foreign Currency" may be defined differently in the Banking Regulations vs. the GIC Regulations). The system uses the definition scoped to the document being queried.

If the *same document* defines a term twice (rare but legal in Indian drafting — one in the definition section and one contextually in a schedule), the **first** definition encountered during ingestion is stored; the second triggers an `ON CONFLICT DO NOTHING`. The extractor will flag this in the audit log for human review.

### 2.2 — Expanded Relationship Types

Based on research into IFSCA regulatory amendment patterns (confirmed from live IFSCA documents), the following relationship types are observed:

| `rel_type` | When Used | Example | Found In Corpus |
|---|---|---|---|
| `REFERS_TO` | Clause cites another clause, section, act, or schedule | "subject to Section 8(1)" / "as defined under FEMA" | All 6 documents |
| `DEFINES_TERM` | Node is the canonical definition of a legal term | Definitions section: "'IBU' means International Banking Unit..." | Docs 1,2,3,4,6 |
| `SUBSTITUTES` | An amendment fully replaces the text of an existing node | "Substituted vide GN/REG041... Before substitution it read as under: [text]" | CMI Amendment, Banking Regs |
| `INSERTED_BY` | An amendment adds an entirely new clause where none existed | "the following proviso shall be inserted, namely:—" | CMI Amendment, IFSCA Act 2nd Schedule, Banking Regs |
| `OMITTED_BY` | An amendment deletes an existing clause | "the word 'recognised' shall be omitted" / "Omitted by GN/REG013" | CMI Amendment, Banking Regs |
| `SUPERSEDES` | One regulation/circular explicitly replaces another instrument | "shall stand superseded" | Techfin, GIC Regs, Sandbox, CMI Amendment |

**Difference between `AMENDS` (removed) and `SUBSTITUTES`/`INSERTED_BY`/`OMITTED_BY` (added):**
The previous spec used a generic `AMENDS` type. This is too coarse. `AMENDS` could mean a partial text change, a full replacement, an insertion, or a deletion — these have completely different retrieval implications:
- `SUBSTITUTES` → the old node is **inactive** and must be swapped for the new node at query time.
- `INSERTED_BY` → the new clause is **additive**; both old and new nodes are active.
- `OMITTED_BY` → the old node is **removed**; it must never appear in retrieval results.

The ifsca-extractor SLM must be prompted to output one of these five types only.

The schema update for the `relationships` table:
```sql
rel_type TEXT NOT NULL CHECK (rel_type IN (
    'REFERS_TO', 'DEFINES_TERM',
    'SUBSTITUTES', 'INSERTED_BY', 'OMITTED_BY', 'SUPERSEDES'
))
```

**Validated against documents/ folder:** The `capital_market_intermediaries__amendment_regulations.pdf` document contains explicit "Substituted by", "Inserted by", and "Omitted by" footnotes matching these categories. The `consolidated-ifsca-banking-regulations-as-on-july-14-202314082023111415.pdf` uses the same pattern extensively.

### 2.3 — tsvector Update Mechanism

The `ts_vector` column is updated automatically by a PostgreSQL trigger (no application-level management):

```sql
CREATE OR REPLACE FUNCTION update_ast_node_tsvector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.ts_vector :=
        setweight(to_tsvector('english', COALESCE(NEW.title, '')), 'A') ||
        setweight(to_tsvector('english', COALESCE(NEW.text_content, '')), 'B');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;
```

**How it works:**
1. `to_tsvector('english', text)` tokenises the text using the English dictionary (applies stemming: "regulations" → "regulat", "requiring" → "requir"). This enables the query "require" to match text containing "requiring" or "required".
2. `setweight(..., 'A')` and `setweight(..., 'B')` assign relevance weights. Title matches (weight 'A') are ranked higher than body text matches (weight 'B') in `ts_rank` scoring.
3. `||` concatenates the two weighted tsvectors into one column, so a single full-text query searches both title and body simultaneously.
4. The trigger fires `BEFORE INSERT OR UPDATE` — so every time an AST node is written or updated, the `ts_vector` is recomputed automatically. The application layer never touches this column.

### 2.4 — Complete Schema (`001_initial_schema.sql`)

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS documents (
    doc_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_name       TEXT NOT NULL UNIQUE,
    title           TEXT NOT NULL,
    publish_date    DATE,
    doc_type        TEXT NOT NULL,   -- Open-ended text; validated at application layer (see Section 4.4)
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_active       BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS ast_nodes (
    node_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id          UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    parent_id       UUID REFERENCES ast_nodes(node_id) ON DELETE CASCADE,
    level           SMALLINT NOT NULL CHECK (level BETWEEN 1 AND 6),
    node_type       TEXT NOT NULL,
    title           TEXT,
    text_content    TEXT,
    breadcrumb      TEXT NOT NULL,
    needs_repair    BOOLEAN NOT NULL DEFAULT FALSE,
    embedding       VECTOR(768),
    ts_vector       TSVECTOR
);

-- HNSW tuning: m=16 (edge density), ef_construction=128 (build quality).
-- For a 50k–200k node corpus at 768 dimensions:
--   m=16 → good recall, manageable memory (~0.8 GB index)
--   ef_construction=128 → higher quality graph than default 64, with acceptable build time
-- Query-time tuning: SET hnsw.ef_search = 80 (tune up to 150 without rebuilding)
-- IMPORTANT: SET maintenance_work_mem = '2GB' before running this index creation
CREATE INDEX IF NOT EXISTS idx_ast_nodes_embedding
    ON ast_nodes USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 128);

CREATE INDEX IF NOT EXISTS idx_ast_nodes_fts    ON ast_nodes USING gin (ts_vector);
CREATE INDEX IF NOT EXISTS idx_ast_nodes_doc_id ON ast_nodes (doc_id);
CREATE INDEX IF NOT EXISTS idx_ast_nodes_repair ON ast_nodes (needs_repair) WHERE needs_repair = TRUE;

CREATE TABLE IF NOT EXISTS glossary (
    term            TEXT NOT NULL,
    doc_id          UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    definition      TEXT NOT NULL,
    source_node_id  UUID NOT NULL REFERENCES ast_nodes(node_id) ON DELETE CASCADE,
    PRIMARY KEY (term, doc_id)
);

CREATE TABLE IF NOT EXISTS relationships (
    rel_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_node_id  UUID NOT NULL REFERENCES ast_nodes(node_id) ON DELETE CASCADE,
    target_node_id  UUID REFERENCES ast_nodes(node_id) ON DELETE SET NULL,
    target_text_ref TEXT,
    rel_type        TEXT NOT NULL CHECK (rel_type IN (
                        'REFERS_TO', 'DEFINES_TERM',
                        'SUBSTITUTES', 'INSERTED_BY', 'OMITTED_BY'
                    )),
    effective_date  DATE,          -- w.e.f. date extracted from amendment footnotes
    is_resolved     BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT unique_relation UNIQUE (source_node_id, target_text_ref, rel_type)
);

CREATE INDEX IF NOT EXISTS idx_relationships_source  ON relationships (source_node_id);
CREATE INDEX IF NOT EXISTS idx_relationships_target  ON relationships (target_node_id);
CREATE INDEX IF NOT EXISTS idx_relationships_pending ON relationships (is_resolved) WHERE is_resolved = FALSE;

CREATE OR REPLACE FUNCTION update_ast_node_tsvector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.ts_vector :=
        setweight(to_tsvector('english', COALESCE(NEW.title, '')), 'A') ||
        setweight(to_tsvector('english', COALESCE(NEW.text_content, '')), 'B');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER trg_ast_nodes_tsvector
    BEFORE INSERT OR UPDATE ON ast_nodes
    FOR EACH ROW EXECUTE FUNCTION update_ast_node_tsvector();
```

> [!NOTE]
> `doc_type` is now an open `TEXT` field instead of a `CHECK` constraint. This enables future document types without schema migrations. Validation lives in application code (see Section 4.4).

---

## 3. Model Registry, Selection & Legal Pretraining

### 3.1 — Complete Model Assignment Table

| Stage | Model | Placement | Memory | Primary Justification |
|---|---|---|---|---|
| **Doc Classification** | `ifsca-classifier-3b` (Llama 3.2 3B Q4_K_M) | CPU · Ollama | 2.2 GB RAM | Lightweight; classification only needs front-matter text |
| **Boundary Detection** | `ifsca-boundary-3b` (Llama 3.2 3B Q4_K_M) | CPU · Ollama | 2.2 GB RAM | Structural classification is a pure formatting task |
| **Relation Extraction** | `ifsca-extractor-3b` (Llama 3.2 3B Q4_K_M) | CPU · Ollama | 2.2 GB RAM | 10-clause batching keeps throughput high |
| **Text Embedding** | `nomic-embed-text:v1.5` | CPU · Ollama | 280 MB RAM | 8k context (NTK interpolation), bi-encoder, Ollama-native. Phase 3: evaluate BGE-M3 |
| **Query Expansion** | `ifsca-expander-3b` (Llama 3.2 3B Q4_K_M) | CPU · Ollama | 2.2 GB RAM | Fast 3-variant expansion; no GPU needed |
| **Cross-Encoder Rerank** | `ms-marco-MiniLM-L-6-v2` (ONNX via fastembed) | CPU · Python | 90 MB RAM | Torch-free ONNX inference; identical accuracy to PyTorch; no 200MB torch binary |
| **Context Compression** | `ifsca-extractor-3b` (reused) | CPU · Ollama | Shared | Sentence extraction is a pure filtering task |
| **Primary Q&A** | `ifsca-saullm-7b-ft` (SaulLM-7B-Instruct + RAFT QLoRA) | **GPU · Ollama** | 5.5 GB VRAM at 4k ctx | Legal pretraining (30B legal tokens) + RAFT fine-tune; `num_ctx 4096` keeps within 6GB VRAM |
| **Eval / Testing** | `mistral-nemo:12b` | Mac M4 Air | 8 GB Unified | LLM-as-a-Judge evaluator; larger reasoning model for golden dataset scoring |

### 3.2 — Embedding Model: Full Analysis (Re-validated June 2026)

#### Option A: `nomic-embed-text:v1.5` (Selected for Phase 1)
- **Context window:** 8,192 tokens via Dynamic NTK RoPE interpolation. Pretraining base was 2,048 tokens. Extended context is stable for inputs under 6k tokens — well within the range of even the longest regulatory schedule entries.
- **Critical deployment note:** Ollama may silently cap context at 2,048 tokens with default settings. Always set `num_ctx = 8192` explicitly in the Ollama configuration when generating embeddings.
- **Dimensionality:** 768 dimensions; Matryoshka Representation Learning (MRL) allows downsizing to 256/64 dimensions if storage becomes a concern.
- **Training objective:** Contrastive bi-encoder (MRL). Designed explicitly for asymmetric retrieval (short query → long document passage). This is exactly our use case.
- **Local availability:** Ships natively in Ollama. Zero extra Python dependencies.
- **Benchmark caveat:** nomic-embed-text-v1.5 does **not** appear in the top-10 of the Massive Legal Embedding Benchmark (MLEB, Oct 2025). MLEB is the domain-specific legal retrieval benchmark. This is a known limitation we accept for Phase 1 in exchange for zero infrastructure overhead.

#### Option B: `BAAI/bge-m3` (Planned Phase 3 Upgrade)
- **Context window:** 8,192 tokens. Fully bi-encoder trained.
- **Multi-functionality:** Supports dense retrieval + learned sparse retrieval (SPLADE-style) + multi-vector ColBERT simultaneously. In practice, BGE-M3's learned sparse retrieval outperforms PostgreSQL tsvector BM25, potentially allowing us to **simplify our hybrid search stage** in Phase 3.
- **MLEB performance:** Competitive on legal benchmarks — strong general-purpose legal retrieval performance.
- **Limitation for Phase 1:** Cannot be served through Ollama's `ollama pull` workflow. Requires a FastAPI embedding service using `sentence-transformers` or `fastembed`. Adds infrastructure complexity.
- **Recommendation:** Start with `nomic-embed-text:v1.5` (zero extra infrastructure). **Evaluate BGE-M3 as a Phase 3 upgrade** if Recall@10 on the golden dataset is below 85%.

#### Option C: `legal-bert-base-uncased` (Rejected for Retrieval)
- **Context window:** 512 tokens — a hard BERT architectural limit. The consolidated Banking Regulations alone have schedule entries well exceeding 512 tokens.
- **Training objective:** Masked Language Model (MLM). This is a *pre-training technique*, not a retrieval architecture. Legal-BERT CAN be used as a bi-encoder backbone — but ONLY after explicit fine-tuning with contrastive objectives (MultipleNegativesRankingLoss). Without that fine-tuning, raw CLS embeddings from Legal-BERT produce poor cosine similarity scores.
- **Verdict:** **Rejected.** Even if fine-tuned as a bi-encoder, the 512-token limit makes it structurally unsuitable for IFSCA regulatory clauses.

#### Benchmark Context: Massive Legal Embedding Benchmark (MLEB, Oct 2025)
MLEB is the current gold-standard benchmark for legal embedding models (10 expert-annotated datasets, English, US/UK/EU/Singapore jurisdictions).

| Rank | Model | MLEB Score | Self-Hostable? |
|---|---|---|---|
| #1 | Kanon 2 Embedder | 81.9% | ❌ API |
| #2 | Voyage 4 Large | 81.1% | ❌ API |
| #5 | Qwen3 Embedding 8B | 75.9% | ⚠️ Qwen (excluded per constraints) |
| — | BGE-M3 | ~70%+ | ✅ Phase 3 target |
| — | nomic-embed-text-v1.5 | Not in top-10 | ✅ Phase 1 choice |

> [!IMPORTANT]
> **Re-validation result (June 2026):** nomic-embed-text:v1.5 supports 8k tokens via NTK interpolation — confirmed. Legal-BERT 512-token limit — confirmed. BGE-M3 bi-encoder with 8k context — confirmed. Neither nomic nor BGE-M3 are in the top-10 of MLEB. The leading open-source legal embedding path is BGE-M3 **fine-tuned** on our IFSCA corpus (Phase 3 activity). Phase 1 proceeds with nomic-embed-text:v1.5 for zero-overhead local deployment.

### 3.3 — SaulLM-7B-Instruct: Legal Pretraining, Plain-English Output & VRAM Reality

**Plain-English output:** End users are generalist IFSCA officers, not lawyers. SaulLM's legal pretraining is an **advantage for comprehension** — the model understands legal text accurately enough to *extract* core facts from complex statutory language, then the Modelfile instructs it to output in plain English. A generic model may misread regulatory nuance; SaulLM reads it correctly, then simplifies it.

**VRAM reality check (GTX 3050, 6 GB VRAM):**

| Component | VRAM consumed |
|---|---|
| SaulLM-7B Q4_K_M weights | ~4.0–4.5 GB |
| KV cache at 8k context | ~1.5 GB |
| Runtime overhead | ~0.5 GB |
| **Total at 8k context** | **~6.5 GB — exceeds GTX 3050 VRAM** |
| **Total at 4k context** | **~5.5 GB — fits with ~0.5 GB headroom** |

**Decision:** The production Modelfile uses `num_ctx 4096` (not 8192). The contextual compression stage (Stage E) ensures that by the time context reaches the generator, it is under 1,500 tokens — so 4k context is more than sufficient. This keeps inference on the GPU.

**India/IFSCA jurisdiction caveat:** SaulLM-7B was trained on the Pile of Law dataset — primarily US federal statutes, SCOTUS opinions, UK legislation, and EU directives. **Indian financial regulations (IFSCA, SEBI, RBI, FEMA) are NOT present in SaulLM's training data.** This means:
- SaulLM cannot answer IFSCA questions from parametric memory — RAG is mandatory, not optional.
- The model's general legal reasoning ability (identifying obligations, spotting inconsistencies, understanding conditional clauses) **does generalize** to IFSCA text — this is why it is still the best available open-source option.
- RAFT fine-tuning on our IFSCA corpus explicitly teaches it the IFSCA terminology and drafting patterns, bridging this gap at the weight level.

### 3.4 — Model Baking Architecture (Addressing Review Comment)

**Review comment:** "What happened to baking the entire regulatory text reasoning power within the model?"

The baking strategy is **two-layer**:

**Layer 1 — SaulLM-7B-Instruct base (Legal Pretraining):**
SaulLM already has 30B+ legal tokens baked into its weights. This gives us IFSCA-adjacent legal reasoning for free. We do not start from a generic model.

**Layer 2 — RAFT QLoRA Fine-tuning (IFSCA-specific baking):**
We further fine-tune SaulLM on synthetic IFSCA-specific Q&A pairs. Each training example contains:
- The correct source clause (golden context)
- 2 distractor clauses from unrelated documents
- The expected answer in plain-English with structured table + citation

This bakes three behaviours into the weights:
1. **Distractor resistance** — the model learns to ignore irrelevant retrieved context
2. **Citation format** — the model always produces structured tables + exact quotes
3. **Plain-English output** — training examples use plain-English answers consistently

**Layer 3 — Ollama Modelfile (System Prompt Lock):**
Even after fine-tuning, the Modelfile's `SYSTEM` block provides a runtime guard that cannot be overridden by user input. This is a defence-in-depth measure against prompt injection.

**Modelfile for JSON schema flexibility:**
```dockerfile
FROM ./ifsca-saullm-7b-ft.Q4_K_M.gguf

SYSTEM """
You are the IFSCA Regulatory Assistant. You help regulatory officers understand compliance requirements.

OUTPUT RULES:
1. Answer using ONLY facts from the CONTEXT blocks provided below.
2. Write in plain, clear English. Avoid legal jargon.
3. If the answer is partially in context, answer what you can and state: "For [missing part], no regulation was found in the available corpus."
4. Every answer must include: (a) a plain-English explanation, (b) a structured table of key facts if applicable, (c) the exact source section citation, (d) the verbatim quoted text from the regulation.
5. Never extrapolate, infer, or use prior knowledge about regulations.
"""

PARAMETER temperature 0.0
PARAMETER top_p 0.1
PARAMETER num_ctx 4096   # GTX 3050 6GB constraint: 8k ctx pushes VRAM to ~6.5GB (overflow)
                         # Compression stage (Stage E) ensures context arrives at <1500 tokens
PARAMETER stop "<|eot_id|>"
```

> [!IMPORTANT]
> **Rule 3 in the Modelfile replaces the old "REGULATION NOT FOUND" hard stop.** The model now answers partial queries and provides definitional context even when the specific regulation is absent. Full "not found" responses only occur when zero relevant context exists.

---

## 4. Ingestion Pipeline — Full SLM-Powered Architecture

### 4.1 — Pipeline Overview

```
PDF Document
    │
    ▼
[Pre-filter] Devanagari density filter (>30% → skip block)
    │         Page truncation (classifier uses pages 1–2 only)
    │
    ▼
[Docling] Visual OCR + Layout Graph extraction
    │
    ▼
[SLM-1: ifsca-classifier-3b]
    │   Input: raw text of first available English pages
    │   Output: doc_type, title, publish_date, is_amendment, amends_document
    │
    ▼
[SLM-2: ifsca-boundary-3b] — called once per layout block
    │   Input: text block content
    │   Output: node_type, level, is_boundary_break, heading_text
    │
    ▼
[Deterministic AST Builder]
    │   Builds adjacency-list hierarchy from SLM-2 outputs
    │   No regex; purely from model-assigned node_type + level
    │
    ▼
[SLM-3: ifsca-extractor-3b] — batched, 10 clauses per call
    │   Input: batch of clause texts
    │   Output: relationship edges per clause
    │
    ▼
[Auditor] Deterministic checks: character loss, sequence gaps, empty nodes
    │   Flags need_repair = TRUE on failure (max 2 self-heal attempts)
    │
    ▼
[Embed + Write] Generate embeddings → atomic PostgreSQL transaction
    │
    ▼
[Corpus Resolver] Resolve PENDING references against full corpus
```

### 4.2 — Edge Case: Document Title on Later Pages (e.g. Techfin.pdf)

**The problem:** Some IFSCA documents are bilingual Gazette notifications where the Hindi/Devanagari title appears first, and the English title may appear on page 3 or later. Furthermore, the English section can start mid-page (e.g., at the bottom 20% of a page), causing whole-page Devanagari ratio checks to fail or skip the title.

**The solution:** Replace the hardcoded "pages 1–2" rule with a **dynamic English block detector** that scans all pages of the document line-by-line and detects the transition point where continuous English text begins:

```python
def find_english_classifier_window(docling_pages: list[DoclingPage]) -> str:
    """
    Concatenates all pages of the document and scans for the starting line of the
    continuous English section, resolving cases where the English text starts at
    the bottom of a Hindi page (e.g., bilingual Gazette notifications).
    
    Returns a window of text (~2 pages) starting from the English title.
    """
    # 1. Concatenate all pages into lines to preserve text flow
    all_lines = []
    for page in docling_pages:
        if page.text:
            all_lines.extend(page.text.split('\n'))
            
    if not all_lines:
        return ""

    # Helper to compute devanagari ratio of a string
    def get_devanagari_ratio(text_str: str) -> float:
        if not text_str:
            return 0.0
        # Ignore whitespace and punctuation for a more accurate ratio of actual text characters
        chars = [c for c in text_str if not c.isspace()]
        if not chars:
            return 0.0
        devanagari_count = sum(1 for c in chars if '\u0900' <= c <= '\u097F')
        return devanagari_count / len(chars)

    # 2. Find the first line where the English section begins.
    # To avoid false positives (e.g., brief English headings or names in the Hindi section),
    # we verify that the current line and the subsequent block of text are predominantly English.
    start_line_idx = 0
    total_lines = len(all_lines)
    
    for i in range(total_lines):
        line = all_lines[i].strip()
        # Skip empty lines or extremely short lines (e.g. line numbers) for start detection
        if len(line) < 10:
            continue
            
        if get_devanagari_ratio(line) < 0.15:
            # We found a potential English start. Let's lookahead to verify.
            # Lookahead next 15 lines or 800 characters to confirm it's a sustained English block
            lookahead_lines = all_lines[i:min(i + 15, total_lines)]
            lookahead_text = " ".join(lookahead_lines)
            
            if len(lookahead_text.strip()) > 100 and get_devanagari_ratio(lookahead_text) < 0.15:
                # Confirmed: this is the start of the English section!
                start_line_idx = i
                break
    else:
        # Fallback if no English section detected: start from the beginning
        start_line_idx = 0

    # 3. Return a window of ~2 pages (approx 120 lines or 6000 characters) starting from the English title
    end_line_idx = min(start_line_idx + 120, total_lines)
    return "\n".join(all_lines[start_line_idx:end_line_idx])
```

This function is called before the classifier and guarantees that the English content is found regardless of how many Hindi/bilingual pages precede it. This handles Techfin.pdf and all similar bilingual gazette notifications.

### 4.3 — Handling New Document Types (Flexible doc_type)

**The concern:** What happens when a new document type appears (e.g. "Guidelines", "Master Direction", "FAQ Circular")?

**The solution:** `doc_type` is a free-text field in PostgreSQL (no `CHECK` constraint). Validation happens in the application layer via a **configurable allowlist**:

```python
# config.py
KNOWN_DOC_TYPES = {
    "Act", "Regulation", "Circular", "Framework", "Guidelines",
    "Master Direction", "Notification", "Order", "FAQ"
}

def validate_doc_type(doc_type: str) -> str:
    """
    If doc_type is in the known set, use it as-is.
    If it is new but non-empty, log a warning and accept it
    (the system learns new types automatically).
    If it is empty or null, raise an error.
    """
    if not doc_type:
        raise ValueError("Classifier returned empty doc_type")
    if doc_type not in KNOWN_DOC_TYPES:
        logger.warning("new_doc_type_discovered", doc_type=doc_type)
        KNOWN_DOC_TYPES.add(doc_type)  # Self-expanding; persisted via config update
    return doc_type
```

This means the system **never breaks on a new document type** — it logs it as a new discovery and continues. The admin dashboard surfaces these discoveries so the operator can formally register the new type.

### 4.4 — SLM JSON Output Schema: Flexibility Design

**The concern:** If we need to change the JSON schema sent to/from SLMs in future, how flexible is the process?

**The answer:** Each SLM interaction is isolated in a dedicated Python class with a **versioned Pydantic model**. The Modelfile is versioned separately. Changing the schema involves:

1. Update the Pydantic model (e.g. `ClassifierOutput`) to add/remove fields.
2. Update the Modelfile system prompt to describe the new schema.
3. Run `ollama create ifsca-classifier-3b -f Modelfile.classifier` to recompile.
4. No database changes needed unless a new field maps to a DB column.

```python
# extraction/schemas.py — The single source of truth for all SLM output schemas
class ClassifierOutput(BaseModel):
    title: str
    doc_type: str
    publish_date: date | None
    is_amendment: bool
    amends_document: str | None

class BoundaryOutput(BaseModel):
    node_type: Literal["CHAPTER","SCHEDULE","SECTION","SUBSECTION","CLAUSE","SUBCLAUSE","PREAMBLE","DEFINITION","BODY_TEXT","IGNORE"]
    level: int = Field(ge=1, le=6)
    is_boundary_break: bool
    heading_text: str | None

class RelationItem(BaseModel):
    rel_type: Literal["REFERS_TO","DEFINES_TERM","SUBSTITUTES","INSERTED_BY","OMITTED_BY"]
    target_text_ref: str
    context: str
    effective_date: str | None = None  # ISO date string e.g. "2023-07-15"

class ExtractorClauseOutput(BaseModel):
    source_clause_index: int
    relations: list[RelationItem]
```

Adding a field to `ClassifierOutput` and updating the Modelfile prompt is all that's needed — **the rest of the pipeline is unaffected** because downstream code only reads the fields it needs.

### 4.5 — Amendment Auto-Detection (No Manual User Input)

**The concern:** "The detection of which regulation/act is amended must be automatic."

**The solution:** The `ifsca-classifier-3b` Modelfile already extracts `amends_document` from the PDF front-matter. Amendment PDFs always carry a title like:

> *"IFSCA (Banking) (Amendment) Regulations, 2023"*

The classifier is prompted to:
1. Detect the word "Amendment" in the document title → set `is_amendment: true`
2. Infer the parent document name from the title pattern → set `amends_document: "IFSCA Banking Regulations"`

The orchestrator then:
1. Looks up `amends_document` in the `documents` table to find the parent `doc_id`.
2. Feeds the parent document's node structure to the extractor as context.
3. The extractor identifies which specific clauses are "Substituted by", "Inserted by", or "Omitted by" this amendment.

If the classifier cannot determine `amends_document` with confidence, it returns `null` and the system flags the document for manual operator review in the admin dashboard. No user input is required during ingestion.

### 4.6 — Relationship Extraction Verification

**The concern:** "How are we verifying that the extractor doesn't hallucinate relationship entities?"

The verification is **structural and probabilistic**, not prompted:

1. **Structural validation (deterministic):** The extracted `target_text_ref` (e.g. "Section 12(3)(b)") is matched against `ast_nodes.title` in the database. If no match is found, `is_resolved = FALSE`. This is a hard check.
2. **Resolution rate monitoring:** `corpus_resolver.py` tracks the ratio of resolved vs. unresolved references. A healthy document should resolve >70% of intra-document references. A resolution rate below 50% triggers an alert — indicating the extractor is hallucinating references that don't exist.
3. **Audit log:** Every extracted relation is logged to `structlog` with the source clause text, so a developer can manually inspect false positives.
4. **Golden dataset validation:** During evaluation (Phase 3), we manually verify 25 cross-reference questions to ensure the expected linked clauses appear in the retrieved context.

---

## 5. Retrieval Pipeline — Contextual Compression Chain

### 5.1 — Worked Example

**User query:** *"Can an IFSC Banking Unit accept deposits from Indian residents?"*

**Stage A — Query Expansion (`ifsca-expander-3b`):**
```json
{
  "original_query": "Can an IFSC Banking Unit accept deposits from Indian residents?",
  "expansions": [
    "IBU deposit acceptance rules IFSCA Banking Regulations",
    "Section on permissible deposits for IFSC Banking Units",
    "definition of permissible customer IFSCA IBU regulations"
  ]
}
```
_Four queries are now issued to the database (original + 3 expansions)._

**Stage B — Hybrid Retrieval (PostgreSQL):**
- Dense HNSW search on all 4 queries → returns 20 candidate nodes each → 80 raw candidates.
- Sparse tsvector search for "IBU deposit resident" → returns 20 nodes.
- RRF merges all → top-20 unique nodes by combined rank.

Example top-3 returned nodes:
```
Rank 1: "Section 4(1). Permissible Activities — An IBU shall not accept deposits from persons resident in India..."
Rank 2: "Section 2(k). Definition of 'Permitted Currency'"
Rank 3: "Schedule I. List of permissible foreign currencies"
```

**Stage C — Relational Hop Expansion:**
- `Rank 1` has a `REFERS_TO` edge to "Section 2(k)" → Section 2(k) is appended to context.
- `Rank 2` has a `DEFINES_TERM` edge → definition is inlined into the context block.
- Temporal filter: no `SUBSTITUTES` edge on Rank 1 → node is current, no swap needed.

**Stage D — Cross-Encoder Reranking (`ms-marco-MiniLM-L6-v2`):**
Re-scores all expanded context nodes against the original query. Final top-5 nodes selected.

**Stage E — Context Compression (`ifsca-extractor-3b`):**
Input: "Section 4(1) full text (450 tokens)" + query.
Output: "An IBU shall not accept deposits from persons resident in India under Section 4(1) of the IFSCA Banking Regulations."

_Compressed from 450 tokens to 28 tokens — reducing generation cost by 94%._

**Stage F — Structured Generation (`ifsca-saullm-7b-ft`):**
Input: compressed_context + conversation_history + original_query.
Output:
```
**Short Answer:** No. An IBU cannot accept deposits from Indian residents.

| Rule | Detail | Source |
|---|---|---|
| Deposit Restriction | IBUs may not accept deposits from persons resident in India | Section 4(1), IFSCA Banking Regulations |

**Plain Language:** An IFSC Banking Unit is only allowed to deal with international (non-resident) customers. Taking deposits from Indian residents is not permitted under the current regulations.

**Source Quote:** "An IBU shall not accept deposits from persons resident in India." — Section 4(1), IFSCA (Banking) Regulations, 2020
```

---

## 6. Inter-Stage Data Contracts

Each stage communicates via **Python dataclasses** passed in memory within a single async pipeline. No inter-process serialization overhead. The pipeline context object accumulates data as it flows through each stage.

```python
# rag/retrieval/pipeline_context.py

@dataclass
class QueryPipelineContext:
    """
    The single mutable context object that flows through all retrieval stages.
    Each stage reads from it and writes its outputs back to it.
    """
    request_id: str                          # UUID for end-to-end tracing
    original_query: str                       # Raw user query
    doc_filter: list[UUID] | None = None      # Optional: scope to specific doc_ids

    # Populated by Stage A (Query Expander)
    expanded_queries: list[str] = field(default_factory=list)

    # Populated by Stage B (Hybrid Search)
    candidate_nodes: list[NodeCandidate] = field(default_factory=list)

    # Populated by Stage C (Hop Expander)
    expanded_nodes: list[NodeCandidate] = field(default_factory=list)
    inlined_definitions: dict[str, str] = field(default_factory=dict)

    # Populated by Stage D (Reranker)
    reranked_nodes: list[NodeCandidate] = field(default_factory=list)

    # Populated by Stage E (Compressor)
    compressed_context: str = ""

    # Populated by Stage F (Generator)
    answer_text: str = ""
    source_citations: list[SourceCitation] = field(default_factory=list)

    # Timing data for observability
    stage_timings: dict[str, float] = field(default_factory=dict)
```

Each stage is a function with a consistent signature:
```python
async def run_stage(ctx: QueryPipelineContext) -> QueryPipelineContext:
    start = time.monotonic()
    # ... stage logic ...
    ctx.stage_timings["stage_name"] = time.monotonic() - start
    return ctx
```

The orchestrator chains stages:
```python
ctx = QueryPipelineContext(request_id=str(uuid4()), original_query=user_query)
ctx = await run_query_expander(ctx)
ctx = await run_hybrid_search(ctx)
ctx = await run_hop_expander(ctx)
ctx = await run_reranker(ctx)
ctx = await run_compressor(ctx)
async for token in run_generator(ctx):   # Generator is async-streamed
    yield token                           # SSE token to client
```

**Ingestion pipeline** uses a similar `IngestionContext` dataclass, populated stage by stage through the ingestion orchestrator.

### 6.2 — Why Not LangGraph?

The user raised a valid design question: **Why not use LangGraph for this architecture?**

While LangGraph is an excellent tool for complex, agentic, multi-turn, or cyclical workflows, it is an unnecessary abstraction and potential latency bottleneck for our core retrieval pipeline. Below is a comparison table outlining why we chose a custom dataclass-driven pipeline and when we should transition to LangGraph:

| Architectural Metric | Custom `QueryPipelineContext` (Chosen) | LangGraph |
| :--- | :--- | :--- |
| **Pipeline Nature** | **Strictly Linear**: A → B → C → D → E → F. No branching, cycles, or state-routing logic. | **Cyclical / State-Graph**: Designed for state machines, agent loops, and conditional routing. |
| **Overhead & Latency** | **Zero**: standard async Python function calls. Crucial for meeting low TTFT (Time-to-First-Token) goals. | **Moderate**: runtime state validations, channel updates, graph compilation, and serialization. |
| **Streaming Control** | **Native Async Generator**: direct SSE (Server-Sent Events) streaming from Ollama to client with zero abstraction. | **Complex**: Graph stream events require specialized event handlers and custom parsers. |
| **Dependency Weight** | **None**: pure python stdlib (`dataclasses`). Keeps Docker images and production server lightweight. | **High**: adds `langgraph`, `langchain-core`, and large dependency sub-trees (~50MB+ footprint). |
| **Best Fit** | High-performance, low-latency, linear RAG pipelines. | Non-linear agentic loops, human-in-the-loop, multi-agent collaboration. |

#### When will we use LangGraph?
We will adopt LangGraph in **Phase 2 / Phase 3 (Compliance Checker Component)**. The compliance checker is inherently non-linear and requires:
1. **Conditional Routing:** Branching based on applicability (e.g., is Chapter II applicable to this entity?).
2. **State Backtracking & Looping:** If the compliance checker finds a compliance mismatch, it may loop back to query the vector store for specific clarifying clauses.
3. **Human-in-the-Loop (HITL):** Pausing the graph state to request human auditor input, and resuming execution once approved.

For the core Q&A retrieval pipeline, the custom `QueryPipelineContext` keeps the code highly readable, performant, and simple to debug.

---

## 7. Model Baking Strategy

### 7.1 — RAFT Training Dataset Format
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are the IFSCA Regulatory Assistant."
    },
    {
      "role": "user",
      "content": "CONTEXT:\n[DISTRACTOR: RBI circular on interest rates — unrelated]\n[TARGET: IFSCA Banking Regulations, Section 4(1): 'An IBU shall not accept deposits from persons resident in India']\n[DISTRACTOR: FinTech Sandbox Framework Section 7 on exit criteria]\n\nQUESTION: Can an IBU accept deposits from Indian residents?"
    },
    {
      "role": "assistant",
      "content": "**Short Answer:** No. An IBU cannot accept deposits from Indian residents.\n\n| Rule | Detail | Source |\n|---|---|---|\n| Deposit Restriction | IBUs may not accept deposits from persons resident in India | Section 4(1) |\n\n**Plain Language:** An IFSC Banking Unit is set up to serve only international (non-resident) customers. Accepting deposits from Indian residents is not permitted.\n\n**Source Quote:** \"An IBU shall not accept deposits from persons resident in India.\" — Section 4(1), IFSCA Banking Regulations 2020"
    }
  ]
}
```

- 2 distractors per training example — trains the model to ignore irrelevant context.
- Plain-English answers baked into training data — model learns this output format at weight level.
- ~2,000 total training pairs across all 6 document categories.

### 7.2 — MLX Fine-tuning (Mac M4 Air)
```bash
python scripts/fine_tune_mlx.py \
  --model mlx-community/SaulLM-7B-Instruct-4bit \
  --data data/raft_training_pairs.jsonl \
  --lora-rank 8 \
  --lora-alpha 16 \
  --lora-targets q_proj v_proj \
  --learning-rate 1e-5 \
  --epochs 3 \
  --batch-size 2 \
  --grad-checkpoint \
  --max-seq-length 1024 \
  --output models/ifsca-saullm-7b-ft-adapters
```

### 7.3 — Deployment to Ubuntu Server
```bash
# Merge LoRA adapters + quantize
bash scripts/convert_to_gguf.sh \
  --base saullm-7b-instruct \
  --adapters models/ifsca-saullm-7b-ft-adapters \
  --output models/ifsca-saullm-7b-ft.Q4_K_M.gguf

# Compile and deploy Modelfile
ollama create ifsca-saullm-7b-ft -f modelfiles/Modelfile.saullm
```

---

## 8. Evaluation Metrics & Benchmarking

### 8.1 — LLM-as-a-Judge Setup
LLM-as-a-Judge uses a separate, independent model to evaluate generated answers. We use `mistral-nemo:12b` on the Mac M4 Air (high-reasoning dev model) to score responses. The judge receives:
- The user query
- The retrieved context
- The generated answer
- The golden reference answer

It scores on each metric (0.0–1.0) and returns structured JSON.

```python
# tests/eval_judge.py
class EvalScores(BaseModel):
    faithfulness: float          # 0–1 — every claim supported by context?
    context_recall: float        # 0–1 — all ground-truth facts in context?
    answer_relevance: float      # 0–1 — answer directly addresses the question?
    citation_precision: float    # 0–1 — all cited sections exist and are active?
```

### 8.2 — Complete Metrics Suite

#### Retrieval Metrics (run on every golden dataset evaluation)

**Recall@K:**
$$\text{Recall@K} = \frac{\sum_{i=1}^{Q} \mathbb{I}(\text{target\_node} \in \text{Top-K}(Q_i))}{Q}$$
Target: Recall@5 ≥ 0.85, Recall@10 ≥ 0.92.

**Mean Reciprocal Rank (MRR):**
$$\text{MRR} = \frac{1}{Q}\sum_{i=1}^{Q} \frac{1}{\text{rank}_i}$$
Target: MRR ≥ 0.75.

**Reference Resolution Rate:**
$$\text{RRR} = \frac{\text{Relationships where is\_resolved = TRUE}}{\text{Total relationships}}$$
Target: RRR ≥ 0.80 (20% unresolved is acceptable — these are external acts like FEMA not yet in corpus).

#### Generation Metrics (LLM-as-a-Judge via `mistral-nemo:12b`)

**Faithfulness:**
$$\text{Faithfulness} = \frac{\text{Claims in answer supported by retrieved context}}{\text{Total claims in answer}}$$
Target: ≥ 0.95 (near-zero hallucinations required for regulatory use).

**Context Recall:**
$$\text{Context Recall} = \frac{\text{Ground-truth statements present in context}}{\text{Total ground-truth statements}}$$
Target: ≥ 0.85.

**Answer Relevance:**
$$\text{Answer Relevance} = \text{Cosine-Sim}(\text{Query}, \text{Generated Answer})$$
Target: ≥ 0.80.

**Citation Precision:**
$$\text{Citation Precision} = \frac{\text{Citations pointing to valid, is\_active=TRUE nodes}}{\text{Total citations in answer}}$$
Target: 1.00 (every citation must be verifiable).

#### Operational Metrics (measured during load testing)

| Metric | Target | Measurement |
|---|---|---|
| Time to First Token (TTFT) | < 800ms | Prometheus histogram |
| Total Query Latency P50 | < 3.0s | Prometheus histogram |
| Total Query Latency P95 | < 6.0s | Prometheus histogram |
| Ingestion throughput | > 5 pages/min | Ingestion logs |
| Repair Rate | < 5% of nodes | `COUNT(*) WHERE needs_repair=TRUE / COUNT(*)` |

### 8.3 — Golden Dataset Composition (100 Questions)
- 30 direct regulation questions ("What is the minimum capital for an IBU?")
- 25 cross-reference questions ("Section A says X subject to Section B — what is Section B's requirement?")
- 20 glossary-term questions ("What does 'Permitted Currency' mean under IFSCA Banking Regulations?")
- 15 amendment/temporal questions ("What is the current capital requirement after the 2023 amendment?")
- 10 compliance check questions ("Is a 10% equity holding by a GIC compliant under IFSCA GIC Regulations?")

---

## 9. Dependency Checklist & Idempotency Rules

### 9.1 — Full Python Dependencies (`requirements.txt`)

```ini
# ── Core Web API ──────────────────────────────────────────────────────────────
fastapi==0.111.0
uvicorn[standard]==0.30.1
pydantic==2.7.4
python-multipart==0.0.9
python-dotenv==1.0.1

# ── Database ──────────────────────────────────────────────────────────────────
asyncpg==0.29.0           # Async PostgreSQL driver
psycopg2-binary==2.9.9    # Sync driver for migration scripts

# ── Ingestion & Visual Parsing ────────────────────────────────────────────────
docling==2.0.1             # Visual PDF layout parsing
pypdf==4.2.0               # PDF page text extraction (pre-filter only)
                           # Handles mixed-language corpus (e.g., first 20 pages of 
                           # Techfin/GIC may contain Hindi headers/titles).

# ── AI / Local Inference ──────────────────────────────────────────────────────
ollama==0.2.1              # Async Ollama API client (classifier, boundary, extractor, generator)
fastembed==0.3.6           # Torch-free cross-encoder reranking via ONNX Runtime.
                           # Eliminates torch (~200MB CPU binary) from the production image.
                           # Usage: from fastembed.rerank.cross_encoder import TextCrossEncoder
                           # model = TextCrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                           # scores = model.rerank(query, candidate_texts)

# ── Logging & Observability ───────────────────────────────────────────────────
structlog==24.2.0          # Structured JSON logging
prometheus-client==0.20.0  # Metrics export for TTFT, latency

# ── Testing & Evaluation ─────────────────────────────────────────────────────
pytest==8.2.2
pytest-asyncio==0.23.7
httpx==0.27.0              # Async HTTP client for integration tests
```

> [!NOTE]
> **Why fastembed instead of sentence-transformers?** The `CrossEncoder` class in `sentence-transformers` is a PyTorch-native implementation — it pulls in `torch` (~200MB CPU binary) as a hard dependency even when no GPU is used. `fastembed` provides the identical `ms-marco-MiniLM-L-6-v2` cross-encoder via ONNX Runtime, with zero torch dependency, similar latency on CPU, and a much smaller footprint. For the Mac M4 Air dev environment only (where torch is already installed for MLX fine-tuning), either library works.

### 9.2 — Idempotency Contracts Per Stage

**Rule:** Every stage must be safe to re-run against the same input without corrupting state.

| Stage | Idempotency Mechanism |
|---|---|
| Document Ingestion | `DELETE FROM documents WHERE file_name = :file_name` before insert. Cascades to all AST nodes and relationships automatically. |
| Relationship Extraction | `UNIQUE (source_node_id, target_text_ref, rel_type)` + `ON CONFLICT DO NOTHING`. |
| Corpus Resolution | `UPDATE WHERE is_resolved = FALSE` — resolved edges are never re-processed. |
| Embedding Generation | Embeddings are written during the atomic transaction. If the transaction fails, no embedding is persisted. Re-run re-generates and re-inserts cleanly. |

> [!IMPORTANT]
> **ON CONFLICT DO NOTHING and idempotency:** This is safe because the unique constraint on `(source_node_id, target_text_ref, rel_type)` guarantees that re-running extraction on the same clause produces the same edge with no duplication. If the extractor output changes for the same clause (e.g. after a Modelfile update), the old edge persists — a full re-ingestion (which cascades delete all relationships) is required to refresh. This is by design: relationships are only refreshed on explicit document re-ingestion.

### 9.3 — Error Handling Philosophy
**No fallback chains.** One specific behaviour per failure mode:

| Failure Mode | Root Cause | Resolution |
|---|---|---|
| SLM returns invalid JSON | Model prepended explanation text | Ollama `format: "json"` + Pydantic parse + 1 self-heal with error feedback in prompt |
| Self-heal fails on 2nd attempt | Persistent model confusion on unusual text | Set `needs_repair = TRUE`, commit node with raw text, continue pipeline |
| Embedding call times out | Ollama model not loaded or VRAM pressure | Raise `EmbeddingServiceError`, fail the entire document ingestion, log for operator |
| PostgreSQL transaction fails | DB connection lost or disk full | Rollback automatically (ACID); log critical error; do not retry automatically |
| Docling cannot parse PDF | Corrupted PDF or unusual encoding | Raise `IngestionError`, reject the upload with a clear error message to the user |

---

## 10. UI/UX Architecture

### 10.1 — Design Principles

The UI is built as an **extensible platform**, not a monolithic app. The current feature set covers Q&A, compliance checking, and document management. Future features (e.g. amendment tracker, comparative analysis, export to Word) are added as new pages without touching existing components.

**Technology stack:** React 18 + Vite + Vanilla CSS (no Tailwind). Streamed responses via SSE.

### 10.2 — Application Structure

```
Three-column shell layout:
┌─────────────┬───────────────────────────────┬──────────────────────────────┐
│ Left Nav    │       Main Content Panel       │   Right Context Panel        │
│ (240px)     │       (flexible)               │   (360px, collapsible)       │
│             │                                │                              │
│ ▣ Q&A       │  [Active page renders here]    │  Source citations for        │
│ ▣ Compliance│                                │  the current response        │
│ ▣ Admin     │                                │                              │
│             │                                │  Collapsible by user         │
└─────────────┴───────────────────────────────┴──────────────────────────────┘
```

### 10.3 — Page Designs

#### Page 1: Q&A Interface (`/qa`)
```
┌─────────────────────────────────────────────────────────────┐
│  IFSCA Regulatory Assistant                    [Admin →]     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Conversation history                                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 🧑 Can an IBU accept deposits from Indian residents? │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 🤖 No. An IFSC Banking Unit cannot accept deposits... │   │
│  │                                                      │   │
│  │ | Rule | Detail | Source |                           │   │
│  │ |---|---|---|                                         │   │
│  │ | Deposit Restriction | ... | Section 4(1) |         │   │
│  │                                                      │   │
│  │ [📎 View Sources ▾]                                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  [Filter by document ▾]   [Ask a follow-up...]     [Send]  │
└─────────────────────────────────────────────────────────────┘
```

**Key behaviours:**
- Token-by-token streaming via SSE — user sees the answer build in real time.
- "View Sources" expands inline to show the exact regulation excerpts with breadcrumbs.
- "Filter by document" allows scoping queries to a specific regulation (e.g. Banking Regulations only).
- Conversation history is maintained client-side (not in DB); each session is ephemeral.

#### Page 2: Compliance Checker (`/compliance`)
```
┌─────────────────────────────────────────────────────────────┐
│  Compliance Document Check                                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Upload the entity's document:                              │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  [Drag & drop PDF here]  or  [Browse]               │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Check against: [Banking Regulations ▾]  [All ▾]           │
│                                                             │
│  [▶ Run Compliance Check]                                   │
│                                                             │
│  ─── Live Results ───────────────────────────────────────── │
│  ✅ Section 4(1): Deposit restriction — COMPLIANT           │
│  ❌ Section 6(2): Capital adequacy — NON-COMPLIANT          │
│     Entity capital: USD 12M | Required: USD 20M             │
│     [View Regulation]                                       │
│  ⚠️  Section 9(1): Reporting frequency — NEEDS REVIEW      │
└─────────────────────────────────────────────────────────────┘
```

**Key behaviours:**
- The uploaded entity PDF is chunked (simple character chunking is acceptable here — we're not building an AST for the entity doc).
- Each chunk is queried against the regulation corpus via the retrieval pipeline.
- Results stream live as each section is checked — the user sees compliance status building in real time.
- Status icons: ✅ COMPLIANT / ❌ NON-COMPLIANT / ⚠️ NEEDS REVIEW.
- Each result card is expandable to show the exact regulation text and the quoted entity text.

#### Page 3: Admin Dashboard (`/admin`)
```
┌─────────────────────────────────────────────────────────────┐
│  Admin — Document Corpus Management                          │
├────────────────────────────┬────────────────────────────────┤
│  Corpus (6 documents)       │  Ingest New Document          │
│                             │  ┌──────────────────────────┐ │
│  ✅ IFSCA Act 2019          │  │ [Drag & drop PDF]        │ │
│  ✅ Banking Regulations 2020 │  └──────────────────────────┘ │
│  ✅ GIC Regulations 2025    │  [▶ Ingest]                   │
│  ✅ FinTech Sandbox 2026    │                               │
│  ✅ Techfin Framework       │  Ingestion Log               │
│  ✅ Capital Mkt Amendment   │  [Live log output here...]   │
│                             │                               │
│  ⚠️ 3 nodes need repair     │  Evaluation Metrics          │
│  [View Flagged Nodes]       │  Recall@5: 0.91              │
│                             │  MRR: 0.78                   │
│                             │  Faithfulness: 0.96          │
└────────────────────────────┴────────────────────────────────┘
```

**Key behaviours:**
- All 6 current corpus documents are shown with status.
- New document upload triggers the full ingestion pipeline; live log output streams to the UI.
- Flagged nodes (`needs_repair = TRUE`) are surfaced for operator attention.
- Evaluation metrics panel shows the latest golden dataset scores.

### 10.4 — Frontend Extensibility Design
Each future feature is a new React page registered in `App.jsx`'s router:

```jsx
// Adding a new feature requires only:
// 1. Create pages/NewFeaturePage.jsx
// 2. Add one route entry below
<Routes>
  <Route path="/qa"           element={<QAPage />} />
  <Route path="/compliance"   element={<CompliancePage />} />
  <Route path="/admin"        element={<AdminDashboard />} />
  {/* Future features added here without touching existing pages */}
  <Route path="/amendments"   element={<AmendmentTracker />} />
  <Route path="/compare"      element={<DocumentComparator />} />
</Routes>
```

The `SourcePanel` (right column) is a shared component — all pages use it to display citation context. The Left Nav is driven by a `PAGES` config array — adding a new entry automatically adds it to the nav.

---

## 11. Repository Structure

```
smart-regulator-v2/
│
├── documents/                              # Source regulatory PDFs (corpus root)
│
├── migrations/
│   └── 001_initial_schema.sql              # PostgreSQL DDL (v3.1)
│
├── backend/
│   ├── main.py                             # FastAPI entry point
│   ├── config.py                           # Settings (DB URL, model names, known doc_types)
│   │
│   ├── api/
│   │   ├── qa.py                           # GET /api/qa — SSE streaming Q&A
│   │   ├── compliance.py                   # POST /api/compliance — Compliance audit
│   │   └── admin.py                        # POST /api/admin/ingest — Upload & ingest
│   │
│   ├── database/
│   │   ├── connection.py                   # asyncpg connection pool
│   │   └── queries.py                      # Typed async SQL query functions
│   │
│   └── rag/
│       ├── ingestion/
│       │   ├── orchestrator.py             # Top-level ingestion coordinator
│       │   ├── prefilters.py               # Devanagari filter + English page finder
│       │   ├── ast_builder.py              # Deterministic AST from SLM boundary outputs
│       │   ├── auditor.py                  # Structural audit + max-2 self-heal
│       │   └── corpus_resolver.py          # Resolves PENDING references
│       │
│       ├── extraction/
│       │   ├── schemas.py                  # All Pydantic SLM I/O schemas (source of truth)
│       │   ├── llm_client.py               # Async Ollama client with Pydantic parse + 1 retry
│       │   ├── classifier.py               # ifsca-classifier-3b calls
│       │   ├── boundary_detector.py        # ifsca-boundary-3b calls
│       │   └── relational_extractor.py     # ifsca-extractor-3b batched calls
│       │
│       └── retrieval/
│           ├── pipeline_context.py         # QueryPipelineContext dataclass
│           ├── query_expander.py           # ifsca-expander-3b: 3 query phrasings
│           ├── hybrid_search.py            # HNSW + tsvector BM25 via RRF in SQL
│           ├── hop_expander.py             # Parent chain + REFERS_TO + DEFINES_TERM SQL
│           ├── temporal_filter.py          # Applies SUBSTITUTES/OMITTED_BY temporal rules
│           ├── reranker.py                 # ms-marco-MiniLM-L6-v2 cross-encoder
│           ├── compressor.py               # ifsca-extractor-3b sentence extraction
│           └── generator.py               # ifsca-saullm-7b-ft SSE streaming
│
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   └── src/
│       ├── App.jsx                         # Router + three-column shell
│       ├── config/pages.js                 # Page registry (nav auto-generated)
│       ├── pages/
│       │   ├── QAPage.jsx                  # Streaming chat interface
│       │   ├── CompliancePage.jsx          # PDF upload + live violation stream
│       │   └── AdminDashboard.jsx          # Corpus management + eval metrics
│       └── components/
│           ├── SourcePanel.jsx             # Shared right-column citation viewer
│           ├── StreamingAnswer.jsx         # SSE consumer with progressive render
│           ├── ViolationCard.jsx           # Compliance result card
│           └── IngestionLog.jsx            # Live ingestion log display
│
├── modelfiles/
│   ├── Modelfile.classifier                # ifsca-classifier-3b
│   ├── Modelfile.boundary                  # ifsca-boundary-3b
│   ├── Modelfile.extractor                 # ifsca-extractor-3b
│   └── Modelfile.saullm                   # ifsca-saullm-7b-ft (primary Q&A)
│
├── scripts/
│   ├── generate_synthetic_data.py          # Queries PostgreSQL → RAFT jsonl pairs
│   ├── fine_tune_mlx.py                   # MLX QLoRA training loop (Mac M4 Air)
│   ├── convert_to_gguf.sh                 # Merges LoRA + Q4_K_M quantization
│   ├── compile_modelfiles.sh              # Runs `ollama create` for all Modelfiles
│   ├── onboard_document.py                # CLI: ingest a single PDF
│   ├── supersede_document.py              # CLI: mark document superseded
│   ├── run_node_repair.py                 # CLI: re-process needs_repair nodes
│   └── debug_pipeline.py                  # CLI: trace a query through all stages
│
├── tests/
│   ├── golden_dataset.json                # 100 hand-verified QA pairs
│   ├── run_eval_judge.py                  # Evaluation runner (mistral-nemo judge)
│   └── test_ingestion.py                  # Unit tests: AST builder, auditor, resolver
│
├── .env.example                           # Environment variable template
├── docker-compose.yml                     # postgres + backend + ollama services
└── requirements.txt                       # Pinned Python dependencies
```

---

## 12. Developer Setup Guide

### 12.1 — Ubuntu Server Setup (Production)

```bash
# 1. Install system dependencies
sudo apt update && sudo apt install -y \
    postgresql-16 postgresql-server-dev-16 postgresql-16-pgvector \
    python3.11 python3.11-venv git curl

# 2. Start PostgreSQL and create the database
sudo systemctl start postgresql
sudo -u postgres psql -c "CREATE DATABASE smart_regulator;"
sudo -u postgres psql -c "CREATE USER sr_app WITH PASSWORD 'your_password_here';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE smart_regulator TO sr_app;"
sudo -u postgres psql -d smart_regulator -f migrations/001_initial_schema.sql

# Verify pgvector is installed
sudo -u postgres psql -d smart_regulator -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 3. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable ollama

# Set Ollama environment (add to /etc/environment)
echo 'OLLAMA_NUM_PARALLEL=4' | sudo tee -a /etc/environment
echo 'OLLAMA_KEEP_ALIVE=60m' | sudo tee -a /etc/environment
echo 'OLLAMA_FLASH_ATTENTION=1' | sudo tee -a /etc/environment
source /etc/environment

# 4. Pull base models
ollama pull llama3.2:3b
ollama pull nomic-embed-text:v1.5

# 5. Compile all Modelfiles
bash scripts/compile_modelfiles.sh

# 6. Set up the Python environment
git clone https://github.com/your-org/smart-regulator-v2.git
cd smart-regulator-v2
python3.11 -m venv venv
source venv/bin/activate

# Install CPU-only torch (avoids 2GB CUDA download)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 7. Configure environment
cp .env.example .env
# Edit .env: set DATABASE_URL, OLLAMA_HOST

# 8. Ingest the initial corpus
for f in documents/*.pdf; do
    python scripts/onboard_document.py --file "$f"
done

# 9. Start the backend
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### 12.2 — macOS M4 Air Setup (Development & Fine-tuning)

```bash
# 1. Install Homebrew dependencies
brew update
brew install postgresql@16 python@3.11 git

# Start PostgreSQL (local development DB)
brew services start postgresql@16
createdb smart_regulator_dev
psql smart_regulator_dev -f migrations/001_initial_schema.sql

# Install pgvector via Homebrew
brew install pgvector

# 2. Install Ollama
# Download from https://ollama.com/download/mac and install the .app
# Or via Homebrew:
brew install ollama
brew services start ollama

# Pull models (M4 Air can handle larger models via Metal)
ollama pull llama3.2:3b
ollama pull nomic-embed-text:v1.5
ollama pull mistral-nemo:12b       # For evaluation (judge model)

# Pull SaulLM for fine-tuning base (via MLX community)
# No Ollama pull needed — MLX handles directly
pip install mlx-lm

# 3. Set up Python environment
python3.11 -m venv venv
source venv/bin/activate

# On M4 Air, use the standard torch (Metal GPU support)
pip install torch torchvision  # No --cpu flag; Metal acceleration used
pip install -r requirements.txt
pip install mlx mlx-lm         # Fine-tuning only — not in main requirements.txt

# 4. Configure environment
cp .env.example .env
# Set DATABASE_URL to the local PostgreSQL instance

# 5. Compile Modelfiles
bash scripts/compile_modelfiles.sh

# 6. Start the backend (dev mode with hot reload)
uvicorn backend.main:app --reload --port 8000
```

---

## 13. Operations Runbook

### 13.1 — Ingest Initial Corpus (All 6 Documents)
```bash
# Run once after infrastructure setup
python scripts/onboard_document.py --file "documents/IFSCA ACT.pdf"
python scripts/onboard_document.py --file "documents/consolidated-ifsca-banking-regulations-as-on-july-14-202314082023111415.pdf"
python scripts/onboard_document.py --file "documents/Global In House.pdf"
python scripts/onboard_document.py --file "documents/IFSCA_FinTech_Sandbox_Framework_20260316_0338.pdf"
python scripts/onboard_document.py --file "documents/Techfin.pdf"
python scripts/onboard_document.py --file "documents/capital_market_intermediaries__amendment_regulations.pdf"

# After all documents are ingested, run the corpus resolver
# (resolves cross-document references)
python -m backend.rag.ingestion.corpus_resolver
```

### 13.2 — Ingest a New Regulation (Future)
```bash
# The system auto-detects doc_type and amendment status from the document
python scripts/onboard_document.py --file "/path/to/new_regulation.pdf"
# Amendment detection is automatic — no --type flag needed
```

### 13.3 — Handling `needs_repair` Nodes
```bash
# Query flagged nodes
psql $DATABASE_URL -c "SELECT node_id, breadcrumb, LEFT(text_content, 100) FROM ast_nodes WHERE needs_repair = TRUE;"

# Re-process a specific node
python scripts/run_node_repair.py --node-id <uuid>

# Re-process ALL flagged nodes
python scripts/run_node_repair.py --all
```

### 13.4 — Database Backup
```bash
# Full backup (run daily via cron)
pg_dump -U sr_app -d smart_regulator -F c -b -v \
    -f "backups/smart_regulator_$(date +%F_%H%M).backup"

# List available backups
ls -lh backups/

# Restore from backup (use in disaster recovery)
pg_restore -U sr_app -d smart_regulator -v "backups/smart_regulator_2026-06-14_0200.backup"
```

### 13.5 — Monitor HNSW Index Performance
```sql
-- Check current ef_search setting
SHOW hnsw.ef_search;

-- Tune ef_search for higher recall (no index rebuild needed)
SET hnsw.ef_search = 100;

-- Check index size and health
SELECT indexname, pg_size_pretty(pg_relation_size(indexname::regclass))
FROM pg_indexes
WHERE tablename = 'ast_nodes';
```

---

## 14. Debugging & Pipeline Tracing Playbook

### 14.1 — End-to-End Query Trace
```bash
python scripts/debug_pipeline.py \
    --query "Can an IBU accept deposits from Indian residents?" \
    --verbose
```

**Expected output per stage:**
```
[request_id: a8f9b1c0]

[Stage A: Query Expander] (112ms)
  Expansions:
    1. IBU deposit acceptance rules IFSCA Banking Regulations
    2. Section on permissible deposits for IFSC Banking Units
    3. definition of permissible customer IFSCA IBU regulations

[Stage B: Hybrid Retrieval] (43ms)
  HNSW candidates: 80 (4 queries × 20 each)
  FTS candidates: 20
  After RRF merge: 20 unique nodes
  Top-3:
    Rank 1: node_id=8fa7b2a6 | Section 4(1) | score=0.91
    Rank 2: node_id=3c21d5a0 | Section 2(k) Definition | score=0.74
    Rank 3: node_id=9f44b8e1 | Schedule I | score=0.61

[Stage C: Relational Expansion] (8ms)
  Expanded Rank 1 via REFERS_TO → appended Section 2(k)
  Inlined glossary: "Permitted Currency" → [definition text]
  SUBSTITUTES check: Section 4(1) has no active substitution

[Stage D: Reranker] (28ms)
  ms-marco-MiniLM-L6-v2 re-scores 20 expanded nodes
  Top-5 selected: node_ids=[8fa7b2a6, 3c21d5a0, 2b18e3f7, ...]

[Stage E: Compressor] (183ms)
  Input tokens: ~2,100 (5 nodes × avg 420 tokens)
  Output tokens: 97 (compressed to relevant sentences only)

[Stage F: Generator] — streaming
  TTFT: 621ms
  Model: ifsca-saullm-7b-ft
```

### 14.2 — Inspect Structured Logs
```bash
# View all logs for a request (replace with actual request_id)
cat logs/app.log | jq 'select(.request_id == "a8f9b1c0")' | jq '{stage, duration_ms, msg}'

# Find all repair events in the last 24 hours
cat logs/app.log | jq 'select(.event == "node_needs_repair")' | tail -20

# Find all newly discovered doc_types
cat logs/app.log | jq 'select(.event == "new_doc_type_discovered")'

# Monitor resolution rate
cat logs/app.log | jq 'select(.event == "corpus_resolution_complete") | {resolved, total, rate}'
```

### 14.3 — Diagnose Low Recall
1. Run `debug_pipeline.py` — check if the golden answer node appears in Stage B candidates.
2. If **not in Stage B**: embedding or FTS retrieval failure. Try increasing `ef_search` (`SET hnsw.ef_search = 150`) and re-run. If it appears then, raise the default in `config.py`.
3. If **in Stage B but not Stage D**: the cross-encoder reranker demoted it. Inspect the `ms-marco-MiniLM-L6-v2` scores. If the node is genuinely relevant but poorly ranked, it indicates the expanded context (Stage C) added too much noise — investigate `hop_expander.py`.
4. If **in Stage D but not in final answer**: the compressor may have removed the critical sentence. Check Stage E output directly. Tighten the compression prompt.

### 14.4 — Debug Ingestion Failures
```bash
# Re-run ingestion with debug logging
LOG_LEVEL=DEBUG python scripts/onboard_document.py --file documents/IFSCA\ ACT.pdf 2>&1 | tee ingest_debug.log

# Check for repair flags after ingestion
psql $DATABASE_URL -c "
    SELECT d.file_name, COUNT(*) AS repair_count
    FROM ast_nodes a
    JOIN documents d ON a.doc_id = d.doc_id
    WHERE a.needs_repair = TRUE
    GROUP BY d.file_name;"

# Inspect a specific failed node
psql $DATABASE_URL -c "
    SELECT node_id, breadcrumb, LEFT(text_content, 200)
    FROM ast_nodes
    WHERE needs_repair = TRUE
    LIMIT 5;"
```

---

## 15. Phased Delivery Plan

### Phase 0 — Infrastructure Setup (Days 1–2)
- [ ] Provision PostgreSQL on Ubuntu server; run `001_initial_schema.sql`
- [ ] Verify pgvector HNSW index creation on GTX 3050
- [ ] Install Ollama; pull `llama3.2:3b` and `nomic-embed-text:v1.5`
- [ ] Set Ollama env vars (`OLLAMA_NUM_PARALLEL=4`, `OLLAMA_KEEP_ALIVE=60m`, `OLLAMA_FLASH_ATTENTION=1`)
- [ ] Compile all 4 Modelfiles via `scripts/compile_modelfiles.sh`
- [ ] Set up macOS dev environment with local PostgreSQL and Ollama

### Phase 1 — Ingestion Pipeline (Days 3–7)
- [ ] Implement `prefilters.py` — Devanagari filter + `find_english_classifier_window()`
- [ ] Implement `schemas.py` — all Pydantic SLM I/O models
- [ ] Implement `llm_client.py` — async Ollama client (Pydantic parse + 1 retry)
- [ ] Implement `classifier.py`, `boundary_detector.py`, `relational_extractor.py`
- [ ] Implement `ast_builder.py` — deterministic tree from SLM boundary outputs
- [ ] Implement `auditor.py` + `corpus_resolver.py`
- [ ] Implement `orchestrator.py` — wire all stages
- [ ] Run `onboard_document.py` on all 6 source documents; verify AST in DB

### Phase 2 — Retrieval Pipeline (Days 8–12)
- [ ] Implement `pipeline_context.py` — `QueryPipelineContext` dataclass
- [ ] Implement all 6 retrieval stages (expander, hybrid_search, hop_expander, temporal_filter, reranker, compressor, generator)
- [ ] Build FastAPI endpoints (`/api/qa`, `/api/compliance`, `/api/admin/ingest`)
- [ ] Implement `debug_pipeline.py` CLI tool

### Phase 3 — Fine-Tuning & Evaluation (Days 13–18)
- [ ] Run `generate_synthetic_data.py` → 2,000 RAFT training pairs
- [ ] Execute `fine_tune_mlx.py` on Mac M4 Air (SaulLM-7B, ~45 mins)
- [ ] Run `convert_to_gguf.sh` → `ifsca-saullm-7b-ft.Q4_K_M.gguf`
- [ ] Compile `Modelfile.saullm`; deploy to Ubuntu Ollama
- [ ] Assemble 100-question `golden_dataset.json` from actual IFSCA documents
- [ ] Run `run_eval_judge.py` — baseline vs. our architecture
- [ ] Document all metric scores; identify improvement areas

### Phase 4 — Frontend & Production Hardening (Days 19–24)
- [ ] Build React + Vite frontend (QAPage, CompliancePage, AdminDashboard)
- [ ] Implement three-column shell layout with extensible page registry
- [ ] Integrate SSE streaming consumer (StreamingAnswer component)
- [ ] Build IngestionLog live streaming component
- [ ] Load test: 5–20 concurrent users; record TTFT and P95 latency
- [ ] Write `docker-compose.yml` covering postgres + backend + ollama
- [ ] Final deployment to Ubuntu server; smoke test all modes
