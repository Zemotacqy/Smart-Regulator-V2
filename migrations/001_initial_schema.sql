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
