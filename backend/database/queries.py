import asyncpg
from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import date
import structlog

logger = structlog.get_logger()

async def get_document_by_filename(conn: asyncpg.Connection, file_name: str) -> Optional[Dict[str, Any]]:
    """
    Checks if a document with the given file name exists in the database.
    """
    row = await conn.fetchrow(
        "SELECT doc_id, file_name, title, publish_date, doc_type, is_active FROM documents WHERE file_name = $1",
        file_name
    )
    return dict(row) if row else None

async def delete_document_by_filename(conn: asyncpg.Connection, file_name: str) -> None:
    """
    Deletes a document by its file name. This cascades to delete all related AST nodes,
    relationships, and glossary terms automatically due to FOREIGN KEY constraints.
    """
    logger.info("deleting_document_from_db", file_name=file_name)
    await conn.execute("DELETE FROM documents WHERE file_name = $1", file_name)

async def insert_document(
    conn: asyncpg.Connection,
    file_name: str,
    title: str,
    publish_date: Optional[date],
    doc_type: str
) -> UUID:
    """
    Inserts a new document record and returns the doc_id.
    """
    doc_id = await conn.fetchval(
        """
        INSERT INTO documents (file_name, title, publish_date, doc_type)
        VALUES ($1, $2, $3, $4)
        RETURNING doc_id
        """,
        file_name, title, publish_date, doc_type
    )
    return doc_id

async def insert_ast_nodes(
    conn: asyncpg.Connection,
    nodes: List[Dict[str, Any]]
) -> None:
    """
    Bulk inserts AST nodes into the database.
    Each node dict should contain:
    - node_id (UUID)
    - doc_id (UUID)
    - parent_id (UUID or None)
    - level (int)
    - node_type (str)
    - title (str or None)
    - text_content (str or None)
    - breadcrumb (str)
    - needs_repair (bool)
    - embedding (list of floats or None)
    """
    query = """
        INSERT INTO ast_nodes (node_id, doc_id, parent_id, level, node_type, title, text_content, breadcrumb, needs_repair, embedding)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::vector)
    """
    records = [
        (
            n["node_id"],
            n["doc_id"],
            n["parent_id"],
            n["level"],
            n["node_type"],
            n["title"],
            n["text_content"],
            n["breadcrumb"],
            n["needs_repair"],
            str(n["embedding"]) if n["embedding"] is not None else None
        )
        for n in nodes
    ]
    await conn.executemany(query, records)

async def insert_relationships(
    conn: asyncpg.Connection,
    relationships: List[Dict[str, Any]]
) -> None:
    """
    Bulk inserts relationship edges.
    Each relationship dict should contain:
    - source_node_id (UUID)
    - target_text_ref (str)
    - rel_type (str)
    - effective_date (date or None)
    """
    query = """
        INSERT INTO relationships (source_node_id, target_text_ref, rel_type, effective_date)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (source_node_id, target_text_ref, rel_type) DO NOTHING
    """
    records = [
        (
            r["source_node_id"],
            r["target_text_ref"],
            r["rel_type"],
            r["effective_date"]
        )
        for r in relationships
    ]
    await conn.executemany(query, records)

async def insert_glossary_entry(
    conn: asyncpg.Connection,
    term: str,
    doc_id: UUID,
    definition: str,
    source_node_id: UUID
) -> None:
    """
    Inserts or updates a glossary term definition.
    """
    await conn.execute(
        """
        INSERT INTO glossary (term, doc_id, definition, source_node_id)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (term, doc_id) DO UPDATE 
        SET definition = EXCLUDED.definition, source_node_id = EXCLUDED.source_node_id
        """,
        term, doc_id, definition, source_node_id
    )
