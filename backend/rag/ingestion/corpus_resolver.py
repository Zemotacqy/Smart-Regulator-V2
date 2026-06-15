import asyncio
import asyncpg
import structlog
from backend.database.connection import get_db_connection, init_db_pool, close_db_pool

logger = structlog.get_logger()

async def resolve_pending_references() -> None:
    """
    Scans the database for all unresolved relationships (is_resolved = FALSE),
    attempts to match their target_text_ref to a specific ast_node,
    and updates the target_node_id and is_resolved state.
    
    If the relationship is DEFINES_TERM, it also populates the glossary table.
    """
    logger.info("corpus_resolver_started")
    
    async with get_db_connection() as conn:
        # 1. Fetch all unresolved relationships, along with their source node details
        unresolved = await conn.fetch(
            """
            SELECT r.rel_id, r.source_node_id, r.target_text_ref, r.rel_type,
                   a.doc_id, a.text_content
            FROM relationships r
            JOIN ast_nodes a ON r.source_node_id = a.node_id
            WHERE r.is_resolved = FALSE
            """
        )
        
        if not unresolved:
            logger.info("corpus_resolver_no_pending_references")
            return
            
        logger.info("corpus_resolver_pending_count", count=len(unresolved))
        resolved_count = 0
        
        for rel in unresolved:
            rel_id = rel["rel_id"]
            source_node_id = rel["source_node_id"]
            target_text_ref = rel["target_text_ref"].strip()
            rel_type = rel["rel_type"]
            doc_id = rel["doc_id"]
            definition_text = rel["text_content"] or ""
            
            logger.debug(
                "resolving_reference",
                rel_id=str(rel_id),
                rel_type=rel_type,
                target_text_ref=target_text_ref
            )
            
            target_node_id = None
            
            # Match strategy 0: Check match within the parent section hierarchy (relative references)
            section_id = await conn.fetchval(
                """
                WITH RECURSIVE ancestors AS (
                    SELECT node_id, parent_id, level FROM ast_nodes WHERE node_id = $1
                    UNION ALL
                    SELECT n.node_id, n.parent_id, n.level FROM ast_nodes n JOIN ancestors a ON n.node_id = a.parent_id
                )
                SELECT node_id FROM ancestors WHERE level = 3 LIMIT 1;
                """,
                source_node_id
            )
            
            if section_id:
                target_node_id = await conn.fetchval(
                    """
                    WITH RECURSIVE descendants AS (
                        SELECT node_id, parent_id, title, level FROM ast_nodes WHERE parent_id = $1
                        UNION ALL
                        SELECT n.node_id, n.parent_id, n.title, n.level FROM ast_nodes n JOIN descendants d ON n.parent_id = d.node_id
                    )
                    SELECT node_id FROM descendants 
                    WHERE title = $2 OR title ILIKE $3 OR title ILIKE $4
                    ORDER BY level ASC
                    LIMIT 1;
                    """,
                    section_id, target_text_ref, f"{target_text_ref}%", f"% {target_text_ref}%"
                )
            
            # Match strategy 1: Check exact title match in the same document
            if not target_node_id:
                target_node_id = await conn.fetchval(
                    """
                    SELECT node_id FROM ast_nodes
                    WHERE doc_id = $1 AND title = $2
                    LIMIT 1
                    """,
                    doc_id, target_text_ref
                )
            
            # Match strategy 2: Check prefix/like title match in the same document (e.g. "Section 4" matches "Section 4. Capital requirement")
            if not target_node_id:
                # Add word boundary check or simple ILIKE starting match
                ref_like = f"{target_text_ref}%"
                target_node_id = await conn.fetchval(
                    """
                    SELECT node_id FROM ast_nodes
                    WHERE doc_id = $1 AND (title ILIKE $2 OR title ILIKE $3)
                    ORDER BY level ASC
                    LIMIT 1
                    """,
                    doc_id, ref_like, f"% {target_text_ref}%"
                )
                
            # Match strategy 3: Search text content for definitions if DEFINES_TERM
            if not target_node_id and rel_type == "DEFINES_TERM":
                # Look for nodes of type DEFINITION that define the term in the same document
                target_node_id = await conn.fetchval(
                    """
                    SELECT node_id FROM ast_nodes
                    WHERE doc_id = $1 AND node_type = 'DEFINITION' AND text_content ILIKE $2
                    LIMIT 1
                    """,
                    doc_id, f"%{target_text_ref}%"
                )
                
            # Match strategy 4: Search cross-document if the reference looks like a doc title
            # For this baseline, we scope it to same document references.
            
            if target_node_id:
                # Found the target node! Update the relationship
                await conn.execute(
                    """
                    UPDATE relationships
                    SET target_node_id = $1, is_resolved = TRUE
                    WHERE rel_id = $2
                    """,
                    target_node_id, rel_id
                )
                resolved_count += 1
                
                # If relationship is DEFINES_TERM, insert into glossary
                if rel_type == "DEFINES_TERM":
                    await conn.execute(
                        """
                        INSERT INTO glossary (term, doc_id, definition, source_node_id)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT (term, doc_id) DO UPDATE
                        SET definition = EXCLUDED.definition, source_node_id = EXCLUDED.source_node_id
                        """,
                        target_text_ref, doc_id, definition_text, source_node_id
                    )
                    logger.debug("glossary_entry_created", term=target_text_ref)
                    
                logger.debug("reference_resolved", rel_id=str(rel_id), target_node_id=str(target_node_id))
            else:
                logger.debug("reference_resolution_failed", rel_id=str(rel_id), target_text_ref=target_text_ref)
                
        logger.info(
            "corpus_resolver_completed",
            total=len(unresolved),
            resolved=resolved_count,
            unresolved=len(unresolved) - resolved_count
        )

if __name__ == "__main__":
    async def run():
        await init_db_pool()
        try:
            await resolve_pending_references()
        finally:
            await close_db_pool()
            
    asyncio.run(run())
