import time
from typing import List, Dict
from uuid import UUID
import structlog

from backend.database.connection import get_db_connection
from backend.rag.retrieval.pipeline_context import QueryPipelineContext, NodeCandidate

logger = structlog.get_logger()

async def run_hop_expander(ctx: QueryPipelineContext) -> QueryPipelineContext:
    """
    Stage C: Relational Hop Expansion.
    1. Fetches parent chains for candidate nodes.
    2. Fetches nodes referenced via 'REFERS_TO' relationships.
    3. Fetches glossary entries for terms defined by the candidate nodes.
    Populates ctx.expanded_nodes and ctx.inlined_definitions.
    """
    start_time = time.monotonic()
    
    if not ctx.candidate_nodes:
        logger.warning("hop_expander_skipped_no_candidates")
        ctx.stage_timings["hop_expander"] = time.monotonic() - start_time
        return ctx
        
    candidate_ids = [node.node_id for node in ctx.candidate_nodes]
    logger.info("hop_expander_start", candidate_count=len(candidate_ids))
    
    try:
        async with get_db_connection() as conn:
            # 1. Fetch parent chains for all candidate nodes
            parent_query = """
            WITH RECURSIVE parents AS (
                SELECT node_id, doc_id, parent_id, level, node_type, title, text_content, breadcrumb, 1 as distance
                FROM ast_nodes
                WHERE node_id = ANY($1::uuid[])
                UNION ALL
                SELECT n.node_id, n.doc_id, n.parent_id, n.level, n.node_type, n.title, n.text_content, n.breadcrumb, p.distance + 1
                FROM ast_nodes n
                JOIN parents p ON n.node_id = p.parent_id
            )
            SELECT DISTINCT ON (p.node_id) p.node_id, p.doc_id, p.parent_id, p.level, p.node_type, p.title, p.text_content, p.breadcrumb, d.file_name
            FROM parents p
            JOIN documents d ON p.doc_id = d.doc_id;
            """
            parent_rows = await conn.fetch(parent_query, candidate_ids)
            
            # 2. Fetch REFERS_TO targets for candidate nodes
            ref_query = """
            SELECT r.target_node_id, n.node_id, n.doc_id, n.parent_id, n.level, n.node_type, n.title, n.text_content, n.breadcrumb, d.file_name
            FROM relationships r
            JOIN ast_nodes n ON r.target_node_id = n.node_id
            JOIN documents d ON n.doc_id = d.doc_id
            WHERE r.source_node_id = ANY($1::uuid[]) AND r.rel_type = 'REFERS_TO' AND r.is_resolved = TRUE AND d.is_active = TRUE;
            """
            ref_rows = await conn.fetch(ref_query, candidate_ids)
            
            # 3. Fetch glossary terms defined by candidate nodes
            glossary_query = """
            SELECT term, definition
            FROM glossary
            WHERE source_node_id = ANY($1::uuid[]);
            """
            glossary_rows = await conn.fetch(glossary_query, candidate_ids)
            
        # Parse fetched nodes into dictionary mapping node_id -> NodeCandidate
        all_candidate_ids = {node.node_id for node in ctx.candidate_nodes}
        expanded_nodes_dict: Dict[UUID, NodeCandidate] = {}
        
        # Add parent chain nodes
        for row in parent_rows:
            node_id = row["node_id"]
            if node_id not in all_candidate_ids:
                expanded_nodes_dict[node_id] = NodeCandidate(
                    node_id=node_id,
                    doc_id=row["doc_id"],
                    parent_id=row["parent_id"],
                    level=row["level"],
                    node_type=row["node_type"],
                    title=row["title"],
                    text_content=row["text_content"] or "",  # Fallback to empty string for safety
                    breadcrumb=row["breadcrumb"],
                    file_name=row["file_name"]
                )
                
        # Add referenced target nodes
        for row in ref_rows:
            node_id = row["node_id"]
            if node_id not in all_candidate_ids and node_id not in expanded_nodes_dict:
                expanded_nodes_dict[node_id] = NodeCandidate(
                    node_id=node_id,
                    doc_id=row["doc_id"],
                    parent_id=row["parent_id"],
                    level=row["level"],
                    node_type=row["node_type"],
                    title=row["title"],
                    text_content=row["text_content"] or "",  # Fallback to empty string for safety
                    breadcrumb=row["breadcrumb"],
                    file_name=row["file_name"]
                )
                
        # Inline definitions
        ctx.inlined_definitions = {row["term"]: row["definition"] for row in glossary_rows}
        ctx.expanded_nodes = list(expanded_nodes_dict.values())
        
        logger.info("hop_expander_complete", 
                    expanded_count=len(ctx.expanded_nodes), 
                    definitions_count=len(ctx.inlined_definitions))
                    
    except Exception as e:
        logger.error("hop_expander_failed", error=str(e))
        ctx.expanded_nodes = []
        ctx.inlined_definitions = {}
        
    ctx.stage_timings["hop_expander"] = time.monotonic() - start_time
    return ctx
