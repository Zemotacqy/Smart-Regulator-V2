import time
from typing import List, Dict, Set, Optional
from uuid import UUID
import structlog
from datetime import date
from dataclasses import replace

from backend.database.connection import get_db_connection
from backend.rag.retrieval.pipeline_context import QueryPipelineContext, NodeCandidate

logger = structlog.get_logger()

async def run_temporal_filter(ctx: QueryPipelineContext) -> QueryPipelineContext:
    """
    Stage: Temporal Filtering.
    Checks for active SUBSTITUTES or OMITTED_BY relationships on all candidate and expanded nodes.
    If a node is substituted, it is swapped for the new (amended) node, resolving transitively.
    If a node is omitted, it is removed from the context entirely.
    """
    start_time = time.monotonic()
    
    # Combine candidate and expanded nodes to check
    all_nodes = ctx.candidate_nodes + ctx.expanded_nodes
    if not all_nodes:
        ctx.stage_timings["temporal_filter"] = time.monotonic() - start_time
        return ctx
        
    original_node_map = {node.node_id: node for node in all_nodes}
    logger.info("temporal_filter_start", total_nodes=len(original_node_map))
    
    current_date = date.today()
    
    try:
        resolved_relations: Dict[UUID, dict] = {}
        all_fetched: Set[UUID] = set()
        current_ids: Set[UUID] = set(original_node_map.keys())
        
        async with get_db_connection() as conn:
            # Loop to resolve transitive substitutions
            loop_guard = 0
            while (current_ids - all_fetched) and loop_guard < 10:
                loop_guard += 1
                to_fetch = list(current_ids - all_fetched)
                all_fetched.update(to_fetch)
                
                query = """
                SELECT source_node_id, target_node_id, rel_type, effective_date, created_at
                FROM relationships
                WHERE target_node_id = ANY($1::uuid[])
                  AND rel_type IN ('SUBSTITUTES', 'OMITTED_BY')
                  AND (effective_date IS NULL OR effective_date <= $2::date)
                ORDER BY effective_date DESC NULLS LAST, created_at DESC;
                """
                rows = await conn.fetch(query, to_fetch, current_date)
                
                for row in rows:
                    target_id = row["target_node_id"]
                    # Keep only the latest relationship targeting this node
                    if target_id not in resolved_relations:
                        resolved_relations[target_id] = dict(row)
                        if row["rel_type"] == "SUBSTITUTES":
                            current_ids.add(row["source_node_id"])
            
            # Fetch details for all replacement nodes
            substitute_source_ids = [
                rel["source_node_id"] 
                for rel in resolved_relations.values() 
                if rel["rel_type"] == "SUBSTITUTES"
            ]
            
            replacement_nodes: Dict[UUID, NodeCandidate] = {}
            if substitute_source_ids:
                replacement_query = """
                SELECT n.node_id, n.doc_id, n.parent_id, n.level, n.node_type, n.title, n.text_content, n.breadcrumb, d.file_name
                FROM ast_nodes n
                JOIN documents d ON n.doc_id = d.doc_id
                WHERE n.node_id = ANY($1::uuid[]) AND d.is_active = TRUE;
                """
                repl_rows = await conn.fetch(replacement_query, substitute_source_ids)
                for r_row in repl_rows:
                    node_id = r_row["node_id"]
                    replacement_nodes[node_id] = NodeCandidate(
                        node_id=node_id,
                        doc_id=r_row["doc_id"],
                        parent_id=r_row["parent_id"],
                        level=r_row["level"],
                        node_type=r_row["node_type"],
                        title=r_row["title"],
                        text_content=r_row["text_content"] or "",
                        breadcrumb=r_row["breadcrumb"],
                        file_name=r_row["file_name"]
                    )
                    
        # Apply rules to resolve each original node
        final_replacements: Dict[UUID, Optional[NodeCandidate]] = {}
        
        for orig_id in original_node_map.keys():
            curr_id = orig_id
            is_omitted = False
            visited = {curr_id}
            
            while curr_id in resolved_relations:
                rel = resolved_relations[curr_id]
                if rel["rel_type"] == "OMITTED_BY":
                    is_omitted = True
                    break
                elif rel["rel_type"] == "SUBSTITUTES":
                    next_id = rel["source_node_id"]
                    if next_id in visited:
                        logger.error("temporal_filter_substitution_loop_detected", path=list(visited))
                        break # Break loop if cycle detected
                    visited.add(next_id)
                    curr_id = next_id
                else:
                    break
                    
            if is_omitted:
                final_replacements[orig_id] = None
            elif curr_id != orig_id:
                # If we swapped, fetch the replacement node if active, otherwise omit or keep original
                if curr_id in replacement_nodes:
                    final_replacements[orig_id] = replacement_nodes[curr_id]
                else:
                    # Replacement node is not active/available; omit the candidate
                    logger.warning("temporal_filter_replacement_inactive_or_missing", replacement_id=str(curr_id))
                    final_replacements[orig_id] = None
            else:
                final_replacements[orig_id] = original_node_map[orig_id]
                
        # Helper to process lists
        def process_nodes(nodes_list: List[NodeCandidate]) -> List[NodeCandidate]:
            result_list = []
            for node in nodes_list:
                node_id = node.node_id
                if node_id in final_replacements:
                    repl = final_replacements[node_id]
                    if repl is None:
                        logger.info("temporal_filter_removed_or_omitted", node_id=str(node_id), breadcrumb=node.breadcrumb)
                        continue
                    if repl.node_id != node_id:
                        # Use replace to avoid mutating shared replacement nodes
                        new_node = replace(repl, score=node.score)
                        logger.info("temporal_filter_replaced", 
                                    old_id=str(node_id), old_breadcrumb=node.breadcrumb,
                                    new_id=str(new_node.node_id), new_breadcrumb=new_node.breadcrumb)
                        result_list.append(new_node)
                    else:
                        result_list.append(node)
                else:
                    result_list.append(node)
            return result_list
            
        ctx.candidate_nodes = process_nodes(ctx.candidate_nodes)
        ctx.expanded_nodes = process_nodes(ctx.expanded_nodes)
        
        logger.info("temporal_filter_complete", 
                    candidates_count=len(ctx.candidate_nodes), 
                    expanded_count=len(ctx.expanded_nodes))
                    
    except Exception as e:
        logger.error("temporal_filter_failed", error=str(e))
        # Keep original nodes on failure
        
    ctx.stage_timings["temporal_filter"] = time.monotonic() - start_time
    return ctx
