import time
from typing import List, Dict, Tuple, Set
from uuid import UUID
from collections import defaultdict
import structlog

from backend.database.connection import get_db_connection
from backend.rag.retrieval.pipeline_context import QueryPipelineContext, NodeCandidate

logger = structlog.get_logger()

async def run_hop_expander(ctx: QueryPipelineContext) -> QueryPipelineContext:
    """
    Stage C: Hierarchical Section Rollup.
    1. For each candidate node, resolves the nearest SECTION (level 3) or SUBSECTION (level 4) parent.
    2. Fallbacks to the candidate node itself if no section/subsection parent exists.
    3. Recursively fetches all descendant nodes for those anchor section nodes.
    4. Merges text contents sorted by physical tuple order (ctid) to preserve visual hierarchy.
    5. Formats descendant nodes using Markdown headers (##, ###, ####...) scaled by depth,
       so each sub-section is visually discrete in the generator's context window.
       This prevents the generator from conflating adjacent numbered sub-sections
       (e.g. mistaking Section 13(6) for the answer to a question answered by Section 13(5)).
       Blocks are joined with double newlines so compressor.split_large_section() splits
       at clean sub-section boundaries rather than mid-sentence when the section is large.
    6. Fetches refers_to references and glossary definitions for the section context.
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
            # 1. Resolve closest SECTION (level 3) or SUBSECTION (level 4) ancestors
            anchor_query = """
            WITH RECURSIVE ancestors AS (
                SELECT node_id as original_id, node_id, parent_id, level, node_type
                FROM ast_nodes
                WHERE node_id = ANY($1::uuid[])
                UNION ALL
                SELECT a.original_id, n.node_id, n.parent_id, n.level, n.node_type
                FROM ast_nodes n
                JOIN ancestors a ON n.node_id = a.parent_id
                WHERE a.level > 3
            )
            SELECT DISTINCT ON (original_id) original_id, node_id as anchor_id, level
            FROM ancestors
            WHERE level IN (3, 4)
            ORDER BY original_id, level ASC;
            """
            anchor_rows = await conn.fetch(anchor_query, candidate_ids)
            candidate_to_anchor = {row["original_id"]: row["anchor_id"] for row in anchor_rows}
            
            # Form final unique set of anchor node IDs
            all_anchors = set()
            for node in ctx.candidate_nodes:
                anchor_id = candidate_to_anchor.get(node.node_id) or node.node_id
                all_anchors.add(anchor_id)
                
            logger.info("hop_expander_resolved_anchors", anchors_count=len(all_anchors))
            
            # 2. Fetch all descendants recursively for all resolved anchor IDs
            descendant_query = """
            WITH RECURSIVE descendants AS (
                SELECT node_id as anchor_id, node_id, parent_id, level, node_type, title, text_content, breadcrumb, doc_id, ctid
                FROM ast_nodes
                WHERE node_id = ANY($1::uuid[])
                UNION ALL
                SELECT d.anchor_id, n.node_id, n.parent_id, n.level, n.node_type, n.title, n.text_content, n.breadcrumb, n.doc_id, n.ctid
                FROM ast_nodes n
                JOIN descendants d ON n.parent_id = d.node_id
            )
            SELECT d.anchor_id, d.node_id, d.parent_id, d.level, d.node_type, d.title, d.text_content, d.breadcrumb, d.doc_id, d.ctid::text, doc.file_name
            FROM descendants d
            JOIN documents doc ON d.doc_id = doc.doc_id
            ORDER BY d.anchor_id, d.ctid;
            """
            descendant_rows = await conn.fetch(descendant_query, list(all_anchors))
            
            # Group rows by anchor_id
            anchor_descendants = defaultdict(list)
            for row in descendant_rows:
                anchor_descendants[row["anchor_id"]].append(row)
                
            # Construct rolled-up section candidates
            rolled_up_candidates = []
            all_descendant_ids = set()
            
            for anchor_id in all_anchors:
                descendants = anchor_descendants.get(anchor_id, [])
                if not descendants:
                    continue
                    
                # Use the anchor row details as node template
                anchor_row = next((r for r in descendants if r["node_id"] == anchor_id), descendants[0])
                anchor_level = anchor_row["level"]
                
                # Merge descendants using Markdown headers scaled by relative depth.
                #
                # Anchor node   (indent_level=0) → ## heading
                # First child   (indent_level=1) → ### heading  (e.g. sub-sections (1), (2)...)
                # Second child  (indent_level=2) → #### heading (e.g. clauses (a), (b)...)
                # Deeper levels capped at ###### (valid Markdown maximum).
                #
                # Using "\n\n" as the block separator means compressor.split_large_section(),
                # which already splits on "\n\n", will split at sub-section boundaries
                # rather than arbitrarily mid-sentence when a section exceeds MAX_SINGLE_SECTION_CHARS.
                lines = []
                for d in descendants:
                    all_descendant_ids.add(d["node_id"])
                    content = (d["text_content"] or "").strip()
                    title = (d["title"] or "").strip()

                    indent_level = max(0, d["level"] - anchor_level)
                    # Clamp to valid Markdown header depth (## through ######).
                    header_depth = min(2 + indent_level, 6)
                    header_prefix = "#" * header_depth

                    if title and content:
                        lines.append(f"{header_prefix} {title}\n{content}")
                    elif title:
                        lines.append(f"{header_prefix} {title}")
                    elif content:
                        # Content-only node (e.g. BODY_TEXT): emit without a header
                        # to avoid imposing spurious structure on plain prose blocks.
                        lines.append(content)

                rolled_text = "\n\n".join(lines)
                
                rolled_up_candidates.append(NodeCandidate(
                    node_id=anchor_id,
                    doc_id=anchor_row["doc_id"],
                    parent_id=anchor_row["parent_id"],
                    level=anchor_row["level"],
                    node_type=anchor_row["node_type"],
                    title=anchor_row["title"],
                    text_content=rolled_text,
                    breadcrumb=anchor_row["breadcrumb"],
                    file_name=anchor_row["file_name"]
                ))
                
            # Replace candidate nodes in context with rolled-up candidate sections
            ctx.candidate_nodes = rolled_up_candidates
            
            # 3. Fetch refers_to reference targets (1 hop) for all descendant IDs
            ref_rows = []
            if all_descendant_ids:
                ref_query = """
                SELECT r.target_node_id, n.node_id, n.doc_id, n.parent_id, n.level, n.node_type, n.title, n.text_content, n.breadcrumb, d.file_name
                FROM relationships r
                JOIN ast_nodes n ON r.target_node_id = n.node_id
                JOIN documents d ON n.doc_id = d.doc_id
                WHERE r.source_node_id = ANY($1::uuid[]) AND r.rel_type = 'REFERS_TO' AND r.is_resolved = TRUE AND d.is_active = TRUE;
                """
                ref_rows = await conn.fetch(ref_query, list(all_descendant_ids))
                
            # 4. Fetch glossary definitions defined by any of the descendant nodes
            glossary_rows = []
            if all_descendant_ids:
                glossary_query = """
                SELECT term, definition
                FROM glossary
                WHERE source_node_id = ANY($1::uuid[]);
                """
                glossary_rows = await conn.fetch(glossary_query, list(all_descendant_ids))
                
        # Format resolved target reference nodes
        candidate_ids_set = {n.node_id for n in rolled_up_candidates}
        expanded_nodes = []
        for row in ref_rows:
            node_id = row["node_id"]
            if node_id not in candidate_ids_set:
                expanded_nodes.append(NodeCandidate(
                    node_id=node_id,
                    doc_id=row["doc_id"],
                    parent_id=row["parent_id"],
                    level=row["level"],
                    node_type=row["node_type"],
                    title=row["title"],
                    text_content=row["text_content"] or "",
                    breadcrumb=row["breadcrumb"],
                    file_name=row["file_name"]
                ))
                
        ctx.expanded_nodes = expanded_nodes
        ctx.inlined_definitions = {row["term"]: row["definition"] for row in glossary_rows}
        
        logger.info("hop_expander_complete",
                    rolled_up_count=len(ctx.candidate_nodes),
                    expanded_count=len(ctx.expanded_nodes),
                    definitions_count=len(ctx.inlined_definitions))
                    
    except Exception as e:
        logger.error("hop_expander_failed", error=str(e))
        ctx.expanded_nodes = []
        ctx.inlined_definitions = {}
        
    ctx.stage_timings["hop_expander"] = time.monotonic() - start_time
    return ctx
