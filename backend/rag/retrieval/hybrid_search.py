import asyncio
import time
from typing import List, Optional
from uuid import UUID
import structlog

from backend.database.connection import get_db_connection
from backend.rag.extraction.llm_client import generate_embedding
from backend.rag.retrieval.pipeline_context import QueryPipelineContext, NodeCandidate

logger = structlog.get_logger()

async def run_hybrid_search(ctx: QueryPipelineContext) -> QueryPipelineContext:
    """
    Stage B: Hybrid Search (HNSW dense + tsvector sparse) with Reciprocal Rank Fusion (RRF) in SQL.
    Combines dense results for the original query and expansions, plus full-text search results for the original query.
    """
    start_time = time.monotonic()
    
    # 1. Combine original query and expansions
    all_queries = [ctx.original_query] + ctx.expanded_queries
    logger.info("hybrid_search_start", total_queries=len(all_queries), has_filter=ctx.doc_filter is not None)
    
    try:
        # 2. Generate embeddings in parallel
        embeddings = await asyncio.gather(*[generate_embedding(q) for q in all_queries])
        
        # 3. Construct dynamic SQL query
        cte_clauses = []
        union_clauses = []
        params = []
        
        # Add doc filter parameter index if present
        filter_param_idx = None
        if ctx.doc_filter:
            filter_param_idx = len(all_queries) + 2  # all_queries embeddings (1 to N) + original query (N+1) + filter (N+2)
            
        doc_filter_condition = f"AND n.doc_id = ANY(${filter_param_idx}::uuid[])" if filter_param_idx else ""
        
        # Dense CTEs
        for idx, emb in enumerate(embeddings):
            cte_name = f"dense{idx}"
            param_idx = len(params) + 1
            params.append(str(emb))  # Cast to pgvector string format
            cte_clauses.append(f"""
{cte_name} AS (
  SELECT n.node_id, ROW_NUMBER() OVER (ORDER BY n.embedding <=> ${param_idx}::vector) as rank
  FROM ast_nodes n JOIN documents d ON n.doc_id = d.doc_id
  WHERE d.is_active = TRUE AND n.embedding <=> ${param_idx}::vector IS NOT NULL {doc_filter_condition}
  LIMIT 20
)""")
            union_clauses.append(f"SELECT node_id, rank FROM {cte_name}")
            
        # Sparse CTE (uses the original query)
        sparse_param_idx = len(params) + 1
        params.append(ctx.original_query)
        cte_clauses.append(f"""
sparse AS (
  SELECT n.node_id, ROW_NUMBER() OVER (ORDER BY ts_rank(n.ts_vector, plainto_tsquery('english', ${sparse_param_idx})) DESC) as rank
  FROM ast_nodes n JOIN documents d ON n.doc_id = d.doc_id
  WHERE d.is_active = TRUE AND n.ts_vector @@ plainto_tsquery('english', ${sparse_param_idx}) {doc_filter_condition}
  LIMIT 20
)""")
        union_clauses.append("SELECT node_id, rank FROM sparse")
        
        if filter_param_idx:
            # Add doc filter list to params
            params.append(ctx.doc_filter)
            
        cte_sql = ",\n".join(cte_clauses)
        union_sql = "\nUNION ALL\n".join(union_clauses)
        
        final_sql = f"""
WITH {cte_sql},
all_ranks AS (
  {union_sql}
),
rrf_scores AS (
  SELECT node_id, SUM(1.0 / (60.0 + rank)) as rrf_score
  FROM all_ranks
  GROUP BY node_id
)
SELECT n.node_id, n.doc_id, n.parent_id, n.level, n.node_type, n.title, n.text_content, n.breadcrumb, d.file_name, r.rrf_score
FROM rrf_scores r
JOIN ast_nodes n ON r.node_id = n.node_id
JOIN documents d ON n.doc_id = d.doc_id
ORDER BY r.rrf_score DESC
LIMIT 20
"""
        
        # 4. Execute query
        async with get_db_connection() as conn:
            rows = await conn.fetch(final_sql, *params)
            
        # 5. Populate candidate_nodes in context
        ctx.candidate_nodes = [
            NodeCandidate(
                node_id=row["node_id"],
                doc_id=row["doc_id"],
                parent_id=row["parent_id"],
                level=row["level"],
                node_type=row["node_type"],
                title=row["title"],
                text_content=row["text_content"] or "",
                breadcrumb=row["breadcrumb"],
                score=float(row["rrf_score"]) if row["rrf_score"] is not None else None,
                file_name=row["file_name"]
            )
            for row in rows
        ]
        logger.info("hybrid_search_complete", total_candidates=len(ctx.candidate_nodes))
        
    except Exception as e:
        logger.error("hybrid_search_failed", error=str(e))
        ctx.candidate_nodes = []
        
    ctx.stage_timings["hybrid_search"] = time.monotonic() - start_time
    return ctx
