import asyncio
import structlog
from backend.database.connection import init_db_pool, get_db_connection, close_db_pool
from backend.rag.extraction.llm_client import generate_embedding
from backend.rag.retrieval.query_expander import run_query_expander
from backend.rag.retrieval.pipeline_context import QueryPipelineContext
import logging

# Set up simple logging
logging.basicConfig(level=logging.INFO)
structlog.configure(
    processors=[
        structlog.processors.JSONRenderer()
    ]
)

async def run_hybrid():
    # 1. Initialize DB pool
    pool = await init_db_pool()
    
    # 2. Context setup
    query = "Can an IFSC Banking Unit accept deposits from Indian residents?"
    ctx = QueryPipelineContext(
        request_id="test-request-id",
        original_query=query
    )
    
    # 3. Run expander
    print("--- Running Expander ---")
    ctx = await run_query_expander(ctx)
    print("Expansions:", ctx.expanded_queries)
    
    # We combine original query and expansions
    all_queries = [ctx.original_query] + ctx.expanded_queries
    print(f"Total queries to search: {len(all_queries)}")
    
    # 4. Generate embeddings in parallel
    print("--- Generating Embeddings ---")
    embeddings = await asyncio.gather(*[generate_embedding(q) for q in all_queries])
    
    # 5. Build dynamic RRF query
    print("--- Constructing SQL RRF ---")
    cte_clauses = []
    union_clauses = []
    params = []
    
    # First, the dense CTEs
    for idx, emb in enumerate(embeddings):
        cte_name = f"dense{idx}"
        param_idx = len(params) + 1
        params.append(str(emb))
        cte_clauses.append(f"""
{cte_name} AS (
  SELECT node_id, ROW_NUMBER() OVER (ORDER BY embedding <=> ${param_idx}::vector) as rank
  FROM ast_nodes n JOIN documents d ON n.doc_id = d.doc_id
  WHERE d.is_active = TRUE AND n.embedding <=> ${param_idx}::vector IS NOT NULL
  LIMIT 20
)""")
        union_clauses.append(f"SELECT node_id, rank FROM {cte_name}")
        
    # Now, the sparse CTE
    sparse_idx = len(params) + 1
    params.append(ctx.original_query)
    cte_clauses.append(f"""
sparse AS (
  SELECT node_id, ROW_NUMBER() OVER (ORDER BY ts_rank(ts_vector, plainto_tsquery('english', ${sparse_idx})) DESC) as rank
  FROM ast_nodes n JOIN documents d ON n.doc_id = d.doc_id
  WHERE d.is_active = TRUE AND ts_vector @@ plainto_tsquery('english', ${sparse_idx})
  LIMIT 20
)""")
    union_clauses.append("SELECT node_id, rank FROM sparse")
    
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
    
    print("Executing final SQL...")
    async with get_db_connection() as conn:
        rows = await conn.fetch(final_sql, *params)
        print(f"Retrieved {len(rows)} candidates")
        for i, row in enumerate(rows[:5]):
            print(f"Candidate {i+1}: RRF Score={row['rrf_score']:.4f} | Breadcrumb={row['breadcrumb']} | File={row['file_name']}")
            print(f"Content: {row['text_content'][:100]}...\n")
            
    await close_db_pool()

if __name__ == "__main__":
    asyncio.run(run_hybrid())
