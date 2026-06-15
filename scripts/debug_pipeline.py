import argparse
import asyncio
import os
import sys
import time
from uuid import UUID, uuid4
import structlog

# Add project root to python path so backend can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.database.connection import init_db_pool, close_db_pool
from backend.rag.retrieval.pipeline_context import QueryPipelineContext
from backend.rag.retrieval.query_expander import run_query_expander
from backend.rag.retrieval.hybrid_search import run_hybrid_search
from backend.rag.retrieval.hop_expander import run_hop_expander
from backend.rag.retrieval.temporal_filter import run_temporal_filter
from backend.rag.retrieval.reranker import run_reranker
from backend.rag.retrieval.compressor import run_compressor
from backend.rag.retrieval.generator import run_generator

# Setup basic structlog console output for debugging
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ]
)
logger = structlog.get_logger()

async def debug_pipeline(query: str, doc_filter_id: Optional[str] = None):
    print("=" * 80)
    print(f"DEBUGGING RAG RETRIEVAL PIPELINE")
    print(f"Original Query: '{query}'")
    if doc_filter_id:
        print(f"Document Filter: {doc_filter_id}")
    print("=" * 80)
    print()

    # Initialize pool
    await init_db_pool()
    
    doc_filter = [UUID(doc_filter_id)] if doc_filter_id else None
    
    ctx = QueryPipelineContext(
        request_id=str(uuid4()),
        original_query=query,
        doc_filter=doc_filter
    )
    
    total_start = time.monotonic()
    
    try:
        # --- Stage A: Query Expansion ---
        print("▶ Running Stage A: Query Expander...")
        ctx = await run_query_expander(ctx)
        print(f"  Duration: {ctx.stage_timings.get('query_expander', 0.0):.4f}s")
        print(f"  Expansions Generated:")
        for idx, exp in enumerate(ctx.expanded_queries, 1):
            print(f"    {idx}. '{exp}'")
        print("-" * 80)
        
        # --- Stage B: Hybrid Search ---
        print("▶ Running Stage B: Hybrid Search...")
        ctx = await run_hybrid_search(ctx)
        print(f"  Duration: {ctx.stage_timings.get('hybrid_search', 0.0):.4f}s")
        print(f"  Candidate Nodes Retrieved: {len(ctx.candidate_nodes)}")
        for idx, node in enumerate(ctx.candidate_nodes[:5], 1):
            print(f"    [{idx}] {node.breadcrumb} | Score: {node.score}")
            print(f"        Snippet: {node.text_content[:100]}...")
        if len(ctx.candidate_nodes) > 5:
            print(f"    ... and {len(ctx.candidate_nodes) - 5} more candidate nodes.")
        print("-" * 80)
        
        # --- Stage C: Hop Expander ---
        print("▶ Running Stage C: Hop Expander...")
        ctx = await run_hop_expander(ctx)
        print(f"  Duration: {ctx.stage_timings.get('hop_expander', 0.0):.4f}s")
        print(f"  Expanded Nodes (AST parent chain / definitions): {len(ctx.expanded_nodes)}")
        print(f"  Glossary Definitions Inlined: {len(ctx.inlined_definitions)}")
        for term, definition in ctx.inlined_definitions.items():
            print(f"    - '{term}': {definition[:80]}...")
        print("-" * 80)
        
        # --- Temporal Filter ---
        print("▶ Running Temporal Filter (Amendments/Omissions)...")
        ctx = await run_temporal_filter(ctx)
        print(f"  Duration: {ctx.stage_timings.get('temporal_filter', 0.0):.4f}s")
        print(f"  Post-Filter Candidate Nodes: {len(ctx.candidate_nodes)}")
        print(f"  Post-Filter Expanded Nodes: {len(ctx.expanded_nodes)}")
        print("-" * 80)
        
        # --- Stage D: Reranker ---
        print("▶ Running Stage D: Cross-Encoder Reranker...")
        ctx = await run_reranker(ctx)
        print(f"  Duration: {ctx.stage_timings.get('reranker', 0.0):.4f}s")
        print(f"  Top Reranked Nodes (Max 5):")
        for idx, node in enumerate(ctx.reranked_nodes, 1):
            print(f"    [{idx}] {node.breadcrumb} | Score: {node.score}")
            print(f"        Snippet: {node.text_content[:120]}...")
        print("-" * 80)
        
        # --- Stage E: Compressor ---
        print("▶ Running Stage E: Compressor...")
        ctx = await run_compressor(ctx)
        print(f"  Duration: {ctx.stage_timings.get('compressor', 0.0):.4f}s")
        print(f"  Compressed Context (Character length: {len(ctx.compressed_context)}):")
        print(f"  {ctx.compressed_context[:300]}...")
        print("-" * 80)
        
        # --- Stage F: Generator ---
        print("▶ Running Stage F: Generator...")
        print("  Streaming Answer: ", end="", flush=True)
        async for token in run_generator(ctx):
            print(token, end="", flush=True)
        print("\n")
        print(f"  Duration: {ctx.stage_timings.get('generator', 0.0):.4f}s")
        print(f"  Source Citations Count: {len(ctx.source_citations)}")
        for idx, citation in enumerate(ctx.source_citations, 1):
            print(f"    [{idx}] {citation.breadcrumb} (File: {citation.file_name})")
        print("=" * 80)
        
        total_duration = time.monotonic() - total_start
        print(f"★ Total Pipeline Execution Time: {total_duration:.4f}s")
        print("=" * 80)
        
    except Exception as e:
        logger.exception("pipeline_debug_failed", error=str(e))
    finally:
        await close_db_pool()

def main():
    parser = argparse.ArgumentParser(description="Debug and trace the RAG retrieval pipeline stage-by-stage.")
    parser.add_argument("--query", required=True, help="The query text to run through the pipeline.")
    parser.add_argument("--doc-id", required=False, help="Optional document ID (UUID) to filter by.")
    args = parser.parse_args()
    
    from typing import Optional
    asyncio.run(debug_pipeline(args.query, args.doc_id))

if __name__ == "__main__":
    main()
