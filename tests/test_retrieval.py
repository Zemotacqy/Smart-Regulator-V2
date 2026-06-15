import pytest
import pytest_asyncio
from uuid import uuid4
from backend.database.connection import init_db_pool, close_db_pool
from backend.rag.retrieval.pipeline_context import QueryPipelineContext
from backend.rag.retrieval.query_expander import run_query_expander
from backend.rag.retrieval.hybrid_search import run_hybrid_search
from backend.rag.retrieval.hop_expander import run_hop_expander
from backend.rag.retrieval.temporal_filter import run_temporal_filter
from backend.rag.retrieval.reranker import run_reranker
from backend.rag.retrieval.compressor import run_compressor
from backend.rag.retrieval.generator import run_generator

@pytest_asyncio.fixture(scope="function")
async def db_pool():
    pool = await init_db_pool()
    yield pool
    await close_db_pool()

@pytest.mark.asyncio
async def test_full_retrieval_pipeline(db_pool):
    ctx = QueryPipelineContext(
        request_id=str(uuid4()),
        original_query="Can an IFSC Banking Unit accept deposits from Indian residents?"
    )
    
    # Run Stage A
    ctx = await run_query_expander(ctx)
    assert isinstance(ctx.expanded_queries, list)
    
    # Run Stage B
    ctx = await run_hybrid_search(ctx)
    assert isinstance(ctx.candidate_nodes, list)
    
    # Run Stage C
    ctx = await run_hop_expander(ctx)
    assert isinstance(ctx.expanded_nodes, list)
    assert isinstance(ctx.inlined_definitions, dict)
    
    # Run Temporal Filter
    ctx = await run_temporal_filter(ctx)
    assert isinstance(ctx.candidate_nodes, list)
    assert isinstance(ctx.expanded_nodes, list)
    
    # Run Stage D
    ctx = await run_reranker(ctx)
    assert isinstance(ctx.reranked_nodes, list)
    assert len(ctx.reranked_nodes) <= 5
    
    # Run Stage E
    ctx = await run_compressor(ctx)
    assert isinstance(ctx.compressed_context, str)
    
    # Run Stage F
    tokens = []
    async for token in run_generator(ctx):
        tokens.append(token)
        
    assert len(tokens) > 0
    assert isinstance(ctx.answer_text, str)
    assert len(ctx.answer_text) > 0
    assert len(ctx.source_citations) > 0
    
    # Verify timings
    assert "query_expander" in ctx.stage_timings
    assert "hybrid_search" in ctx.stage_timings
    assert "hop_expander" in ctx.stage_timings
    assert "temporal_filter" in ctx.stage_timings
    assert "reranker" in ctx.stage_timings
    assert "compressor" in ctx.stage_timings
    assert "generator" in ctx.stage_timings
