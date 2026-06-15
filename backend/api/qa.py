import json
import asyncio
from uuid import UUID, uuid4
from typing import List, Optional
from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import StreamingResponse
import structlog

from backend.rag.retrieval.pipeline_context import QueryPipelineContext
from backend.rag.retrieval.query_expander import run_query_expander
from backend.rag.retrieval.hybrid_search import run_hybrid_search
from backend.rag.retrieval.hop_expander import run_hop_expander
from backend.rag.retrieval.temporal_filter import run_temporal_filter
from backend.rag.retrieval.reranker import run_reranker
from backend.rag.retrieval.compressor import run_compressor
from backend.rag.retrieval.generator import run_generator

logger = structlog.get_logger()
router = APIRouter()

@router.get("/qa")
async def qa_endpoint(
    query: str = Query(..., description="The user query to run through the RAG pipeline"),
    doc_filter: Optional[List[str]] = Query(None, description="Optional document IDs to filter by")
):
    """
    GET /api/qa
    Runs the 7-stage retrieval-generation pipeline and streams the response as Server-Sent Events (SSE).
    """
    logger.info("qa_endpoint_hit", query=query, doc_filter=doc_filter)
    
    # Validate and convert doc_filter to List[UUID]
    uuid_filters: Optional[List[UUID]] = None
    if doc_filter:
        try:
            uuid_filters = [UUID(df) for df in doc_filter if df]
        except ValueError as e:
            logger.error("invalid_doc_filter_format", doc_filter=doc_filter, error=str(e))
            raise HTTPException(status_code=400, detail="Invalid UUID format in doc_filter")

    async def sse_generator():
        ctx = QueryPipelineContext(
            request_id=str(uuid4()),
            original_query=query,
            doc_filter=uuid_filters
        )
        
        try:
            # Stage A: Query Expander
            ctx = await run_query_expander(ctx)
            
            # Stage B: Hybrid Search
            ctx = await run_hybrid_search(ctx)
            
            # Stage C: Hop Expander
            ctx = await run_hop_expander(ctx)
            
            # Stage D: Temporal Filter
            ctx = await run_temporal_filter(ctx)
            
            # Stage E: Reranker
            ctx = await run_reranker(ctx)
            
            # Stage F: Compressor
            ctx = await run_compressor(ctx)
            
            # Stage G: Generator (streams tokens)
            async for token in run_generator(ctx):
                yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"
                
            # Send citations
            citations_data = [
                {
                    "node_id": str(c.node_id),
                    "doc_id": str(c.doc_id),
                    "file_name": c.file_name,
                    "breadcrumb": c.breadcrumb,
                    "title": c.title,
                    "text_content": c.text_content
                }
                for c in ctx.source_citations
            ]
            yield f"event: citations\ndata: {json.dumps(citations_data)}\n\n"
            
            # Send timings
            yield f"event: timings\ndata: {json.dumps(ctx.stage_timings)}\n\n"
            
            # Signal done
            yield "event: done\ndata: {}\n\n"
            
        except Exception as e:
            logger.error("qa_pipeline_failed", query=query, error=str(e))
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
            
    return StreamingResponse(sse_generator(), media_type="text/event-stream")
