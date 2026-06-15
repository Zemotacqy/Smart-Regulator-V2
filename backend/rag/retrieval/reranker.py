import time
import asyncio
from typing import List, Optional
import structlog
from fastembed.rerank.cross_encoder import TextCrossEncoder

from backend.config import RERANK_MODEL
from backend.rag.retrieval.pipeline_context import QueryPipelineContext, NodeCandidate

logger = structlog.get_logger()

# Global model instance and lock for async thread-safe lazy loading
_model: Optional[TextCrossEncoder] = None
_model_lock = asyncio.Lock()

async def get_reranker_model() -> TextCrossEncoder:
    global _model
    async with _model_lock:
        if _model is None:
            model_name = RERANK_MODEL
            # Map to Xenova version if standard name is used in config
            if model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2":
                model_name = "Xenova/ms-marco-MiniLM-L-6-v2"
            logger.info("loading_reranker_model_start", model_name=model_name)
            
            # Instantiate model in a background thread to prevent blocking event loop during disk/CPU load
            _model = await asyncio.to_thread(TextCrossEncoder, model_name)
            logger.info("loading_reranker_model_complete")
        return _model

async def run_reranker(ctx: QueryPipelineContext) -> QueryPipelineContext:
    """
    Stage D: Cross-Encoder Reranking.
    Re-scores all candidate and expanded nodes against the original query.
    Selects the top 5 highest-scoring nodes and stores them in ctx.reranked_nodes.
    """
    start_time = time.monotonic()
    
    all_candidates = ctx.candidate_nodes + ctx.expanded_nodes
    if not all_candidates:
        logger.warning("reranker_skipped_no_candidates")
        ctx.reranked_nodes = []
        ctx.stage_timings["reranker"] = time.monotonic() - start_time
        return ctx
        
    # Deduplicate candidates by node_id
    seen_ids = set()
    unique_candidates = []
    for node in all_candidates:
        if node.node_id not in seen_ids:
            seen_ids.add(node.node_id)
            unique_candidates.append(node)
            
    logger.info("reranking_start", total_candidates=len(unique_candidates), original_count=len(all_candidates))
    
    try:
        model = await get_reranker_model()
        
        # Extract text content from candidates for scoring
        candidate_texts = [node.text_content or "" for node in unique_candidates]
        
        # Execute the CPU-bound reranking in a background thread to avoid blocking the asyncio loop
        scores = await asyncio.to_thread(
            lambda: list(model.rerank(ctx.original_query, candidate_texts))
        )
        
        # Pair nodes with scores safely using zip
        scored_nodes = []
        for node, score in zip(unique_candidates, scores):
            node.score = float(score)
            scored_nodes.append(node)
            
        # Sort by score descending and take top 5
        scored_nodes.sort(key=lambda x: x.score if x.score is not None else -999999.0, reverse=True)
        ctx.reranked_nodes = scored_nodes[:5]
        
        logger.info("reranking_complete", top_scores=[node.score for node in ctx.reranked_nodes])
        
    except Exception as e:
        logger.error("reranking_failed", error=str(e))
        # Fallback: take first 5 unique candidates and explicitly set score to None
        ctx.reranked_nodes = []
        for node in unique_candidates[:5]:
            node.score = None
            ctx.reranked_nodes.append(node)
        
    ctx.stage_timings["reranker"] = time.monotonic() - start_time
    return ctx
