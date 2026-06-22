"""
Stage D: local cross-encoder Reranker

Rescores candidate nodes from the hop expander against the user query
using a locally-hosted cross-encoder model (mixedbread-ai/mxbai-rerank-large-v1).

Design decisions:
  - The mxbai cross-encoder is loaded on CPU lazily (with startup preheating support).
  - Handles candidate text length overflow by partition splitting (Map phase) and
    Weighted Peak-Density Aggregation (Reduce phase).
  - Individual batch failure is isolated to prevent pipeline crashes.
"""

import time
import asyncio
import threading
from typing import List
import structlog
import torch

from backend.rag.retrieval.pipeline_context import QueryPipelineContext, NodeCandidate

logger = structlog.get_logger()

_mxbai_tokenizer = None
_mxbai_model = None
_mxbai_lock = threading.Lock()

def _get_mxbai_model():
    global _mxbai_tokenizer, _mxbai_model
    if _mxbai_model is None:
        with _mxbai_lock:
            if _mxbai_model is None:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                model_name = "mixedbread-ai/mxbai-rerank-large-v1"
                
                device = "cpu"
                if torch.cuda.is_available():
                    device = "cuda"
                elif torch.backends.mps.is_available():
                    device = "mps"
                    
                logger.info("loading_mxbai_reranker_model_start", model=model_name, device=device)
                _mxbai_tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                _mxbai_model = model.to(device)
                _mxbai_model.eval()
                logger.info("loading_mxbai_reranker_model_complete", device=device)
    return _mxbai_tokenizer, _mxbai_model


async def get_reranker_model() -> None:
    """
    Warms up the local cross-encoder model by loading it and its tokenizer
    into memory at startup to avoid first-query latency.
    """
    logger.info("warming_up_mxbai_reranker_model_start")
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _get_mxbai_model)
    logger.info("warming_up_mxbai_reranker_model_complete")


def _split_node_into_chunks(node: NodeCandidate, tokenizer, max_tokens: int = 380) -> List[str]:
    text = node.text_content or ""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return [text]
        
    chunks = []
    # Split by token boundaries, with a 40-token overlap
    step = max_tokens - 40
    for i in range(0, len(token_ids), step):
        chunk_ids = token_ids[i:i + max_tokens]
        chunks.append(tokenizer.decode(chunk_ids))
    return chunks


def _build_map_input(node: NodeCandidate, chunk_text: str) -> str:
    doc_title = node.title or node.file_name or "IFSCA Regulation"
    return f"Document: {doc_title} > {node.breadcrumb}\nContent: {chunk_text}"


# Maximum number of candidates sent to the reranker (controls total latency).
MAX_CANDIDATES = 10

# Top-N sections forwarded to the generator after reranking.
TOP_K = 4


async def run_reranker(ctx: QueryPipelineContext) -> QueryPipelineContext:
    """
    Stage D: Cross-Encoder Reranker.
    Scores candidates via mixedbread CPU cross-encoder using MapReduce.
    """
    start_time = time.monotonic()

    # --- 1. Collect and deduplicate candidates ---
    all_candidates = ctx.candidate_nodes + ctx.expanded_nodes
    if not all_candidates:
        logger.warning("reranker_skipped_no_candidates")
        ctx.reranked_nodes = []
        ctx.stage_timings["reranker"] = time.monotonic() - start_time
        return ctx

    seen_ids = set()
    unique_candidates: List[NodeCandidate] = []
    for node in all_candidates:
        if node.node_id not in seen_ids:
            seen_ids.add(node.node_id)
            unique_candidates.append(node)

    logger.info("reranking_start", total_candidates=len(unique_candidates))

    # --- 2. Score via dedicated Mixedbread Cross-Encoder ---
    candidates_to_score = unique_candidates[:MAX_CANDIDATES]
    if len(unique_candidates) > MAX_CANDIDATES:
        logger.warning(
            "reranker_candidates_truncated",
            total=len(unique_candidates),
            scored=MAX_CANDIDATES,
        )
    
    try:
        # Load CPU model and tokenizer first for token-based chunking
        tokenizer, model = _get_mxbai_model()
        
        # Map: Partition long nodes into 380-token chunks with metadata prepended
        all_inputs = []
        index_mapping = []  # Maps flat sub-chunk index -> (candidate_list_index)
        
        for c_idx, node in enumerate(candidates_to_score):
            sub_chunks = _split_node_into_chunks(node, tokenizer)
            for chunk_text in sub_chunks:
                all_inputs.append(_build_map_input(node, chunk_text))
                index_mapping.append(c_idx)
        
        # GPU/MPS/CPU forward pass (vectorized batch)
        pairs = [[ctx.original_query, inp] for inp in all_inputs]
        
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)
        
        # Move inputs to the same device as the model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.squeeze(-1).cpu().tolist()
            if not isinstance(logits, list):
                logits = [logits]
        
        # Group scores by candidate ID
        candidate_chunk_scores = {c_idx: [] for c_idx in range(len(candidates_to_score))}
        for idx, logit in enumerate(logits):
            c_idx = index_mapping[idx]
            candidate_chunk_scores[c_idx].append(logit)
            
        # Reduce: Weighted Peak-Density Aggregator
        scored_nodes: List[NodeCandidate] = []
        for c_idx, node in enumerate(candidates_to_score):
            scores = candidate_chunk_scores[c_idx]
            if not scores:
                node.score = None
            else:
                max_score = max(scores)
                top_k_scores = sorted(scores, reverse=True)[:2]
                avg_top_k = sum(top_k_scores) / len(top_k_scores)
                node.score = 0.7 * max_score + 0.3 * avg_top_k
            scored_nodes.append(node)
            
    except Exception as exc:
        logger.error("reranker_mxbai_failed", error=str(exc))
        scored_nodes = candidates_to_score
        for node in scored_nodes:
            node.score = None

    # --- 3. Sort: scored nodes first (descending), unscored appended last ---
    scored_with_value = [n for n in scored_nodes if n.score is not None]
    scored_without_value = [n for n in scored_nodes if n.score is None]

    scored_with_value.sort(key=lambda n: n.score, reverse=True)
    final_ranked = scored_with_value + scored_without_value

    logger.info(
        "reranking_scores_parsed_successfully",
        scores=[(n.title or n.breadcrumb[:40], n.score) for n in final_ranked],
    )

    # --- 4. Write top-K to context ---
    ctx.reranked_nodes = final_ranked[:TOP_K]
    logger.info("reranking_complete", top_count=len(ctx.reranked_nodes))
    ctx.stage_timings["reranker"] = time.monotonic() - start_time
    return ctx
