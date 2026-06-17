import asyncio
import time
from typing import List
import structlog

from backend.config import EXTRACTOR_MODEL
from backend.rag.extraction.llm_client import call_llm_with_validation
from backend.rag.extraction.schemas import BatchedCompressorOutput
from backend.rag.retrieval.pipeline_context import QueryPipelineContext

logger = structlog.get_logger()

async def run_compressor(ctx: QueryPipelineContext) -> QueryPipelineContext:
    """
    Stage E: Context Compression.
    
    Optimizations implemented:
    1. Targeted bypass: Only nodes longer than 1000 characters are sent for compression.
       Shorter nodes are kept as-is, saving substantial latency.
    2. Batched execution: Collapses all compression requests for long nodes into a single
       LLM batch call, avoiding Ollama queue congestion and sequential call overhead.
    3. Bounded tail latency: The batched LLM call is guarded by an 8-second stage-level timeout.
    4. Robust fallback: On timeout or extraction error, we fall back to the original full text
       so we never lose context.
    """
    start_time = time.monotonic()
    
    if not ctx.reranked_nodes:
        logger.warning("compressor_skipped_no_nodes")
        ctx.compressed_context = ""
        ctx.stage_timings["compressor"] = time.monotonic() - start_time
        return ctx
        
    # Helper for fallback text construction (uses full node content)
    def get_fallback_context() -> str:
        raw_blocks = [
            f"Source: {node.breadcrumb} <!-- ID: {node.node_id} -->\n{node.text_content}"
            for node in ctx.reranked_nodes
            if node.text_content
        ]
        return "\n\n".join(raw_blocks)
        
    # Capping character limits to prevent VRAM overflow in downstream generator
    MAX_NODE_CHARS = 1600   # Max characters per node
    MAX_TOTAL_CHARS = 8000  # Max total context characters (~2000 tokens)

    # ──────────────────────────────────────────────────────────────────────────
    # [DEMO BYPASS WORKFLOW]
    # Direct, latency-free context construction with character bounds
    # ──────────────────────────────────────────────────────────────────────────
    logger.info("compressor_demo_bypass_active", node_count=len(ctx.reranked_nodes))
    
    compressed_blocks = []
    total_len = 0
    
    for node in ctx.reranked_nodes:
        if not node.text_content:
            continue
            
        content = node.text_content.strip()
        # Truncate individual node if it exceeds max node chars
        if len(content) > MAX_NODE_CHARS:
            content = content[:MAX_NODE_CHARS] + " ... [TRUNCATED FOR CONTEXT LIMIT]"
            
        block = f"Source: {node.breadcrumb} <!-- ID: {node.node_id} -->\n{content}"
        
        # Check if adding this block exceeds total limit
        if total_len + len(block) > MAX_TOTAL_CHARS:
            # If we already have some context, stop adding to avoid overflow
            if compressed_blocks:
                logger.warning("compressor_total_limit_reached_truncating", current_len=total_len)
                break
            else:
                # If even the first block is huge, truncate it to fit
                block = block[:MAX_TOTAL_CHARS] + " ... [TRUNCATED]"
                
        compressed_blocks.append(block)
        total_len += len(block)

    ctx.compressed_context = "\n\n".join(compressed_blocks)
    
    # ──────────────────────────────────────────────────────────────────────────
    # [COMMENTED OUT: SLM-BASED COMPRESSION]
    # To restore, uncomment the block below and comment out the demo bypass block above.
    # ──────────────────────────────────────────────────────────────────────────
    """
    # Character length threshold below which compression is bypassed
    COMPRESSION_CHAR_THRESHOLD = 1000
    
    # Identify which nodes meet the threshold and require compression
    nodes_to_compress = []
    to_compress_map = {}  # maps index in nodes_to_compress to index in ctx.reranked_nodes
    
    for idx, node in enumerate(ctx.reranked_nodes):
        if node.text_content and len(node.text_content) >= COMPRESSION_CHAR_THRESHOLD:
            to_compress_map[len(nodes_to_compress)] = idx
            nodes_to_compress.append(node)
            
    # If no nodes meet the size threshold, bypass SLM compression entirely
    if not nodes_to_compress:
        logger.info("compressor_all_nodes_below_threshold_bypassed", node_count=len(ctx.reranked_nodes))
        ctx.compressed_context = get_fallback_context()
        ctx.stage_timings["compressor"] = time.monotonic() - start_time
        return ctx

    logger.info("compressor_started_batched", 
                total_nodes=len(ctx.reranked_nodes), 
                compress_count=len(nodes_to_compress))
                
    # Format the input nodes to compress as a numbered list
    user_content_lines = []
    for batch_idx, node in enumerate(nodes_to_compress):
        user_content_lines.append(f"Clause Index {batch_idx}:\n{node.text_content}\n---")
        
    user_content = "\n".join(user_content_lines)
    
    system_prompt = (
        "You are a regulatory text compressor for the International Financial Services Centres Authority (IFSCA).\n"
        "Your task is to analyze a batch of regulatory clauses and extract only the sentences from each clause that are directly relevant to answering the user query.\n\n"
        "Input format:\n"
        "1. User Query: The user's question.\n"
        "2. List of regulatory clauses, each numbered by its index.\n\n"
        "Output format:\n"
        "You must return a JSON object containing a list under the key \\"nodes\\". Each item in the list must correspond to one of the input clauses and specify:\n"
        "- node_index: The 0-based index of the clause in the input list.\n"
        "- relevant_sentences: A list of exact sentences extracted from the clause text that help answer the query. Do not summarize, paraphrase, or change any words. If a clause contains no relevant information, return an empty list for that clause.\n\n"
        "Output strictly valid JSON matching this schema. No markdown formatting or extra text."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"User Query: {ctx.original_query}\\n\\nList of clauses to process:\\n\\n{user_content}\\n\\nExtract relevant sentences:"}
    ]
    
    compressed_sentences_by_orig_idx = {}
    compression_successful = False
    
    try:
        # Single batched LLM call with an 8-second timeout
        result = await asyncio.wait_for(
            call_llm_with_validation(
                model=EXTRACTOR_MODEL,
                messages=messages,
                response_schema=BatchedCompressorOutput,
                temperature=0.0,
                num_ctx=4096
            ),
            timeout=8.0
        )
        
        for item in result.nodes:
            batch_idx = item.node_index
            if batch_idx in to_compress_map:
                orig_idx = to_compress_map[batch_idx]
                compressed_sentences_by_orig_idx[orig_idx] = [s.strip() for s in item.relevant_sentences if s]
                
        compression_successful = True
        logger.info("compressor_batch_succeeded", returned_count=len(result.nodes))
        
    except asyncio.TimeoutError:
        logger.warning("compressor_batch_timeout", timeout_seconds=8.0)
    except Exception as e:
        logger.warning("compressor_batch_failed", error=str(e))
        
    # Assemble the final context
    compressed_blocks = []
    for idx, node in enumerate(ctx.reranked_nodes):
        if not node.text_content:
            continue
            
        if idx in to_compress_map.values():
            # Long node that we attempted to compress
            sentences = compressed_sentences_by_orig_idx.get(idx)
            if compression_successful and sentences is not None:
                if sentences:
                    joined_text = " ".join(sentences)
                    compressed_blocks.append(f"Source: {node.breadcrumb} <!-- ID: {node.node_id} -->\\n{joined_text}")
                else:
                    # Compressed to empty (deemed irrelevant by model)
                    logger.debug("node_compressed_to_empty", node_id=node.node_id, breadcrumb=node.breadcrumb)
            else:
                # Compression failed or timed out for this node specifically. Use full text.
                compressed_blocks.append(f"Source: {node.breadcrumb} <!-- ID: {node.node_id} -->\\n{node.text_content}")
        else:
            # Short node, bypass compression and use full text
            compressed_blocks.append(f"Source: {node.breadcrumb} <!-- ID: {node.node_id} -->\\n{node.text_content}")
            
    ctx.compressed_context = "\\n\\n".join(compressed_blocks)
    if not ctx.compressed_context.strip():
        logger.warning("compressor_context_empty_fallback")
        ctx.compressed_context = get_fallback_context()
    """

    logger.info("compressor_complete_bypass", 
                compressed_length=len(ctx.compressed_context))
                
    ctx.stage_timings["compressor"] = time.monotonic() - start_time
    return ctx
