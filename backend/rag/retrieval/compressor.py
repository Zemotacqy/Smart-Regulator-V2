import asyncio
import time
from typing import List
import structlog

from backend.config import EXTRACTOR_MODEL
from backend.rag.extraction.llm_client import call_llm_with_validation
from backend.rag.extraction.schemas import CompressorOutput
from backend.rag.retrieval.pipeline_context import QueryPipelineContext

logger = structlog.get_logger()

# Process LLM compression sequentially (limit to 2) to avoid Ollama queuing queue-induced client-side timeouts
_semaphore = asyncio.Semaphore(2)

async def compress_node_text(query: str, text_content: str) -> List[str]:
    """
    Calls the extractor model to extract relevant sentences from a single node's text content.
    Includes a timeout guard to prevent hung requests.
    """
    if not text_content or not isinstance(text_content, str) or not text_content.strip():
        return []
        
    messages = [
        {
            "role": "user",
            "content": (
                f"Query: {query}\n\n"
                f"Regulatory Text:\n{text_content}\n\n"
                f"Extract all sentences from the Regulatory Text that are directly relevant to answering the Query. "
                f"Do not summarize or paraphrase. Output strictly valid JSON."
            )
        }
    ]
    
    async with _semaphore:
        try:
            # Wrap the LLM call with a 25-second timeout to prevent stalling the pipeline under CPU execution
            result = await asyncio.wait_for(
                call_llm_with_validation(
                    model=EXTRACTOR_MODEL,
                    messages=messages,
                    response_schema=CompressorOutput,
                    temperature=0.0
                ),
                timeout=10.0
            )
            return [str(s).strip() for s in result.relevant_sentences if s]
        except asyncio.TimeoutError:
            logger.warning("node_compression_timeout", timeout_seconds=10.0)
            # Fallback to the entire text (no compression) on timeout
            return [text_content]
        except Exception as e:
            logger.warning("node_compression_failed", error=str(e))
            # On general failure, fallback to the entire text (no compression)
            return [text_content]

async def run_compressor(ctx: QueryPipelineContext) -> QueryPipelineContext:
    """
    Stage E: Context Compression.
    
    TEMPORARY NOTICE: This stage has been temporarily bypassed to save latency.
    To re-enable context compression:
    1. Uncomment the try block with `asyncio.gather(*tasks)` and the lines below it.
    2. Comment out the direct assignment `ctx.compressed_context = get_fallback_context()` and logging line.
    """
    start_time = time.monotonic()
    
    if not ctx.reranked_nodes:
        logger.warning("compressor_skipped_no_nodes")
        ctx.compressed_context = ""
        ctx.stage_timings["compressor"] = time.monotonic() - start_time
        return ctx
        
    logger.info("compressor_start_bypassed_temporarily", node_count=len(ctx.reranked_nodes))
    
    # Helper for fallback text construction (uses full node content)
    def get_fallback_context() -> str:
        raw_blocks = [
            f"Source: {node.breadcrumb} [ID: {node.node_id}]\n{node.text_content}"
            for node in ctx.reranked_nodes
            if node.text_content
        ]
        return "\n\n".join(raw_blocks)
        
    # Directly use full uncompressed context (bypassing the SLM extraction)
    ctx.compressed_context = get_fallback_context()
    logger.info("compressor_bypassed_using_fallback_context", length=len(ctx.compressed_context))

    # --- TO RE-ENABLE COMPRESSION, UNCOMMENT THE BLOCK BELOW AND COMMENT OUT THE TWO LINES ABOVE ---
    # try:
    #     # Run compression for all reranked nodes in parallel with semaphore protection
    #     tasks = [
    #         compress_node_text(ctx.original_query, node.text_content)
    #         for node in ctx.reranked_nodes
    #     ]
    #     compressed_results = await asyncio.gather(*tasks)
    #     
    #     # Combine the results into a structured format
    #     compressed_blocks = []
    #     for idx, sentences in enumerate(compressed_results):
    #         node = ctx.reranked_nodes[idx]
    #         if sentences:
    #             joined_text = " ".join(sentences)
    #             # Form a block with citation prefix and exact node ID
    #             compressed_blocks.append(f"Source: {node.breadcrumb} [ID: {node.node_id}]\n{joined_text}")
    #             
    #     ctx.compressed_context = "\n\n".join(compressed_blocks)
    #     
    #     # Fallback if the compression yielded zero content
    #     if not ctx.compressed_context.strip():
    #         logger.warning("compressor_yielded_empty_context_using_fallback")
    #         ctx.compressed_context = get_fallback_context()
    #         
    #     logger.info("compressor_complete", 
    #                 compressed_length=len(ctx.compressed_context),
    #                 original_length=sum(len(n.text_content or "") for n in ctx.reranked_nodes))
    #                 
    # except Exception as e:
    #     logger.error("compressor_failed", error=str(e))
    #     ctx.compressed_context = get_fallback_context()
    # ------------------------------------------------------------------------------------------------
        
    ctx.stage_timings["compressor"] = time.monotonic() - start_time
    return ctx
