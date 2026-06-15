import asyncio
import time
from typing import List
import structlog

from backend.config import EXTRACTOR_MODEL
from backend.rag.extraction.llm_client import call_llm_with_validation
from backend.rag.extraction.schemas import CompressorOutput
from backend.rag.retrieval.pipeline_context import QueryPipelineContext

logger = structlog.get_logger()

# Limit concurrent LLM calls to 3 to avoid overloading Ollama
_semaphore = asyncio.Semaphore(3)

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
            # Wrap the LLM call with a 15-second timeout to prevent stalling the pipeline
            result = await asyncio.wait_for(
                call_llm_with_validation(
                    model=EXTRACTOR_MODEL,
                    messages=messages,
                    response_schema=CompressorOutput,
                    temperature=0.0
                ),
                timeout=15.0
            )
            return [str(s).strip() for s in result.relevant_sentences if s]
        except asyncio.TimeoutError:
            logger.warning("node_compression_timeout", timeout_seconds=15.0)
            # Fallback to the entire text (no compression) on timeout
            return [text_content]
        except Exception as e:
            logger.warning("node_compression_failed", error=str(e))
            # On general failure, fallback to the entire text (no compression)
            return [text_content]

async def run_compressor(ctx: QueryPipelineContext) -> QueryPipelineContext:
    """
    Stage E: Context Compression.
    Calls the extractor model in parallel (with concurrency limit) to compress the text of the top-5 reranked nodes.
    Constructs the consolidated ctx.compressed_context.
    """
    start_time = time.monotonic()
    
    if not ctx.reranked_nodes:
        logger.warning("compressor_skipped_no_nodes")
        ctx.compressed_context = ""
        ctx.stage_timings["compressor"] = time.monotonic() - start_time
        return ctx
        
    logger.info("compressor_start", node_count=len(ctx.reranked_nodes))
    
    # Helper for fallback text construction
    def get_fallback_context() -> str:
        raw_blocks = [
            f"Source: {node.breadcrumb}\n{node.text_content}"
            for node in ctx.reranked_nodes
            if node.text_content
        ]
        return "\n\n".join(raw_blocks)
        
    try:
        # Run compression for all reranked nodes in parallel with semaphore protection
        tasks = [
            compress_node_text(ctx.original_query, node.text_content)
            for node in ctx.reranked_nodes
        ]
        compressed_results = await asyncio.gather(*tasks)
        
        # Combine the results into a structured format
        compressed_blocks = []
        for idx, sentences in enumerate(compressed_results):
            node = ctx.reranked_nodes[idx]
            if sentences:
                joined_text = " ".join(sentences)
                # Form a block with citation prefix
                compressed_blocks.append(f"Source: {node.breadcrumb}\n{joined_text}")
                
        ctx.compressed_context = "\n\n".join(compressed_blocks)
        
        # Fallback if the compression yielded zero content
        if not ctx.compressed_context.strip():
            logger.warning("compressor_yielded_empty_context_using_fallback")
            ctx.compressed_context = get_fallback_context()
            
        logger.info("compressor_complete", 
                    compressed_length=len(ctx.compressed_context),
                    original_length=sum(len(n.text_content or "") for n in ctx.reranked_nodes))
                    
    except Exception as e:
        logger.error("compressor_failed", error=str(e))
        ctx.compressed_context = get_fallback_context()
        
    ctx.stage_timings["compressor"] = time.monotonic() - start_time
    return ctx
