import time
from typing import AsyncGenerator, List, Dict
import structlog
from ollama import AsyncClient

from backend.config import GENERATOR_MODEL, OLLAMA_HOST
from backend.rag.retrieval.pipeline_context import QueryPipelineContext, SourceCitation

logger = structlog.get_logger()

# Global AsyncClient instance
_client = AsyncClient(host=OLLAMA_HOST)

async def check_model_exists(model_name: str) -> bool:
    """
    Checks if the model exists in the local Ollama registry.
    Handles both ListResponse object (ollama>=0.2.0) and dict (older versions) structures.
    """
    try:
        models_response = await _client.list()
        
        # Determine format
        if isinstance(models_response, dict):
            models_list = models_response.get("models", [])
        else:
            models_list = getattr(models_response, "models", [])
            
        for m in models_list:
            if isinstance(m, dict):
                model_id = m.get("model", "")
            else:
                model_id = getattr(m, "model", "")
                
            if model_name in model_id or model_id.startswith(model_name) or model_name.startswith(model_id):
                return True
        return False
    except Exception as e:
        logger.error("check_model_exists_failed", error=str(e))
        return False

async def run_generator(ctx: QueryPipelineContext) -> AsyncGenerator[str, None]:
    """
    Stage F: Structured Generation.
    Streams plain-English answers citing sources.
    Uses ifsca-saullm-7b-ft model (or fallback if not available).
    Populates ctx.answer_text and ctx.source_citations.
    """
    start_time = time.monotonic()
    
    # Identify which reranked nodes contributed content to the compressed context.
    # The compressed context format is: "Source: {breadcrumb} [ID: {node_id}]\n{text}"
    # We check for the node_id UUID string directly in the compressed context.
    # If the context is empty (e.g. all compressor calls timed out), fall back
    # to all reranked nodes so citations are never silently empty.
    compressed = ctx.compressed_context or ""
    if compressed.strip():
        cited_node_ids = {
            node.node_id
            for node in ctx.reranked_nodes
            if str(node.node_id) in compressed
        }
        # If the string check yields nothing (e.g. format mismatch), fall back to all reranked
        if not cited_node_ids:
            cited_node_ids = {node.node_id for node in ctx.reranked_nodes}
    else:
        cited_node_ids = {node.node_id for node in ctx.reranked_nodes}

    ctx.source_citations = [
        SourceCitation(
            node_id=node.node_id,
            doc_id=node.doc_id,
            file_name=node.file_name or "Unknown",
            breadcrumb=node.breadcrumb,
            title=node.title,
            text_content=node.text_content,
            verbatim_quote=(node.text_content[:200].strip() if node.text_content else None)
        )
        for node in ctx.reranked_nodes
        if node.node_id in cited_node_ids and node.breadcrumb
    ]
    
    # 1. Format the glossary definitions
    glossary_text = ""
    if ctx.inlined_definitions:
        glossary_lines = [f"'{term}': {definition}" for term, definition in ctx.inlined_definitions.items()]
        glossary_text = "\n".join(glossary_lines)
        
    system_prompt = (
        "You are the IFSCA Regulatory Assistant. You help regulatory officers understand compliance requirements.\n\n"
        "OUTPUT RULES:\n"
        "1. Answer using ONLY facts from the CONTEXT blocks provided below.\n"
        "2. Write in plain, clear English. Avoid legal jargon.\n"
        "3. If the answer is partially in context, answer what you can and state: \"For [missing part], no regulation was found in the available corpus.\"\n"
        "4. Every answer must include: (a) a plain-English explanation, (b) a structured table of key facts if applicable, (c) the exact source section citation, (d) the verbatim quoted text from the regulation.\n"
        "5. Never extrapolate, infer, or use prior knowledge about regulations.\n"
    )
    
    user_content = (
        f"GLOSSARY DEFINITIONS:\n{glossary_text or 'None'}\n\n"
        f"CONTEXT:\n{ctx.compressed_context or 'No context found.'}\n\n"
        f"QUERY: {ctx.original_query}"
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    # 2. Determine which model to use (fallback to llama3.2:3b if saullm-ft is not compiled yet)
    model_to_use = GENERATOR_MODEL
    model_exists = await check_model_exists(model_to_use)
    if not model_exists:
        # Check if saullm:latest exists
        saul_latest_exists = await check_model_exists("ifsca-saullm-7b-ft:latest")
        if saul_latest_exists:
            model_to_use = "ifsca-saullm-7b-ft:latest"
        else:
            # Fallback to llama3.2:3b which is guaranteed to exist
            logger.warning("generator_model_not_found_using_fallback", 
                           configured_model=GENERATOR_MODEL, fallback_model="llama3.2:3b")
            model_to_use = "llama3.2:3b"
            
    # 3. Stream the generation
    logger.info("generation_start", model=model_to_use)
    accumulated_response = []
    
    try:
        response_stream = await _client.chat(
            model=model_to_use,
            messages=messages,
            stream=True,
            options={"temperature": 0.0}
        )
        
        async for chunk in response_stream:
            token = chunk.message.content or ""
            accumulated_response.append(token)
            yield token
            
        ctx.answer_text = "".join(accumulated_response)
        logger.info("generation_complete", answer_length=len(ctx.answer_text), citations_count=len(ctx.source_citations))
        
    except Exception as e:
        logger.error("generation_failed", error=str(e))
        error_msg = f"\nError generating answer: {str(e)}"
        yield error_msg
        ctx.answer_text = "".join(accumulated_response) + error_msg
        
    ctx.stage_timings["generator"] = time.monotonic() - start_time
