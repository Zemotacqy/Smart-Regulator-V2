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
    # The compressed context format is: "Source: {breadcrumb} <!-- ID: {node_id} -->\n{text}"
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
        "1. Answer using ONLY facts from the CONTEXT blocks provided below. Do not extrapolate, infer, or use prior knowledge.\n"
        "2. Write in plain, clear, and friendly English. Avoid dense legal jargon and make the response easily understandable for non-technical users.\n"
        "3. Be highly verbose, comprehensive, and detailed in your explanation. Share the complete set of rules, requirements, conditions, limits, qualifications, and exemptions. Never write a short summary or simply point to where the answer can be found.\n"
        "4. Write your response naturally using paragraphs and bullet points. Do not format your answer as a table.\n"
        "5. When citing regulations, always explain exactly where to find the rule by mentioning the full name of the regulation and the section/clause title naturally in your text (for example: 'as specified in Section 8 (Reserve requirements) of the IFSCA Banking Regulations, 2020'). Never just say 'Section 8' without context.\n"
        "6. At the very end of your response, create a section titled \"# Exact Regulation Quote\" to share the verbatim quoted text of the relevant regulations for reference.\n"
        "7. NEVER include or output UUIDs, chunk IDs, or internal identifiers (like \"[ID: ...]\" or comment tags) in your response.\n"
        "8. If you do not know the answer or the context does not contain facts to answer the question, clearly state: \"I do not know the answer as no regulation was found in the available corpus.\"\n"
        "9. CRITICAL LIST CONSTRAINT: When asked to list items (such as a Code of Conduct, principles, or conditions), you MUST identify EVERY separate set of rules/principles present in the context. For each set of rules/principles found, you MUST extract and list EVERY single item/point present in the context completely without omission."
    )
    
    has_context = ctx.compressed_context and ctx.compressed_context.strip() and ctx.compressed_context.strip() != "No context found."
    
    if not has_context:
        user_content = (
            f"GLOSSARY DEFINITIONS:\n{glossary_text or 'None'}\n\n"
            f"CONTEXT:\nNo context found.\n\n"
            f"QUERY: {ctx.original_query}\n\n"
            "INSTRUCTION: Since the CONTEXT is empty (No context found), you MUST start your response with: "
            "\"I do not know the answer as no regulation was found in the available corpus.\" "
            "Then, provide the definitions of any key terms from the GLOSSARY DEFINITIONS to help the user."
        )
    else:
        user_content = (
            f"GLOSSARY DEFINITIONS:\n{glossary_text or 'None'}\n\n"
            f"CONTEXT:\n{ctx.compressed_context}\n\n"
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
        
        buffer = ""
        async for chunk in response_stream:
            token = chunk.message.content or ""
            buffer += token
            
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                trimmed = line.strip()
                # Skip table rows completely to omit the Rule/Detail/Source table from the final answer
                if trimmed.startswith('|') and trimmed.endswith('|'):
                    continue
                yield line + "\n"
                accumulated_response.append(line + "\n")
                
        # Process remaining buffer content
        if buffer:
            trimmed = buffer.strip()
            if not (trimmed.startswith('|') and trimmed.endswith('|')):
                yield buffer
                accumulated_response.append(buffer)
                
        ctx.answer_text = "".join(accumulated_response)
        logger.info("generation_complete", answer_length=len(ctx.answer_text), citations_count=len(ctx.source_citations))
        
    except Exception as e:
        logger.error("generation_failed", error=str(e))
        error_msg = f"\nError generating answer: {str(e)}"
        yield error_msg
        ctx.answer_text = "".join(accumulated_response) + error_msg
        
    ctx.stage_timings["generator"] = time.monotonic() - start_time
