import time
import asyncio
from typing import AsyncGenerator, List, Dict
import structlog
from ollama import AsyncClient

from backend.config import GENERATOR_MODEL, OLLAMA_HOST
from backend.rag.retrieval.pipeline_context import QueryPipelineContext, SourceCitation
from backend.rag.retrieval.compressor import assemble_batch

logger = structlog.get_logger()
_client = AsyncClient(host=OLLAMA_HOST)

async def check_model_exists(model_name: str) -> bool:
    """
    Checks if the model exists in the local Ollama registry.
    """
    try:
        models_response = await _client.list()
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

async def generate_once(context_block: str, glossary_text: str, query: str, system_prompt: str, model: str) -> str:
    """Helper to run a single non-streaming generator call for overflow map batches."""
    user_content = (
        f"GLOSSARY DEFINITIONS:\n{glossary_text or 'None'}\n\n"
        f"CONTEXT:\n{context_block}\n\n"
        f"QUERY: {query}"
    )
    response = await _client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        options={"temperature": 0.0}
    )
    # Post-process answer to remove table rows
    lines = response.message.content.split("\n")
    cleaned_lines = []
    for line in lines:
        trimmed = line.strip()
        if trimmed.startswith('|') and trimmed.endswith('|'):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

async def merge_answers(partial_answers: List[str], query: str, model: str) -> str:
    """Synthesizes multiple partial answers into a single coherent response."""
    formatted_partials = ""
    for i, ans in enumerate(partial_answers):
        formatted_partials += f"--- CONTENT SEGMENT ---\n{ans}\n\n"
        
    merge_prompt = (
        f"Synthesize the following content segments into ONE complete, factual response for the query below. "
        f"Do not mention the segments themselves or number them. Remove duplicates, combine information, "
        f"and preserve all regulatory citations and verbatim quotes exactly as they appear.\n\n"
        f"Query: {query}\n\n"
        f"Content segments:\n{formatted_partials}\n"
        f"Output: A single, coherent, plain-English regulatory answer."
    )
    
    response = await _client.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a legal synthesis assistant. Combine the provided content segments "
                    "into a single, coherent, fully-formed response. Do not invent any new facts. "
                    "Preserve all citations and exact quotes. Do not comment on the process of synthesis."
                )
            },
            {"role": "user", "content": merge_prompt}
        ],
        options={"temperature": 0.0}
    )
    return response.message.content.strip()

async def run_generator(ctx: QueryPipelineContext) -> AsyncGenerator[str, None]:
    """
    Stage F: Structured Generation with Map-Reduce Merge.
    Streams plain-English answers citing sources.
    Handles context window overflow gracefully using map-reduce synthesis.
    """
    start_time = time.monotonic()
    
    # Assemble citations for all contributing nodes
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
        if node.breadcrumb
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
    
    # 2. Determine which generator model to use
    model_to_use = GENERATOR_MODEL
    model_exists = await check_model_exists(model_to_use)
    if not model_exists:
        saul_latest_exists = await check_model_exists("ifsca-saullm-7b-ft:latest")
        if saul_latest_exists:
            model_to_use = "ifsca-saullm-7b-ft:latest"
        else:
            logger.warning("generator_model_not_found_using_fallback", 
                           configured_model=GENERATOR_MODEL, fallback_model="llama3.2:3b")
            model_to_use = "llama3.2:3b"
            
    # Resolve merge model: ifsca-extractor-3b or llama3.2:3b
    merge_model = "ifsca-extractor-3b"
    merge_model_exists = await check_model_exists(merge_model)
    if not merge_model_exists:
        merge_model = "llama3.2:3b"
        
    has_context = ctx.compressed_context and ctx.compressed_context.strip() and ctx.compressed_context.strip() != "No context found."
    has_overflow = len(ctx.overflow_batches) > 0
    
    accumulated_response = []
    
    try:
        if not has_context:
            user_content = (
                f"GLOSSARY DEFINITIONS:\n{glossary_text or 'None'}\n\n"
                f"CONTEXT:\nNo context found.\n\n"
                f"QUERY: {ctx.original_query}\n\n"
                "INSTRUCTION: Since the CONTEXT is empty (No context found), you MUST start your response with: "
                "\"I do not know the answer as no regulation was found in the available corpus.\" "
                "Then, provide the definitions of any key terms from the GLOSSARY DEFINITIONS to help the user."
            )
            response_stream = await _client.chat(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                stream=True,
                options={"temperature": 0.0}
            )
            async for chunk in response_stream:
                token = chunk.message.content or ""
                yield token
                accumulated_response.append(token)
                
            ctx.answer_text = "".join(accumulated_response)
        else:
            # First batch generation (streaming)
            logger.info("generation_start_batch_1", model=model_to_use)
            user_content_1 = (
                f"GLOSSARY DEFINITIONS:\n{glossary_text or 'None'}\n\n"
                f"CONTEXT:\n{ctx.compressed_context}\n\n"
                f"QUERY: {ctx.original_query}"
            )
            response_stream = await _client.chat(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content_1}
                ],
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
                    if trimmed.startswith('|') and trimmed.endswith('|'):
                        continue
                    yield line + "\n"
                    accumulated_response.append(line + "\n")
                    
            if buffer:
                trimmed = buffer.strip()
                if not (trimmed.startswith('|') and trimmed.endswith('|')):
                    yield buffer
                    accumulated_response.append(buffer)
                    
            if has_overflow:
                # Notify UI that refinement is in progress
                yield "\n\n---\n\n*Refining answer using additional retrieved contexts...*\n"
                
                partial_answers = ["".join(accumulated_response)]
                
                # Map phase: Generate partial answers for remaining batches
                for idx, batch_nodes in enumerate(ctx.overflow_batches):
                    logger.info("generation_start_overflow_batch", batch_idx=idx+1, model=model_to_use)
                    batch_context = assemble_batch(batch_nodes)
                    try:
                        partial = await generate_once(batch_context, glossary_text, ctx.original_query, system_prompt, model_to_use)
                        partial_answers.append(partial)
                    except Exception as batch_err:
                        logger.error("overflow_batch_failed_skipping", batch_idx=idx+1, error=str(batch_err))
                        # Skip failed batches — do not crash the whole generator
                    
                # Reduce phase: Merge all partial responses
                logger.info("generation_merge_start", model=merge_model)
                try:
                    merged_answer = await asyncio.wait_for(
                        merge_answers(partial_answers, ctx.original_query, merge_model),
                        timeout=30.0
                    )
                    yield f"\n\n# Synthesized Final Answer\n{merged_answer}\n"
                    ctx.answer_text = merged_answer
                except Exception as merge_err:
                    logger.error("merge_failed_falling_back_to_concatenation", error=str(merge_err))
                    fallback_merged = "\n\n---\n\n".join(partial_answers)
                    yield f"\n\n# Synthesized Answer (Fallback)\n{fallback_merged}\n"
                    ctx.answer_text = fallback_merged
            else:
                ctx.answer_text = "".join(accumulated_response)
                
        logger.info("generation_complete", answer_length=len(ctx.answer_text), citations_count=len(ctx.source_citations))
        
    except Exception as e:
        logger.error("generation_failed", error=str(e))
        error_msg = f"\nError generating answer: {str(e)}"
        yield error_msg
        ctx.answer_text = "".join(accumulated_response) + error_msg
        
    ctx.stage_timings["generator"] = time.monotonic() - start_time
