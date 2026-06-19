import time
import asyncio
import re
from typing import AsyncGenerator, List, Dict
import structlog
from ollama import AsyncClient

from backend.config import GENERATOR_MODEL, OLLAMA_HOST
from backend.rag.retrieval.pipeline_context import QueryPipelineContext, SourceCitation, NodeCandidate
from backend.rag.retrieval.compressor import assemble_batch

logger = structlog.get_logger()
_client = AsyncClient(host=OLLAMA_HOST)

def clean_for_matching(text: str) -> str:
    """Strips HTML comments, normalizes spaces/newlines, and removes punctuation/case."""
    # Strip HTML comments like <!-- ID: ... -->
    text = re.sub(r"<!--.*?-->", "", text)
    # Normalize whitespace/newlines
    text = re.sub(r"\s+", " ", text).strip().lower()
    # Strip basic punctuation to prevent mismatch on minor details
    text = re.sub(r"[^\w\s]", "", text)
    return text

def extract_quote_from_text(text: str) -> str:
    """Extracts verbatim quote from model output block by removing headers and formatting."""
    lines = text.split("\n")
    quote_lines = []
    for line in lines:
        line_strip = line.strip()
        if line_strip.startswith("#") or "quote" in line_strip.lower() or not line_strip:
            continue
        # Strip blockquote prefix
        if line_strip.startswith(">"):
            line_strip = line_strip[1:].strip()
        # Strip double quotes
        if line_strip.startswith('"') and line_strip.endswith('"'):
            line_strip = line_strip[1:-1].strip()
        quote_lines.append(line_strip)
    return "\n".join(quote_lines).strip()

def verify_and_correct_answer(answer_text: str, nodes: List[NodeCandidate]) -> str:
    """Verifies that the verbatim quote in answer_text exists in nodes, otherwise corrects it."""
    lower_ans = answer_text.lower()
    trigger_words = [
        "\n# verbatim regulatory quote",
        "\n# exact regulation quote",
        "\n# verbatim quote",
        "\n# exact quote",
    ]
    
    first_idx = -1
    for trigger in trigger_words:
        idx = lower_ans.find(trigger)
        if idx != -1:
            first_idx = idx
            break
            
    if first_idx == -1:
        # If the quote section was omitted entirely, append it programmatically
        if nodes:
            return answer_text.strip() + f"\n\n# Verbatim Regulatory Quote\n> {nodes[0].text_content.strip()}"
        return answer_text
        
    pre_quote = answer_text[:first_idx]
    quote_section = answer_text[first_idx:]
    
    extracted_quote = extract_quote_from_text(quote_section)
    if not extracted_quote:
        if nodes:
            return pre_quote.strip() + f"\n\n# Verbatim Regulatory Quote\n> {nodes[0].text_content.strip()}"
        return answer_text
        
    cleaned_model_quote = clean_for_matching(extracted_quote)
    
    # Check if this cleaned quote is present in any node's raw content
    match_found = False
    for node in nodes:
        cleaned_node_text = clean_for_matching(node.text_content)
        if cleaned_model_quote in cleaned_node_text:
            match_found = True
            break
            
    if match_found:
        return answer_text
    else:
        # Programmatic correction of the quote
        if nodes:
            return pre_quote.strip() + f"\n\n# Verbatim Regulatory Quote\n> {nodes[0].text_content.strip()}"
        return pre_quote.strip()

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
    return response.message.content.strip()

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
        "6. At the very end of your response, you MUST create a section titled \"# Verbatim Regulatory Quote\" containing the exact verbatim quoted text of the relevant regulations from the context inside a blockquote (for example:\n"
        "# Verbatim Regulatory Quote\n"
        "> [verbatim text here]).\n"
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
            response_stream = await asyncio.wait_for(
                _client.chat(
                    model=model_to_use,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    stream=True,
                    options={"temperature": 0.0}
                ),
                timeout=30.0
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
            response_stream = await asyncio.wait_for(
                _client.chat(
                    model=model_to_use,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content_1}
                    ],
                    stream=True,
                    options={"temperature": 0.0}
                ),
                timeout=30.0
            )
            
            in_quote_section = False
            quote_buffer = []
            accumulated_stream = ""
            yielded_len = 0
            
            async for chunk in response_stream:
                token = chunk.message.content or ""
                accumulated_stream += token
                
                if not in_quote_section:
                    lower_accumulated = accumulated_stream.lower()
                    trigger_words = [
                        "\n# verbatim regulatory quote",
                        "\n# exact regulation quote",
                        "\n# verbatim quote",
                        "\n# exact quote",
                    ]
                    
                    first_idx = -1
                    for trigger in trigger_words:
                        idx = lower_accumulated.find(trigger)
                        if idx != -1:
                            first_idx = idx
                            break
                            
                    if first_idx != -1:
                        # Yield everything before trigger
                        text_to_yield = accumulated_stream[yielded_len:first_idx]
                        if text_to_yield:
                            yield text_to_yield
                            accumulated_response.append(text_to_yield)
                        
                        yielded_len = first_idx
                        in_quote_section = True
                        quote_buffer.append(accumulated_stream[first_idx:])
                    else:
                        # Yield safely keeping a buffer margin for potential triggers
                        safety_margin = 35
                        if len(accumulated_stream) - yielded_len > safety_margin:
                            text_to_yield = accumulated_stream[yielded_len : len(accumulated_stream) - safety_margin]
                            if text_to_yield:
                                yield text_to_yield
                                accumulated_response.append(text_to_yield)
                                yielded_len += len(text_to_yield)
                else:
                    quote_buffer.append(token)
            
            # Flush out final pieces
            if not in_quote_section:
                remaining_text = accumulated_stream[yielded_len:]
                if remaining_text:
                    yield remaining_text
                    accumulated_response.append(remaining_text)
            else:
                quote_block = "".join(quote_buffer)
                extracted_quote = extract_quote_from_text(quote_block)
                
                if extracted_quote:
                    cleaned_model_quote = clean_for_matching(extracted_quote)
                    
                    # Verify quote against raw retrieved nodes
                    match_found = False
                    for node in ctx.reranked_nodes:
                        cleaned_node_text = clean_for_matching(node.text_content)
                        if cleaned_model_quote in cleaned_node_text:
                            match_found = True
                            break
                            
                    if match_found:
                        yield quote_block
                        accumulated_response.append(quote_block)
                    else:
                        logger.warning("hallucinated_quote_detected_correcting_at_runtime", extracted_quote=extracted_quote[:100])
                        if ctx.reranked_nodes:
                            corrected_quote_block = (
                                f"\n\n# Verbatim Regulatory Quote\n"
                                f"> {ctx.reranked_nodes[0].text_content.strip()}"
                            )
                            yield corrected_quote_block
                            accumulated_response.append(corrected_quote_block)
                else:
                    if ctx.reranked_nodes:
                        corrected_quote_block = (
                            f"\n\n# Verbatim Regulatory Quote\n"
                            f"> {ctx.reranked_nodes[0].text_content.strip()}"
                        )
                        yield corrected_quote_block
                        accumulated_response.append(corrected_quote_block)
                        
            if has_overflow:
                yield "\n\n---\n\n*Refining answer using additional retrieved contexts...*\n"
                
                partial_answers = ["".join(accumulated_response)]
                
                # Map phase
                for idx, batch_nodes in enumerate(ctx.overflow_batches):
                    logger.info("generation_start_overflow_batch", batch_idx=idx+1, model=model_to_use)
                    batch_context = assemble_batch(batch_nodes)
                    try:
                        partial = await generate_once(batch_context, glossary_text, ctx.original_query, system_prompt, model_to_use)
                        partial_answers.append(partial)
                    except Exception as batch_err:
                        logger.error("overflow_batch_failed_skipping", batch_idx=idx+1, error=str(batch_err))
                    
                # Reduce phase
                logger.info("generation_merge_start", model=merge_model)
                try:
                    merged_answer = await asyncio.wait_for(
                        merge_answers(partial_answers, ctx.original_query, merge_model),
                        timeout=120.0
                    )
                    # Verify map-reduce merged output quote
                    verified_merged = verify_and_correct_answer(merged_answer, ctx.reranked_nodes)
                    yield f"\n\n# Synthesized Final Answer\n{verified_merged}\n"
                    ctx.answer_text = verified_merged
                except Exception as merge_err:
                    logger.error("merge_failed_falling_back_to_concatenation", error=str(merge_err))
                    fallback_merged = "\n\n---\n\n".join(partial_answers)
                    verified_fallback = verify_and_correct_answer(fallback_merged, ctx.reranked_nodes)
                    yield f"\n\n# Synthesized Answer (Fallback)\n{verified_fallback}\n"
                    ctx.answer_text = verified_fallback
            else:
                ctx.answer_text = "".join(accumulated_response)
                
        logger.info("generation_complete", answer_length=len(ctx.answer_text), citations_count=len(ctx.source_citations))
        
    except Exception as e:
        logger.error("generation_failed", error=str(e))
        error_msg = f"\nError generating answer: {str(e)}"
        yield error_msg
        ctx.answer_text = "".join(accumulated_response) + error_msg
        
    ctx.stage_timings["generator"] = time.monotonic() - start_time
