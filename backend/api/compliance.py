import os
import json
import asyncio
import tempfile
from uuid import UUID, uuid4
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from pypdf import PdfReader
import structlog

from backend.config import GENERATOR_MODEL
from backend.rag.extraction.schemas import ComplianceAuditResult
from backend.rag.extraction.llm_client import call_llm_with_validation
from backend.rag.retrieval.pipeline_context import QueryPipelineContext
from backend.rag.retrieval.query_expander import run_query_expander
from backend.rag.retrieval.hybrid_search import run_hybrid_search
from backend.rag.retrieval.hop_expander import run_hop_expander
from backend.rag.retrieval.temporal_filter import run_temporal_filter
from backend.rag.retrieval.reranker import run_reranker
from backend.rag.retrieval.compressor import run_compressor
from backend.rag.retrieval.generator import check_model_exists

logger = structlog.get_logger()
router = APIRouter()

def chunk_text(text: str, chunk_size: int = 1500, overlap: int = 300) -> List[str]:
    """
    Splits text into chunks using a sliding window.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

@router.post("/compliance")
async def compliance_endpoint(
    file: UploadFile = File(..., description="The entity document PDF to audit"),
    doc_filter: Optional[List[str]] = Form(None, description="Optional document IDs to scope the audit against")
):
    """
    POST /api/compliance
    Extracts text from the uploaded PDF, chunks it, retrieves relevant regulatory context for each chunk,
    and runs compliance audits, streaming the result of each audit block as Server-Sent Events (SSE).
    """
    logger.info("compliance_endpoint_hit", filename=file.filename, doc_filter=doc_filter)
    
    # Validate file format
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
    # Parse doc_filter to UUIDs
    uuid_filters: Optional[List[UUID]] = None
    if doc_filter:
        try:
            # Forms might submit as a single comma-separated string or multiple values
            parsed_filters = []
            for item in doc_filter:
                for part in item.split(","):
                    part_stripped = part.strip()
                    if part_stripped:
                        parsed_filters.append(UUID(part_stripped))
            if parsed_filters:
                uuid_filters = parsed_filters
        except ValueError as e:
            logger.error("invalid_doc_filter_format", doc_filter=doc_filter, error=str(e))
            raise HTTPException(status_code=400, detail="Invalid UUID format in doc_filter")

    async def sse_compliance_stream():
        # Save uploaded file to temp file
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"{uuid4()}_{file.filename}")
        
        try:
            with open(temp_path, "wb") as f:
                content = await file.read()
                f.write(content)
                
            # Extract text
            logger.info("compliance_extracting_text", path=temp_path)
            reader = PdfReader(temp_path)
            full_text = ""
            for idx, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
                    
            if not full_text.strip():
                logger.error("compliance_pdf_empty", filename=file.filename)
                yield f"event: error\ndata: {json.dumps({'error': 'PDF text extraction yielded no content.'})}\n\n"
                return
                
            # Chunk the text
            chunks = chunk_text(full_text)
            logger.info("compliance_text_chunked", chunks_count=len(chunks))
            
            # Determine which model to use (fallback to llama3.2:3b if saullm-ft is not available)
            model_to_use = GENERATOR_MODEL
            model_exists = await check_model_exists(model_to_use)
            if not model_exists:
                saul_latest_exists = await check_model_exists("ifsca-saullm-7b-ft:latest")
                if saul_latest_exists:
                    model_to_use = "ifsca-saullm-7b-ft:latest"
                else:
                    logger.warning("compliance_model_not_found_using_fallback", 
                                   configured_model=GENERATOR_MODEL, fallback_model="llama3.2:3b")
                    model_to_use = "llama3.2:3b"
                    
            # Audit each chunk
            for chunk_idx, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                    
                # Run retrieval pipeline to find matching regulations
                ctx = QueryPipelineContext(
                    request_id=str(uuid4()),
                    original_query=chunk[:200], # Query expander works best on shorter queries, or we can use chunk prefix
                    doc_filter=uuid_filters
                )
                
                try:
                    ctx = await run_query_expander(ctx)
                    ctx = await run_hybrid_search(ctx)
                    ctx = await run_hop_expander(ctx)
                    ctx = await run_temporal_filter(ctx)
                    # Overwrite original query with full chunk so reranker and compressor target the entire content
                    ctx.original_query = chunk
                    ctx = await run_reranker(ctx)
                    ctx = await run_compressor(ctx)
                    
                    # If no relevant context is retrieved, skip auditing this chunk
                    if not ctx.compressed_context or not ctx.compressed_context.strip():
                        logger.debug("compliance_skip_chunk_no_context", chunk_index=chunk_idx)
                        continue
                        
                    # Run LLM validation audit
                    system_prompt = (
                        "You are the IFSCA Regulatory Compliance Auditor.\n"
                        "Your task is to analyze the Entity Text against the provided Regulation Context.\n"
                        "Check if the entity practices, rules, or limits comply with the regulation.\n"
                        "You MUST return a JSON matching the ComplianceAuditResult schema."
                    )
                    
                    user_prompt = (
                        f"REGULATION CONTEXT:\n{ctx.compressed_context}\n\n"
                        f"ENTITY TEXT:\n{chunk}\n\n"
                        f"Perform the compliance audit. Compare the texts carefully."
                    )
                    
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                    
                    audit_res = await call_llm_with_validation(
                        model=model_to_use,
                        messages=messages,
                        response_schema=ComplianceAuditResult,
                        temperature=0.0
                    )
                    
                    # Yield structured audit event
                    yield f"event: audit\ndata: {json.dumps(audit_res.model_dump())}\n\n"
                    
                except Exception as ex:
                    logger.error("compliance_chunk_audit_failed", chunk_index=chunk_idx, error=str(ex))
                    # Skip or report error - we report it but keep scanning other chunks
                    yield f"event: chunk_error\ndata: {json.dumps({'chunk_index': chunk_idx, 'error': str(ex)})}\n\n"
                    
            # Send done
            yield "event: done\ndata: {}\n\n"
            
        except Exception as e:
            logger.error("compliance_endpoint_failed", error=str(e))
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
        finally:
            # Cleanup temp file
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as err:
                    logger.warning("failed_to_cleanup_temp_file", path=temp_path, error=str(err))
                    
    return StreamingResponse(sse_compliance_stream(), media_type="text/event-stream")
