import os
import time
from uuid import UUID, uuid4
from datetime import date
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

# Set docling device to CPU before importing docling
os.environ["DOCLING_DEVICE"] = "cpu"
from docling.document_converter import DocumentConverter

import structlog
from backend.config import validate_doc_type
from backend.database.connection import get_db_connection
from backend.database.queries import (
    get_document_by_filename,
    get_document_by_hash,
    delete_document_by_filename,
    insert_document,
    insert_ast_nodes,
    insert_relationships,
    insert_glossary_entry
)
from backend.rag.ingestion.prefilters import (
    DoclingPage,
    is_devanagari_block,
    find_english_classifier_window
)
from backend.rag.extraction.classifier import classify_document
from backend.rag.extraction.boundary_detector import detect_boundary
from backend.rag.extraction.relational_extractor import extract_relations
from backend.rag.extraction.llm_client import generate_embedding
from backend.rag.ingestion.ast_builder import build_ast, ASTNode, RawBlock
from backend.rag.ingestion.auditor import audit_ast_nodes
from backend.rag.ingestion.corpus_resolver import resolve_pending_references

logger = structlog.get_logger()

class IngestionError(Exception):
    """Base exception for ingestion pipeline failures."""
    pass

class EmbeddingServiceError(IngestionError):
    """Raised when generating embeddings fails."""
    pass

def compute_file_hash(file_path: str) -> str:
    import hashlib
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

async def ingest_document(pdf_path: str, dedup: bool = True) -> UUID:
    """
    Orchestrates the full visual RAG ingestion pipeline for a single regulatory PDF.
    
    Workflow:
    1. Parse PDF using Docling (CPU backend).
    2. Extract and group page text.
    3. Apply English detection window and classify document metadata.
    4. Validate document type, delete existing doc if naming collision (idempotency).
    5. Filter out Devanagari blocks (>30% density).
    6. Run boundary detection on blocks.
    7. Build AST tree structures.
    8. Audit AST node integrity (character loss, empty nodes, parent structure).
    9. Run batched relationship extraction on content nodes.
    10. Generate dense text embeddings using nomic-embed-text.
    11. Write everything atomically to PostgreSQL.
    12. Run corpus resolver to connect cross-references.
    
    Args:
        pdf_path: Absolute or relative file path to the PDF document.
        
    Returns:
        The doc_id UUID of the newly ingested document.
    """
    file_name = os.path.basename(pdf_path)
    file_hash = compute_file_hash(pdf_path)
    
    # 0. Check MD5 duplicate ingestion
    async with get_db_connection() as conn:
        existing = await get_document_by_hash(conn, file_hash)
        if existing:
            if dedup:
                logger.info("document_already_ingested_skipping", file_name=file_name, doc_id=str(existing["doc_id"]), file_hash=file_hash)
                return existing["doc_id"]
            else:
                logger.info("document_already_ingested_deleting_for_reingest", file_name=file_name, doc_id=str(existing["doc_id"]), file_hash=file_hash)
                # Deleting document automatically cascades to AST nodes, relationships, and glossary
                await conn.execute("DELETE FROM documents WHERE doc_id = $1", existing["doc_id"])
                
    logger.info("ingest_document_started", file_name=file_name, path=pdf_path, file_hash=file_hash)
    start_time = time.monotonic()
    
    # 1. Parse PDF using Docling
    try:
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        doc = result.document
    except Exception as e:
        logger.error("docling_parsing_failed", file_name=file_name, error=str(e))
        raise IngestionError(f"Docling failed to parse PDF {file_name}: {str(e)}") from e
        
    # 2. Extract and group page text to find English classifier window
    page_texts: Dict[int, List[str]] = {}
    total_raw_chars = 0
    
    for item, _ in doc.iterate_items():
        text = getattr(item, "text", None)
        if text:
            total_raw_chars += len(text)
            prov = getattr(item, "prov", None)
            if prov and len(prov) > 0:
                page_no = prov[0].page_no
                if page_no not in page_texts:
                    page_texts[page_no] = []
                page_texts[page_no].append(text)
                
    docling_pages: List[DoclingPage] = [
        DoclingPage(page_no=pno, text="\n".join(page_texts[pno]))
        for pno in sorted(page_texts.keys())
    ]
    
    # 3. English window detector and metadata classification
    classifier_window = find_english_classifier_window(docling_pages)
    if not classifier_window:
        raise IngestionError(f"Could not extract sufficient text window from {file_name}")
        
    try:
        metadata = await classify_document(classifier_window)
    except Exception as e:
        logger.error("metadata_classification_failed", file_name=file_name, error=str(e))
        raise IngestionError(f"Metadata classification failed for {file_name}: {str(e)}") from e
        
    # Validate doc type
    try:
        validated_doc_type = validate_doc_type(metadata.doc_type)
    except ValueError as e:
        logger.error("invalid_doc_type", file_name=file_name, doc_type=metadata.doc_type, error=str(e))
        raise IngestionError(f"Document classification returned invalid type: {metadata.doc_type}") from e
        
    # 4. Generate local doc_id
    doc_id = uuid4()
    
    # 5. Filter layout blocks and run boundary detection
    raw_blocks: List[RawBlock] = []
    
    for item_idx, (item, _) in enumerate(doc.iterate_items()):
        text = getattr(item, "text", None)
        if not text or not text.strip():
            continue
            
        # Skip Hindi blocks (>30% Devanagari characters)
        if is_devanagari_block(text, threshold=0.30):
            logger.debug("skipping_devanagari_block", index=item_idx, text_snippet=text[:50])
            continue
            
        # Boundary detection (self-healing happens internally inside llm_client)
        try:
            boundary = await detect_boundary(text)
        except Exception as e:
            logger.warning("boundary_detection_failed_block", index=item_idx, error=str(e))
            # On boundary detection failure, we treat it as generic body text needing repair later
            from backend.rag.extraction.schemas import BoundaryOutput
            boundary = BoundaryOutput(
                node_type="BODY_TEXT",
                level=6,
                is_boundary_break=False,
                heading_text=None
            )
            
        raw_blocks.append(RawBlock(text=text, boundary=boundary))
        
    # 6. Build AST
    ast_nodes = build_ast(doc_id, metadata.title, raw_blocks)
    
    # 7. Audit AST nodes
    total_english_chars = sum(len(b.text) for b in raw_blocks)
    ast_nodes = audit_ast_nodes(ast_nodes, total_english_chars)
    
    # 8. Batched relationship extraction on content nodes
    # We batch extract relationships for content nodes only
    content_nodes = [
        n for n in ast_nodes
        if n.node_type in ["SECTION", "SUBSECTION", "CLAUSE", "SUBCLAUSE", "DEFINITION"]
    ]
    
    relationships_to_insert: List[Dict[str, Any]] = []
    
    # Dynamic batching based on character limit (~20k chars to fit 8k tokens) and max 10 nodes to avoid JSON response token limit overflow
    batches, current_batch, current_length = [], [], 0
    for node in content_nodes:
        text_len = len(node.text_content or "")
        if current_batch and (current_length + text_len > 20000 or len(current_batch) >= 10):
            batches.append(current_batch)
            current_batch, current_length = [], 0
        current_batch.append(node)
        current_length += text_len
    if current_batch:
        batches.append(current_batch)
        
    for i, batch in enumerate(batches):
        batch_texts = [node.text_content for node in batch]
        
        try:
            batch_relations = await extract_relations(batch_texts)
            
            for clause_rel in batch_relations:
                src_idx = clause_rel.source_clause_index
                if 0 <= src_idx < len(batch):
                    source_node = batch[src_idx]
                    
                    for rel in clause_rel.relations:
                        # Validate effective date format
                        parsed_date = None
                        if rel.effective_date:
                            try:
                                parsed_date = date.fromisoformat(rel.effective_date)
                            except ValueError:
                                logger.warning("invalid_effective_date_format", value=rel.effective_date)
                                parsed_date = None
                                
                        relationships_to_insert.append({
                            "source_node_id": source_node.node_id,
                            "target_text_ref": rel.target_text_ref,
                            "rel_type": rel.rel_type,
                            "effective_date": parsed_date
                        })
        except Exception as e:
            logger.error("batch_relationship_extraction_failed", index_start=i, error=str(e))
            # We continue ingestion even if relationship extraction fails for a batch,
            # since the nodes themselves are still valid.
            continue
            
    # 9. Generate dense embeddings for all nodes that have content
    nodes_to_insert: List[Dict[str, Any]] = []
    for node in ast_nodes:
        embedding = None
        # Generate embedding for nodes with text content (ignoring root node text usually)
        if node.text_content and node.level > 1:
            try:
                # Combine breadcrumb + title + text content to give embedding better structural context
                contextual_text = f"{node.breadcrumb}\n\n{node.text_content}"
                embedding = await generate_embedding(contextual_text)
            except Exception as e:
                logger.error("embedding_generation_failed_fatal", node_id=str(node.node_id), error=str(e))
                raise EmbeddingServiceError(f"Embedding service failed or timed out: {str(e)}") from e
                
        nodes_to_insert.append({
            "node_id": node.node_id,
            "doc_id": node.doc_id,
            "parent_id": node.parent_id,
            "level": node.level,
            "node_type": node.node_type,
            "title": node.title,
            "text_content": node.text_content,
            "breadcrumb": node.breadcrumb,
            "needs_repair": node.needs_repair,
            "embedding": embedding
        })
        
    # 10. Write AST nodes and relationships in database transaction
    async with get_db_connection() as conn:
        async with conn.transaction():
            # Idempotency rule: delete if existing naming collision
            existing = await get_document_by_filename(conn, file_name)
            if existing:
                logger.info("idempotency_trigger_deleting_existing_doc", file_name=file_name, doc_id=str(existing["doc_id"]))
                await delete_document_by_filename(conn, file_name)
                
            # Insert document
            await insert_document(
                conn,
                doc_id=doc_id,
                file_name=file_name,
                title=metadata.title,
                publish_date=metadata.publish_date,
                doc_type=validated_doc_type,
                file_hash=file_hash
            )
            logger.info("document_inserted_to_db", doc_id=str(doc_id))
            
            await insert_ast_nodes(conn, nodes_to_insert)
            if relationships_to_insert:
                await insert_relationships(conn, relationships_to_insert)
                
            logger.info("document_ingestion_transaction_committed", file_name=file_name, doc_id=str(doc_id))
            
    # 11. Run corpus resolver to connect references globally (outside main transaction to see all documents)
    await resolve_pending_references()
    
    duration = time.monotonic() - start_time
    logger.info("ingest_document_completed", file_name=file_name, doc_id=str(doc_id), duration_seconds=duration)
    return doc_id
