import os
import sys
import json
import re
import asyncio
from uuid import UUID, uuid4
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from pathlib import Path

# Resolve project root dynamically so this script works on any machine
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import structlog
from backend.config import DATABASE_URL, EMBEDDING_MODEL
from backend.database.connection import init_db_pool, get_db_connection, close_db_pool
from backend.database.queries import insert_document, insert_ast_nodes, insert_relationships
from backend.rag.extraction.llm_client import generate_embedding
from backend.rag.ingestion.corpus_resolver import resolve_pending_references

# Setup Console Renderer for clean logging during ingestion
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ]
)
logger = structlog.get_logger()

# Hardcoded document metadata
DOC_METADATA = {
    "ifsca_act": {
        "title": "IFSCA Act - International Financial Services Centres Authority Act, 2019",
        "doc_type": "Act",
        "publish_date": "2019-12-19"
    },
    "cmi": {
        "title": "IFSCA (Capital Market Intermediaries) Regulations, 2025",
        "doc_type": "Regulation",
        "publish_date": "2025-04-16"
    },
    "banking": {
        "title": "IFSCA (Banking) Regulations, 2020",
        "doc_type": "Regulation",
        "publish_date": "2020-11-18"
    },
    "esr": {
        "title": "IFSCA (Employees' Service) Regulations, 2020",
        "doc_type": "Regulation",
        "publish_date": "2020-11-12"
    },
    "fme": {
        "title": "IFSCA (Fund Management) Regulations, 2025",
        "doc_type": "Regulation",
        "publish_date": "2025-01-01"
    },
    "techfin": {
        "title": "IFSCA (TechFin and Ancillary Services) Regulations, 2025",
        "doc_type": "Regulation",
        "publish_date": "2025-01-01"
    }
}

DUMPS_DIR = str(PROJECT_ROOT / "Dumps")

@dataclass
class NormalisedSection:
    section_id: str
    doc_name: str
    chapter_text: str
    section_title: str
    content: str
    references: List[str]

@dataclass
class NormalisedChunk:
    chunk_id: str
    parent_chunk_id: str
    content: str
    summary: Optional[str]
    sub_chunk_index: int

def clean_whitespace(text: str) -> str:
    if not text:
        return ""
    # Remove double or more spaces (AGENTS.md FTS requirement)
    text = re.sub(r' {2,}', ' ', text)
    # Convert multiple tabs/spaces to a single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Normalize line endings
    text = text.replace('\r\n', '\n')
    return text.strip()

def parse_references(ref_data: Any) -> List[str]:
    if not ref_data:
        return []
    if isinstance(ref_data, list):
        return [str(r).strip() for r in ref_data if r]
    if isinstance(ref_data, str):
        try:
            loaded = json.loads(ref_data)
            if isinstance(loaded, list):
                return [str(r).strip() for r in loaded if r]
        except Exception:
            # If it's a comma-separated string or just a plain string
            if ref_data.strip().startswith('['):
                return []
            return [ref_data.strip()]
    return []

def extract_clause_title(text: str) -> Optional[str]:
    # Match bullet markers like (a), (1), (i), (A)
    match = re.match(r"^\(([a-zA-Z0-9]+)\)\s+", text)
    if match:
        return f"({match.group(1)})"
    return None

def extract_definitions_from_text(text: str) -> List[Tuple[str, str]]:
    # Match: (a) "assets" means ... or “Client” means ... or 'Term' means ...
    pattern = r'[“\"\'\‘\“]([^”\"\'\’\”]+)[”\"\'\’\”]\s+(?:means|shall mean)\s+(.+?)(?=;|\.|$)'
    matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
    return [(term.strip(), def_text.strip()) for term, def_text in matches]

async def process_document_dumps(doc_name: str) -> Tuple[List[NormalisedSection], List[NormalisedChunk]]:
    sections: List[NormalisedSection] = []
    chunks: List[NormalisedChunk] = []

    # Map doc_name to directory path and file names
    if doc_name == "ifsca_act":
        sec_file = os.path.join(DUMPS_DIR, "ACT", "check.json")
        if os.path.exists(sec_file):
            with open(sec_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for idx, item in enumerate(data):
                    sections.append(NormalisedSection(
                        section_id=f"act_section_{idx}",
                        doc_name=doc_name,
                        chapter_text=item.get("contained_chapter") or "",
                        section_title=item.get("contained_section") or f"Section {idx}",
                        content=item.get("content") or "",
                        references=parse_references(item.get("references"))
                    ))

    elif doc_name == "esr":
        sec_file = os.path.join(DUMPS_DIR, "ESR", "ESR.spans.chunks.sections.json")
        if os.path.exists(sec_file):
            with open(sec_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for idx, item in enumerate(data):
                    sections.append(NormalisedSection(
                        section_id=f"esr_section_{idx}",
                        doc_name=doc_name,
                        chapter_text=item.get("contained_chapter") or "",
                        section_title=item.get("contained_section") or f"Section {idx}",
                        content=item.get("content") or "",
                        references=parse_references(item.get("references"))
                    ))

    else:
        # CMI, Consolidated Banking, FME, TechFin
        dir_map = {
            "cmi": "CMI",
            "banking": "Consolidated Banking",
            "fme": "FME",
            "techfin": "TechFin"
        }
        dir_name = dir_map[doc_name]
        
        # Determine sections file name (FME is section.json, others are sections.json)
        sec_filename = "section.json" if doc_name == "fme" else "sections.json"
        sec_file = os.path.join(DUMPS_DIR, dir_name, sec_filename)
        chunk_file = os.path.join(DUMPS_DIR, dir_name, "chunks.json")

        if os.path.exists(sec_file):
            with open(sec_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for idx, item in enumerate(data):
                    meta = item.get("metadata") or {}
                    sections.append(NormalisedSection(
                        section_id=item.get("id") or f"{doc_name}_section_{idx}",
                        doc_name=doc_name,
                        chapter_text=meta.get("contained_chapter") or "",
                        section_title=meta.get("contained_section") or f"Section {idx}",
                        content=item.get("content") or "",
                        references=parse_references(meta.get("references"))
                    ))

        if os.path.exists(chunk_file):
            with open(chunk_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for idx, item in enumerate(data):
                    meta = item.get("metadata") or {}
                    chunks.append(NormalisedChunk(
                        chunk_id=item.get("id") or f"{doc_name}_chunk_{idx}",
                        parent_chunk_id=meta.get("parent_chunk_id") or "",
                        content=item.get("content") or "",
                        summary=item.get("summary"),
                        sub_chunk_index=meta.get("chunk_index") or 0
                    ))

    return sections, chunks

async def ingest_all():
    await init_db_pool()
    sem = asyncio.Semaphore(4)  # Throttle embedding requests to avoid overloading Ollama

    async def get_embedding_throttled(text: str) -> List[float]:
        async with sem:
            return await generate_embedding(text)

    try:
        async with get_db_connection() as conn:
            # Clear target v3 database (clean slate)
            logger.info("cleaning_up_v3_tables")
            await conn.execute("TRUNCATE TABLE documents CASCADE")
            
            for doc_name, meta in DOC_METADATA.items():
                logger.info("ingesting_document_started", doc_name=doc_name)
                
                sections, chunks = await process_document_dumps(doc_name)
                logger.info("parsed_dumps", sections_count=len(sections), chunks_count=len(chunks))

                # 1. Insert Document Entry
                doc_id = uuid4()
                pub_date = datetime.strptime(meta["publish_date"], "%Y-%m-%d").date()
                await insert_document(
                    conn=conn,
                    doc_id=doc_id,
                    file_name=f"{doc_name}.pdf",
                    title=meta["title"],
                    publish_date=pub_date,
                    doc_type=meta["doc_type"],
                    file_hash=doc_name  # unique identifier for hash
                )

                # Map to keep track of section/chunk IDs to node info (UUID, level, breadcrumb)
                id_map: Dict[str, Tuple[UUID, int, str]] = {}
                nodes_to_insert = []
                rels_to_insert = []

                # 2. Create Document Root (Level 1)
                doc_root_id = uuid4()
                nodes_to_insert.append({
                    "node_id": doc_root_id,
                    "doc_id": doc_id,
                    "parent_id": None,
                    "level": 1,
                    "node_type": "DOCUMENT_ROOT",
                    "title": meta["title"],
                    "text_content": "",
                    "breadcrumb": meta["title"],
                    "needs_repair": False,
                    "embedding": None
                })

                # 3. Resolve and create Chapters (Level 2)
                # Group chapters to create unique chapter nodes
                chapters_set = set()
                for sec in sections:
                    chapters_set.add(clean_whitespace(sec.chapter_text))
                for chk in chunks:
                    # Chunks can also have chapter metadata
                    pass # but we primarily rely on section-level chapters or chunks nested under sections

                chapter_map: Dict[str, UUID] = {}
                for ch_name in chapters_set:
                    ch_id = uuid4()
                    nodes_to_insert.append({
                        "node_id": ch_id,
                        "doc_id": doc_id,
                        "parent_id": doc_root_id,
                        "level": 2,
                        "node_type": "CHAPTER",
                        "title": ch_name if ch_name else None,
                        "text_content": ch_name,
                        "breadcrumb": f"{meta['title']} > {ch_name}" if ch_name else f"{meta['title']} > Chapter Block",
                        "needs_repair": False,
                        "embedding": None
                    })
                    chapter_map[ch_name] = ch_id

                # 4. Generate Section Nodes (Level 3)
                embedding_tasks = []
                for sec in sections:
                    sec_node_id = uuid4()
                    ch_name = clean_whitespace(sec.chapter_text)
                    parent_ch_id = chapter_map.get(ch_name) or doc_root_id
                    ch_breadcrumb = f"{meta['title']} > {ch_name}" if ch_name else f"{meta['title']} > Chapter Block"

                    cleaned_title = clean_whitespace(sec.section_title)
                    cleaned_content = clean_whitespace(sec.content)
                    breadcrumb = f"{ch_breadcrumb} > {cleaned_title}"

                    # Save mappings
                    id_map[sec.section_id] = (sec_node_id, 3, breadcrumb)

                    # Embed: breadcrumb + "\n" + content
                    embed_text = f"{breadcrumb}\n{cleaned_content}"
                    embedding_tasks.append((len(nodes_to_insert), embed_text))

                    nodes_to_insert.append({
                        "node_id": sec_node_id,
                        "doc_id": doc_id,
                        "parent_id": parent_ch_id,
                        "level": 3,
                        "node_type": "SECTION",
                        "title": cleaned_title,
                        "text_content": cleaned_content,
                        "breadcrumb": breadcrumb,
                        "needs_repair": False,
                        "embedding": None
                    })

                    # Handle DEFINES_TERM relationships for definitions sections
                    if "definition" in cleaned_title.lower() or "section 3" in cleaned_title.lower() or "section 2" in cleaned_title.lower():
                        definitions = extract_definitions_from_text(cleaned_content)
                        for term, def_text in definitions:
                            rels_to_insert.append({
                                "source_node_id": sec_node_id,
                                "target_text_ref": term,
                                "rel_type": "DEFINES_TERM",
                                "effective_date": None
                            })

                    # Handle REFERS_TO relationships
                    for ref in sec.references:
                        rels_to_insert.append({
                            "source_node_id": sec_node_id,
                            "target_text_ref": ref,
                            "rel_type": "REFERS_TO",
                            "effective_date": None
                        })

                # 5. Generate Chunk Nodes (Level 4/5)
                # First, map parent references. We'll iteratively resolve chunk levels to handle any nesting
                unresolved_chunks = list(chunks)
                resolved_in_loop = True

                while unresolved_chunks and resolved_in_loop:
                    resolved_in_loop = False
                    still_unresolved = []

                    for chk in unresolved_chunks:
                        parent_id_str = chk.parent_chunk_id
                        
                        # Resolve parent UUID and parent level
                        parent_node_id = None
                        parent_level = 3 # default to section
                        parent_breadcrumb = ""

                        if parent_id_str in id_map:
                            parent_node_id, parent_level, parent_breadcrumb = id_map[parent_id_str]
                        else:
                            # Let's search if the parent is one of the section IDs or if we should fallback
                            # e.g., if parent_chunk_id matches section_id (which is mapped)
                            pass

                        if parent_node_id:
                            # Found parent, resolve chunk level
                            chunk_node_id = uuid4()
                            level = min(parent_level + 1, 6)
                            
                            cleaned_content = clean_whitespace(chk.content)
                            clause_title = extract_clause_title(cleaned_content)
                            breadcrumb = f"{parent_breadcrumb} > {clause_title}" if clause_title else f"{parent_breadcrumb} > Clause"

                            id_map[chk.chunk_id] = (chunk_node_id, level, breadcrumb)

                            # Embed: parent breadcrumb breadcrumb + "\n" + content
                            embed_text = f"{parent_breadcrumb}\n{cleaned_content}"
                            embedding_tasks.append((len(nodes_to_insert), embed_text))

                            nodes_to_insert.append({
                                "node_id": chunk_node_id,
                                "doc_id": doc_id,
                                "parent_id": parent_node_id,
                                "level": level,
                                "node_type": "CLAUSE",
                                "title": clause_title,
                                "text_content": cleaned_content,
                                "breadcrumb": breadcrumb,
                                "needs_repair": False,
                                "embedding": None
                            })
                            resolved_in_loop = True
                        else:
                            still_unresolved.append(chk)

                    unresolved_chunks = still_unresolved

                # For chunks that couldn't be resolved hierarchically, attach them to the document root or first section
                if unresolved_chunks:
                    logger.warning("attaching_unresolved_chunks_to_root", count=len(unresolved_chunks))
                    for chk in unresolved_chunks:
                        chunk_node_id = uuid4()
                        cleaned_content = clean_whitespace(chk.content)
                        clause_title = extract_clause_title(cleaned_content)
                        breadcrumb = f"{meta['title']} > Clause"

                        embed_text = cleaned_content
                        embedding_tasks.append((len(nodes_to_insert), embed_text))

                        nodes_to_insert.append({
                            "node_id": chunk_node_id,
                            "doc_id": doc_id,
                            "parent_id": doc_root_id,
                            "level": 4,
                            "node_type": "CLAUSE",
                            "title": clause_title,
                            "text_content": cleaned_content,
                            "breadcrumb": breadcrumb,
                            "needs_repair": False,
                            "embedding": None
                        })

                # 6. Generate embeddings concurrently
                logger.info("generating_embeddings", total=len(embedding_tasks))
                embedding_texts = [text[:6000] for _, text in embedding_tasks]
                
                # Execute in batches to prevent event loop delay or overloading
                embeddings_results = []
                batch_size = 50
                for i in range(0, len(embedding_texts), batch_size):
                    batch = embedding_texts[i:i+batch_size]
                    batch_results = await asyncio.gather(*[get_embedding_throttled(text) for text in batch])
                    embeddings_results.extend(batch_results)
                    logger.info("embeddings_progress", processed=len(embeddings_results), total=len(embedding_tasks))

                # Map embeddings back to nodes
                for idx, (node_idx, _) in enumerate(embedding_tasks):
                    nodes_to_insert[node_idx]["embedding"] = embeddings_results[idx]

                # 7. Write everything to DB
                logger.info("writing_ast_nodes_to_db", nodes_count=len(nodes_to_insert))
                await insert_ast_nodes(conn, nodes_to_insert)

                if rels_to_insert:
                    logger.info("writing_relationships_to_db", rels_count=len(rels_to_insert))
                    await insert_relationships(conn, rels_to_insert)

            # 8. Resolve references cross-document
            logger.info("resolving_pending_references_in_db")
            # We call the resolver functions inside the database connection
            # Wait, resolve_pending_references() creates its own pool if not called properly.
            # Let's call the resolve_pending_references function directly.
            
    finally:
        await close_db_pool()

    # Now let's run resolve_pending_references separately as it manages its own connection pool
    await resolve_pending_references()
    logger.info("ingestion_completed_successfully")

if __name__ == "__main__":
    asyncio.run(ingest_all())
