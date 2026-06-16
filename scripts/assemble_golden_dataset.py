import argparse
import asyncio
import os
import sys
import json
import random
from typing import Any, Dict, List, Optional
import structlog

# Add project root to python path so backend can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.database.connection import init_db_pool, get_db_connection, close_db_pool
from ollama import AsyncClient
from backend.config import OLLAMA_HOST

logger = structlog.get_logger()

# Configure basic structured logs
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ]
)

async def generate_golden_pair(
    ollama_client: AsyncClient,
    model: str,
    category: str,
    context_text: str,
    breadcrumb: str,
    target_node_id: str,
    semaphore: asyncio.Semaphore
) -> Optional[Dict[str, Any]]:
    """
    Generates a single golden QA pair using the local LLM.
    
    Args:
        ollama_client (AsyncClient): Ollama async client.
        model (str): LLM model name.
        category (str): Category classification for this question.
        context_text (str): Regulatory text context.
        breadcrumb (str): Section breadcrumb/title.
        target_node_id (str): Target node UUID.
        semaphore (asyncio.Semaphore): Concurrency controller.
        
    Returns:
        Optional[Dict[str, Any]]: Golden dataset entry, or None on failure.
    """
    system_prompt = (
        "You are an expert regulatory compliance dataset builder.\n"
        "Given the regulatory text, generate a single high-quality question and its correct plain-English answer.\n"
        "The question must be natural and realistic, typical of what a compliance officer or auditor would ask.\n"
        "The answer must be complete, precise, and directly supported by the provided text.\n"
        "Return a JSON object with exactly two keys: 'query' (string) and 'golden_answer' (string)."
    )
    
    user_prompt = (
        f"CATEGORY: {category}\n"
        f"BREADCRUMB: {breadcrumb}\n"
        f"CONTEXT TEXT:\n{context_text}\n\n"
        "Generate a QA pair. Output strictly JSON with keys 'query' and 'golden_answer'."
    )
    
    async with semaphore:
        try:
            response = await ollama_client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                format="json",
                options={"temperature": 0.2}
            )
            content_str = response.message.content
            parsed = json.loads(content_str)
            
            query = parsed.get("query")
            golden_answer = parsed.get("golden_answer")
            
            if not query or not golden_answer:
                return None
                
            return {
                "query": query,
                "golden_answer": golden_answer,
                "target_node_id": target_node_id,
                "category": category,
                "breadcrumb": breadcrumb
            }
        except Exception as e:
            logger.debug("generate_golden_pair_failed", error=str(e), node_id=target_node_id)
            return None

async def assemble_dataset(ollama_model: str, output_path: str, concurrency: int) -> None:
    """
    Queries the database for candidate nodes across all 5 evaluation categories,
    invokes Ollama to generate golden QA pairs, and saves them to a JSON file.
    
    Args:
        ollama_model (str): Ollama model name.
        output_path (str): Output filepath.
        concurrency (int): Max concurrent LLM requests.
    """
    logger.info("querying_candidates_from_database")
    
    async with get_db_connection() as conn:
        # Category 1: Direct regulation questions (target 30)
        direct_rows = await conn.fetch(
            """
            SELECT node_id, text_content, breadcrumb
            FROM ast_nodes
            WHERE text_content IS NOT NULL AND length(text_content) > 200 AND needs_repair = FALSE
            ORDER BY RANDOM()
            LIMIT 40
            """
        )
        
        # Category 2: Cross-reference questions (target 25)
        cross_rows = await conn.fetch(
            """
            SELECT r.source_node_id as node_id, n1.text_content, n1.breadcrumb, n2.text_content as target_text, n2.breadcrumb as target_breadcrumb
            FROM relationships r
            JOIN ast_nodes n1 ON r.source_node_id = n1.node_id
            JOIN ast_nodes n2 ON r.target_node_id = n2.node_id
            WHERE r.rel_type = 'REFERS_TO' AND n1.text_content IS NOT NULL AND n2.text_content IS NOT NULL
            ORDER BY RANDOM()
            LIMIT 30
            """
        )
        
        # Category 3: Glossary-term questions (target 20)
        glossary_rows = await conn.fetch(
            """
            SELECT g.source_node_id as node_id, g.definition as text_content, n.breadcrumb || ' > ' || g.term as breadcrumb
            FROM glossary g
            JOIN ast_nodes n ON g.source_node_id = n.node_id
            WHERE g.definition IS NOT NULL AND length(g.definition) > 80
            ORDER BY RANDOM()
            LIMIT 25
            """
        )
        
        # Category 4: Amendment/temporal questions (target 15)
        # First try SUBSTITUTES relationships
        sub_rows = await conn.fetch(
            """
            SELECT r.source_node_id as node_id, n1.text_content, n1.breadcrumb
            FROM relationships r
            JOIN ast_nodes n1 ON r.source_node_id = n1.node_id
            WHERE r.rel_type = 'SUBSTITUTES' AND n1.text_content IS NOT NULL
            ORDER BY RANDOM()
            LIMIT 20
            """
        )
        
        # If not enough, fetch from amendment documents
        needed_amendments = 20 - len(sub_rows)
        amend_rows = []
        if needed_amendments > 0:
            amend_rows = await conn.fetch(
                """
                SELECT n.node_id, n.text_content, n.breadcrumb
                FROM ast_nodes n
                JOIN documents d ON n.doc_id = d.doc_id
                WHERE d.is_active = TRUE AND (d.file_name LIKE '%amendment%' OR d.title LIKE '%amendment%' OR d.title LIKE '%Amendment%')
                  AND n.text_content IS NOT NULL AND length(n.text_content) > 150
                ORDER BY RANDOM()
                LIMIT $1
                """,
                needed_amendments
            )
            
        amendment_candidates = [dict(r) for r in sub_rows] + [dict(r) for r in amend_rows]
        
        # Category 5: Compliance check questions (target 10)
        compliance_rows = await conn.fetch(
            """
            SELECT n.node_id, n.text_content, n.breadcrumb, d.title as doc_title
            FROM ast_nodes n
            JOIN documents d ON n.doc_id = d.doc_id
            WHERE d.is_active = TRUE 
              AND (d.file_name LIKE '%GIC%' OR d.file_name LIKE '%Techfin%' OR d.file_name LIKE '%Sandbox%' 
                   OR d.title LIKE '%GIC%' OR d.title LIKE '%Techfin%' OR d.title LIKE '%Sandbox%')
              AND n.text_content IS NOT NULL AND length(n.text_content) > 150
            ORDER BY RANDOM()
            LIMIT 15
            """
        )

    logger.info("fetched_all_candidate_rows", 
                direct=len(direct_rows), 
                cross=len(cross_rows), 
                glossary=len(glossary_rows), 
                amendments=len(amendment_candidates), 
                compliance=len(compliance_rows))

    ollama_client = AsyncClient(host=OLLAMA_HOST)
    semaphore = asyncio.Semaphore(concurrency)
    
    tasks = []
    
    # 1. Queue Direct (target 30)
    for r in direct_rows[:30]:
        tasks.append((
            "Direct Regulation",
            r["text_content"],
            r["breadcrumb"],
            str(r["node_id"])
        ))
        
    # 2. Queue Cross-Reference (target 25)
    for r in cross_rows[:25]:
        ctx_text = f"Source Section: {r['text_content']}\nReferenced Section: {r['target_text']}"
        tasks.append((
            "Cross-Reference",
            ctx_text,
            f"{r['breadcrumb']} -> {r['target_breadcrumb']}",
            str(r["node_id"])
        ))
        
    # 3. Queue Glossary (target 20)
    for r in glossary_rows[:20]:
        tasks.append((
            "Glossary Definition",
            r["text_content"],
            r["breadcrumb"],
            str(r["node_id"])
        ))
        
    # 4. Queue Amendments (target 15)
    for r in amendment_candidates[:15]:
        tasks.append((
            "Amendment/Temporal",
            r["text_content"],
            r["breadcrumb"],
            str(r["node_id"])
        ))
        
    # 5. Queue Compliance (target 10)
    for r in compliance_rows[:10]:
        tasks.append((
            "Compliance Check",
            r["text_content"],
            f"{r.get('doc_title', 'Regulations')} > {r['breadcrumb']}",
            str(r["node_id"])
        ))

    logger.info("launching_qa_generation_jobs", count=len(tasks))
    
    # Create future tasks
    future_tasks = [
        generate_golden_pair(
            ollama_client=ollama_client,
            model=ollama_model,
            category=cat,
            context_text=ctx,
            breadcrumb=bread,
            target_node_id=nid,
            semaphore=semaphore
        )
        for cat, ctx, bread, nid in tasks
    ]
    
    results = []
    completed = 0
    for future in asyncio.as_completed(future_tasks):
        res = await future
        completed += 1
        if res:
            results.append(res)
            
        if completed % 10 == 0 or completed == len(future_tasks):
            logger.info("generation_progress", completed=completed, total=len(future_tasks), generated=len(results))
            
    logger.info("writing_golden_dataset", count=len(results), path=output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    logger.info("golden_dataset_assembled_successfully", count=len(results))

async def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble golden dataset from DB candidate nodes.")
    parser.add_argument("--model", type=str, default="llama3.2:3b", help="Ollama model to use.")
    parser.add_argument("--output", type=str, default="tests/golden_dataset.json", help="Output JSON path.")
    parser.add_argument("--concurrency", type=int, default=8, help="Ollama concurrency limit.")
    args = parser.parse_args()
    
    await init_db_pool()
    try:
        await assemble_dataset(args.model, args.output, args.concurrency)
    finally:
        await close_db_pool()

if __name__ == "__main__":
    asyncio.run(main())
