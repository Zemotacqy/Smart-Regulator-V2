import argparse
import asyncio
import os
import sys
import json
import random
import time
from typing import Any, Dict, List, Optional
import structlog

# Add project root to python path so backend can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.database.connection import init_db_pool, get_db_connection, close_db_pool
from backend.config import GEMINI_API_KEYS, GEMINI_MODEL
from scripts.utils.gemini_client import GeminiClient

logger = structlog.get_logger()

# Configure basic structured logs
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ]
)

async def fetch_node_text(node_id: str) -> Optional[Dict[str, Any]]:
    """Fetches the node's text content and context from the PostgreSQL database."""
    query = """
        SELECT n.node_id, n.text_content, n.breadcrumb, d.title as doc_title
        FROM ast_nodes n
        JOIN documents d ON n.doc_id = d.doc_id
        WHERE n.node_id = $1::uuid
    """
    async with get_db_connection() as conn:
        row = await conn.fetchrow(query, node_id)
        return dict(row) if row else None

async def rewrite_golden_answer(
    gemini_client: GeminiClient,
    model: str,
    query: str,
    original_golden: str,
    node_ctx: Dict[str, Any],
    semaphore: asyncio.Semaphore
) -> Optional[str]:
    """Uses Gemini API to reformat a single golden answer into the new 3-part layout."""
    system_prompt = (
        "You are a regulatory text reformatting assistant.\n"
        "Your task is to rewrite the original golden answer to fit a clean, structured RAG format.\n"
        "You MUST structure the output into exactly three parts:\n"
        "1. **Executive Summary:** A direct, user-friendly 1-2 sentence plain-English summary answering the question directly.\n"
        "2. **Key Requirements / Conditions:** A clean, bulleted list detailing all rules, thresholds, and conditions with natural inline citations (no tables).\n"
        "3. # Verbatim Regulatory Quote\n"
        "   > [Verbatim quotation of the key sentence from the target clause]\n\n"
        "Strictly use the provided Context clause to extract the verbatim quote. Do not invent details."
    )
    
    user_prompt = (
        f"QUESTION: {query}\n\n"
        f"ORIGINAL GOLDEN ANSWER: {original_golden}\n\n"
        f"CONTEXT CLAUSE:\n"
        f"Document: {node_ctx['doc_title']}\n"
        f"Section: {node_ctx['breadcrumb']}\n"
        f"Content: {node_ctx['text_content']}\n\n"
        "Rewrite the original answer into the 3-part format. Return ONLY the rewritten markdown string."
    )

    async with semaphore:
        try:
            rewritten = await gemini_client.generate_content(
                model=model,
                system_instruction=system_prompt,
                user_content=user_prompt,
                temperature=0.0,
                json_mode=False
            )
            return rewritten.strip()
        except Exception as e:
            logger.error("rewrite_golden_answer_failed", query=query, error=str(e))
            return None

async def process_item(
    item: Dict[str, Any],
    gemini_client: GeminiClient,
    model: str,
    semaphore: asyncio.Semaphore
) -> Dict[str, Any]:
    """Processes a single item and returns the updated item, preserving order."""
    node_id = item.get("target_node_id")
    if not node_id:
        return item
        
    node_ctx = await fetch_node_text(node_id)
    if not node_ctx:
        logger.warning("target_node_not_found_in_db", node_id=node_id)
        return item
        
    rewritten = await rewrite_golden_answer(
        gemini_client=gemini_client,
        model=model,
        query=item["query"],
        original_golden=item["golden_answer"],
        node_ctx=node_ctx,
        semaphore=semaphore
    )
    
    if rewritten:
        new_item = item.copy()
        new_item["golden_answer"] = rewritten
        return new_item
    else:
        return item

async def main() -> None:
    parser = argparse.ArgumentParser(description="Reformat golden dataset reference answers to the new 3-section structured format.")
    parser.add_argument("--dataset", type=str, default="tests/golden_dataset.json", help="Path to golden dataset JSON file.")
    parser.add_argument("--output", type=str, help="Path to save the aligned dataset. Defaults to the value of --dataset (overwrite).")
    parser.add_argument("--concurrency", type=int, default=15, help="Number of parallel API workers.")
    parser.add_argument("--model", type=str, default=GEMINI_MODEL, help="Gemini model to use.")
    args = parser.parse_args()

    golden_path = args.dataset
    output_path = args.output if args.output else args.dataset

    if not os.path.exists(golden_path):
        print(f"ERROR: Golden dataset not found at {golden_path}", file=sys.stderr)
        sys.exit(1)
        
    if not GEMINI_API_KEYS:
        print("ERROR: GEMINI_API_KEYS is not defined in .env.", file=sys.stderr)
        sys.exit(1)

    print(f"Reading golden dataset from {golden_path}...")
    with open(golden_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    print(f"Found {len(dataset)} items to align.")
    
    # Initialize DB pool
    await init_db_pool()
    
    gemini_client = GeminiClient(api_keys=GEMINI_API_KEYS, rpm_limit=10)
    semaphore = asyncio.Semaphore(args.concurrency)
    
    try:
        print("Launching alignment jobs...")
        jobs = [process_item(item, gemini_client, args.model, semaphore) for item in dataset]
        aligned_dataset = await asyncio.gather(*jobs)
        
        # Write back to file (possibly with backup first if overwriting)
        if output_path == golden_path:
            backup_path = golden_path + ".bak"
            with open(backup_path, "w", encoding="utf-8") as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            print(f"Backup saved to {backup_path}")
            
        # Ensure parent directory of output exists
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        print(f"Writing aligned dataset to {output_path}...")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(aligned_dataset, f, ensure_ascii=False, indent=2)
            
        print("Golden dataset alignment completed successfully!")
        
    finally:
        await gemini_client.close()
        await close_db_pool()

if __name__ == "__main__":
    asyncio.run(main())
