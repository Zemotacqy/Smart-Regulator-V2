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

async def fetch_target_nodes() -> List[Dict[str, Any]]:
    """
    Fetches candidate target nodes from the PostgreSQL database.
    Excludes test ingestion documents and nodes marked for repair.
    
    Returns:
        List[Dict[str, Any]]: List of dictionaries containing target node data.
    """
    query = """
        SELECT n.node_id, n.doc_id, n.title, n.text_content, n.breadcrumb, d.title as doc_title
        FROM ast_nodes n
        JOIN documents d ON n.doc_id = d.doc_id
        WHERE d.is_active = TRUE
          AND d.file_name NOT LIKE '%test_ingest%'
          AND n.text_content IS NOT NULL
          AND n.needs_repair = FALSE
          AND length(n.text_content) >= 150
    """
    async with get_db_connection() as conn:
        rows = await conn.fetch(query)
        return [dict(r) for r in rows]

async def fetch_all_distractor_nodes(target_node_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Fetches candidate distractor nodes from the database.
    
    Args:
        target_node_ids (List[str]): List of node IDs used as targets to exclude if needed.
        
    Returns:
        List[Dict[str, Any]]: List of distractor nodes.
    """
    query = """
        SELECT n.node_id, n.title, n.text_content, n.breadcrumb, d.title as doc_title
        FROM ast_nodes n
        JOIN documents d ON n.doc_id = d.doc_id
        WHERE d.is_active = TRUE
          AND d.file_name NOT LIKE '%test_ingest%'
          AND n.text_content IS NOT NULL
          AND n.needs_repair = FALSE
          AND length(n.text_content) >= 100
    """
    async with get_db_connection() as conn:
        rows = await conn.fetch(query)
        return [dict(r) for r in rows]

async def generate_single_pair(
    ollama_client: AsyncClient,
    model: str,
    target: Dict[str, Any],
    distractors: List[Dict[str, Any]],
    semaphore: asyncio.Semaphore
) -> Optional[Dict[str, Any]]:
    """
    Generates a single RAFT training example (messages structure) by querying Ollama.
    
    Args:
        ollama_client (AsyncClient): Async Ollama API client.
        model (str): Name of the local LLM model to query.
        target (Dict[str, Any]): Target node information.
        distractors (List[Dict[str, Any]]): List of 2 distractor nodes.
        semaphore (asyncio.Semaphore): Concurrency controller.
        
    Returns:
        Optional[Dict[str, Any]]: Structured training dict, or None on failure.
    """
    system_prompt = (
        "You are a dataset generator for a regulatory compliance RAG system.\n"
        "You return a JSON object with exactly two keys: 'question' and 'answer'.\n"
        "The value of 'question' is a string containing a natural user question.\n"
        "The value of 'answer' MUST be a single string containing the exact markdown formatted text with headers, table, and source quotes. Do NOT make the value of 'answer' a JSON object or dictionary.\n"
        "Your generated answer MUST contain all four of these parts:\n"
        "1. **Short Answer:** [Direct plain-English answer, max 2 sentences]\n"
        "2. A markdown table with columns 'Rule', 'Detail', and 'Source' mapping the key fact(s) from the target clause.\n"
        "3. **Plain Language:** [Simple explanation of the rule, explaining any complex legal terms simply]\n"
        "4. **Source Quote:** \"[Verbatim quotation of the key sentence from the target clause]\"\n\n"
        "Example output format:\n"
        "{\n"
        "  \"question\": \"What are the capital requirements for an IBU?\",\n"
        "  \"answer\": \"**Short Answer:** The minimum capital is USD 20 million.\\n\\n| Rule | Detail | Source |\\n|---|---|---|\\n| Capital Requirement | Minimum USD 20 million | Section 3(a) |\\n\\n**Plain Language:** An IBU must have at least USD 20 million in capital to operate.\\n\\n**Source Quote:** \\\"The minimum capital requirement for an IFSC Banking Unit shall be USD 20 million.\\\"\"\n"
        "}"
    )

    user_prompt = (
        "Based on the following target clause, generate one compliance question and its detailed answer string.\n\n"
        "TARGET CLAUSE:\n"
        f"Breadcrumb: {target['breadcrumb']}\n"
        f"Title: {target['title']}\n"
        f"Content: {target['text_content']}\n\n"
        "Ensure the generated answer uses ONLY facts from the target clause.\n"
        "Return a JSON object with keys 'question' (string) and 'answer' (string)."
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
                options={"temperature": 0.4}
            )
            content_str = response.message.content
            parsed = json.loads(content_str)
            
            question = parsed.get("question")
            answer = parsed.get("answer")
            
            if not question or not answer:
                return None
                
            # If the model returned the answer as a dictionary (sometimes occurs despite prompt), format it.
            if isinstance(answer, dict):
                parts = []
                short_ans = answer.get("**Short Answer**:") or answer.get("Short Answer") or answer.get("**Short Answer**") or ""
                if short_ans:
                    parts.append(f"**Short Answer:** {short_ans}")
                    
                table_lines = []
                for k, v in answer.items():
                    if "Rule" in k or "table" in k:
                        if isinstance(v, list):
                            table_lines.extend(v)
                        elif isinstance(v, str):
                            table_lines.append(v)
                if table_lines:
                    parts.append("\n".join(table_lines))
                    
                plain_lang = answer.get("**Plain Language**:") or answer.get("Plain Language") or answer.get("**Plain Language**") or ""
                if plain_lang:
                    parts.append(f"**Plain Language:** {plain_lang}")
                    
                quote = answer.get("**Source Quote**:") or answer.get("Source Quote") or answer.get("**Source Quote**") or ""
                if quote:
                    parts.append(f"**Source Quote:** \"{quote}\"")
                    
                answer = "\n\n".join(parts)
                
            # Build Context list
            contexts = []
            
            # Format targets and distractors
            target_ctx = f"[TARGET: {target['doc_title']}, {target['breadcrumb']}: '{target['text_content']}']"
            dist1_ctx = f"[DISTRACTOR: {distractors[0]['doc_title']}, {distractors[0]['breadcrumb']}: '{distractors[0]['text_content']}']"
            dist2_ctx = f"[DISTRACTOR: {distractors[1]['doc_title']}, {distractors[1]['breadcrumb']}: '{distractors[1]['text_content']}']"
            
            # Shuffle targets and distractors
            context_list = [target_ctx, dist1_ctx, dist2_ctx]
            random.shuffle(context_list)
            
            context_str = "CONTEXT:\n" + "\n".join(context_list)
            
            # Assemble RAFT conversation dict
            raft_item = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"You are the IFSCA Regulatory Assistant.\n\n{context_str}\n\nQUESTION: {question}"
                    },
                    {
                        "role": "assistant",
                        "content": answer
                    }
                ]
            }
            return raft_item
            
        except Exception as e:
            logger.debug("generation_single_pair_failed", error=str(e), target_id=str(target.get("node_id")))
            return None

async def main() -> None:
    """
    Main entry point for the synthetic data generation script.
    Parses arguments, queries database, launches concurrent generations,
    and writes outputs to JSONL format.
    """
    parser = argparse.ArgumentParser(description="Generate synthetic RAFT training dataset from PostgreSQL.")
    parser.add_argument("--count", type=int, default=2000, help="Number of training pairs to generate.")
    parser.add_argument("--model", type=str, default="llama3.2:3b", help="Ollama model to use.")
    parser.add_argument("--output", type=str, default="data/raft_training_pairs.jsonl", help="Output JSONL filepath.")
    parser.add_argument("--concurrency", type=int, default=8, help="Number of parallel generation tasks.")
    args = parser.parse_args()
    
    logger.info("synthetic_generation_started", target_count=args.count, model=args.model, output_file=args.output)
    
    # Initialize connection pool
    await init_db_pool()
    
    try:
        # Fetch targets and distractors
        targets = await fetch_target_nodes()
        if not targets:
            logger.error("no_target_nodes_found")
            sys.exit(1)
            
        logger.info("fetched_candidate_nodes", count=len(targets))
        
        distractors_all = await fetch_all_distractor_nodes([str(t["node_id"]) for t in targets])
        if len(distractors_all) < 2:
            logger.error("insufficient_distractor_nodes", count=len(distractors_all))
            sys.exit(1)
            
        # Create output directory
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        ollama_client = AsyncClient(host=OLLAMA_HOST)
        semaphore = asyncio.Semaphore(args.concurrency)
        
        generated_count = 0
        tasks = []
        
        # We sample target nodes to match the desired count
        # Each target can be used multiple times if count > len(targets)
        random.seed(42)
        sampled_targets = random.choices(targets, k=args.count)
        
        # Build tasks
        for t in sampled_targets:
            # Select 2 distractors that are not the same node
            dists = random.sample(distractors_all, k=2)
            while dists[0]["node_id"] == t["node_id"] or dists[1]["node_id"] == t["node_id"]:
                dists = random.sample(distractors_all, k=2)
            
            task = generate_single_pair(
                ollama_client=ollama_client,
                model=args.model,
                target=t,
                distractors=dists,
                semaphore=semaphore
            )
            tasks.append(task)
            
        logger.info("launching_generation_jobs", num_jobs=len(tasks))
        
        # Open output file in append/write mode
        with open(args.output, "w", encoding="utf-8") as outfile:
            # Process tasks as they complete
            completed_jobs = 0
            for future in asyncio.as_completed(tasks):
                result = await future
                completed_jobs += 1
                
                if result:
                    outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
                    outfile.flush()
                    generated_count += 1
                    
                if completed_jobs % 10 == 0 or completed_jobs == len(tasks):
                    logger.info("progress_report", completed=completed_jobs, total=len(tasks), successfully_generated=generated_count)
                    
        logger.info("synthetic_generation_completed", successfully_generated=generated_count, file_saved=args.output)
        
    except Exception as e:
        logger.error("synthetic_generation_failed", error=str(e))
        sys.exit(1)
    finally:
        await close_db_pool()

if __name__ == "__main__":
    asyncio.run(main())
