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
    gemini_client: GeminiClient,
    model: str,
    target: Dict[str, Any],
    distractors: List[Dict[str, Any]],
    semaphore: asyncio.Semaphore,
    is_negative: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Generates a single RAFT training example (messages structure) by querying Gemini.
    Supports negative sampling where the target clause is omitted from context.
    """
    if is_negative:
        # Negative sample: generate query for target, but omit target from context block.
        system_prompt = (
            "You are a dataset generator for a regulatory compliance RAG system.\n"
            "You return a JSON object with exactly one key: 'question'.\n"
            "The value of 'question' is a string containing a natural user question based on the target clause.\n"
            "To ensure the model generalizes well to different styles of user queries, vary your query styles. "
            "Some queries should be formal legal questions, others should be casual, procedural, or formulated as hypothetical scenarios (e.g. 'Can my company do X...')."
        )
        user_prompt = (
            "Based on the following target clause, generate one compliance question that a user might ask to learn about this rule.\n\n"
            "TARGET CLAUSE:\n"
            f"Breadcrumb: {target['breadcrumb']}\n"
            f"Title: {target['title']}\n"
            f"Content: {target['text_content']}\n\n"
            "Return a JSON object with key 'question' (string)."
        )
    else:
        # Positive sample: generate both query and structured answer.
        system_prompt = (
            "You are a dataset generator for a regulatory compliance RAG system.\n"
            "You return a JSON object with exactly two keys: 'question' and 'answer'.\n"
            "The value of 'question' is a string containing a natural user question.\n"
            "To ensure the model generalizes well to different styles of user queries, vary your query styles. "
            "Some queries should be formal legal questions, others should be casual, procedural, or formulated as hypothetical scenarios (e.g. 'Can my company do X...').\n"
            "The value of 'answer' MUST be a single string containing the exact markdown formatted text. Do NOT make the value of 'answer' a JSON object or dictionary.\n"
            "Your generated answer MUST contain all three of these parts:\n"
            "1. **Executive Summary:** A direct, user-friendly 1-2 sentence plain-English summary answering the question directly.\n"
            "2. **Key Requirements / Conditions:** A clean, bulleted list detailing all rules, thresholds, and conditions with natural inline citations (no tables).\n"
            "3. # Verbatim Regulatory Quote\n"
            "   > [Verbatim quotation of the key sentence from the target clause]\n\n"
            "Example output format:\n"
            "{\n"
            "  \"question\": \"What are the capital requirements for an IBU?\",\n"
            "  \"answer\": \"**Executive Summary:** The minimum capital requirement for an IFSC Banking Unit (IBU) is USD 20 million.\\n\\n**Key Requirements / Conditions:**\\n* As specified in Section 3(a) of the IFSC Banking Regulations, the IBU must maintain a minimum capital of USD 20 million at all times.\\n\\n# Verbatim Regulatory Quote\\n> The minimum capital requirement for an IFSC Banking Unit shall be USD 20 million.\"\n"
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
            content_str = await gemini_client.generate_content(
                model=model,
                system_instruction=system_prompt,
                user_content=user_prompt,
                temperature=0.4,
                json_mode=True
            )
            parsed = json.loads(content_str)
            question = parsed.get("question")
            
            if not question:
                return None
                
            if is_negative:
                answer = "I do not know the answer as no regulation was found in the available corpus."
                # Format contexts (only distractors)
                dist1_ctx = f"[DISTRACTOR: {distractors[0]['doc_title']}, {distractors[0]['breadcrumb']}: '{distractors[0]['text_content']}']"
                dist2_ctx = f"[DISTRACTOR: {distractors[1]['doc_title']}, {distractors[1]['breadcrumb']}: '{distractors[1]['text_content']}']"
                context_list = [dist1_ctx, dist2_ctx]
            else:
                answer = parsed.get("answer")
                if not answer:
                    return None
                # Format contexts (target + distractors)
                target_ctx = f"[TARGET: {target['doc_title']}, {target['breadcrumb']}: '{target['text_content']}']"
                dist1_ctx = f"[DISTRACTOR: {distractors[0]['doc_title']}, {distractors[0]['breadcrumb']}: '{distractors[0]['text_content']}']"
                dist2_ctx = f"[DISTRACTOR: {distractors[1]['doc_title']}, {distractors[1]['breadcrumb']}: '{distractors[1]['text_content']}']"
                context_list = [target_ctx, dist1_ctx, dist2_ctx]
            
            # Shuffle contexts
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
    parser = argparse.ArgumentParser(description="Generate synthetic RAFT training dataset from PostgreSQL using Gemini API.")
    parser.add_argument("--count", type=int, default=2000, help="Number of training pairs to generate.")
    parser.add_argument("--model", type=str, default=GEMINI_MODEL, help="Gemini model to use.")
    parser.add_argument("--output", type=str, default="data/raft_training_pairs.jsonl", help="Output JSONL filepath.")
    parser.add_argument("--concurrency", type=int, default=20, help="Number of parallel generation tasks.")
    parser.add_argument("--resume", action="store_true", help="Resume generation from the existing output file if it exists.")
    args = parser.parse_args()
    
    logger.info("synthetic_generation_started", target_count=args.count, model=args.model, output_file=args.output)
    
    # Check for Gemini keys
    if not GEMINI_API_KEYS:
        logger.error("missing_gemini_keys", error="GEMINI_API_KEYS is not defined or is empty in config/.env")
        print("ERROR: GEMINI_API_KEYS environment variable is not defined or is empty in .env.", file=sys.stderr)
        print("Please configure at least one Gemini API key before running this script.", file=sys.stderr)
        sys.exit(1)
        
    logger.info("gemini_keys_loaded", count=len(GEMINI_API_KEYS))
    
    # Initialize connection pool
    await init_db_pool()
    
    # Initialize Rotated rate-limited Gemini Client
    # 15 RPM is the free tier limit per key
    gemini_client = GeminiClient(api_keys=GEMINI_API_KEYS, rpm_limit=10)
    
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
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # Determine file mode and starting count
        mode = "w"
        existing_count = 0
        if os.path.exists(args.output):
            if args.resume:
                try:
                    with open(args.output, "r", encoding="utf-8") as infile:
                        existing_count = sum(1 for _ in infile if _.strip())
                    mode = "a"
                    logger.info("resuming_generation", existing_count=existing_count)
                except Exception as e:
                    logger.warning("failed_to_read_existing_file_starting_fresh", error=str(e))
                    mode = "w"
            else:
                logger.warning("output_file_exists_overwriting", filepath=args.output)
                
        if existing_count >= args.count:
            logger.info("dataset_already_complete", existing_count=existing_count, target_count=args.count)
            return
            
        semaphore = asyncio.Semaphore(args.concurrency)
        generated_count = existing_count
        tasks = []
        
        # We sample target nodes to match the desired count
        random.seed(42)
        sampled_targets = random.choices(targets, k=args.count)
        
        # If resuming, slice targets to process only remaining ones
        if existing_count > 0:
            sampled_targets = sampled_targets[existing_count:]
            
        # Build tasks
        for t in sampled_targets:
            # Select 2 distractors that are not the same node, with a bounded retry limit
            max_tries = 20
            for _ in range(max_tries):
                dists = random.sample(distractors_all, k=2)
                if dists[0]["node_id"] != t["node_id"] and dists[1]["node_id"] != t["node_id"]:
                    break
            else:
                # Fall back to any two nodes that are not the target node
                dists = [d for d in distractors_all if d["node_id"] != t["node_id"]][:2]
                if len(dists) < 2:
                    logger.warning("distractor_pool_too_small_skipping", node_id=str(t["node_id"]))
                    continue
            
            # 18% probability of negative sampling (target omitted)
            is_negative = random.random() < 0.18
            
            task = generate_single_pair(
                gemini_client=gemini_client,
                model=args.model,
                target=t,
                distractors=dists,
                semaphore=semaphore,
                is_negative=is_negative
            )
            tasks.append(task)
            
        logger.info("launching_generation_jobs", num_jobs=len(tasks))
        
        # Open output file
        with open(args.output, mode, encoding="utf-8") as outfile:
            completed_jobs = 0
            for future in asyncio.as_completed(tasks):
                result = await future
                completed_jobs += 1
                
                if result:
                    outfile.write(json.dumps(result, ensure_ascii=False) + "\n")
                    outfile.flush()
                    generated_count += 1
                    
                if completed_jobs % 10 == 0 or completed_jobs == len(tasks):
                    logger.info(
                        "progress_report",
                        completed=completed_jobs,
                        total=len(tasks),
                        successfully_generated=generated_count
                    )
                    
        logger.info("synthetic_generation_completed", successfully_generated=generated_count, file_saved=args.output)
        
    except Exception as e:
        logger.error("synthetic_generation_failed", error=str(e))
        sys.exit(1)
    finally:
        await gemini_client.close()
        await close_db_pool()

if __name__ == "__main__":
    asyncio.run(main())
