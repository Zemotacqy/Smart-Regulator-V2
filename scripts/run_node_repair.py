"""
run_node_repair.py — Re-process AST nodes flagged with needs_repair = TRUE.

For each flagged node, this script:
  1. Reads its current text_content and breadcrumb from the database.
  2. Re-generates the dense embedding using the configured embedding model.
  3. Updates the embedding in the database and resets needs_repair = FALSE.

The script does NOT delete any rows. It operates with surgical UPDATE statements
on specific node_ids only.

Usage:
  python scripts/run_node_repair.py --dry-run      # List flagged nodes, no changes
  python scripts/run_node_repair.py                 # Apply repairs
  python scripts/run_node_repair.py --doc-id <UUID> # Scope to one document
"""

import argparse
import asyncio
import os
import sys
from uuid import UUID
from typing import List, Optional

import structlog

# Add project root to python path so backend can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.database.connection import init_db_pool, get_db_connection, close_db_pool
from backend.rag.extraction.llm_client import generate_embedding

# Configure structured console logging for CLI output
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ]
)
logger = structlog.get_logger()


async def fetch_repair_nodes(doc_id: Optional[UUID] = None) -> List[dict]:
    """
    Fetches all AST nodes with needs_repair = TRUE.
    If doc_id is provided, scopes the query to that document only.

    Returns:
        List of dicts with node_id, doc_id, node_type, breadcrumb, text_content, file_name.
    """
    base_query = """
        SELECT n.node_id, n.doc_id, n.node_type, n.breadcrumb, n.text_content, d.file_name
        FROM ast_nodes n
        JOIN documents d ON n.doc_id = d.doc_id
        WHERE n.needs_repair = TRUE
    """
    params = []
    if doc_id:
        base_query += " AND n.doc_id = $1"
        params.append(doc_id)

    async with get_db_connection() as conn:
        rows = await conn.fetch(base_query, *params)
        return [dict(r) for r in rows]


async def repair_node(node: dict, dry_run: bool) -> bool:
    """
    Re-generates the embedding for a single repair node and updates the DB.

    Args:
        node: Dict with node_id, breadcrumb, text_content fields.
        dry_run: If True, logs the action but does not update the DB.

    Returns:
        True if repair succeeded (or simulated), False on failure.
    """
    node_id = node["node_id"]
    breadcrumb = node["breadcrumb"] or ""
    text_content = node["text_content"] or ""

    logger.info(
        "repairing_node",
        node_id=str(node_id),
        node_type=node["node_type"],
        file_name=node["file_name"],
        breadcrumb=breadcrumb[:80],
        dry_run=dry_run
    )

    if not text_content.strip():
        logger.warning(
            "repair_skipped_empty_text",
            node_id=str(node_id),
            breadcrumb=breadcrumb[:80]
        )
        return False

    try:
        # Re-generate embedding using breadcrumb + text for structural context
        contextual_text = f"{breadcrumb}\n\n{text_content}"
        embedding = await generate_embedding(contextual_text)

        if dry_run:
            logger.info(
                "repair_dry_run_would_update",
                node_id=str(node_id),
                embedding_dim=len(embedding) if embedding else 0
            )
            return True

        # Apply UPDATE — embedding refresh + clear needs_repair flag
        async with get_db_connection() as conn:
            await conn.execute(
                """
                UPDATE ast_nodes
                SET embedding = $1::vector,
                    needs_repair = FALSE
                WHERE node_id = $2
                """,
                str(embedding),
                node_id
            )
        logger.info("node_repaired_successfully", node_id=str(node_id))
        return True

    except Exception as e:
        logger.error("node_repair_failed", node_id=str(node_id), error=str(e))
        return False


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-process AST nodes flagged with needs_repair = TRUE."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="List flagged nodes and simulate repairs without writing to the database."
    )
    parser.add_argument(
        "--doc-id",
        type=str,
        default=None,
        help="Optional: scope repairs to a specific document UUID."
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Max concurrent embedding generation calls (default: 2)."
    )
    args = parser.parse_args()

    doc_id: Optional[UUID] = None
    if args.doc_id:
        try:
            doc_id = UUID(args.doc_id)
        except ValueError:
            logger.error("invalid_doc_id_format", provided=args.doc_id)
            sys.exit(1)

    logger.info(
        "node_repair_started",
        dry_run=args.dry_run,
        doc_id=str(doc_id) if doc_id else "all"
    )

    # Initialise the DB connection pool
    await init_db_pool()

    try:
        nodes = await fetch_repair_nodes(doc_id)

        if not nodes:
            logger.info("no_repair_nodes_found")
            print("\n✅  No nodes with needs_repair = TRUE found. Corpus is clean.")
            return

        print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Found {len(nodes)} node(s) requiring repair:\n")
        for n in nodes:
            print(f"  • [{n['node_type']}] {n['breadcrumb'][:100]}")
            print(f"    node_id: {n['node_id']} | file: {n['file_name']}")
        print()

        if args.dry_run:
            print("[DRY RUN] No changes written. Re-run without --dry-run to apply repairs.\n")
            return

        # Apply repairs with bounded concurrency
        semaphore = asyncio.Semaphore(args.concurrency)

        async def _bounded_repair(node: dict) -> bool:
            async with semaphore:
                return await repair_node(node, dry_run=False)

        results = await asyncio.gather(*[_bounded_repair(n) for n in nodes])

        succeeded = sum(1 for r in results if r)
        failed = len(results) - succeeded

        print(f"\n{'='*50}")
        print(f"  Node Repair Complete")
        print(f"  Repaired:  {succeeded} / {len(nodes)}")
        print(f"  Failed:    {failed}")
        print(f"{'='*50}\n")

        if failed > 0:
            logger.warning("some_repairs_failed", failed_count=failed)
            sys.exit(1)

    except Exception as e:
        logger.error("node_repair_script_failed", error=str(e))
        sys.exit(1)
    finally:
        await close_db_pool()


if __name__ == "__main__":
    asyncio.run(main())
