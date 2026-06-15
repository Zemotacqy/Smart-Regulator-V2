import argparse
import asyncio
import os
import sys
import structlog

# Add project root to python path so backend can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.database.connection import init_db_pool, close_db_pool
from backend.rag.ingestion.orchestrator import ingest_document

# Setup basic console logging
structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ]
)
logger = structlog.get_logger()

async def main():
    parser = argparse.ArgumentParser(description="Onboard/Ingest a regulatory PDF document into the RAG database.")
    parser.add_argument("--file", required=True, help="Path to the PDF document to ingest.")
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        logger.error("file_not_found", path=args.file)
        sys.exit(1)
        
    logger.info("onboarding_script_started", file=args.file)
    
    # Initialize connection pool
    await init_db_pool()
    try:
        doc_id = await ingest_document(args.file)
        logger.info("onboarding_script_success", file=args.file, doc_id=str(doc_id))
    except Exception as e:
        logger.error("onboarding_script_failed", file=args.file, error=str(e))
        sys.exit(1)
    finally:
        await close_db_pool()

if __name__ == "__main__":
    asyncio.run(main())
