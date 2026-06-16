import asyncio
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from backend.database.connection import init_db_pool, get_db_connection, close_db_pool
from backend.database.queries import get_corpus_stats, get_all_documents

async def main():
    await init_db_pool()
    async with get_db_connection() as conn:
        stats = await get_corpus_stats(conn)
        print("Database stats:", stats)
        docs = await get_all_documents(conn)
        print("Documents in DB:")
        for d in docs:
            print(f"- {d['file_name']} (active={d['is_active']}, type={d['doc_type']})")
    await close_db_pool()

if __name__ == "__main__":
    asyncio.run(main())
