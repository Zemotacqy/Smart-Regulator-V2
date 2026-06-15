import asyncpg
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager
import structlog
from backend.config import DATABASE_URL

logger = structlog.get_logger()

# Global connection pool
_pool: Optional[asyncpg.Pool] = None

async def init_db_pool() -> asyncpg.Pool:
    """
    Initializes the asyncpg connection pool.
    """
    global _pool
    if _pool is None:
        logger.info("db_pool_initializing", dsn=DATABASE_URL)
        try:
            _pool = await asyncpg.create_pool(
                dsn=DATABASE_URL,
                min_size=2,
                max_size=10,
                max_inactive_connection_lifetime=300.0
            )
            logger.info("db_pool_initialized")
        except Exception as e:
            logger.error("db_pool_initialization_failed", error=str(e))
            raise
    return _pool

async def close_db_pool() -> None:
    """
    Closes the asyncpg connection pool.
    """
    global _pool
    if _pool is not None:
        logger.info("db_pool_closing")
        await _pool.close()
        _pool = None
        logger.info("db_pool_closed")

@asynccontextmanager
async def get_db_connection() -> AsyncGenerator[asyncpg.Connection, None]:
    """
    Context manager to acquire and release a database connection from the pool.
    Usage:
        async with get_db_connection() as conn:
            await conn.execute(...)
    """
    global _pool
    if _pool is None:
        await init_db_pool()
        
    assert _pool is not None
    conn = await _pool.acquire()
    try:
        yield conn
    finally:
        await _pool.release(conn)
