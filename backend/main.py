import sys
import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog

from backend.database.connection import init_db_pool, close_db_pool
from backend.rag.retrieval.reranker import get_reranker_model
from backend.api import qa, compliance, admin

# Configure structlog to capture ingestion logs for SSE streaming
def configure_structlog():
    # Configure standard library logging format
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )
    
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            # Intercept logs for the Admin Ingestion Logs page
            admin.sse_log_processor,
            # Format logs for terminal console
            structlog.dev.ConsoleRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        cache_logger_on_first_use=True,
    )

# Run structlog configuration
configure_structlog()
logger = structlog.get_logger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup actions
    logger.info("api_server_starting")
    # Initialize the database pool
    await init_db_pool()
    # Preload and cache the reranker model parameters to avoid first-query latency spikes
    try:
        await get_reranker_model()
    except Exception as e:
        logger.error("api_server_reranker_preload_failed", error=str(e))
        
    yield
    
    # Shutdown actions
    logger.info("api_server_shutting_down")
    await close_db_pool()
    logger.info("api_server_shutdown_complete")

app = FastAPI(
    title="IFSCA Smart Regulator API",
    description="Visual and Hierarchical RAG system for IFSCA Regulations",
    version="2.0",
    lifespan=lifespan
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict to frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routers
app.include_router(qa.router, prefix="/api")
app.include_router(compliance.router, prefix="/api")
app.include_router(admin.router, prefix="/api")

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy"}
