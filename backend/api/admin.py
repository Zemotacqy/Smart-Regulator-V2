import os
import json
import asyncio
import collections
import tempfile
from uuid import UUID, uuid4
from typing import List, Dict, Any, Set
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
import structlog

from backend.database.connection import get_db_connection
from backend.database.queries import get_all_documents, get_corpus_stats

logger = structlog.get_logger()
router = APIRouter()

# Global log history buffer (in-memory ring buffer)
log_history = collections.deque(maxlen=500)
# Active SSE log subscribers
log_subscribers: Set[asyncio.Queue] = set()

def sse_log_processor(logger_inst, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Structlog processor that intercepts logs and broadcasts them to active SSE subscribers.
    """
    import datetime
    timestamp = event_dict.get("timestamp")
    if not timestamp:
        timestamp = datetime.datetime.utcnow().isoformat() + "Z"
        
    log_entry = {
        "timestamp": timestamp,
        "level": method_name.upper(),
        "event": event_dict.get("event"),
    }
    
    # Extract any extra fields
    for k, v in event_dict.items():
        if k not in ("timestamp", "event"):
            log_entry[k] = str(v)
            
    # Add to in-memory history
    log_history.append(log_entry)
    
    # Broadcast to all active SSE subscribers
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            for q in list(log_subscribers):
                loop.call_soon_threadsafe(q.put_nowait, log_entry)
    except RuntimeError:
        pass # No running event loop (e.g. CLI script context)
        
    return event_dict

async def run_ingestion_task(temp_path: str):
    """
    Background task that executes the ingestion pipeline.
    """
    logger.info("background_ingestion_started", path=temp_path)
    try:
        from backend.rag.ingestion.orchestrator import ingest_document
        doc_id = await ingest_document(temp_path)
        logger.info("background_ingestion_success", path=temp_path, doc_id=str(doc_id))
    except Exception as e:
        logger.error("background_ingestion_failed", path=temp_path, error=str(e))
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info("background_ingestion_cleanup_complete", path=temp_path)
            except Exception as err:
                logger.warning("background_ingestion_cleanup_failed", path=temp_path, error=str(err))

@router.post("/admin/ingest")
async def admin_ingest(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Regulatory PDF to ingest")
):
    """
    POST /api/admin/ingest
    Uploads a regulatory PDF and schedules the visual RAG ingestion pipeline in the background.
    """
    logger.info("admin_ingest_hit", filename=file.filename)
    
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
    # Create a unique temporary file path
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"{uuid4()}_{file.filename}")
    
    try:
        # Write file contents to temp path
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        # Add to background tasks
        background_tasks.add_task(run_ingestion_task, temp_path)
        return {"status": "queued", "filename": file.filename}
        
    except Exception as e:
        logger.error("admin_ingest_failed_to_queue", filename=file.filename, error=str(e))
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Failed to queue document: {str(e)}")

@router.get("/admin/ingest/logs")
async def admin_ingest_logs(follow: bool = True):
    """
    GET /api/admin/ingest/logs
    Streams ingestion logs in real-time as Server-Sent Events (SSE).
    """
    async def log_stream():
        # 1. Yield log history buffer
        for entry in list(log_history):
            yield f"event: log\ndata: {json.dumps(entry)}\n\n"
            
        if not follow:
            yield "event: done\ndata: {}\n\n"
            return
            
        # 2. Subscribe to new logs
        q = asyncio.Queue()
        log_subscribers.add(q)
        logger.info("log_subscriber_connected", subscriber_count=len(log_subscribers))
        try:
            while True:
                entry = await q.get()
                yield f"event: log\ndata: {json.dumps(entry)}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            log_subscribers.remove(q)
            logger.info("log_subscriber_disconnected", subscriber_count=len(log_subscribers))
            
    return StreamingResponse(log_stream(), media_type="text/event-stream")


@router.get("/admin/documents")
async def admin_list_documents():
    """
    GET /api/admin/documents
    Lists all documents in the corpus.
    """
    async with get_db_connection() as conn:
        docs = await get_all_documents(conn)
    # Serialize datetime/date objects
    for d in docs:
        if d.get("publish_date"):
            d["publish_date"] = d["publish_date"].isoformat()
        if d.get("ingested_at"):
            d["ingested_at"] = d["ingested_at"].isoformat()
        d["doc_id"] = str(d["doc_id"])
    return docs

@router.get("/admin/stats")
async def admin_stats():
    """
    GET /api/admin/stats
    Retrieves general statistics about the database.
    """
    async with get_db_connection() as conn:
        stats = await get_corpus_stats(conn)
    return stats
