import time
import structlog
from backend.config import EXPANDER_MODEL
from backend.rag.extraction.llm_client import call_llm_with_validation
from backend.rag.extraction.schemas import QueryExpansionOutput
from backend.rag.retrieval.pipeline_context import QueryPipelineContext

logger = structlog.get_logger()

async def run_query_expander(ctx: QueryPipelineContext) -> QueryPipelineContext:
    """
    Stage A: Query Expansion.
    Generates 3 distinct search query variations from the original query using ifsca-expander-3b.
    """
    start_time = time.monotonic()
    
    # Guard against empty or whitespace query
    if not ctx.original_query or not ctx.original_query.strip():
        logger.warning("query_expansion_skipped_empty_query")
        ctx.expanded_queries = []
        ctx.stage_timings["query_expander"] = time.monotonic() - start_time
        return ctx
        
    messages = [
        {
            "role": "system",
            "content": (
                "You are a regulatory query expansion assistant specialised in IFSCA (International Financial "
                "Services Centres Authority) regulations, circulars, frameworks, and guidelines.\n\n"
                "Your task is to generate EXACTLY 3 distinct search query variations for the given regulatory query. "
                "Each variation should approach the topic differently — use synonyms, rephrase as a legal concept, "
                "or reformulate as a specific clause lookup — to maximise retrieval coverage.\n\n"
                "You MUST respond with ONLY a valid JSON object matching this schema:\n"
                "{\"expansions\": [\"<query 1>\", \"<query 2>\", \"<query 3>\"]}\n\n"
                "Do not include any explanation, markdown, or text outside the JSON object."
            )
        },
        {
            "role": "user",
            "content": f"Query: {ctx.original_query}"
        }
    ]
    
    try:
        logger.info("running_query_expansion", query=ctx.original_query)
        result = await call_llm_with_validation(
            model=EXPANDER_MODEL,
            messages=messages,
            response_schema=QueryExpansionOutput,
            temperature=0.0
        )
        
        # Ensure we have a list of strings and clean them up
        ctx.expanded_queries = [str(eq).strip() for eq in result.expansions if eq]
        logger.info("query_expansion_complete", expansions=ctx.expanded_queries)
    except Exception as e:
        logger.error("query_expansion_failed", error=str(e))
        # Fallback to empty list so hybrid search only searches the original query
        ctx.expanded_queries = []
        
    ctx.stage_timings["query_expander"] = time.monotonic() - start_time
    return ctx
