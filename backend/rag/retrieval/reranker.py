"""
Stage D: Prompt-Baked SLM Reranker

Rescores candidate nodes from the hop expander against the user query
using a locally-hosted SLM (ifsca-reranker-3b via Ollama).

Design decisions:
  - Candidates are batched (max 5 per batch) to stay within the 4k token
    context window. Each batch is scored in parallel via asyncio.gather.
  - Each candidate receives a plain-text preview truncated to `MAX_PREVIEW_LEN`
    characters. No keyword heuristics are applied — the full leading text is
    passed so that the model receives the most contextually stable signal.
  - The model outputs a JSON object keyed by string section index ("0", "1" …).
    parse_batch_scores() extracts scores defensively without recursion.
  - Individual batch failures are isolated; failing batches fall back to
    score=None so the rest of the pipeline still proceeds.
"""

import json
import time
import asyncio
from typing import List, Dict, Optional
import structlog
from ollama import AsyncClient

from backend.config import OLLAMA_HOST, RERANKER_SLM_MODEL
from backend.rag.retrieval.pipeline_context import QueryPipelineContext, NodeCandidate

logger = structlog.get_logger()
_client = AsyncClient(host=OLLAMA_HOST)

# Maximum characters of section text passed to the reranker per candidate.
# At ~4 chars/token this is ~500 tokens; with 5 candidates per batch the
# payload is ~2,500 tokens, comfortably within the 4096-token context.
MAX_PREVIEW_LEN = 2000

# Number of candidates scored in a single LLM call.
BATCH_SIZE = 5

# Maximum number of candidates sent to the reranker (controls total latency).
MAX_CANDIDATES = 10

# Top-N sections forwarded to the generator after reranking.
TOP_K = 4


async def get_reranker_model() -> str:
    """
    Deprecated placeholder retained for backward compatibility with
    the main.py lifespan startup check.
    """
    from backend.rag.retrieval.generator import check_model_exists
    exists = await check_model_exists(RERANKER_SLM_MODEL)
    if not exists:
        logger.warning("reranker_slm_model_missing_in_ollama", model=RERANKER_SLM_MODEL)
    return RERANKER_SLM_MODEL


def _build_section_block(index: int, node: NodeCandidate, max_len: int) -> str:
    """Return a formatted text block for a single candidate section."""
    preview = (node.text_content or "")[:max_len]
    return (
        f"Section Index: {index}\n"
        f"Breadcrumb: {node.breadcrumb or ''}\n"
        f"Content:\n{preview}\n"
        f"{'─' * 60}"
    )


def _parse_batch_scores(content: str, batch_size: int) -> Dict[int, Optional[float]]:
    """
    Parse the JSON response from the reranker SLM into a dict of
    {local_index: score}.

    The model is instructed to return:
        { "0": {"relevance_score": 8.5, "reasoning": "..."}, "1": {...} }

    Falls back gracefully if the model returns a list of objects instead,
    or if individual entries are malformed.
    """
    scores: Dict[int, Optional[float]] = {}

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        logger.warning("reranker_json_parse_failed", raw=content[:200])
        return scores

    # Normalise: the model sometimes returns a list instead of a keyed dict.
    if isinstance(data, list):
        data = {str(i): item for i, item in enumerate(data)}

    if not isinstance(data, dict):
        logger.warning("reranker_unexpected_response_type", type=type(data).__name__)
        return scores

    for key, value in data.items():
        # Key should be a stringified integer ("0", "1", …)
        try:
            idx = int(key)
        except (ValueError, TypeError):
            continue

        if idx < 0 or idx >= batch_size:
            continue

        # Value should be a dict with a "relevance_score" field.
        if isinstance(value, dict):
            raw_score = value.get("relevance_score")
        elif isinstance(value, (int, float)):
            # Model returned a bare number instead of an object.
            raw_score = value
        else:
            continue

        try:
            score = float(raw_score)
            scores[idx] = max(0.0, min(10.0, score))
        except (ValueError, TypeError):
            pass

    return scores


async def _score_batch(
    batch_index: int,
    nodes: List[NodeCandidate],
    query: str,
) -> Dict[int, Optional[float]]:
    """
    Send one batch of candidate nodes to the reranker SLM and return
    a mapping of {local_batch_index: score}.
    """
    section_blocks = [
        _build_section_block(i, node, MAX_PREVIEW_LEN)
        for i, node in enumerate(nodes)
    ]
    sections_text = "\n\n".join(section_blocks)

    user_prompt = (
        f"User Query: {query}\n\n"
        f"Score each section below for relevance to the query.\n\n"
        f"{sections_text}"
    )

    try:
        response = await _client.chat(
            model=RERANKER_SLM_MODEL,
            messages=[{"role": "user", "content": user_prompt}],
            format="json",
            options={"temperature": 0.0, "top_p": 0.1},
        )
        content = response.message.content
        logger.debug("reranker_slm_raw_response", batch=batch_index, content=content)
        return _parse_batch_scores(content, len(nodes))

    except Exception as exc:
        logger.error("reranker_batch_failed", batch=batch_index, error=str(exc))
        return {}


async def run_reranker(ctx: QueryPipelineContext) -> QueryPipelineContext:
    """
    Stage D: Prompt-Baked SLM Reranking.

    Collects candidates from the hop expander, deduplicates by node_id,
    scores them in parallel batches using the reranker SLM, sorts by score,
    and writes the top-K results to ctx.reranked_nodes.
    """
    start_time = time.monotonic()

    # --- 1. Collect and deduplicate candidates ---
    all_candidates = ctx.candidate_nodes + ctx.expanded_nodes
    if not all_candidates:
        logger.warning("reranker_skipped_no_candidates")
        ctx.reranked_nodes = []
        ctx.stage_timings["reranker"] = time.monotonic() - start_time
        return ctx

    seen_ids = set()
    unique_candidates: List[NodeCandidate] = []
    for node in all_candidates:
        if node.node_id not in seen_ids:
            seen_ids.add(node.node_id)
            unique_candidates.append(node)

    candidates_to_score = unique_candidates[:MAX_CANDIDATES]
    logger.info(
        "reranking_start",
        original_count=len(all_candidates),
        total_candidates=len(candidates_to_score),
    )

    # --- 2. Split into batches and score in parallel ---
    batches = [
        candidates_to_score[i : i + BATCH_SIZE]
        for i in range(0, len(candidates_to_score), BATCH_SIZE)
    ]

    batch_tasks = [
        _score_batch(b_idx, batch, ctx.original_query)
        for b_idx, batch in enumerate(batches)
    ]
    batch_results: List[Dict[int, Optional[float]]] = await asyncio.gather(*batch_tasks)

    # --- 3. Apply scores back to candidate nodes ---
    scored_nodes: List[NodeCandidate] = []
    for batch_idx, batch in enumerate(batches):
        batch_scores = batch_results[batch_idx]
        for local_idx, node in enumerate(batch):
            node.score = batch_scores.get(local_idx)  # None if batch failed
            scored_nodes.append(node)

    # --- 4. Sort: scored nodes first (descending), unscored nodes appended last ---
    scored_with_value = [n for n in scored_nodes if n.score is not None]
    scored_without_value = [n for n in scored_nodes if n.score is None]

    scored_with_value.sort(key=lambda n: n.score, reverse=True)  # type: ignore[arg-type]
    final_ranked = scored_with_value + scored_without_value

    logger.info(
        "reranking_scores_parsed_successfully",
        scores=[(n.title or n.breadcrumb, n.score) for n in final_ranked],
    )

    # --- 5. Write top-K to context ---
    ctx.reranked_nodes = final_ranked[:TOP_K]
    logger.info("reranking_complete", top_count=len(ctx.reranked_nodes))
    ctx.stage_timings["reranker"] = time.monotonic() - start_time
    return ctx
