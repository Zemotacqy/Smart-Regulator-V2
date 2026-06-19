import time
from typing import List
import structlog

from backend.config import MAX_CONTEXT_CHARS, MAX_SINGLE_SECTION_CHARS
from backend.rag.retrieval.pipeline_context import QueryPipelineContext, NodeCandidate

logger = structlog.get_logger()

def split_large_section(node: NodeCandidate, limit: int = 4000) -> List[NodeCandidate]:
    """
    Splits a large rolled-up section at paragraph boundaries to fit within context budgets.
    """
    paragraphs = node.text_content.split("\n\n")
    if len(paragraphs) == 1:
        paragraphs = node.text_content.split("\n")
        
    sub_nodes = []
    current_text = []
    current_len = 0
    part_idx = 1
    
    for p in paragraphs:
        p_clean = p.strip()
        if not p_clean:
            continue
            
        if len(p_clean) > limit:
            # Flush current accumulator first
            if current_text:
                text_block = "\n\n".join(current_text)
                sub_nodes.append(NodeCandidate(
                    node_id=node.node_id,
                    doc_id=node.doc_id,
                    parent_id=node.parent_id,
                    level=node.level,
                    node_type=node.node_type,
                    title=f"{node.title} (Part {part_idx})" if node.title else f"Part {part_idx}",
                    text_content=text_block,
                    breadcrumb=f"{node.breadcrumb} (Part {part_idx})",
                    file_name=node.file_name
                ))
                part_idx += 1
                current_text = []
                current_len = 0
            
            # Segment the huge paragraph by characters
            for offset in range(0, len(p_clean), limit):
                sub_nodes.append(NodeCandidate(
                    node_id=node.node_id,
                    doc_id=node.doc_id,
                    parent_id=node.parent_id,
                    level=node.level,
                    node_type=node.node_type,
                    title=f"{node.title} (Part {part_idx})" if node.title else f"Part {part_idx}",
                    text_content=p_clean[offset:offset+limit],
                    breadcrumb=f"{node.breadcrumb} (Part {part_idx})",
                    file_name=node.file_name
                ))
                part_idx += 1
        else:
            if current_len + len(p_clean) > limit:
                # Flush accumulator
                text_block = "\n\n".join(current_text)
                sub_nodes.append(NodeCandidate(
                    node_id=node.node_id,
                    doc_id=node.doc_id,
                    parent_id=node.parent_id,
                    level=node.level,
                    node_type=node.node_type,
                    title=f"{node.title} (Part {part_idx})" if node.title else f"Part {part_idx}",
                    text_content=text_block,
                    breadcrumb=f"{node.breadcrumb} (Part {part_idx})",
                    file_name=node.file_name
                ))
                part_idx += 1
                current_text = [p_clean]
                current_len = len(p_clean)
            else:
                current_text.append(p_clean)
                current_len += len(p_clean)
                
    if current_text:
        text_block = "\n\n".join(current_text)
        sub_nodes.append(NodeCandidate(
            node_id=node.node_id,
            doc_id=node.doc_id,
            parent_id=node.parent_id,
            level=node.level,
            node_type=node.node_type,
            title=f"{node.title} (Part {part_idx})" if node.title else f"Part {part_idx}",
            text_content=text_block,
            breadcrumb=f"{node.breadcrumb} (Part {part_idx})",
            file_name=node.file_name
        ))
    return sub_nodes

def assemble_batch(batch_nodes: List[NodeCandidate]) -> str:
    blocks = []
    for node in batch_nodes:
        blocks.append(f"Source: {node.breadcrumb} <!-- ID: {node.node_id} -->\n{node.text_content}")
    return "\n\n".join(blocks)

async def run_compressor(ctx: QueryPipelineContext) -> QueryPipelineContext:
    """
    Stage E: Bounded Context Assembler.
    Replaces the SLM compressor stage with a deterministic context assembling step.
    If context overflows MAX_CONTEXT_CHARS (12,000), prepares overflow batches for map-reduce.
    """
    start_time = time.monotonic()
    
    if not ctx.reranked_nodes:
        logger.warning("assembler_skipped_no_nodes")
        ctx.compressed_context = ""
        ctx.overflow_batches = []
        ctx.stage_timings["compressor"] = time.monotonic() - start_time
        return ctx
        
    batches: List[List[NodeCandidate]] = []
    current_batch: List[NodeCandidate] = []
    current_batch_len = 0
    
    for node in ctx.reranked_nodes:
        if not node.text_content:
            continue
            
        nodes_to_add = [node]
        if len(node.text_content) > MAX_SINGLE_SECTION_CHARS:
            logger.info("splitting_large_section", 
                        node_id=str(node.node_id), 
                        chars=len(node.text_content))
            nodes_to_add = split_large_section(node, MAX_SINGLE_SECTION_CHARS)
            
        for n in nodes_to_add:
            block_len = len(f"Source: {n.breadcrumb} <!-- ID: {n.node_id} -->\n{n.text_content}")
            if current_batch_len + block_len > MAX_CONTEXT_CHARS:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [n]
                current_batch_len = block_len
            else:
                current_batch.append(n)
                current_batch_len += block_len
                
    if current_batch:
        batches.append(current_batch)
        
    # Set primary context batch and overflow batches
    if not batches:
        ctx.compressed_context = ""
        ctx.overflow_batches = []
    elif len(batches) == 1:
        ctx.compressed_context = assemble_batch(batches[0])
        ctx.overflow_batches = []
        logger.info("assembler_single_pass_prepared", total_chars=len(ctx.compressed_context))
    else:
        ctx.compressed_context = assemble_batch(batches[0])
        ctx.overflow_batches = batches[1:]
        logger.info("assembler_overflow_batches_prepared", 
                    primary_chars=len(ctx.compressed_context),
                    overflow_batches_count=len(ctx.overflow_batches))
                    
    ctx.stage_timings["compressor"] = time.monotonic() - start_time
    return ctx
