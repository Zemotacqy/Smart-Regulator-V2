import re
import time
from typing import List
import structlog

from backend.config import MAX_CONTEXT_CHARS, MAX_SINGLE_SECTION_CHARS
from backend.rag.retrieval.pipeline_context import QueryPipelineContext, NodeCandidate

logger = structlog.get_logger()

# Matches numbered sub-section markers of the form "(N) " that represent genuine
# structural sub-sections within a flat text blob, as opposed to cross-references
# like "sub-section (2)" or "Section 13(5)" that appear mid-sentence.
#
# A genuine structural marker appears:
#   - At the start of the string:      ^(1) Text...
#   - After a sentence-ending period:  ...previous rule. (2) Next rule...
#   - After a closing parenthesis:     ...(a) last clause.) (3) Next sub-section...
#
# The (?<=\.\s) and (?<=\)\s) lookbehinds require exactly one space after the
# boundary, matching IFSCA Act drafting conventions.
_INLINE_SUBSECTION_RE = re.compile(
    r'(?:(?:^|(?<=\.\s)|(?<=\)\s))\((\d+)\)\s)',
    re.MULTILINE,
)


def annotate_inline_subsections(text: str) -> str:
    """
    Detects numbered sub-sections packed inline within a single text blob
    (a common ingestion artefact when the boundary detector stores an entire
    section as one SECTION node rather than splitting each numbered sub-section
    into a child node) and inserts a blank line + bold label before each one.

    This gives the LLM a clear visual boundary between adjacent sub-sections so
    it can identify which specific sub-section answers the query, rather than
    conflating facts from neighbouring sub-sections.

    The transformation is purely deterministic string manipulation — no LLM
    call, no DB query, zero latency impact.

    Example (input):
        "(1) Scope text... (2) Amendment power... (5) Foreign currency rule..."

    Example (output):
        "**Sub-section (1)**\nScope text...\n\n**Sub-section (2)**\nAmendment power..."

    If no inline sub-section pattern is detected, the original text is returned
    unchanged so the function is safe to apply to all nodes unconditionally.
    """
    matches = list(_INLINE_SUBSECTION_RE.finditer(text))

    # Require at least two distinct sub-section markers before reformatting.
    # A single "(1)" in a paragraph is more likely incidental punctuation than
    # a genuine numbered sub-section sequence worth splitting.
    if len(matches) < 2:
        return text

    parts: List[str] = []

    # Capture any leading text before the first sub-section marker (e.g. a
    # preamble sentence that precedes sub-section (1)).
    first_match_start = matches[0].start()
    preamble = text[:first_match_start].strip()
    if preamble:
        parts.append(preamble)

    for i, match in enumerate(matches):
        sub_num = match.group(1)
        # Content runs from just after this marker up to the start of the next.
        content_start = match.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[content_start:content_end].strip()
        parts.append(f"**Sub-section ({sub_num})**\n{content}")

    return "\n\n".join(parts)

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
    """
    Assembles the final context string passed to the generator.

    Each node's text is passed through `annotate_inline_subsections` before
    inclusion. For nodes whose text is already well-structured (e.g. already
    split into child nodes by the hop expander), the function is a no-op and
    returns the text unchanged.
    """
    blocks = []
    for node in batch_nodes:
        annotated_text = annotate_inline_subsections(node.text_content)
        blocks.append(f"Source: {node.breadcrumb} <!-- ID: {node.node_id} -->\n{annotated_text}")
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
