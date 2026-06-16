from typing import List
from backend.rag.ingestion.ast_builder import ASTNode
import structlog

logger = structlog.get_logger()

def audit_ast_nodes(
    nodes: List[ASTNode],
    total_raw_char_count: int
) -> List[ASTNode]:
    """
    Performs structural audits on the generated AST nodes:
    1. Checks for empty text content in leaf/content nodes.
    2. Checks for significant character loss compared to the original raw document text.
    3. Checks for structural anomalies (e.g., missing parent links, invalid levels).
    
    If a node fails the audit, sets its `needs_repair` attribute to True.
    
    Args:
        nodes: The list of constructed ASTNodes.
        total_raw_char_count: The total length of the raw text blocks parsed from Docling.
        
    Returns:
        The updated list of ASTNodes with `needs_repair` set where audit failed.
    """
    logger.info("auditing_ast_nodes_started", node_count=len(nodes), raw_chars=total_raw_char_count)
    
    # 1. Character loss check
    total_ast_char_count = sum(len(node.text_content) for node in nodes)
    char_loss_ratio = 1.0 - (total_ast_char_count / max(1, total_raw_char_count))
    
    logger.info(
        "ast_character_audit",
        raw_chars=total_raw_char_count,
        ast_chars=total_ast_char_count,
        loss_ratio=char_loss_ratio
    )
    
    # If character loss is > 15% (accounting for ignored headers/footers/Hindi blocks),
    # flag all content nodes as needing repair to trigger a self-heal or review.
    severe_character_loss = char_loss_ratio > 0.15
    if severe_character_loss:
        logger.warning("severe_character_loss_detected", loss_ratio=char_loss_ratio)
        
    # 2. Individual node checks
    # Pre-build a set of all node_ids for O(1) parent validation (Rule C)
    node_id_set = {n.node_id for n in nodes}

    for idx, node in enumerate(nodes):
        # Skip root node
        if node.level == 1:
            continue
            
        # Rule A: Content nodes must not be empty
        is_content_type = node.node_type in ["SECTION", "SUBSECTION", "CLAUSE", "SUBCLAUSE", "DEFINITION"]
        if is_content_type and not node.text_content.strip():
            node.needs_repair = True
            logger.warning("audit_failed_empty_node", breadcrumb=node.breadcrumb)
            
        # Rule B: Section and Subsection headings/titles shouldn't be excessively long
        if node.node_type == "SECTION" and node.title and len(node.title) > 200:
            node.needs_repair = True
            logger.warning("audit_failed_title_too_long", breadcrumb=node.breadcrumb)
            
        # Rule C: Parent UUID must refer to a node in the list
        if node.parent_id:
            if node.parent_id not in node_id_set:
                node.needs_repair = True
                logger.error("audit_failed_orphan_node", breadcrumb=node.breadcrumb)
                
        # If severe character loss was detected, flag leaf nodes
        if severe_character_loss and is_content_type:
            node.needs_repair = True
            
    logger.info(
        "auditing_ast_nodes_completed",
        nodes_needing_repair=sum(1 for n in nodes if n.needs_repair)
    )
    
    return nodes
