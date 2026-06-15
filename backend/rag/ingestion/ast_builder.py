from typing import List, Dict, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field
from backend.rag.extraction.schemas import BoundaryOutput
import structlog

logger = structlog.get_logger()

class ASTNode(BaseModel):
    node_id: UUID = Field(default_factory=uuid4)
    doc_id: UUID
    parent_id: Optional[UUID] = None
    level: int
    node_type: str
    title: Optional[str] = None
    text_content: str = ""
    breadcrumb: str = ""
    needs_repair: bool = False

class RawBlock(BaseModel):
    text: str
    boundary: BoundaryOutput

def build_ast(doc_id: UUID, doc_title: str, blocks: List[RawBlock]) -> List[ASTNode]:
    """
    Deterministically builds an AST tree (flat list of ASTNode objects with parent relations)
    from a list of raw document layout blocks and their boundary classifications.
    
    Args:
        doc_id: The UUID of the document being ingested.
        doc_title: The official title of the document.
        blocks: A list of RawBlock objects (block text + boundary detection).
        
    Returns:
        A flat list of all constructed ASTNode objects, ordered by appearance.
    """
    logger.info("building_ast_started", doc_id=str(doc_id), blocks_count=len(blocks))
    
    nodes: List[ASTNode] = []
    
    # Initialize the stack with the document root at level 1
    root_node = ASTNode(
        doc_id=doc_id,
        parent_id=None,
        level=1,
        node_type="DOCUMENT_ROOT",
        title=doc_title,
        text_content="",
        breadcrumb=doc_title
    )
    nodes.append(root_node)
    
    # Stack maps level -> ASTNode
    stack: Dict[int, ASTNode] = {1: root_node}
    
    for idx, block in enumerate(blocks):
        text = block.text.strip()
        boundary = block.boundary
        
        if boundary.node_type == "IGNORE":
            logger.debug("ast_builder_skip_ignore_block", index=idx)
            continue
            
        if boundary.is_boundary_break:
            # Determine the parent node by looking up the stack for the closest ancestor
            parent = None
            for lvl in range(boundary.level - 1, 0, -1):
                if lvl in stack:
                    parent = stack[lvl]
                    break
            
            # If no parent found, default to the document root (level 1)
            if not parent:
                parent = root_node
                
            # Construct node title
            node_title = boundary.heading_text or f"{boundary.node_type} Block"
            
            # Construct breadcrumb path
            breadcrumb = f"{parent.breadcrumb} > {node_title}"
            
            new_node = ASTNode(
                doc_id=doc_id,
                parent_id=parent.node_id,
                level=boundary.level,
                node_type=boundary.node_type,
                title=boundary.heading_text,
                text_content=text,
                breadcrumb=breadcrumb
            )
            
            nodes.append(new_node)
            
            # Update active stack
            stack[boundary.level] = new_node
            # Clear deeper levels from stack
            levels_to_clear = [lvl for lvl in stack.keys() if lvl > boundary.level]
            for lvl in levels_to_clear:
                del stack[lvl]
                
            logger.debug(
                "ast_builder_new_node",
                node_type=new_node.node_type,
                level=new_node.level,
                breadcrumb=new_node.breadcrumb
            )
        else:
            # Continuation block: append to current active leaf node
            active_level = max(stack.keys())
            active_node = stack[active_level]
            
            # If the active node is the root and we have body text, create an implicit PREAMBLE node at level 2
            if active_node.level == 1:
                preamble_node = ASTNode(
                    doc_id=doc_id,
                    parent_id=root_node.node_id,
                    level=2,
                    node_type="PREAMBLE",
                    title="Preamble",
                    text_content=text,
                    breadcrumb=f"{root_node.breadcrumb} > Preamble"
                )
                nodes.append(preamble_node)
                stack[2] = preamble_node
                logger.debug("ast_builder_implicit_preamble_created")
            else:
                # Append text
                if active_node.text_content:
                    active_node.text_content += "\n" + text
                else:
                    active_node.text_content = text
                logger.debug("ast_builder_append_to_node", breadcrumb=active_node.breadcrumb, text_len=len(text))

    logger.info("building_ast_completed", total_nodes=len(nodes))
    return nodes
