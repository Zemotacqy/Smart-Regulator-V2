import pytest
from uuid import uuid4
from backend.rag.extraction.schemas import BoundaryOutput
from backend.rag.ingestion.ast_builder import build_ast, RawBlock, ASTNode
from backend.rag.ingestion.auditor import audit_ast_nodes

def test_ast_builder_hierarchy():
    doc_id = uuid4()
    doc_title = "Test Regulation 2026"
    
    # Define a sequence of blocks
    blocks = [
        RawBlock(
            text="CHAPTER I\nPRELIMINARY",
            boundary=BoundaryOutput(node_type="CHAPTER", level=2, is_boundary_break=True, heading_text="CHAPTER I")
        ),
        RawBlock(
            text="1. Short title and commencement.",
            boundary=BoundaryOutput(node_type="SECTION", level=3, is_boundary_break=True, heading_text="Short title and commencement")
        ),
        RawBlock(
            text="(1) These regulations may be called the Test Regulation, 2026.",
            boundary=BoundaryOutput(node_type="SUBSECTION", level=4, is_boundary_break=True, heading_text=None)
        ),
        RawBlock(
            text="They shall come into force on the date of their publication.",
            boundary=BoundaryOutput(node_type="BODY_TEXT", level=6, is_boundary_break=False, heading_text=None)
        ),
        RawBlock(
            text="2. Definitions.",
            boundary=BoundaryOutput(node_type="SECTION", level=3, is_boundary_break=True, heading_text="Definitions")
        ),
        RawBlock(
            text="In these regulations, unless the context otherwise requires,—",
            boundary=BoundaryOutput(node_type="BODY_TEXT", level=6, is_boundary_break=False, heading_text=None)
        ),
        RawBlock(
            text="(a) 'Act' means the IFSCA Act, 2019;",
            boundary=BoundaryOutput(node_type="DEFINITION", level=5, is_boundary_break=True, heading_text="Act")
        )
    ]
    
    nodes = build_ast(doc_id, doc_title, blocks)
    
    # Check that root node + 5 nodes were created (CHAPTER I, Section 1, Subsection (1), Section 2, Definition (a))
    # Note: Continuation body texts should have been appended to existing nodes.
    # Total nodes = 1 (root) + 1 (chapter) + 1 (sec 1) + 1 (subsec 1) + 1 (sec 2) + 1 (def a) = 6 nodes
    assert len(nodes) == 6
    
    root = nodes[0]
    chapter = nodes[1]
    sec1 = nodes[2]
    subsec1 = nodes[3]
    sec2 = nodes[4]
    def_a = nodes[5]
    
    # Check parents
    assert chapter.parent_id == root.node_id
    assert sec1.parent_id == chapter.node_id
    assert subsec1.parent_id == sec1.node_id
    assert sec2.parent_id == chapter.node_id
    assert def_a.parent_id == sec2.node_id
    
    # Check content append
    assert "They shall come into force" in subsec1.text_content
    assert "unless the context otherwise requires" in sec2.text_content
    
    # Check breadcrumbs
    assert chapter.breadcrumb == "Test Regulation 2026 > CHAPTER I"
    assert sec1.breadcrumb == "Test Regulation 2026 > CHAPTER I > Short title and commencement"
    assert subsec1.breadcrumb == "Test Regulation 2026 > CHAPTER I > Short title and commencement > SUBSECTION Block"
    assert def_a.breadcrumb == "Test Regulation 2026 > CHAPTER I > Definitions > Act"

def test_auditor_empty_and_character_loss():
    doc_id = uuid4()
    
    # Case 1: normal nodes
    nodes = [
        ASTNode(doc_id=doc_id, level=1, node_type="DOCUMENT_ROOT", text_content="", breadcrumb="Root"),
        ASTNode(doc_id=doc_id, level=3, node_type="SECTION", text_content="Section 4 rules", breadcrumb="Root > Section 4")
    ]
    
    audited = audit_ast_nodes(nodes, total_raw_char_count=15)
    assert not audited[0].needs_repair
    assert not audited[1].needs_repair
    
    # Case 2: empty content node
    empty_nodes = [
        ASTNode(doc_id=doc_id, level=1, node_type="DOCUMENT_ROOT", text_content="", breadcrumb="Root"),
        ASTNode(doc_id=doc_id, level=3, node_type="SECTION", text_content="", breadcrumb="Root > Section 4")
    ]
    audited_empty = audit_ast_nodes(empty_nodes, total_raw_char_count=15)
    assert audited_empty[1].needs_repair
