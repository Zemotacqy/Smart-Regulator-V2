from backend.rag.extraction.schemas import BoundaryOutput
from backend.rag.extraction.llm_client import call_llm_with_validation
from backend.config import BOUNDARY_MODEL
import structlog

logger = structlog.get_logger()

BOUNDARY_SYSTEM_PROMPT = """
You are a structural boundary detector for Indian financial regulatory text (IFSCA).
Your task is to analyze a text layout block and classify its structural type, level, and whether it introduces a new boundary.

The structural hierarchy levels are defined as:
- Level 1: Document Root / Preamble / First Schedule (usually not detected per block unless the document title itself)
- Level 2: CHAPTER, PART, SCHEDULE
- Level 3: SECTION (e.g., "4. Capital requirement" or "Section 12")
- Level 4: SUBSECTION (e.g., "(1) An IBU shall..." or "Sub-section (2)")
- Level 5: CLAUSE (e.g., "(a) a banking unit..." or "Clause (ii)")
- Level 6: SUBCLAUSE (e.g., "(i) a foreign bank...")

Output Fields:
1. node_type: Must be one of:
   - "CHAPTER" (CHAPTER I, CHAPTER II, PART I, etc.)
   - "SCHEDULE" (SCHEDULE I, SCHEDULE II, etc.)
   - "SECTION" (Starts a section, e.g. "3. Establishments...")
   - "SUBSECTION" (Starts a subsection, e.g. "(1)", "(2)", unless it's inside definitions)
   - "CLAUSE" (Starts a clause, e.g. "(a)", "(b)", "(c)")
   - "SUBCLAUSE" (Starts a subclause, e.g. "(i)", "(ii)")
   - "PREAMBLE" (Document introduction / Gazette preamble)
   - "DEFINITION" (A definition entry, e.g. "'IBU' means...", often inside a definitions section)
   - "BODY_TEXT" (A continuation paragraph, bullet point, or text that does not break a boundary)
   - "IGNORE" (Page numbers, running headers/footers, metadata, or noise)
2. level: An integer from 1 to 6 corresponding to the hierarchy level (default to 6 if it is body text or ignore, section is 3, subsection is 4, clause is 5, subclause is 6).
3. is_boundary_break: A boolean indicating if this block starts a new structural node. Set to true for headings, chapter starts, section starts, sub-section starts, clauses, sub-clauses, and definition entries. Set to false for BODY_TEXT, IGNORE, or continuations.
4. heading_text: The title or heading of the section/chapter if present (e.g., "Capital requirement" or "Permissible activities"). Null if there is no explicit title/heading.

You must output ONLY a JSON object matching the requested schema. No explanations, no markdown code block backticks.
"""

async def detect_boundary(block_text: str) -> BoundaryOutput:
    """
    Invokes the boundary detector SLM to classify structural boundaries of a layout block.
    
    Args:
        block_text: The text content of the layout block.
        
    Returns:
        BoundaryOutput detailing structural node_type, level, is_boundary_break, and heading_text.
    """
    logger.debug("detecting_boundary_started", text_length=len(block_text))
    
    messages = [
        {"role": "system", "content": BOUNDARY_SYSTEM_PROMPT},
        {"role": "user", "content": f"Block text:\n---\n{block_text}\n---\nClassify:"}
    ]
    
    result: BoundaryOutput = await call_llm_with_validation(
        model=BOUNDARY_MODEL,
        messages=messages,
        response_schema=BoundaryOutput
    )
    
    # Normalize node_type
    node_type = result.node_type.strip().upper()
    if node_type == "PART":
        node_type = "CHAPTER"
        
    ALLOWED_NODE_TYPES = {
        "CHAPTER", "SCHEDULE", "SECTION", "SUBSECTION", "CLAUSE", "SUBCLAUSE",
        "PREAMBLE", "DEFINITION", "BODY_TEXT", "IGNORE"
    }
    
    if node_type not in ALLOWED_NODE_TYPES:
        logger.warning("unrecognized_node_type_fallback", original=node_type)
        node_type = "BODY_TEXT"
        
    # Normalize level
    level = result.level
    if level is None or not (1 <= level <= 6):
        default_levels = {
            "CHAPTER": 2,
            "SCHEDULE": 2,
            "SECTION": 3,
            "SUBSECTION": 4,
            "CLAUSE": 5,
            "SUBCLAUSE": 6,
            "PREAMBLE": 2,
            "DEFINITION": 5,
            "BODY_TEXT": 6,
            "IGNORE": 6
        }
        level = default_levels.get(node_type, 6)
        
    result.node_type = node_type
    result.level = level
    
    logger.debug(
        "detecting_boundary_completed",
        node_type=result.node_type,
        level=result.level,
        is_boundary_break=result.is_boundary_break,
        heading_text=result.heading_text
    )
    
    return result
