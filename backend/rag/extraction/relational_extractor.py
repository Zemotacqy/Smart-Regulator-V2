from typing import List
from pydantic import BaseModel
from backend.rag.extraction.schemas import ExtractorClauseOutput, RelationItem
from backend.rag.extraction.llm_client import call_llm_with_validation
from backend.config import EXTRACTOR_MODEL
import structlog

logger = structlog.get_logger()

class ExtractorBatchOutput(BaseModel):
    clauses: List[ExtractorClauseOutput]

EXTRACTOR_SYSTEM_PROMPT = """
You are a regulatory relation extractor for the International Financial Services Centres Authority (IFSCA).
Your task is to analyze a batch of clauses and extract semantic and structural relationships between them.

For each clause in the batch, you must identify relationships to other clauses, sections, acts, or terms.
The allowed relationship types (rel_type) are:
1. "REFERS_TO": When a clause cites another section, clause, schedule, or external act (e.g., "subject to Section 8(1)", "as defined under FEMA").
2. "DEFINES_TERM": When a clause defines a legal/regulatory term (e.g., "'IBU' means International Banking Unit..."). The `target_text_ref` must be the exact term defined (e.g., "IBU").
3. "SUBSTITUTES": When an amendment clause explicitly replaces/substitutes another section or clause (e.g., "Substituted vide GN/REG041...").
4. "INSERTED_BY": When an amendment clause adds a new section, clause, or proviso (e.g., "the following proviso shall be inserted, namely...").
5. "OMITTED_BY": When an amendment clause deletes/omits an existing section or clause (e.g., "Clause (b) shall be omitted").

Fields for each RelationItem:
- rel_type: Must be one of the 5 relationship types above.
- target_text_ref: The target of the relationship (e.g. "Section 8(1)", "FEMA", or the term defined "IBU"). Keep it concise and exact.
- context: The exact snippet of text from the clause that shows this relationship.
- effective_date: If the clause is an amendment footnote mentioning a w.e.f. (with effect from) date, extract it as an ISO date string (YYYY-MM-DD), otherwise null.

Input format:
You will receive a numbered list of clauses.
Output format:
You must return a JSON object with a single key "clauses" containing a list. Each item in the list must correspond to one of the input clauses and specify:
- source_clause_index: The 0-based index of the clause in the input list.
- relations: A list of RelationItem objects found in that clause (empty list if no relationships).

Output strictly valid JSON matching this schema. No markdown formatting or extra text.
"""

async def extract_relations(clauses: List[str]) -> List[ExtractorClauseOutput]:
    """
    Extracts relationships from a batch of clauses.
    
    Args:
        clauses: A list of text clauses (up to 10 recommended).
        
    Returns:
        A list of ExtractorClauseOutput objects, one for each clause with relationships.
    """
    if not clauses:
        return []
        
    logger.info("extracting_relations_started", batch_size=len(clauses))
    
    # Format input clauses as a numbered list
    user_content_lines = []
    for idx, clause_text in enumerate(clauses):
        user_content_lines.append(f"Clause Index {idx}:\n{clause_text}\n---")
    
    user_content = "\n".join(user_content_lines)
    
    messages = [
        {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
        {"role": "user", "content": f"List of clauses to process:\n\n{user_content}\n\nExtract relationships:"}
    ]
    
    try:
        result: ExtractorBatchOutput = await call_llm_with_validation(
            model=EXTRACTOR_MODEL,
            messages=messages,
            response_schema=ExtractorBatchOutput
        )
        
        ALLOWED_REL_TYPES = {"REFERS_TO", "DEFINES_TERM", "SUBSTITUTES", "INSERTED_BY", "OMITTED_BY"}
        
        normalized_clauses = []
        for clause_out in result.clauses:
            valid_relations = []
            for rel in clause_out.relations:
                rel_type = rel.rel_type.strip().upper()
                if rel_type in ALLOWED_REL_TYPES:
                    rel.rel_type = rel_type
                    
                    # Handle empty/null target_text_ref
                    if not rel.target_text_ref or not rel.target_text_ref.strip():
                        if rel_type == "DEFINES_TERM":
                            logger.warning("skipping_empty_defines_term_relation")
                            continue
                        rel.target_text_ref = "UNKNOWN_REFERENCE"
                    
                    # Handle null context
                    if rel.context is None:
                        rel.context = ""
                        
                    valid_relations.append(rel)
                else:
                    logger.warning("skipping_invalid_relation_type", rel_type=rel.rel_type)
            
            clause_out.relations = valid_relations
            normalized_clauses.append(clause_out)
            
        result.clauses = normalized_clauses
        
        logger.info("extracting_relations_completed", clauses_processed=len(result.clauses))
        return result.clauses
    except Exception as e:
        logger.error("extracting_relations_failed", error=str(e))
        raise
