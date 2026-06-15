from pydantic import BaseModel, Field
from datetime import date
from typing import Literal, Optional, List

class ClassifierOutput(BaseModel):
    title: str
    doc_type: str
    publish_date: Optional[date] = None
    is_amendment: bool
    amends_document: Optional[str] = None

class BoundaryOutput(BaseModel):
    node_type: Literal["CHAPTER","SCHEDULE","SECTION","SUBSECTION","CLAUSE","SUBCLAUSE","PREAMBLE","DEFINITION","BODY_TEXT","IGNORE"]
    level: Optional[int] = Field(default=None, ge=1, le=6)
    is_boundary_break: bool
    heading_text: Optional[str] = None

class RelationItem(BaseModel):
    rel_type: Literal["REFERS_TO", "DEFINES_TERM", "SUBSTITUTES", "INSERTED_BY", "OMITTED_BY"]
    target_text_ref: Optional[str] = None
    context: Optional[str] = None
    effective_date: Optional[date] = None  # Parse as date if present

class ExtractorClauseOutput(BaseModel):
    source_clause_index: int
    relations: List[RelationItem]

class QueryExpansionOutput(BaseModel):
    expansions: List[str] = Field(default_factory=list, description="Exactly 3 distinct search query variations.")

class CompressorOutput(BaseModel):
    relevant_sentences: List[str] = Field(default_factory=list, description="List of exact relevant sentences extracted from the source text.")

class ComplianceAuditResult(BaseModel):
    status: Literal["COMPLIANT", "NON_COMPLIANT", "NEEDS_REVIEW"]
    section: Optional[str] = Field(default=None, description="The regulation section or clause referenced (e.g. Section 4(1)).")
    rationale: str = Field(..., description="A concise, direct explanation of why this status was chosen, citing specific values or rules.")
    regulation_excerpt: Optional[str] = Field(default=None, description="Verbatim short excerpt from the regulation context.")
    entity_excerpt: Optional[str] = Field(default=None, description="Verbatim short excerpt from the entity text.")


