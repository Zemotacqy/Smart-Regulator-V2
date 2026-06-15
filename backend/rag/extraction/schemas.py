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
    node_type: str
    level: Optional[int] = None
    is_boundary_break: bool
    heading_text: Optional[str] = None

class RelationItem(BaseModel):
    rel_type: str
    target_text_ref: Optional[str] = None
    context: Optional[str] = None
    effective_date: Optional[str] = None  # ISO date string e.g. "2023-07-15"

class ExtractorClauseOutput(BaseModel):
    source_clause_index: int
    relations: List[RelationItem]
