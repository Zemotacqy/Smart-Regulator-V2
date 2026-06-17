from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import date
from typing import Literal, Optional, List, Any
import structlog

logger = structlog.get_logger()

_ALLOWED_NODE_TYPES = frozenset({
    "CHAPTER", "SCHEDULE", "SECTION", "SUBSECTION", "CLAUSE", "SUBCLAUSE",
    "PREAMBLE", "DEFINITION", "BODY_TEXT", "IGNORE"
})

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

    @field_validator("node_type", mode="before")
    @classmethod
    def normalize_node_type(cls, v: Any) -> Any:
        if isinstance(v, str):
            v_upper = v.strip().upper()
            if v_upper == "PART":
                return "CHAPTER"
            if v_upper not in _ALLOWED_NODE_TYPES:
                logger.warning("unrecognized_node_type_fallback", original=v)
                return "BODY_TEXT"
            return v_upper
        return v

class RelationItem(BaseModel):
    rel_type: Literal["REFERS_TO", "DEFINES_TERM", "SUBSTITUTES", "INSERTED_BY", "OMITTED_BY"]
    target_text_ref: Optional[str] = None
    context: Optional[str] = None
    effective_date: Optional[str] = None  # ISO date string e.g. "2023-07-15"

class ExtractorClauseOutput(BaseModel):
    source_clause_index: int
    relations: List[RelationItem]

class QueryExpansionOutput(BaseModel):
    expansions: List[str] = Field(default_factory=list, description="Exactly 3 distinct search query variations.")

class CompressorOutput(BaseModel):
    relevant_sentences: List[str] = Field(default_factory=list, description="List of exact relevant sentences extracted from the source text.")

    @model_validator(mode="before")
    @classmethod
    def handle_sentences_alias(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "relevant_sentences" not in data or not data["relevant_sentences"]:
                if "sentences" in data and isinstance(data["sentences"], list):
                    sentences = []
                    for item in data["sentences"]:
                        if isinstance(item, str):
                            sentences.append(item)
                        elif isinstance(item, dict) and "text" in item:
                            sentences.append(item["text"])
                    data["relevant_sentences"] = sentences
        return data

class CompressedNodeItem(BaseModel):
    node_index: int = Field(..., description="The 0-based index of the node in the input list.")
    relevant_sentences: List[str] = Field(default_factory=list, description="List of exact relevant sentences extracted from this node's text.")

class BatchedCompressorOutput(BaseModel):
    nodes: List[CompressedNodeItem] = Field(default_factory=list)

class ComplianceAuditResult(BaseModel):
    status: Literal["COMPLIANT", "NON_COMPLIANT", "NEEDS_REVIEW"]
    section: Optional[str] = Field(default=None, description="The regulation section or clause referenced (e.g. Section 4(1)).")
    rationale: str = Field(..., description="A concise, direct explanation of why this status was chosen, citing specific values or rules.")
    regulation_excerpt: Optional[str] = Field(default=None, description="Verbatim short excerpt from the regulation context.")
    entity_excerpt: Optional[str] = Field(default=None, description="Verbatim short excerpt from the entity text.")


