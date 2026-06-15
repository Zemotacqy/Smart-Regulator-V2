from dataclasses import dataclass, field
from uuid import UUID
from typing import List, Dict, Optional

@dataclass
class NodeCandidate:
    node_id: UUID
    doc_id: UUID
    parent_id: Optional[UUID]
    level: int
    node_type: str
    title: Optional[str]
    text_content: str
    breadcrumb: str
    score: Optional[float] = None
    file_name: Optional[str] = None

@dataclass
class SourceCitation:
    node_id: UUID
    doc_id: UUID
    file_name: str
    breadcrumb: str
    title: Optional[str]
    text_content: str
    verbatim_quote: Optional[str] = None

@dataclass
class QueryPipelineContext:
    """
    The single mutable context object that flows through all retrieval stages.
    Each stage reads from it and writes its outputs back to it.
    """
    request_id: str                          # UUID for end-to-end tracing
    original_query: str                       # Raw user query
    doc_filter: Optional[List[UUID]] = None      # Optional: scope to specific doc_ids

    # Populated by Stage A (Query Expander)
    expanded_queries: List[str] = field(default_factory=list)

    # Populated by Stage B (Hybrid Search)
    candidate_nodes: List[NodeCandidate] = field(default_factory=list)

    # Populated by Stage C (Hop Expander)
    expanded_nodes: List[NodeCandidate] = field(default_factory=list)
    inlined_definitions: Dict[str, str] = field(default_factory=dict)

    # Populated by Stage D (Reranker)
    reranked_nodes: List[NodeCandidate] = field(default_factory=list)

    # Populated by Stage E (Compressor)
    compressed_context: str = ""

    # Populated by Stage F (Generator)
    answer_text: str = ""
    source_citations: List[SourceCitation] = field(default_factory=list)

    # Timing data for observability
    stage_timings: Dict[str, float] = field(default_factory=dict)
