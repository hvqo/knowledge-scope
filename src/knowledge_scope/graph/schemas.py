"""Read-only response models for the KnowledgeScope knowledge graph surface."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

GraphNeighborDirection = Literal["forward", "reverse", "canonical_bridge"]
GraphEdgeKind = Literal["relation", "canonical_bridge"]


class GraphNodeResponse(BaseModel):
    """One canvas node: a local entity reduced to displayable fields."""

    model_config = ConfigDict(extra="forbid")

    entity_id: str
    canonical_name: str
    entity_type: str
    aliases: list[str] = Field(default_factory=list)
    document_id: UUID
    canonical_entity_ids: list[str] = Field(default_factory=list)
    evidence_count: int = Field(ge=0)
    degree: int = Field(ge=0)


class GraphEdgeResponse(BaseModel):
    """One canvas edge: a relation or a canonical merge reduced to display fields."""

    model_config = ConfigDict(extra="forbid")

    relation_id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    kind: GraphEdgeKind = "relation"
    document_id: UUID | None = None


class GraphEntityTypeOption(BaseModel):
    """Selectable entity type with its occurrence count in the scanned graph."""

    model_config = ConfigDict(extra="forbid")

    entity_type: str
    entity_count: int = Field(ge=0)


class GraphOverviewResponse(BaseModel):
    """One bounded, evidence-backed subgraph prepared for canvas rendering."""

    model_config = ConfigDict(extra="forbid")

    knowledge_base_id: UUID
    entity_count: int = Field(ge=0)
    relation_count: int = Field(ge=0)
    entity_types: list[GraphEntityTypeOption] = Field(default_factory=list)
    truncated: bool
    nodes: list[GraphNodeResponse] = Field(default_factory=list)
    edges: list[GraphEdgeResponse] = Field(default_factory=list)


class GraphEvidenceResponse(BaseModel):
    """Source location of one graph statement."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: UUID
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    section_path: list[str] = Field(default_factory=list)


class GraphCanonicalEntityResponse(BaseModel):
    """A canonical entity that the inspected local entity has been linked to."""

    model_config = ConfigDict(extra="forbid")

    canonical_entity_id: str
    canonical_name: str
    entity_type: str


class GraphNeighborResponse(BaseModel):
    """One neighbour of the inspected entity, with the connecting statement."""

    model_config = ConfigDict(extra="forbid")

    entity_id: str
    canonical_name: str
    entity_type: str
    document_id: UUID
    direction: GraphNeighborDirection
    relation_id: str | None = None
    relation_type: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    shared_canonical_name: str | None = None


class GraphEntityDetailResponse(BaseModel):
    """Everything one entity node needs for its detail panel."""

    model_config = ConfigDict(extra="forbid")

    entity: GraphNodeResponse
    canonical_entities: list[GraphCanonicalEntityResponse] = Field(default_factory=list)
    evidence: list[GraphEvidenceResponse] = Field(default_factory=list)
    neighbors: list[GraphNeighborResponse] = Field(default_factory=list)


__all__ = [
    "GraphCanonicalEntityResponse",
    "GraphEdgeKind",
    "GraphEdgeResponse",
    "GraphEntityDetailResponse",
    "GraphEntityTypeOption",
    "GraphEvidenceResponse",
    "GraphNeighborDirection",
    "GraphNeighborResponse",
    "GraphNodeResponse",
    "GraphOverviewResponse",
]
