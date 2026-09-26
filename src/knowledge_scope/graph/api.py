"""Read-only knowledge graph endpoints for the product surface."""

# Chinese user-facing error text intentionally uses Chinese punctuation.
# ruff: noqa: RUF001

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Sequence
from typing import Final
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request, status

from knowledge_scope.graph.neo4j import (
    GraphCanonicalEdge,
    GraphStoreError,
    Neo4jGraphStore,
)
from knowledge_scope.graph.retrieval import (
    GraphEntitySnapshot,
    GraphNeighbor,
    GraphRelationReference,
)

from .schemas import (
    GraphCanonicalEntityResponse,
    GraphEdgeResponse,
    GraphEntityDetailResponse,
    GraphEntityTypeOption,
    GraphEvidenceResponse,
    GraphNeighborResponse,
    GraphNodeResponse,
    GraphOverviewResponse,
)

router = APIRouter(
    prefix="/knowledge-bases/{knowledge_base_id}/graph",
    tags=["graph"],
)

ENTITY_SCAN_LIMIT: Final = 10_000
RELATION_SCAN_LIMIT: Final = 5_000
CANONICAL_EDGE_LIMIT: Final = 5_000
NEIGHBOR_RELATION_LIMIT: Final = 100
ENTITY_TYPE_OPTION_LIMIT: Final = 20

GRAPH_UNAVAILABLE_DETAIL: Final = "图谱服务暂时不可用，请稍后重试。"
ENTITY_NOT_FOUND_DETAIL: Final = "实体不存在。"


def _matches_term(snapshot: GraphEntitySnapshot, term: str) -> bool:
    entity = snapshot.entity
    candidates = (entity.canonical_name, *entity.aliases)
    return any(term in value.casefold() for value in candidates)


def _filter_snapshots(
    snapshots: Sequence[GraphEntitySnapshot],
    *,
    term: str | None,
    entity_type: str | None,
) -> list[GraphEntitySnapshot]:
    return [
        snapshot
        for snapshot in snapshots
        if (term is None or _matches_term(snapshot, term))
        and (entity_type is None or snapshot.entity.entity_type == entity_type)
    ]


def _node_response(snapshot: GraphEntitySnapshot, degree: int) -> GraphNodeResponse:
    entity = snapshot.entity
    return GraphNodeResponse(
        entity_id=entity.entity_id,
        canonical_name=entity.canonical_name,
        entity_type=entity.entity_type,
        aliases=list(entity.aliases),
        document_id=entity.document_id,
        canonical_entity_ids=[
            canonical.canonical_entity_id for canonical in snapshot.canonical_entities
        ],
        evidence_count=len(snapshot.evidence),
        degree=degree,
    )


def _edge_response(relation: GraphRelationReference) -> GraphEdgeResponse:
    return GraphEdgeResponse(
        relation_id=relation.relation_id,
        source_entity_id=relation.source_entity_id,
        target_entity_id=relation.target_entity_id,
        relation_type=relation.relation_type,
        kind="relation",
        document_id=relation.document_id,
    )


def _canonical_edge_response(edge: GraphCanonicalEdge) -> GraphEdgeResponse:
    return GraphEdgeResponse(
        relation_id=f"canonical_bridge:{edge.canonical_entity_id}:{edge.source_entity_id}:{edge.target_entity_id}",
        source_entity_id=edge.source_entity_id,
        target_entity_id=edge.target_entity_id,
        relation_type=edge.canonical_name,
        kind="canonical_bridge",
        document_id=None,
    )


def _entity_type_options(snapshots: Sequence[GraphEntitySnapshot]) -> list[GraphEntityTypeOption]:
    counts: Counter[str] = Counter(snapshot.entity.entity_type for snapshot in snapshots)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [
        GraphEntityTypeOption(entity_type=entity_type, entity_count=count)
        for entity_type, count in ranked[:ENTITY_TYPE_OPTION_LIMIT]
    ]


def _neighbor_response(neighbor: GraphNeighbor) -> GraphNeighborResponse:
    relation = neighbor.relation
    relation_pages = neighbor.relation_evidence
    return GraphNeighborResponse(
        entity_id=neighbor.neighbor.entity_id,
        canonical_name=neighbor.neighbor.canonical_name,
        entity_type=neighbor.neighbor.entity_type,
        document_id=neighbor.neighbor.document_id,
        direction=neighbor.direction,
        relation_id=relation.relation_id if relation is not None else None,
        relation_type=relation.relation_type if relation is not None else None,
        page_start=min((item.page_start for item in relation_pages), default=None),
        page_end=max((item.page_end for item in relation_pages), default=None),
        shared_canonical_name=neighbor.canonical.canonical_name if neighbor.canonical else None,
    )


def _neighbor_degree(neighbors: Sequence[GraphNeighbor]) -> int:
    keys: set[str] = set()
    for neighbor in neighbors:
        if neighbor.relation is not None:
            keys.add(neighbor.relation.relation_id)
        elif neighbor.canonical is not None:
            keys.add(neighbor.canonical.canonical_entity_id)
    return len(keys)


async def _select_snapshots(
    store: Neo4jGraphStore,
    knowledge_base_id: UUID,
    *,
    term: str | None,
    entity_type: str | None,
    document_id: UUID | None,
    limit: int,
) -> tuple[tuple[GraphEntitySnapshot, ...], bool]:
    """Pick the entities to render and report whether more existed.

    Without a filter the busiest entities win, so the canvas shows a connected
    subgraph instead of an arbitrary slice of the knowledge base.
    """

    def by_name(snapshots: Sequence[GraphEntitySnapshot]) -> list[GraphEntitySnapshot]:
        return sorted(
            snapshots,
            key=lambda snapshot: (snapshot.entity.canonical_name, snapshot.entity.entity_id),
        )

    if term is not None or entity_type is not None:
        snapshots = await asyncio.to_thread(
            store.list_retrieval_entities,
            knowledge_base_id,
            document_id=document_id,
            max_entities=ENTITY_SCAN_LIMIT,
        )
        matched = by_name(_filter_snapshots(snapshots, term=term, entity_type=entity_type))
        return tuple(matched[:limit]), len(matched) > limit

    ranked_ids = await asyncio.to_thread(
        store.list_retrieval_entity_ids_by_degree,
        knowledge_base_id,
        document_id=document_id,
        max_entities=limit,
    )
    if not ranked_ids:
        snapshots = await asyncio.to_thread(
            store.list_retrieval_entities,
            knowledge_base_id,
            document_id=document_id,
            max_entities=limit,
        )
        ranked = by_name(snapshots)
        return tuple(ranked[:limit]), len(ranked) > limit
    selected = await asyncio.to_thread(
        store.get_retrieval_entities,
        knowledge_base_id,
        ranked_ids,
    )
    order = {entity_id: position for position, entity_id in enumerate(ranked_ids)}
    return (
        tuple(sorted(selected, key=lambda snapshot: order[snapshot.entity.entity_id])),
        len(ranked_ids) >= limit,
    )


@router.get("", response_model=GraphOverviewResponse)
async def read_graph_overview(
    knowledge_base_id: UUID,
    request: Request,
    search: str | None = Query(default=None, min_length=1, max_length=200),
    entity_type: str | None = Query(default=None, min_length=1, max_length=100),
    document_id: UUID | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=500),
) -> GraphOverviewResponse:
    """Return one bounded, connected subgraph so a canvas can render it."""
    store: Neo4jGraphStore = request.app.state.graph_store
    term = search.strip().casefold() if search else None
    try:
        selected, more_entities = await _select_snapshots(
            store,
            knowledge_base_id,
            term=term,
            entity_type=entity_type,
            document_id=document_id,
            limit=limit,
        )
        node_ids = {snapshot.entity.entity_id for snapshot in selected}
        ordered_ids = sorted(node_ids)
        relations, canonical_edges = await asyncio.gather(
            asyncio.to_thread(
                store.list_retrieval_relations,
                knowledge_base_id,
                entity_ids=ordered_ids,
                document_id=document_id,
                max_relations=RELATION_SCAN_LIMIT,
            ),
            asyncio.to_thread(
                store.list_retrieval_canonical_edges,
                knowledge_base_id,
                entity_ids=ordered_ids,
                max_edges=CANONICAL_EDGE_LIMIT,
            )
            if document_id is None
            else asyncio.sleep(0, result=()),
        )
    except GraphStoreError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=GRAPH_UNAVAILABLE_DETAIL,
        ) from error

    edges = [
        _edge_response(relation)
        for relation in relations
        if relation.source_entity_id in node_ids and relation.target_entity_id in node_ids
    ]
    edges.extend(
        _canonical_edge_response(edge)
        for edge in canonical_edges
        if edge.source_entity_id in node_ids and edge.target_entity_id in node_ids
    )
    degrees: Counter[str] = Counter()
    for edge in edges:
        degrees[edge.source_entity_id] += 1
        degrees[edge.target_entity_id] += 1
    return GraphOverviewResponse(
        knowledge_base_id=knowledge_base_id,
        entity_count=len(selected),
        relation_count=len(edges),
        entity_types=_entity_type_options(selected),
        truncated=more_entities
        or len(relations) >= RELATION_SCAN_LIMIT
        or len(canonical_edges) >= CANONICAL_EDGE_LIMIT,
        nodes=[
            _node_response(snapshot, degrees.get(snapshot.entity.entity_id, 0))
            for snapshot in selected
        ],
        edges=edges,
    )


@router.get("/entities/{entity_id}", response_model=GraphEntityDetailResponse)
async def read_graph_entity(
    knowledge_base_id: UUID,
    entity_id: str,
    request: Request,
    max_neighbors: int = Query(default=20, ge=1, le=100),
) -> GraphEntityDetailResponse:
    """Return one entity with its canonical bridges, evidence, and neighbours."""
    store: Neo4jGraphStore = request.app.state.graph_store
    if not entity_id.startswith("entity_v2_"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ENTITY_NOT_FOUND_DETAIL,
        )
    try:
        snapshot, neighbors = await asyncio.gather(
            asyncio.to_thread(store.get_retrieval_entity, knowledge_base_id, entity_id),
            asyncio.to_thread(
                store.get_retrieval_neighbors,
                knowledge_base_id,
                entity_id,
                max_neighbors=max_neighbors,
                max_relations=NEIGHBOR_RELATION_LIMIT,
            ),
        )
    except GraphStoreError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=GRAPH_UNAVAILABLE_DETAIL,
        ) from error
    if snapshot is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ENTITY_NOT_FOUND_DETAIL,
        )
    return GraphEntityDetailResponse(
        entity=_node_response(snapshot, _neighbor_degree(neighbors)),
        canonical_entities=[
            GraphCanonicalEntityResponse(
                canonical_entity_id=canonical.canonical_entity_id,
                canonical_name=canonical.canonical_name,
                entity_type=canonical.entity_type,
            )
            for canonical in snapshot.canonical_entities
        ],
        evidence=[
            GraphEvidenceResponse(
                chunk_id=evidence.chunk_id,
                document_id=evidence.document_id,
                page_start=evidence.page_start,
                page_end=evidence.page_end,
                section_path=list(evidence.section_path),
            )
            for evidence in snapshot.evidence
        ],
        neighbors=[_neighbor_response(neighbor) for neighbor in neighbors],
    )


__all__ = ["router"]
