from __future__ import annotations

from collections import Counter
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from knowledge_scope.graph.api import router as graph_router
from knowledge_scope.graph.models import (
    ExtractionProvenance,
    GraphProvenance,
    entity_id_for,
    evidence_id_for,
    relation_id_for,
)
from knowledge_scope.graph.neo4j import GraphCanonicalEdge, GraphStoreError
from knowledge_scope.graph.retrieval import (
    GraphCanonicalReference,
    GraphEntityReference,
    GraphEntitySnapshot,
    GraphEvidence,
    GraphNeighbor,
    GraphRelationReference,
)

KNOWLEDGE_BASE_ID = UUID("33333333-3333-4333-8333-333333333333")
DOCUMENT_ID = UUID("11111111-1111-4111-8111-111111111111")
OTHER_DOCUMENT_ID = UUID("22222222-2222-4222-8222-222222222222")


def _evidence(
    *,
    document_id: UUID = DOCUMENT_ID,
    knowledge_base_id: UUID = KNOWLEDGE_BASE_ID,
    chunk_id: str = "chunk-1",
    page_start: int = 2,
    page_end: int = 3,
) -> GraphEvidence:
    provenance = GraphProvenance(
        document_id=document_id,
        knowledge_base_id=knowledge_base_id,
        chunk_id=chunk_id,
        page_start=page_start,
        page_end=page_end,
        source_block_ids=["block-1"],
        section_path=["章节一"],
        extraction_provenance=ExtractionProvenance(method="extraction-pipeline"),
    )
    return GraphEvidence(
        evidence_id=evidence_id_for(provenance),
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        chunk_id=chunk_id,
        page_start=page_start,
        page_end=page_end,
        source_block_ids=["block-1"],
        section_path=["章节一"],
        extraction_provenance=provenance.extraction_provenance,
    )


def _entity(
    name: str,
    *,
    entity_type: str = "概念",
    aliases: list[str] | None = None,
) -> GraphEntityReference:
    return GraphEntityReference(
        entity_id=entity_id_for(
            name,
            entity_type,
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            document_id=DOCUMENT_ID,
        ),
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=DOCUMENT_ID,
        canonical_name=name,
        entity_type=entity_type,
        aliases=aliases or [],
    )


def _snapshot(entity: GraphEntityReference) -> GraphEntitySnapshot:
    return GraphEntitySnapshot(
        entity=entity,
        evidence=[_evidence()],
        canonical_entities=[
            GraphCanonicalReference(
                canonical_entity_id="canonical_entity_v1_" + value * 32,
                knowledge_base_id=KNOWLEDGE_BASE_ID,
                canonical_name=entity.canonical_name,
                entity_type=entity.entity_type,
            )
            for value in ("a",)
        ],
    )


def _relation(
    source: GraphEntityReference,
    target: GraphEntityReference,
    relation_type: str,
) -> GraphRelationReference:
    return GraphRelationReference(
        relation_id=relation_id_for(
            source.entity_id,
            target.entity_id,
            relation_type,
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            document_id=DOCUMENT_ID,
        ),
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=DOCUMENT_ID,
        source_entity_id=source.entity_id,
        target_entity_id=target.entity_id,
        relation_type=relation_type,
    )


def _neighbor(
    seed: GraphEntityReference,
    neighbor: GraphEntityReference,
    *,
    relation_type: str,
    direction: str = "forward",
) -> GraphNeighbor:
    source, target = (seed, neighbor) if direction == "forward" else (neighbor, seed)
    relation = _relation(source, target, relation_type)
    evidence = _evidence()
    return GraphNeighbor(
        seed_entity_id=seed.entity_id,
        neighbor=neighbor,
        relation=relation,
        canonical=None,
        direction=direction,  # type: ignore[arg-type]
        evidence=[evidence],
        neighbor_evidence=[evidence],
        relation_evidence=[evidence],
    )


class _FakeGraphStore:
    """Small replacement for the Neo4j adapter that keeps their reads explicit."""

    def __init__(
        self,
        *,
        snapshots: tuple[GraphEntitySnapshot, ...] = (),
        relations: tuple[GraphRelationReference, ...] = (),
        canonical_edges: tuple[GraphCanonicalEdge, ...] = (),
        entity: GraphEntitySnapshot | None = None,
        neighbors: tuple[GraphNeighbor, ...] = (),
        error: GraphStoreError | None = None,
    ) -> None:
        self.snapshots = snapshots
        self.relations = relations
        self.canonical_edges = canonical_edges
        self.entity = entity
        self.neighbors = neighbors
        self.error = error
        self.entity_scans: list[int] = []
        self.entity_lookups: list[str] = []
        self.degree_ranks: list[int] = []
        self.id_lookups: list[tuple[str, ...]] = []
        self.relation_calls: list[tuple[tuple[str, ...] | None, UUID | None]] = []
        self.canonical_edge_calls: list[tuple[str, ...] | None] = []
        self.document_filters: list[UUID | None] = []

    def list_retrieval_entities(
        self,
        knowledge_base_id: UUID,
        *,
        document_id: UUID | None = None,
        max_entities: int = 10_000,
    ) -> tuple[GraphEntitySnapshot, ...]:
        self.entity_scans.append(max_entities)
        self.document_filters.append(document_id)
        self._raise_if_broken()
        return self.snapshots

    def list_retrieval_entity_ids_by_degree(
        self,
        knowledge_base_id: UUID,
        *,
        document_id: UUID | None = None,
        max_entities: int = 200,
    ) -> tuple[str, ...]:
        self.degree_ranks.append(max_entities)
        self.document_filters.append(document_id)
        self._raise_if_broken()
        degrees: Counter[str] = Counter()
        for relation in self.relations:
            degrees[relation.source_entity_id] += 1
            degrees[relation.target_entity_id] += 1
        ranked = sorted(degrees.items(), key=lambda item: (-item[1], item[0]))
        return tuple(entity_id for entity_id, _ in ranked[:max_entities])

    def get_retrieval_entities(
        self,
        knowledge_base_id: UUID,
        entity_ids: Sequence[str],
    ) -> tuple[GraphEntitySnapshot, ...]:
        self.id_lookups.append(tuple(entity_ids))
        self._raise_if_broken()
        by_id = {snapshot.entity.entity_id: snapshot for snapshot in self.snapshots}
        return tuple(by_id[entity_id] for entity_id in entity_ids if entity_id in by_id)

    def list_retrieval_relations(
        self,
        knowledge_base_id: UUID,
        *,
        entity_ids: Sequence[str] | None = None,
        document_id: UUID | None = None,
        max_relations: int = 5_000,
    ) -> tuple[GraphRelationReference, ...]:
        self.relation_calls.append((tuple(entity_ids) if entity_ids else None, document_id))
        self._raise_if_broken()
        return self.relations

    def list_retrieval_canonical_edges(
        self,
        knowledge_base_id: UUID,
        *,
        entity_ids: Sequence[str] | None = None,
        max_edges: int = 5_000,
    ) -> tuple[GraphCanonicalEdge, ...]:
        self.canonical_edge_calls.append(tuple(entity_ids) if entity_ids else None)
        self._raise_if_broken()
        return self.canonical_edges

    def get_retrieval_entity(
        self,
        knowledge_base_id: UUID,
        entity_id: str,
    ) -> GraphEntitySnapshot | None:
        self.entity_lookups.append(entity_id)
        self._raise_if_broken()
        return self.entity

    def get_retrieval_neighbors(
        self,
        knowledge_base_id: UUID,
        entity_id: str,
        *,
        max_neighbors: int = 20,
        max_relations: int = 100,
    ) -> tuple[GraphNeighbor, ...]:
        self._raise_if_broken()
        return self.neighbors

    def _raise_if_broken(self) -> None:
        if self.error is not None:
            raise self.error


def _build_app(store: Any) -> FastAPI:
    application = FastAPI()
    application.state.graph_store = store
    application.include_router(graph_router, prefix="/api/v1")
    return application


@asynccontextmanager
async def _client(store: Any) -> AsyncIterator[AsyncClient]:
    async with AsyncClient(
        transport=ASGITransport(app=_build_app(store)),
        base_url="http://testserver",
    ) as http_client:
        yield http_client


@pytest.mark.anyio
async def test_overview_selects_the_busiest_entities_and_keeps_edges_inside_them() -> None:
    hub = _entity("水体")
    first = _entity("水泵")
    second = _entity("阀门")
    isolated = _entity("管道材料")
    store = _FakeGraphStore(
        snapshots=tuple(_snapshot(entity) for entity in (hub, first, second, isolated)),
        relations=(
            _relation(hub, first, "包含"),
            _relation(hub, second, "包含"),
        ),
    )

    async with _client(store) as http_client:
        response = await http_client.get(f"/api/v1/knowledge-bases/{KNOWLEDGE_BASE_ID}/graph")

    assert response.status_code == 200
    payload = response.json()
    assert payload["nodes"][0]["canonical_name"] == "水体"
    assert {node["canonical_name"] for node in payload["nodes"]} == {"水体", "水泵", "阀门"}
    assert [node["degree"] for node in payload["nodes"]] == [2, 1, 1]
    assert payload["relation_count"] == 2
    assert payload["truncated"] is False
    assert payload["entity_types"] == [{"entity_type": "概念", "entity_count": 3}]
    assert store.degree_ranks == [200]
    assert store.entity_scans == []


@pytest.mark.anyio
async def test_overview_falls_back_to_entity_scan_without_relations() -> None:
    store = _FakeGraphStore(snapshots=(_snapshot(_entity("水泵")), _snapshot(_entity("阀门"))))

    async with _client(store) as http_client:
        response = await http_client.get(f"/api/v1/knowledge-bases/{KNOWLEDGE_BASE_ID}/graph")

    payload = response.json()
    assert [node["canonical_name"] for node in payload["nodes"]] == ["水泵", "阀门"]
    assert [node["degree"] for node in payload["nodes"]] == [0, 0]
    assert payload["relation_count"] == 0
    assert store.entity_scans == [200]


@pytest.mark.anyio
async def test_overview_filters_orphan_edges_outside_the_selected_nodes() -> None:
    kept = _entity("传感器")
    other = _entity("执行器")
    outside = _entity("外部系统")
    store = _FakeGraphStore(
        snapshots=(_snapshot(kept), _snapshot(other), _snapshot(outside)),
        relations=(_relation(kept, other, "连接"), _relation(other, outside, "包含")),
    )

    async with _client(store) as http_client:
        response = await http_client.get(
            f"/api/v1/knowledge-bases/{KNOWLEDGE_BASE_ID}/graph?limit=2"
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["truncated"] is True
    assert payload["nodes"][0]["canonical_name"] == "执行器"
    assert len(payload["nodes"]) == 2
    assert payload["relation_count"] == 1


@pytest.mark.anyio
async def test_overview_draws_canonical_merges_as_bridge_edges() -> None:
    first = _entity("研究院")
    second = _entity("设计院")
    store = _FakeGraphStore(
        snapshots=(_snapshot(first), _snapshot(second)),
        relations=(),
        canonical_edges=(
            GraphCanonicalEdge(
                canonical_entity_id="canonical_entity_v1_" + "b" * 32,
                canonical_name="研究院",
                source_entity_id=first.entity_id,
                target_entity_id=second.entity_id,
            ),
        ),
    )

    async with _client(store) as http_client:
        response = await http_client.get(f"/api/v1/knowledge-bases/{KNOWLEDGE_BASE_ID}/graph")

    payload = response.json()
    assert payload["relation_count"] == 1
    edge = payload["edges"][0]
    assert edge["kind"] == "canonical_bridge"
    assert edge["relation_type"] == "研究院"
    assert [node["degree"] for node in payload["nodes"]] == [1, 1]
    assert store.canonical_edge_calls and store.canonical_edge_calls[0] is not None


@pytest.mark.anyio
async def test_overview_scopes_document_views_and_skips_bridges_there() -> None:
    store = _FakeGraphStore(snapshots=(_snapshot(_entity("水体")),), relations=())

    async with _client(store) as http_client:
        response = await http_client.get(
            f"/api/v1/knowledge-bases/{KNOWLEDGE_BASE_ID}/graph?document_id={DOCUMENT_ID}"
        )

    assert response.status_code == 200
    assert store.canonical_edge_calls == []
    assert store.document_filters and set(store.document_filters) == {DOCUMENT_ID}
    assert store.relation_calls[0][1] == DOCUMENT_ID


@pytest.mark.anyio
async def test_overview_search_matches_canonical_names_and_aliases() -> None:
    target = _entity("离心泵", aliases=["Pump A"])
    store = _FakeGraphStore(
        snapshots=(_snapshot(_entity("阀门")), _snapshot(target)),
        relations=(),
    )

    async with _client(store) as http_client:
        by_alias = await http_client.get(
            f"/api/v1/knowledge-bases/{KNOWLEDGE_BASE_ID}/graph?search=pump%20a"
        )
        by_name = await http_client.get(
            f"/api/v1/knowledge-bases/{KNOWLEDGE_BASE_ID}/graph?search=离心"
        )

    assert [node["canonical_name"] for node in by_alias.json()["nodes"]] == ["离心泵"]
    assert [node["canonical_name"] for node in by_name.json()["nodes"]] == ["离心泵"]
    assert store.entity_scans[0] == 10_000


@pytest.mark.anyio
async def test_overview_filters_by_entity_type() -> None:
    concept = _entity("压力", entity_type="概念")
    organisation = _entity("研究院", entity_type="机构")
    store = _FakeGraphStore(snapshots=(_snapshot(concept), _snapshot(organisation)), relations=())

    async with _client(store) as http_client:
        response = await http_client.get(
            f"/api/v1/knowledge-bases/{KNOWLEDGE_BASE_ID}/graph?entity_type=机构"
        )

    payload = response.json()
    assert [node["canonical_name"] for node in payload["nodes"]] == ["研究院"]
    assert [option["entity_type"] for option in payload["entity_types"]] == ["机构"]


@pytest.mark.anyio
async def test_overview_rejects_invalid_limits() -> None:
    store = _FakeGraphStore()

    async with _client(store) as http_client:
        for query in ("limit=0", "limit=501"):
            response = await http_client.get(
                f"/api/v1/knowledge-bases/{KNOWLEDGE_BASE_ID}/graph?{query}"
            )
            assert response.status_code == 422
    assert store.entity_scans == []


@pytest.mark.anyio
async def test_overview_reports_unavailable_graph_store() -> None:
    store = _FakeGraphStore(error=GraphStoreError("Neo4j read operation failed"))

    async with _client(store) as http_client:
        response = await http_client.get(f"/api/v1/knowledge-bases/{KNOWLEDGE_BASE_ID}/graph")

    assert response.status_code == 503
    assert "图谱服务暂时不可用" in response.json()["detail"]


@pytest.mark.anyio
async def test_entity_detail_returns_bridges_evidence_and_neighbours() -> None:
    seed = _entity("水质标准")
    neighbor = _entity("浊度")
    store = _FakeGraphStore(
        entity=_snapshot(seed),
        neighbors=(_neighbor(seed, neighbor, relation_type="引用"),),
    )

    async with _client(store) as http_client:
        response = await http_client.get(
            f"/api/v1/knowledge-bases/{KNOWLEDGE_BASE_ID}/graph/entities/{seed.entity_id}"
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["entity"]["entity_id"] == seed.entity_id
    assert payload["entity"]["degree"] == 1
    assert payload["canonical_entities"][0]["canonical_name"] == "水质标准"
    assert payload["evidence"][0]["page_start"] == 2
    assert payload["neighbors"][0] == {
        "entity_id": neighbor.entity_id,
        "canonical_name": "浊度",
        "entity_type": "概念",
        "document_id": str(DOCUMENT_ID),
        "direction": "forward",
        "relation_id": payload["neighbors"][0]["relation_id"],
        "relation_type": "引用",
        "page_start": 2,
        "page_end": 3,
        "shared_canonical_name": None,
    }


@pytest.mark.anyio
async def test_entity_detail_returns_404_for_missing_entity() -> None:
    store = _FakeGraphStore(entity=None)
    entity_id = f"entity_v2_{'0' * 64}"

    async with _client(store) as http_client:
        response = await http_client.get(
            f"/api/v1/knowledge-bases/{KNOWLEDGE_BASE_ID}/graph/entities/{entity_id}"
        )

    assert response.status_code == 404
    assert store.entity_lookups == [entity_id]


@pytest.mark.anyio
async def test_entity_detail_rejects_foreign_identifiers_without_reads() -> None:
    store = _FakeGraphStore(entity=None)

    async with _client(store) as http_client:
        response = await http_client.get(
            f"/api/v1/knowledge-bases/{KNOWLEDGE_BASE_ID}/graph/entities/not-an-entity"
        )

    assert response.status_code == 404
    assert store.entity_lookups == []


@pytest.mark.anyio
async def test_entity_detail_reports_unavailable_graph_store() -> None:
    store = _FakeGraphStore(error=GraphStoreError("Neo4j read operation failed"))

    async with _client(store) as http_client:
        response = await http_client.get(
            f"/api/v1/knowledge-bases/{KNOWLEDGE_BASE_ID}/graph/entities/entity_v2_{'0' * 64}"
        )

    assert response.status_code == 503
    assert "图谱服务暂时不可用" in response.json()["detail"]
