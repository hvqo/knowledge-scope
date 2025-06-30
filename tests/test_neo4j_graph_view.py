from __future__ import annotations

from typing import Any
from uuid import UUID

import pytest

from knowledge_scope.graph.models import (
    ExtractionProvenance,
    GraphProvenance,
    entity_id_for,
    evidence_id_for,
    relation_id_for,
)
from knowledge_scope.graph.neo4j import GraphStoreError, Neo4jGraphStore
from knowledge_scope.shared.config import Settings

KNOWLEDGE_BASE_ID = UUID("33333333-3333-4333-8333-333333333333")
DOCUMENT_ID = UUID("11111111-1111-4111-8111-111111111111")


def _evidence_properties(
    *,
    chunk_id: str = "chunk-1",
    page_start: int = 2,
    page_end: int = 3,
) -> dict[str, Any]:
    provenance = GraphProvenance(
        document_id=DOCUMENT_ID,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        chunk_id=chunk_id,
        page_start=page_start,
        page_end=page_end,
        source_block_ids=["block-1"],
        section_path=["章节一"],
        extraction_provenance=ExtractionProvenance(method="extraction-pipeline"),
    )
    return {
        "evidence_id": evidence_id_for(provenance),
        "schema_version": "1.0",
        "knowledge_base_id": str(KNOWLEDGE_BASE_ID),
        "document_id": str(DOCUMENT_ID),
        "chunk_id": chunk_id,
        "page_start": page_start,
        "page_end": page_end,
        "source_block_ids": ["block-1"],
        "section_path": ["章节一"],
        "extraction_provenance_json": '{"method":"extraction-pipeline"}',
    }


def _entity_properties(name: str, entity_type: str = "概念") -> dict[str, Any]:
    return {
        "entity_id": entity_id_for(
            name,
            entity_type,
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            document_id=DOCUMENT_ID,
        ),
        "schema_version": "1.0",
        "knowledge_base_id": str(KNOWLEDGE_BASE_ID),
        "document_id": str(DOCUMENT_ID),
        "canonical_name": name,
        "entity_type": entity_type,
        "entity_type_normalized": entity_type,
        "aliases": [],
    }


def _relation_properties(
    source: dict[str, Any],
    target: dict[str, Any],
    relation_type: str,
) -> dict[str, Any]:
    return {
        "relation_id": relation_id_for(
            str(source["entity_id"]),
            str(target["entity_id"]),
            relation_type,
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            document_id=DOCUMENT_ID,
        ),
        "schema_version": "1.0",
        "knowledge_base_id": str(KNOWLEDGE_BASE_ID),
        "document_id": str(DOCUMENT_ID),
        "source_entity_id": source["entity_id"],
        "target_entity_id": target["entity_id"],
        "relation_type": relation_type,
    }


class _Rows:
    """Minimal stand-in for a Neo4j result."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def __iter__(self):
        return iter(self.records)

    def single(self) -> dict[str, Any] | None:
        return self.records[0] if self.records else None

    def consume(self) -> None:
        return None


class _ViewSession:
    def __init__(self, driver: _ViewDriver) -> None:
        self.driver = driver

    def __enter__(self) -> _ViewSession:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def run(self, query: str, **params: object) -> _Rows:
        self.driver.queries.append((query, params))
        if "UNWIND [source.entity_id, target.entity_id]" in query:
            return _Rows(list(self.driver.rank_records))
        if "source_membership:CANONICAL_MEMBER_OF" in query:
            return _Rows(list(self.driver.canonical_records))
        if "[:SOURCE_OF]->(relation:KnowledgeRelation" in query:
            return _Rows(list(self.driver.relation_records))
        if "properties(entity) AS entity," in query and "canonical_values," in query:
            return _Rows(list(self.driver.entity_records))
        return _Rows([])


class _ViewDriver:
    def __init__(
        self,
        *,
        relation_records: list[dict[str, Any]] | None = None,
        entity_records: list[dict[str, Any]] | None = None,
        rank_records: list[dict[str, Any]] | None = None,
        canonical_records: list[dict[str, Any]] | None = None,
    ) -> None:
        self.queries: list[tuple[str, dict[str, object]]] = []
        self.relation_records = relation_records or []
        self.entity_records = entity_records or []
        self.rank_records = rank_records or []
        self.canonical_records = canonical_records or []

    def session(self, *, database: str) -> _ViewSession:
        assert database == "neo4j"
        return _ViewSession(self)

    def close(self) -> None:
        return None


class _FailingDriver:
    def session(self, *, database: str) -> _ViewSession:
        del database
        raise RuntimeError("connection failed with password=should-not-leak")


def _store(driver: Any) -> Neo4jGraphStore:
    return Neo4jGraphStore(
        Settings(_env_file=None, environment="test"),
        driver=driver,
    )


def test_list_retrieval_relations_returns_scoped_relations() -> None:
    source = _entity_properties("进水口")
    target = _entity_properties("格栅")
    driver = _ViewDriver(
        relation_records=[
            {
                "relation": _relation_properties(source, target, "包含"),
                "source": source,
                "target": target,
            }
        ]
    )

    relations = _store(driver).list_retrieval_relations(KNOWLEDGE_BASE_ID, max_relations=10)

    assert len(relations) == 1
    relation = relations[0]
    assert relation.relation_type == "包含"
    assert relation.source_entity_id == source["entity_id"]
    assert relation.target_entity_id == target["entity_id"]
    query, params = next(
        (query, params) for query, params in driver.queries if "knowledge_base_id" in params
    )
    assert params["max_relations"] == 10
    assert params["knowledge_base_id"] == str(KNOWLEDGE_BASE_ID)
    assert "source.entity_id <> target.entity_id" in query


def test_list_retrieval_relations_skips_rows_without_a_relation() -> None:
    driver = _ViewDriver(relation_records=[{"relation": None, "source": None, "target": None}])

    assert _store(driver).list_retrieval_relations(KNOWLEDGE_BASE_ID) == ()


def test_list_retrieval_relations_rejects_out_of_range_limits() -> None:
    store = _store(_ViewDriver())

    with pytest.raises(ValueError, match="max_relations"):
        store.list_retrieval_relations(KNOWLEDGE_BASE_ID, max_relations=0)
    with pytest.raises(ValueError, match="max_relations"):
        store.list_retrieval_relations(KNOWLEDGE_BASE_ID, max_relations=50_001)


def test_list_retrieval_relations_wraps_driver_failures() -> None:
    with pytest.raises(GraphStoreError, match="Neo4j read operation failed"):
        _store(_FailingDriver()).list_retrieval_relations(KNOWLEDGE_BASE_ID)


def test_get_retrieval_entity_returns_snapshot_with_bridge_and_evidence() -> None:
    entity = _entity_properties("出水水质")
    driver = _ViewDriver(
        entity_records=[
            {
                "entity_id": entity["entity_id"],
                "entity": entity,
                "canonical_values": [
                    {
                        "canonical_entity_id": "canonical_entity_v1_" + "a" * 32,
                        "schema_version": "1.0",
                        "knowledge_base_id": str(KNOWLEDGE_BASE_ID),
                        "canonical_name": "出水水质",
                        "entity_type": "概念",
                        "aliases": [],
                    }
                ],
                "evidence_values": [_evidence_properties()],
            }
        ]
    )

    snapshot = _store(driver).get_retrieval_entity(KNOWLEDGE_BASE_ID, str(entity["entity_id"]))

    assert snapshot is not None
    assert snapshot.entity.canonical_name == "出水水质"
    assert [canonical.canonical_entity_id for canonical in snapshot.canonical_entities] == [
        "canonical_entity_v1_" + "a" * 32
    ]
    assert [evidence.page_start for evidence in snapshot.evidence] == [2]


def test_get_retrieval_entity_returns_none_when_missing() -> None:
    snapshot = _store(_ViewDriver()).get_retrieval_entity(
        KNOWLEDGE_BASE_ID,
        str(_entity_properties("缺失实体")["entity_id"]),
    )

    assert snapshot is None


def test_get_retrieval_entity_wraps_driver_failures() -> None:
    with pytest.raises(GraphStoreError, match="Neo4j read operation failed"):
        _store(_FailingDriver()).get_retrieval_entity(KNOWLEDGE_BASE_ID, "entity_v2_" + "0" * 64)


def test_list_retrieval_entity_ids_by_degree_returns_ranked_ids() -> None:
    driver = _ViewDriver(rank_records=[{"entity_id": "entity_v2_" + "a" * 64, "degree": 4}])

    ranked = _store(driver).list_retrieval_entity_ids_by_degree(KNOWLEDGE_BASE_ID, max_entities=50)

    assert ranked == ("entity_v2_" + "a" * 64,)
    query, params = next((query, params) for query, params in driver.queries if "UNWIND" in query)
    assert params["max_entities"] == 50
    assert "count(entity_id) AS degree" in query


def test_list_retrieval_entity_ids_by_degree_rejects_invalid_rankings() -> None:
    driver = _ViewDriver(rank_records=[{"entity_id": "   ", "degree": 1}])

    with pytest.raises(GraphStoreError, match="entity ranking"):
        _store(driver).list_retrieval_entity_ids_by_degree(KNOWLEDGE_BASE_ID)


def test_list_retrieval_entity_ids_by_degree_rejects_out_of_range_limits() -> None:
    store = _store(_ViewDriver())

    with pytest.raises(ValueError, match="max_entities"):
        store.list_retrieval_entity_ids_by_degree(KNOWLEDGE_BASE_ID, max_entities=0)
    with pytest.raises(ValueError, match="max_entities"):
        store.list_retrieval_entity_ids_by_degree(KNOWLEDGE_BASE_ID, max_entities=5_001)


def test_get_retrieval_entities_requests_only_the_selected_ids() -> None:
    entity = _entity_properties("曝气池")
    driver = _ViewDriver(
        entity_records=[
            {
                "entity_id": entity["entity_id"],
                "entity": entity,
                "canonical_values": [],
                "evidence_values": [_evidence_properties()],
            }
        ]
    )
    target = str(entity["entity_id"])

    snapshots = _store(driver).get_retrieval_entities(KNOWLEDGE_BASE_ID, [target, target, "  "])

    assert [snapshot.entity.entity_id for snapshot in snapshots] == [target]
    _, params = next(
        (query, params)
        for query, params in driver.queries
        if "entity.entity_id IN $entity_ids" in query
    )
    assert params["entity_ids"] == [target]


def test_get_retrieval_entities_skips_empty_requests() -> None:
    driver = _ViewDriver()

    assert _store(driver).get_retrieval_entities(KNOWLEDGE_BASE_ID, []) == ()
    assert driver.queries == []


def test_list_retrieval_canonical_edges_returns_merged_pairs() -> None:
    driver = _ViewDriver(
        canonical_records=[
            {
                "canonical_entity_id": "canonical_entity_v1_" + "c" * 32,
                "canonical_name": "研究院",
                "source_entity_id": "entity_v2_" + "a" * 64,
                "target_entity_id": "entity_v2_" + "b" * 64,
            }
        ]
    )

    edges = _store(driver).list_retrieval_canonical_edges(
        KNOWLEDGE_BASE_ID,
        entity_ids=["entity_v2_" + "a" * 64],
        max_edges=10,
    )

    assert len(edges) == 1
    assert edges[0].canonical_name == "研究院"
    assert edges[0].source_entity_id == "entity_v2_" + "a" * 64
    _, params = next(
        (query, params)
        for query, params in driver.queries
        if "CANONICAL_MEMBER_OF" in query and "LIMIT" in query
    )
    assert params["entity_ids"] == ["entity_v2_" + "a" * 64]
    assert params["max_edges"] == 10


def test_list_retrieval_canonical_edges_rejects_malformed_rows() -> None:
    driver = _ViewDriver(
        canonical_records=[
            {
                "canonical_entity_id": "",
                "canonical_name": "研究院",
                "source_entity_id": "entity_v2_" + "a" * 64,
                "target_entity_id": "entity_v2_" + "b" * 64,
            }
        ]
    )

    with pytest.raises(GraphStoreError, match="canonical bridge"):
        _store(driver).list_retrieval_canonical_edges(KNOWLEDGE_BASE_ID)


def test_list_retrieval_canonical_edges_rejects_out_of_range_limits() -> None:
    store = _store(_ViewDriver())

    with pytest.raises(ValueError, match="max_edges"):
        store.list_retrieval_canonical_edges(KNOWLEDGE_BASE_ID, max_edges=0)
    with pytest.raises(ValueError, match="max_edges"):
        store.list_retrieval_canonical_edges(KNOWLEDGE_BASE_ID, max_edges=50_001)


def test_list_retrieval_canonical_edges_wraps_driver_failures() -> None:
    with pytest.raises(GraphStoreError, match="Neo4j read operation failed"):
        _store(_FailingDriver()).list_retrieval_canonical_edges(KNOWLEDGE_BASE_ID)
