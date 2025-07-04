from __future__ import annotations

from uuid import UUID

import pytest

from knowledge_scope.rag.cache import (
    CachedRetrieval,
    MemoryRetrievalCache,
    RedisRetrievalCache,
    build_retrieval_cache_key,
    build_rewrite_cache_key,
    create_retrieval_cache,
    decode_retrieval,
    encode_retrieval,
)
from knowledge_scope.rag.context import ContextSelection, SelectedContextItem
from knowledge_scope.rag.schemas import RAGCitation
from knowledge_scope.shared.config import Settings

KNOWLEDGE_BASE_ID = UUID("11111111-1111-4111-8111-111111111111")
DOCUMENT_ID = UUID("22222222-2222-4222-8222-222222222222")


def _citation(marker: str = "C1") -> RAGCitation:
    return RAGCitation(
        marker=marker,
        document_id=DOCUMENT_ID,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        candidate_kind="chunk",
        chunk_id="chunk-1",
        page_start=1,
        page_end=2,
        source_block_ids=["block-1"],
        section_path=["章节"],
        snippet="来源片段",
        snippet_kind="source",
    )


def _context() -> ContextSelection:
    items = (
        SelectedContextItem(citation=_citation("C1"), text="第一段", truncated=False),
        SelectedContextItem(citation=_citation("C2"), text="第二段", truncated=True),
    )
    return ContextSelection(items=items, character_count=sum(len(item.text) for item in items))


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, environment="test", **overrides)  # type: ignore[arg-type]


def test_cache_key_is_stable_and_case_insensitive() -> None:
    first = build_retrieval_cache_key(
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=None,
        retrieval_mode="unified",
        query="  曝气池  溶解氧  ",
    )
    second = build_retrieval_cache_key(
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=None,
        retrieval_mode="unified",
        query="曝气池 溶解氧",
    )

    assert first == second
    assert first.startswith("rag:context:")


@pytest.mark.parametrize(
    "override",
    [
        {"knowledge_base_id": UUID(int=0)},
        {"document_id": DOCUMENT_ID},
        {"retrieval_mode": "dense"},
        {"query": "别的问题"},
    ],
)
def test_cache_key_separates_every_scope(override: dict[str, object]) -> None:
    base: dict[str, object] = {
        "knowledge_base_id": KNOWLEDGE_BASE_ID,
        "document_id": None,
        "retrieval_mode": "unified",
        "query": "问题",
    }
    baseline = build_retrieval_cache_key(**base)  # type: ignore[arg-type]

    assert build_retrieval_cache_key(**{**base, **override}) != baseline  # type: ignore[arg-type]


def _cached(mode: str = "unified", *, escalated: bool = False) -> CachedRetrieval:
    return CachedRetrieval(
        retrieval_mode=mode,  # type: ignore[arg-type]
        escalated=escalated,
        context=_context(),
    )


def test_retrieval_round_trip_preserves_citations_and_path() -> None:
    payload = encode_retrieval(_cached("unified", escalated=True))

    restored = decode_retrieval(payload)

    assert restored.retrieval_mode == "unified"
    assert restored.escalated is True
    assert [item.citation.marker for item in restored.context.items] == ["C1", "C2"]
    assert [item.truncated for item in restored.context.items] == [False, True]
    assert restored.context.character_count == 6
    assert restored.context.citations[0].snippet == "来源片段"


def test_retrieval_decoding_rejects_malformed_payloads() -> None:
    with pytest.raises(ValueError):
        decode_retrieval("not json")
    with pytest.raises(ValueError):
        decode_retrieval('{"mode": "unified", "escalated": false, "items": {}}')
    with pytest.raises(ValueError, match="mode"):
        decode_retrieval('{"mode": "auto", "escalated": false, "items": []}')
    with pytest.raises(KeyError):
        decode_retrieval('{"mode": "dense", "escalated": false, "items": [{"text": "缺失引用"}]}')


def test_rewrite_cache_key_separates_history() -> None:
    first = build_rewrite_cache_key(query="它呢", history_digest="a" * 64)
    second = build_rewrite_cache_key(query="它呢", history_digest="b" * 64)

    assert first != second
    assert first.startswith("rag:rewrite:")


def test_memory_cache_expires_and_evicts() -> None:
    cache = MemoryRetrievalCache(max_entries=2)

    cache.set("a", "1", ttl_seconds=60)
    cache.set("b", "2", ttl_seconds=60)
    cache.set("c", "3", ttl_seconds=60)

    assert cache.get("a") is None
    assert cache.get("b") == "2"
    assert cache.get("missing") is None
    # A non-positive ttl means "do not cache this entry at all".
    cache.set("expired", "4", ttl_seconds=0)
    assert cache.get("expired") is None

    with pytest.raises(ValueError):
        MemoryRetrievalCache(max_entries=0)


class _FakeRedisClient:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.expirations: list[int] = []

    def get(self, key: str) -> bytes | None:
        value = self.values.get(key)
        return value.encode("utf-8") if value is not None else None

    def set(self, key: str, payload: str, *, ex: int) -> None:
        self.values[key] = payload
        self.expirations.append(ex)


def test_redis_cache_uses_the_injected_client() -> None:
    client = _FakeRedisClient()
    cache = RedisRetrievalCache(client)

    cache.set("key", "payload", ttl_seconds=120)

    assert cache.get("key") == "payload"
    assert cache.get("missing") is None
    assert client.expirations == [120]


def test_cache_factory_respects_the_configured_backend() -> None:
    assert create_retrieval_cache(_settings(rag_cache_backend="off")) is None
    assert isinstance(
        create_retrieval_cache(_settings(rag_cache_backend="memory", rag_cache_max_entries=4)),
        MemoryRetrievalCache,
    )
    # Redis without a reachable server degrades to the process-local cache
    # instead of removing the fast path.
    fallback = create_retrieval_cache(
        _settings(rag_cache_backend="redis", rag_cache_url=None, rag_cache_max_entries=8)
    )
    assert isinstance(fallback, MemoryRetrievalCache)
