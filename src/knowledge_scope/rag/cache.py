"""Optional retrieval cache so repeated questions skip the retrieval branches.

Only the assembled *context* is cached, never the generated answer: retrieval is
deterministic for a given query and corpus snapshot, while answers are not.
Every cache failure is swallowed and logged at debug level, so a broken or
missing cache can never fail a question.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from knowledge_scope.shared.config import Settings

from .context import ContextSelection, SelectedContextItem
from .schemas import RAGCitation, RAGRetrievalMode

logger = logging.getLogger(__name__)

RETRIEVAL_CACHE_KEY_VERSION = "v2"
REWRITE_CACHE_KEY_VERSION = "v1"


@dataclass(frozen=True, slots=True)
class CachedRetrieval:
    """One cached context together with how it was actually produced."""

    retrieval_mode: RAGRetrievalMode
    escalated: bool
    context: ContextSelection


class RetrievalCache(Protocol):
    """Minimal synchronous cache surface used by the RAG service."""

    def get(self, key: str) -> str | None:  # pragma: no cover - protocol
        """Return the cached payload, or ``None`` when it is absent or expired."""

    def set(self, key: str, payload: str, ttl_seconds: int) -> None:  # pragma: no cover
        """Store one payload for a bounded time."""


def build_retrieval_cache_key(
    *,
    knowledge_base_id: UUID | None,
    document_id: UUID | None,
    retrieval_mode: str,
    query: str,
) -> str:
    """Return a stable cache key for one retrieval request."""

    payload = {
        "version": RETRIEVAL_CACHE_KEY_VERSION,
        "knowledge_base_id": str(knowledge_base_id) if knowledge_base_id else None,
        "document_id": str(document_id) if document_id else None,
        "retrieval_mode": retrieval_mode,
        "query": " ".join(query.split()).casefold(),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return f"rag:context:{RETRIEVAL_CACHE_KEY_VERSION}:{digest}"


def build_rewrite_cache_key(*, query: str, history_digest: str) -> str:
    """Return a stable cache key for one follow-up rewrite."""

    payload = {
        "version": REWRITE_CACHE_KEY_VERSION,
        "query": " ".join(query.split()).casefold(),
        "history": history_digest,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return f"rag:rewrite:{REWRITE_CACHE_KEY_VERSION}:{digest}"


def encode_retrieval(cached: CachedRetrieval) -> str:
    """Serialize one cached retrieval result."""

    return json.dumps(
        {
            "mode": cached.retrieval_mode,
            "escalated": cached.escalated,
            "items": [
                {
                    "citation": item.citation.model_dump(mode="json"),
                    "text": item.text,
                    "truncated": item.truncated,
                }
                for item in cached.context.items
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def decode_retrieval(payload: str) -> CachedRetrieval:
    """Rebuild one cached retrieval result, rejecting anything malformed."""

    parsed = json.loads(payload)
    entries = parsed["items"]
    if not isinstance(entries, list):
        raise ValueError("cached context items must be a list")
    mode = parsed["mode"]
    if mode not in {"dense", "unified"}:
        raise ValueError("cached retrieval mode is unknown")
    items = tuple(
        SelectedContextItem(
            citation=RAGCitation.model_validate(entry["citation"]),
            text=str(entry["text"]),
            truncated=bool(entry["truncated"]),
        )
        for entry in entries
    )
    return CachedRetrieval(
        retrieval_mode=mode,
        escalated=bool(parsed["escalated"]),
        context=ContextSelection(
            items=items,
            character_count=sum(len(item.text) for item in items),
        ),
    )


class MemoryRetrievalCache:
    """Process-local TTL cache used when no Redis instance is configured."""

    def __init__(self, *, max_entries: int = 256) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        self._max_entries = max_entries
        self._entries: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> str | None:
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            expires_at, payload = entry
            if expires_at <= now:
                self._entries.pop(key, None)
                return None
            self._entries.move_to_end(key)
            return payload

    def set(self, key: str, payload: str, ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            return
        expires_at = time.monotonic() + ttl_seconds
        with self._lock:
            self._entries[key] = (expires_at, payload)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)


class RedisRetrievalCache:
    """Redis-backed cache shared by every worker of one deployment.

    The Redis client is injected, so the package stays an optional dependency
    and tests can drive the cache without a server.
    """

    def __init__(self, client: object) -> None:
        self._client = client

    def get(self, key: str) -> str | None:
        value = self._client.get(key)  # type: ignore[attr-defined]
        if value is None:
            return None
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    def set(self, key: str, payload: str, ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            return
        self._client.set(key, payload, ex=ttl_seconds)  # type: ignore[attr-defined]


def _build_redis_cache(settings: Settings) -> RetrievalCache | None:
    if not settings.rag_cache_url:
        logger.warning("rag cache backend is redis but no cache url is configured")
        return None
    try:
        # Optional dependency, imported on demand.
        import redis
    except ImportError:
        logger.warning(
            "rag cache backend is redis but the redis package is missing; "
            "install the 'cache' extra to enable it"
        )
        return None
    try:
        client = redis.Redis.from_url(
            settings.rag_cache_url,
            socket_timeout=settings.rag_cache_timeout_seconds,
            socket_connect_timeout=settings.rag_cache_timeout_seconds,
        )
        client.ping()
    except Exception as error:  # pragma: no cover - depends on a live server
        logger.warning("rag cache is unavailable (%s); continuing without it", type(error).__name__)
        return None
    return RedisRetrievalCache(client)


def create_retrieval_cache(settings: Settings) -> RetrievalCache | None:
    """Build the configured cache, degrading to memory or to no cache."""

    backend = settings.rag_cache_backend
    if backend == "off":
        return None
    if backend == "redis":
        cache = _build_redis_cache(settings)
        if cache is not None:
            return cache
        # A configured Redis that cannot be reached must not remove the fast
        # path entirely, so fall back to the process-local cache.
        return MemoryRetrievalCache(max_entries=settings.rag_cache_max_entries)
    return MemoryRetrievalCache(max_entries=settings.rag_cache_max_entries)


__all__ = [
    "RETRIEVAL_CACHE_KEY_VERSION",
    "REWRITE_CACHE_KEY_VERSION",
    "CachedRetrieval",
    "MemoryRetrievalCache",
    "RedisRetrievalCache",
    "RetrievalCache",
    "build_retrieval_cache_key",
    "build_rewrite_cache_key",
    "create_retrieval_cache",
    "decode_retrieval",
    "encode_retrieval",
]
