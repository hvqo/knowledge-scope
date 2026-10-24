# ruff: noqa: RUF001
"""Independent, model-fingerprinted Qdrant collection for native visual vectors.

Only original image pixels are encoded here.  The collection never mixes with
the chunk or representation collections, and its name embeds the encoder
contract so two different visual models can never silently share one space.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from time import perf_counter
from typing import Any, Literal
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from knowledge_scope.shared.config import Settings

from .visual_embedding import VisualEmbeddingContract, VisualEmbeddingEncoder

VISUAL_INDEX_SCHEMA_VERSION = "1.0"
VISUAL_POINT_NAMESPACE = UUID("7a1f81d2-9a64-5f47-8e45-3c2b1d9f0a66")
_VISUAL_COLLECTION_ROLE = "visual"
_MODEL_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


class VisualIndexError(RuntimeError):
    """Raised when the visual collection cannot be used safely."""


class VisualCollectionConfigurationError(VisualIndexError):
    """Raised when the visual collection conflicts with the configured contract."""


class VisualRepresentationRef(BaseModel):
    """Minimal representation provenance retained for citation construction."""

    model_config = ConfigDict(extra="forbid")

    representation_id: str = Field(pattern=r"^representation_v1_[0-9a-f]{64}$")
    representation_type: str = Field(min_length=1, max_length=40)
    content: str | None = Field(default=None, max_length=4_000)
    reference: str | None = Field(default=None, max_length=4_000)


class VisualVectorPayload(BaseModel):
    """One original image asset, mapped to its authoritative Evidence lineage."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = VISUAL_INDEX_SCHEMA_VERSION
    collection_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    visual_branch: Literal["image"] = "image"
    knowledge_base_id: UUID
    document_id: UUID
    evidence_id: str = Field(pattern=r"^evidence_v1_[0-9a-f]{64}$")
    page_number: int = Field(ge=1)
    source_block_ids: list[str] = Field(min_length=1)
    asset_index: int = Field(ge=0)
    asset_ref: str = Field(min_length=1)
    asset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    asset_mime: str = Field(min_length=1)
    image_width: int | None = Field(default=None, ge=1)
    image_height: int | None = Field(default=None, ge=1)
    caption: str | None = Field(default=None, max_length=2_000)
    context: str | None = Field(default=None, max_length=4_000)
    representation_ids: list[str] = Field(default_factory=list)
    representations: list[VisualRepresentationRef] = Field(default_factory=list)
    policy_mode: Literal["text_only", "text_and_visual"]
    policy_reason: str | None = Field(default=None, max_length=500)
    policy_version: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    embedding_dim: int = Field(ge=1)
    indexed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("source_block_ids")
    @classmethod
    def validate_unique_blocks(cls, values: list[str]) -> list[str]:
        if len(set(values)) != len(values):
            raise ValueError("source_block_ids must be unique")
        return values

    @field_validator("representation_ids")
    @classmethod
    def validate_unique_representations(cls, values: list[str]) -> list[str]:
        if len(set(values)) != len(values):
            raise ValueError("representation_ids must be unique")
        return values


class VisualSearchHit(BaseModel):
    """One raw visual hit with its model-fingerprinted payload."""

    model_config = ConfigDict(extra="forbid")

    rank: int = Field(ge=1)
    score: float
    point_id: str
    payload: VisualVectorPayload


class VisualSearchResult(BaseModel):
    """Bounded native visual retrieval result for one knowledge base."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = VISUAL_INDEX_SCHEMA_VERSION
    query: str = Field(min_length=1, max_length=4_000)
    knowledge_base_id: UUID
    collection_name: str
    collection_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_id: str
    embedding_dim: int = Field(ge=1)
    items: list[VisualSearchHit] = Field(default_factory=list)
    latency_ms: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_knowledge_base_scope(self) -> VisualSearchResult:
        for item in self.items:
            if item.payload.knowledge_base_id != self.knowledge_base_id:
                raise ValueError("visual hit belongs to another knowledge base")
            if item.payload.collection_fingerprint != self.collection_fingerprint:
                raise ValueError("visual hit belongs to another collection contract")
        return self


def visual_encoder_fingerprint(contract: VisualEmbeddingContract) -> str:
    """Return the deterministic fingerprint of the encoder contract."""

    payload = dict(contract.as_dict())
    payload["fingerprint_schema_version"] = VISUAL_INDEX_SCHEMA_VERSION
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def visual_collection_name(settings: Settings, contract: VisualEmbeddingContract) -> str:
    """Return a collection name that cannot collide across encoder contracts."""

    slug = _MODEL_SLUG_PATTERN.sub("-", contract.model_id.casefold()).strip("-")[:40] or "visual"
    fingerprint = visual_encoder_fingerprint(contract)[:10]
    budget = 63 - len(slug) - len(fingerprint) - 2
    base = settings.visual_collection_name[: max(3, budget)].rstrip("_-")
    return f"{base}_{slug}_{fingerprint}"


def visual_point_id(payload: VisualVectorPayload) -> UUID:
    """Return a deterministic id so re-indexing identical bytes is idempotent."""

    key = "|".join(
        (
            payload.collection_fingerprint,
            str(payload.knowledge_base_id),
            str(payload.document_id),
            payload.evidence_id,
            str(payload.asset_index),
            payload.asset_sha256,
        )
    )
    return uuid5(VISUAL_POINT_NAMESPACE, key)


def _payload_matches_contract(payload: VisualVectorPayload, fingerprint: str) -> bool:
    return (
        payload.collection_fingerprint == fingerprint and payload.model_fingerprint == fingerprint
    )


class QdrantVisualStore:
    """Thin Qdrant wrapper for one independent visual collection."""

    def __init__(
        self,
        settings: Settings,
        encoder: VisualEmbeddingEncoder,
        *,
        client: Any | None = None,
    ) -> None:
        self.settings = settings
        self.encoder = encoder
        self._client = client
        self._fingerprint = visual_encoder_fingerprint(encoder.contract)
        self._collection = visual_collection_name(settings, encoder.contract)

    @property
    def collection_name(self) -> str:
        return self._collection

    @property
    def collection_fingerprint(self) -> str:
        return self._fingerprint

    @property
    def model_id(self) -> str:
        return self.encoder.contract.model_id

    @property
    def embedding_dim(self) -> int:
        return self.encoder.contract.dimension

    def _get_client(self) -> Any:
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(url=self.settings.qdrant_url)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            close = getattr(self._client, "close", None)
            if callable(close):
                close()
            self._client = None

    def _collection_info(self) -> Any:
        return self._get_client().get_collection(self._collection)

    def _vector_config(self, info: Any) -> tuple[int | None, str | None]:
        vectors = getattr(getattr(info, "config", None), "params", None)
        vectors = getattr(vectors, "vectors", None)
        size = getattr(vectors, "size", None)
        distance = getattr(vectors, "distance", None)
        distance_text = getattr(distance, "value", distance)
        return size, distance_text

    def _validate_collection(self, info: Any) -> None:
        size, distance = self._vector_config(info)
        if size is None:
            raise VisualCollectionConfigurationError("视觉向量集合必须是单向量配置")
        if int(size) != self.encoder.contract.dimension:
            raise VisualCollectionConfigurationError(
                f"视觉向量维度不匹配：集合 {size}，编码器 {self.encoder.contract.dimension}"
            )
        if str(distance).casefold() != "cosine":
            raise VisualCollectionConfigurationError("视觉向量集合必须使用 cosine 距离")

    def ensure_collection(self) -> str:
        client = self._get_client()
        try:
            info = client.get_collection(self._collection)
        except Exception:
            info = None
        if info is None:
            from qdrant_client import models

            client.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(
                    size=self.encoder.contract.dimension,
                    distance=models.Distance.COSINE,
                ),
            )
            return self._collection
        self._validate_collection(info)
        return self._collection

    def readiness(self) -> dict[str, Any]:
        try:
            info = self._collection_info()
        except Exception:
            return {
                "collection_name": self._collection,
                "collection_exists": False,
                "collection_fingerprint": self._fingerprint,
                "model_id": self.model_id,
                "embedding_dim": self.embedding_dim,
            }
        size, distance = self._vector_config(info)
        existing = getattr(info, "payload_schema", None) or {}
        return {
            "collection_name": self._collection,
            "collection_exists": True,
            "collection_fingerprint": self._fingerprint,
            "vector_size": size,
            "distance": getattr(distance, "value", distance),
            "model_id": self.model_id,
            "embedding_dim": self.embedding_dim,
            "points_count": getattr(info, "points_count", None),
            "payload_fields": sorted(existing.keys()) if isinstance(existing, dict) else [],
        }

    def _scroll_ids(self, *, document_id: UUID) -> list[Any]:
        from qdrant_client import models

        point_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=str(document_id)),
                )
            ]
        )
        ids: list[Any] = []
        offset = None
        while True:
            records, offset = self._get_client().scroll(
                collection_name=self._collection,
                scroll_filter=point_filter,
                limit=256,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            ids.extend(record.id for record in records)
            if offset is None:
                break
        return ids

    def replace_document(
        self,
        document_id: UUID,
        entries: list[tuple[VisualVectorPayload, bytes]],
    ) -> int:
        """Atomically-ish replace one document's visual points.

        Points are deleted first so an evidence asset whose bytes changed cannot
        leave a stale visual vector behind.  Upserted ids are deterministic for
        unchanged bytes, which makes rebuilding idempotent.
        """

        client = self._get_client()
        stale_ids = self._scroll_ids(document_id=document_id)
        from qdrant_client import models

        if stale_ids:
            client.delete(
                collection_name=self._collection,
                points_selector=models.PointIdsList(points=stale_ids),
                wait=True,
            )
        if not entries:
            return 0
        payloads = [payload for payload, _bytes in entries]
        vectors, _stats = self.encoder.encode_images(
            [payload_bytes for _payload, payload_bytes in entries]
        )

        points = [
            models.PointStruct(
                id=str(visual_point_id(payload)),
                vector=vector,
                payload=payload.model_dump(mode="json"),
            )
            for payload, vector in zip(payloads, vectors, strict=True)
        ]
        client.upsert(collection_name=self._collection, points=points, wait=True)
        return len(points)

    def delete_document(self, document_id: UUID) -> int:
        ids = self._scroll_ids(document_id=document_id)
        if not ids:
            return 0
        from qdrant_client import models

        self._get_client().delete(
            collection_name=self._collection,
            points_selector=models.PointIdsList(points=ids),
            wait=True,
        )
        return len(ids)

    def search(
        self,
        query: str,
        knowledge_base_id: UUID,
        *,
        top_k: int,
        document_id: UUID | None = None,
    ) -> VisualSearchResult:
        if not query.strip():
            raise ValueError("query must not be blank")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        try:
            info = self._collection_info()
        except Exception:
            return VisualSearchResult(
                query=query,
                knowledge_base_id=UUID(str(knowledge_base_id)),
                collection_name=self._collection,
                collection_fingerprint=self._fingerprint,
                model_id=self.model_id,
                embedding_dim=self.embedding_dim,
                items=[],
                latency_ms=0.0,
            )
        self._validate_collection(info)
        started = perf_counter()
        vector, _stats = self.encoder.encode_query(query)
        from qdrant_client import models

        must = [
            models.FieldCondition(
                key="knowledge_base_id",
                match=models.MatchValue(value=str(knowledge_base_id)),
            ),
            models.FieldCondition(
                key="collection_fingerprint",
                match=models.MatchValue(value=self._fingerprint),
            ),
        ]
        if document_id is not None:
            must.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=str(document_id)),
                )
            )
        points = (
            self._get_client()
            .query_points(
                collection_name=self._collection,
                query=vector,
                query_filter=models.Filter(must=must),
                limit=top_k,
                with_payload=True,
            )
            .points
        )
        items: list[VisualSearchHit] = []
        for rank, point in enumerate(points, start=1):
            if not isinstance(point.payload, dict):
                continue
            try:
                payload = VisualVectorPayload.model_validate(point.payload)
            except Exception:
                continue
            if payload.knowledge_base_id != UUID(str(knowledge_base_id)):
                continue
            if not _payload_matches_contract(payload, self._fingerprint):
                continue
            items.append(
                VisualSearchHit(
                    rank=rank,
                    score=float(point.score),
                    point_id=str(point.id),
                    payload=payload,
                )
            )
        return VisualSearchResult(
            query=query,
            knowledge_base_id=knowledge_base_id,
            collection_name=self._collection,
            collection_fingerprint=self._fingerprint,
            model_id=self.model_id,
            embedding_dim=self.embedding_dim,
            items=items,
            latency_ms=(perf_counter() - started) * 1_000,
        )


__all__ = [
    "VISUAL_INDEX_SCHEMA_VERSION",
    "QdrantVisualStore",
    "VisualCollectionConfigurationError",
    "VisualIndexError",
    "VisualSearchHit",
    "VisualSearchResult",
    "VisualVectorPayload",
    "visual_collection_name",
    "visual_encoder_fingerprint",
    "visual_point_id",
]
