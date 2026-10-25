# ruff: noqa: RUF001
"""Independent, model-fingerprinted Qdrant collection for native visual vectors.

Only original image pixels are encoded here.  The collection never mixes with
the chunk or representation collections, and its name embeds the encoder
contract so two different visual models can never silently share one space.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
from datetime import UTC, datetime
from time import perf_counter
from typing import Any, Literal
from uuid import UUID, uuid4, uuid5

import httpx
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
    section_path: list[str] = Field(default_factory=list)
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
    generation: str = Field(default="", pattern=r"^[0-9a-f]{0,32}$")
    active: bool = False
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
    collection_missing: bool = False
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
            payload.generation,
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
        points_count = getattr(info, "points_count", None)
        from qdrant_client import models

        try:
            counted = self._get_client().count(
                collection_name=self._collection,
                count_filter=models.Filter(
                    must_not=[
                        models.FieldCondition(
                            key="kind",
                            match=models.MatchValue(value="generation_pointer"),
                        )
                    ]
                ),
                exact=True,
            )
            points_count = int(counted.count)
        except Exception:
            pass
        return {
            "collection_name": self._collection,
            "collection_exists": True,
            "collection_fingerprint": self._fingerprint,
            "vector_size": size,
            "distance": getattr(distance, "value", distance),
            "model_id": self.model_id,
            "embedding_dim": self.embedding_dim,
            "points_count": points_count,
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

    def _scroll_records(self, *, document_id: UUID) -> list[Any]:
        from qdrant_client import models

        point_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=str(document_id)),
                )
            ]
        )
        records: list[Any] = []
        offset = None
        while True:
            batch, offset = self._get_client().scroll(
                collection_name=self._collection,
                scroll_filter=point_filter,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            records.extend(batch)
            if offset is None:
                break
        return records

    def _read_pointer(self, document_id: UUID) -> dict[str, Any] | None:
        from qdrant_client import models

        point_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=str(document_id)),
                ),
                models.FieldCondition(
                    key="kind",
                    match=models.MatchValue(value="generation_pointer"),
                ),
            ]
        )
        records, _offset = self._get_client().scroll(
            collection_name=self._collection,
            scroll_filter=point_filter,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if not records:
            return None
        payload = records[0].payload
        if not isinstance(payload, dict):
            return None
        generation = payload.get("active_generation")
        if not isinstance(generation, str) or not generation:
            return None
        return payload

    def _write_pointer(
        self,
        document_id: UUID,
        knowledge_base_id: UUID,
        generation: str,
        *,
        point_count: int,
    ) -> None:
        from qdrant_client import models

        pointer_id = uuid5(
            VISUAL_POINT_NAMESPACE,
            f"pointer|{self._fingerprint}|{knowledge_base_id}|{document_id}",
        )
        payload = {
            "kind": "generation_pointer",
            "schema_version": VISUAL_INDEX_SCHEMA_VERSION,
            "collection_fingerprint": self._fingerprint,
            "knowledge_base_id": str(knowledge_base_id),
            "document_id": str(document_id),
            "active_generation": generation,
            "point_count": point_count,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        self._get_client().upsert(
            collection_name=self._collection,
            points=[
                models.PointStruct(
                    id=str(pointer_id),
                    vector=[0.0] * self.embedding_dim,
                    payload=payload,
                )
            ],
            wait=True,
        )

    def _set_generation_active(
        self,
        document_id: UUID,
        generation: str,
        *,
        active: bool,
    ) -> None:
        from qdrant_client import models

        point_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=str(document_id)),
                ),
                models.FieldCondition(
                    key="generation",
                    match=models.MatchValue(value=generation),
                ),
                models.FieldCondition(
                    key="collection_fingerprint",
                    match=models.MatchValue(value=self._fingerprint),
                ),
            ]
        )
        self._get_client().set_payload(
            collection_name=self._collection,
            payload={"active": active},
            points=point_filter,
            wait=True,
        )

    def _delete_ids(self, ids: list[Any]) -> None:
        if not ids:
            return
        from qdrant_client import models

        self._get_client().delete(
            collection_name=self._collection,
            points_selector=models.PointIdsList(points=ids),
            wait=True,
        )

    def _delete_generation(self, document_id: UUID, generation: str) -> None:
        from qdrant_client import models

        point_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=str(document_id)),
                ),
                models.FieldCondition(
                    key="generation",
                    match=models.MatchValue(value=generation),
                ),
                models.FieldCondition(
                    key="collection_fingerprint",
                    match=models.MatchValue(value=self._fingerprint),
                ),
            ]
        )
        self._get_client().delete(
            collection_name=self._collection,
            points_selector=models.FilterSelector(filter=point_filter),
            wait=True,
        )

    def _normalize_flags(self, document_id: UUID, generation: str) -> None:
        self._set_generation_active(document_id, generation, active=True)
        records = self._scroll_records(document_id=document_id)
        for record in records:
            payload = record.payload if isinstance(record.payload, dict) else {}
            if payload.get("kind") == "generation_pointer":
                continue
            if payload.get("generation") != generation:
                self._delete_ids([record.id])

    def recover_document(self, document_id: UUID) -> str | None:
        """Recover one document to a single, clearly valid visual generation.

        Search reads the pointer before returning hits, so a crash between
        "write new points" and "switch pointer" can never expose the half
        written generation.  This routine cleans those orphans and normalizes
        the ``active`` flags for the pointer generation.
        """

        records = self._scroll_records(document_id=document_id)
        data_records = [
            record
            for record in records
            if not (
                isinstance(record.payload, dict)
                and record.payload.get("kind") == "generation_pointer"
            )
        ]
        if not data_records:
            self._delete_ids([record.id for record in records])
            return None
        generations: dict[str, dict[str, Any]] = {}
        for record in data_records:
            payload = record.payload if isinstance(record.payload, dict) else {}
            generation = payload.get("generation")
            if not isinstance(generation, str) or not generation:
                continue
            entry = generations.setdefault(generation, {"count": 0, "indexed_at": ""})
            entry["count"] += 1
            indexed_at = str(payload.get("indexed_at") or "")
            if indexed_at > entry["indexed_at"]:
                entry["indexed_at"] = indexed_at
        if not generations:
            self._delete_ids([record.id for record in records])
            return None
        pointer = self._read_pointer(document_id)
        pointer_generation = pointer.get("active_generation") if pointer else None
        if isinstance(pointer_generation, str) and pointer_generation in generations:
            keep = pointer_generation
        else:
            active = [
                generation
                for generation in generations
                if any(
                    isinstance(record.payload, dict)
                    and record.payload.get("generation") == generation
                    and record.payload.get("active") is True
                    for record in data_records
                )
            ]
            candidates = active or list(generations)
            keep = max(candidates, key=lambda item: generations[item]["indexed_at"])
        knowledge_base_id = next(
            (
                record.payload["knowledge_base_id"]
                for record in data_records
                if isinstance(record.payload, dict)
                and isinstance(record.payload.get("knowledge_base_id"), str)
            ),
            None,
        )
        if knowledge_base_id is not None:
            self._write_pointer(
                document_id,
                UUID(knowledge_base_id),
                keep,
                point_count=generations[keep]["count"],
            )
        self._normalize_flags(document_id, keep)
        return keep

    def replace_document(
        self,
        document_id: UUID,
        entries: list[tuple[VisualVectorPayload, bytes]],
    ) -> int:
        """Replace one document's visual points without losing the old index.

        Encoding happens before any write.  New points are written with
        ``active=False``; only after the generation pointer switches are they
        marked active and the previous generation removed.  A failed encode or
        upsert therefore leaves the previous generation queryable.
        """

        client = self._get_client()
        from qdrant_client import models

        self.recover_document(document_id)
        payloads = [payload for payload, _bytes in entries]
        if not payloads:
            self._delete_ids(self._scroll_ids(document_id=document_id))
            return 0
        knowledge_base_id = payloads[0].knowledge_base_id
        try:
            vectors, _stats = self.encoder.encode_images(
                [payload_bytes for _payload, payload_bytes in entries]
            )
        except Exception:
            raise VisualIndexError("视觉索引编码失败，旧索引保持不变") from None
        generation = uuid4().hex
        staged_payloads = [
            payload.model_copy(update={"generation": generation, "active": False})
            for payload in payloads
        ]
        points = [
            models.PointStruct(
                id=str(visual_point_id(payload)),
                vector=vector,
                payload=payload.model_dump(mode="json"),
            )
            for payload, vector in zip(staged_payloads, vectors, strict=True)
        ]
        previous_pointer = self._read_pointer(document_id)
        previous_generation = (
            previous_pointer.get("active_generation") if previous_pointer else None
        )
        new_ids = [point.id for point in points]
        try:
            client.upsert(collection_name=self._collection, points=points, wait=True)
        except Exception:
            self._safe_delete(new_ids)
            raise VisualIndexError("视觉索引写入失败，旧索引保持不变") from None
        try:
            self._write_pointer(
                document_id,
                knowledge_base_id,
                generation,
                point_count=len(points),
            )
            self._set_generation_active(document_id, generation, active=True)
            self._delete_other_generations(document_id, keep=generation)
        except Exception:
            self._safe_delete(new_ids)
            if not (isinstance(previous_generation, str) and previous_generation):
                with contextlib.suppress(Exception):
                    self._delete_pointer(document_id)
            if isinstance(previous_generation, str) and previous_generation:
                try:
                    self._write_pointer(
                        document_id,
                        knowledge_base_id,
                        previous_generation,
                        point_count=0,
                    )
                    self._set_generation_active(
                        document_id,
                        previous_generation,
                        active=True,
                    )
                except Exception:
                    pass
            raise VisualIndexError("视觉索引切换失败，已回滚到旧索引") from None
        return len(points)

    def _delete_pointer(self, document_id: UUID) -> None:
        from qdrant_client import models

        pointer_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=str(document_id)),
                ),
                models.FieldCondition(
                    key="kind",
                    match=models.MatchValue(value="generation_pointer"),
                ),
            ]
        )
        self._get_client().delete(
            collection_name=self._collection,
            points_selector=models.FilterSelector(filter=pointer_filter),
            wait=True,
        )

    def _safe_delete(self, ids: list[Any]) -> None:
        with contextlib.suppress(Exception):
            self._delete_ids(ids)

    def _delete_other_generations(self, document_id: UUID, *, keep: str | None) -> None:
        for record in self._scroll_records(document_id=document_id):
            payload = record.payload if isinstance(record.payload, dict) else {}
            if payload.get("kind") == "generation_pointer":
                continue
            if keep is not None and payload.get("generation") == keep:
                continue
            self._delete_ids([record.id])

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

    @staticmethod
    def _is_collection_missing(error: BaseException) -> bool:
        status_code = getattr(error, "status_code", None)
        if status_code == 404:
            return True
        message = str(error).lower()
        return "not found" in message or "does not exist" in message or "doesn't exist" in message

    @staticmethod
    def _is_backend_unavailable(error: BaseException) -> bool:
        if isinstance(error, (ConnectionError, TimeoutError, httpx.HTTPError)):
            return True
        message = type(error).__name__.lower()
        return "responsehandling" in message or "transport" in message

    def _empty_result(
        self,
        query: str,
        knowledge_base_id: UUID,
        *,
        collection_missing: bool,
        started: float,
    ) -> VisualSearchResult:
        return VisualSearchResult(
            query=query,
            knowledge_base_id=UUID(str(knowledge_base_id)),
            collection_name=self._collection,
            collection_fingerprint=self._fingerprint,
            model_id=self.model_id,
            embedding_dim=self.embedding_dim,
            items=[],
            collection_missing=collection_missing,
            latency_ms=(perf_counter() - started) * 1_000,
        )

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
        started = perf_counter()
        try:
            info = self._collection_info()
        except Exception as error:
            if self._is_collection_missing(error):
                return self._empty_result(
                    query,
                    knowledge_base_id,
                    collection_missing=True,
                    started=started,
                )
            if self._is_backend_unavailable(error):
                raise VisualIndexError("视觉检索后端不可用，请稍后重试") from error
            raise
        self._validate_collection(info)
        vector, _stats = self.encoder.encode_query(query)

        from qdrant_client import models

        must: list[Any] = [
            models.FieldCondition(
                key="knowledge_base_id",
                match=models.MatchValue(value=str(knowledge_base_id)),
            ),
            models.FieldCondition(
                key="collection_fingerprint",
                match=models.MatchValue(value=self._fingerprint),
            ),
            models.FieldCondition(
                key="active",
                match=models.MatchValue(value=True),
            ),
        ]
        if document_id is not None:
            must.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=str(document_id)),
                )
            )
        try:
            points = (
                self._get_client()
                .query_points(
                    collection_name=self._collection,
                    query=vector,
                    query_filter=models.Filter(must=must),
                    limit=max(1, min(top_k * 4, 64)),
                    with_payload=True,
                )
                .points
            )
        except Exception as error:
            if self._is_collection_missing(error):
                return self._empty_result(
                    query,
                    knowledge_base_id,
                    collection_missing=True,
                    started=started,
                )
            raise VisualIndexError("视觉检索后端不可用，请稍后重试") from error

        pointer_cache: dict[UUID, str | None] = {}
        items: list[VisualSearchHit] = []
        seen: set[tuple[str, str, int]] = set()
        expected_kb = UUID(str(knowledge_base_id))
        for point in points:
            if len(items) >= top_k:
                break
            if not isinstance(point.payload, dict):
                continue
            try:
                payload = VisualVectorPayload.model_validate(point.payload)
            except Exception:
                continue
            if payload.knowledge_base_id != expected_kb:
                continue
            if not _payload_matches_contract(payload, self._fingerprint):
                continue
            if payload.generation:
                if payload.document_id not in pointer_cache:
                    try:
                        pointer = self._read_pointer(payload.document_id)
                    except Exception as error:
                        raise VisualIndexError("视觉检索后端不可用，请稍后重试") from error
                    active_generation = pointer.get("active_generation") if pointer else None
                    pointer_cache[payload.document_id] = (
                        active_generation if isinstance(active_generation, str) else None
                    )
                if pointer_cache[payload.document_id] != payload.generation:
                    continue
            key = (
                str(payload.document_id),
                payload.evidence_id,
                payload.asset_index,
            )
            if key in seen:
                continue
            seen.add(key)
            items.append(
                VisualSearchHit(
                    rank=len(items) + 1,
                    score=float(point.score),
                    point_id=str(point.id),
                    payload=payload,
                )
            )
        return VisualSearchResult(
            query=query,
            knowledge_base_id=UUID(str(knowledge_base_id)),
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
