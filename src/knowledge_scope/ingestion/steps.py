# ruff: noqa: RUF001
"""Real pipeline steps for the M8 single-host PDF auto-ingestion worker.

The steps deliberately separate *preparation* (expensive GPU work that is
persisted into one per-document bundle) from *publish* (writing derived indexes
into their query-visible stores).  A document therefore never becomes partially
searchable: if any preparation stage fails, the previously published indexes of
that document remain untouched.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import UUID

from pydantic import ValidationError
from sqlalchemy import select

from knowledge_scope.chunking.models import ChunkedDocument
from knowledge_scope.chunking.service import chunk_artifact_path, chunk_document_by_id
from knowledge_scope.documents.models import Document
from knowledge_scope.evidence.lifecycle import remove_evidence_artifact, write_evidence_artifact
from knowledge_scope.evidence.models import MultimodalEvidenceDocument
from knowledge_scope.parsing.service import PARSING_DIRECTORY_NAME, parse_document_by_id
from knowledge_scope.retrieval.embedding import QwenEmbeddingModel
from knowledge_scope.retrieval.indexing import build_vector_points
from knowledge_scope.retrieval.qdrant import ChunkVectorPayload, QdrantVectorStore, VectorPoint
from knowledge_scope.retrieval.representation_index import (
    IndexedRepresentationPayload,
    QdrantRepresentationStore,
    RepresentationEmbeddingEncoder,
    RepresentationVectorPoint,
    build_indexable_evidence,
    build_representation_points,
    load_canonical_document,
)
from knowledge_scope.retrieval.sparse import (
    SparseChunkRecord,
    SparseIndexConfig,
    SparseIndexStore,
    records_from_chunked_documents,
)
from knowledge_scope.retrieval.visual_build import BuildMode, VisualIndexBuilder
from knowledge_scope.retrieval.visual_embedding import VisualEmbeddingEncoder
from knowledge_scope.retrieval.visual_index import QdrantVisualStore, VisualVectorPayload
from knowledge_scope.shared.config import Settings
from knowledge_scope.shared.database import create_database_engine, create_session_factory

from .models import INGESTION_PIPELINE_VERSION, IngestionTask
from .store import IngestionTaskStore

logger = logging.getLogger(__name__)

BUNDLE_SCHEMA_VERSION = "1.0"


class IngestionStepError(RuntimeError):
    """Raised when one ingestion step cannot complete safely."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class DocumentSnapshot:
    document_id: UUID
    knowledge_base_id: UUID
    source_sha256: str
    original_filename: str


def _release_torch_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - defensive cleanup only
        return


def _write_bundle(store: IngestionTaskStore, document_id: UUID, bundle: dict[str, Any]) -> None:
    bundle["schema_version"] = BUNDLE_SCHEMA_VERSION
    bundle["pipeline_version"] = INGESTION_PIPELINE_VERSION
    store.write_bundle(document_id, bundle)


def _read_bundle(store: IngestionTaskStore, document_id: UUID) -> dict[str, Any]:
    bundle = store.read_bundle(document_id)
    if bundle is None:
        raise IngestionStepError("prepared_bundle_missing", "预处理结果不存在，请重新处理")
    if bundle.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise IngestionStepError("prepared_bundle_invalid", "预处理结果版本不匹配")
    return bundle


def _dense_points_to_json(points: list[VectorPoint]) -> list[dict[str, Any]]:
    return [
        {
            "point_id": str(point.point_id),
            "vector": list(point.vector),
            "payload": point.payload.model_dump(mode="json"),
        }
        for point in points
    ]


def _dense_points_from_json(items: list[dict[str, Any]]) -> list[VectorPoint]:
    return [
        VectorPoint(
            point_id=UUID(str(item["point_id"])),
            vector=tuple(float(value) for value in item["vector"]),
            payload=ChunkVectorPayload.model_validate(item["payload"]),
        )
        for item in items
    ]


def _representation_points_to_json(
    points: list[RepresentationVectorPoint],
) -> list[dict[str, Any]]:
    return [
        {
            "point_id": str(point.point_id),
            "vector": list(point.vector),
            "payload": point.payload.model_dump(mode="json"),
        }
        for point in points
    ]


def _representation_points_from_json(
    items: list[dict[str, Any]],
) -> list[RepresentationVectorPoint]:
    return [
        RepresentationVectorPoint(
            point_id=UUID(str(item["point_id"])),
            vector=tuple(float(value) for value in item["vector"]),
            payload=IndexedRepresentationPayload.model_validate(item["payload"]),
        )
        for item in items
    ]


def _sparse_records_to_json(records: list[SparseChunkRecord]) -> list[dict[str, Any]]:
    return [record.model_dump(mode="json") for record in records]


def _sparse_records_from_json(items: list[dict[str, Any]]) -> list[SparseChunkRecord]:
    return [SparseChunkRecord.model_validate(item) for item in items]


class IngestionSteps:
    """Default real implementation of the ingestion stages."""

    def __init__(self, settings: Settings, task_store: IngestionTaskStore) -> None:
        self.settings = settings
        self.task_store = task_store

    # -- inputs ---------------------------------------------------------

    async def _load_document(self, document_id: UUID) -> DocumentSnapshot:
        engine = create_database_engine(self.settings)
        try:
            session_factory = create_session_factory(engine)
            async with session_factory() as session:
                document = await session.scalar(select(Document).where(Document.id == document_id))
                if document is None:
                    raise IngestionStepError("document_missing", "文档不存在")
                if document.storage_kind != "managed":
                    raise IngestionStepError(
                        "document_not_managed",
                        "只有应用托管的 PDF 才能自动入库",
                    )
                return DocumentSnapshot(
                    document_id=document.id,
                    knowledge_base_id=document.knowledge_base_id,
                    source_sha256=document.sha256,
                    original_filename=document.original_filename,
                )
        finally:
            await engine.dispose()

    def load_document(self, document_id: UUID) -> DocumentSnapshot:
        return asyncio.run(self._load_document(document_id))

    def _load_chunked(self, document_id: UUID) -> ChunkedDocument:
        path = chunk_artifact_path(self.settings, document_id)
        if not path.is_file():
            raise IngestionStepError("chunk_artifact_missing", "分块结果不存在")
        try:
            return ChunkedDocument.model_validate_json(path.read_bytes())
        except (OSError, ValidationError, ValueError) as error:
            raise IngestionStepError("chunk_artifact_invalid", "分块结果无效") from error

    # -- prepare stages -------------------------------------------------

    def parse(self, task: IngestionTask) -> dict[str, object]:
        started = perf_counter()
        try:
            result = asyncio.run(parse_document_by_id(task.document_id, self.settings))
        except Exception as error:
            raise IngestionStepError("parse_failed", f"PDF 解析失败：{error}") from error
        return {
            "elapsed_ms": round((perf_counter() - started) * 1_000, 3),
            "parser_version": result.parser_version,
            "backend": result.backend,
            "canonical_ref": result.canonical_ref,
            "raw_ref": result.raw_ref,
        }

    def chunk(self, task: IngestionTask) -> dict[str, object]:
        started = perf_counter()
        try:
            result = chunk_document_by_id(task.document_id, self.settings)
        except Exception as error:
            raise IngestionStepError("chunk_failed", f"文档分块失败：{error}") from error
        return {
            "elapsed_ms": round((perf_counter() - started) * 1_000, 3),
            "chunks": result.chunk_count,
            "config_fingerprint": result.config_fingerprint,
        }

    def text_index(self, task: IngestionTask) -> dict[str, object]:
        started = perf_counter()
        chunked = self._load_chunked(task.document_id)
        if not chunked.chunks:
            raise IngestionStepError("no_chunks", "文档没有可索引的结构化分块")
        embedder = QwenEmbeddingModel(self.settings)
        try:
            points = build_vector_points(
                chunked,
                knowledge_base_id=task.knowledge_base_id,
                embedder=embedder,
            )
        except Exception as error:
            raise IngestionStepError(
                "text_embedding_failed", f"文本向量生成失败：{error}"
            ) from error
        finally:
            _release_torch_memory()
        bundle = self.task_store.read_bundle(task.document_id) or {}
        bundle["dense_points"] = _dense_points_to_json(points)
        bundle["chunk_count"] = len(chunked.chunks)
        _write_bundle(self.task_store, task.document_id, bundle)
        return {
            "elapsed_ms": round((perf_counter() - started) * 1_000, 3),
            "vectors": len(points),
        }

    def sparse_index(self, task: IngestionTask) -> dict[str, object]:
        started = perf_counter()
        chunked = self._load_chunked(task.document_id)
        records = records_from_chunked_documents([chunked], task.knowledge_base_id)
        if not records:
            raise IngestionStepError("no_chunks", "文档没有可索引的 BM25 分块")
        bundle = self._read_bundle_or_empty(task)
        bundle["sparse_records"] = _sparse_records_to_json(records)
        _write_bundle(self.task_store, task.document_id, bundle)
        return {
            "elapsed_ms": round((perf_counter() - started) * 1_000, 3),
            "records": len(records),
        }

    def multimodal_index(self, task: IngestionTask) -> dict[str, object]:
        started = perf_counter()
        canonical_path = (
            Path(self.settings.data_dir)
            / PARSING_DIRECTORY_NAME
            / str(task.document_id)
            / "canonical.json"
        )
        try:
            document = load_canonical_document(canonical_path)
        except Exception as error:
            raise IngestionStepError("canonical_missing", "解析产物不存在") from error
        artifact = build_indexable_evidence(document, task.knowledge_base_id)
        embedder: RepresentationEmbeddingEncoder = QwenEmbeddingModel(self.settings)
        try:
            points = build_representation_points(
                artifact,
                settings=self.settings,
                embedder=embedder,
            )
        except Exception as error:
            raise IngestionStepError(
                "multimodal_embedding_failed", f"多模态表示向量生成失败：{error}"
            ) from error
        finally:
            _release_torch_memory()
        image_count = sum(1 for item in artifact.evidence if item.modality == "image")
        bundle = self._read_bundle_or_empty(task)
        bundle["evidence"] = artifact.model_dump(mode="json")
        bundle["representation_points"] = _representation_points_to_json(points)
        bundle["image_count"] = image_count
        _write_bundle(self.task_store, task.document_id, bundle)
        return {
            "elapsed_ms": round((perf_counter() - started) * 1_000, 3),
            "representations": len(points),
            "images": image_count,
        }

    def _read_bundle_or_empty(self, task: IngestionTask) -> dict[str, Any]:
        bundle = self.task_store.read_bundle(task.document_id)
        if bundle is None:
            return {}
        if bundle.get("schema_version") != BUNDLE_SCHEMA_VERSION:
            return {}
        return bundle

    def visual_index(self, task: IngestionTask) -> dict[str, object]:
        if not self.settings.visual_embedding_enabled:
            bundle = self._read_bundle_or_empty(task)
            bundle["visual_entries"] = []
            bundle["visual_skipped_reason"] = "visual_embedding_disabled"
            _write_bundle(self.task_store, task.document_id, bundle)
            return {"skipped": True, "reason": "visual_embedding_disabled", "images": 0}
        started = perf_counter()
        bundle = _read_bundle(self.task_store, task.document_id)
        try:
            artifact = MultimodalEvidenceDocument.model_validate(bundle["evidence"])
        except (KeyError, ValidationError) as error:
            raise IngestionStepError("evidence_missing", "多模态 Evidence 不存在") from error
        image_count = sum(1 for item in artifact.evidence if item.modality == "image")
        if image_count == 0:
            bundle["visual_entries"] = []
            bundle["visual_skipped_reason"] = "no_images"
            _write_bundle(self.task_store, task.document_id, bundle)
            return {"skipped": True, "reason": "no_images", "images": 0}
        encoder: VisualEmbeddingEncoder | None = None
        store: QdrantVisualStore | None = None
        try:
            encoder = VisualEmbeddingEncoder(self.settings)
            store = QdrantVisualStore(self.settings, encoder)
            builder = VisualIndexBuilder(self.settings, store)
            mode: BuildMode = self.settings.ingestion_visual_mode
            entries, stats = builder.prepare_document(
                task.knowledge_base_id,
                task.document_id,
                artifact,
                mode=mode,
            )
        except Exception as error:
            raise IngestionStepError(
                "visual_embedding_failed", f"图片视觉向量生成失败：{error}"
            ) from error
        finally:
            if store is not None:
                store.close()
            if encoder is not None:
                encoder.close()
            _release_torch_memory()
        bundle["visual_entries"] = [
            {
                "payload": payload.model_dump(mode="json"),
                "image_base64": base64.b64encode(image_bytes).decode("ascii"),
            }
            for payload, image_bytes in entries
        ]
        bundle["visual_skipped_reason"] = None
        _write_bundle(self.task_store, task.document_id, bundle)
        return {
            "elapsed_ms": round((perf_counter() - started) * 1_000, 3),
            "images": stats.images_indexed or len(entries),
            "images_total": stats.images_total,
            "images_missing": stats.images_missing,
            "images_invalid": stats.images_invalid,
            "skipped_by_policy": stats.images_skipped_by_policy,
            "embedding_latency_ms": stats.embedding_latency_ms,
            "vram_peak_mb": stats.vram_peak_mb,
        }

    # -- publish --------------------------------------------------------

    def publish(self, task: IngestionTask) -> dict[str, object]:
        bundle = _read_bundle(self.task_store, task.document_id)
        started = perf_counter()
        try:
            evidence = MultimodalEvidenceDocument.model_validate(bundle["evidence"])
            dense_points = _dense_points_from_json(bundle.get("dense_points") or [])
            representation_points = _representation_points_from_json(
                bundle.get("representation_points") or []
            )
            sparse_records = _sparse_records_from_json(bundle.get("sparse_records") or [])
            visual_entries = [
                (
                    VisualVectorPayload.model_validate(item["payload"]),
                    base64.b64decode(item["image_base64"]),
                )
                for item in (bundle.get("visual_entries") or [])
            ]
        except (KeyError, ValidationError, ValueError) as error:
            raise IngestionStepError("prepared_bundle_invalid", "预处理结果无效") from error
        if not dense_points or not representation_points or not sparse_records:
            raise IngestionStepError("prepared_bundle_incomplete", "预处理结果不完整")

        dense_store: QdrantVectorStore | None = None
        representation_store: QdrantRepresentationStore | None = None
        sparse_store: SparseIndexStore | None = None
        visual_store: QdrantVisualStore | None = None
        visual_encoder: VisualEmbeddingEncoder | None = None
        published: list[str] = []
        evidence_written = False
        try:
            write_evidence_artifact(self.settings.data_dir, evidence)
            evidence_written = True
            dense_store = QdrantVectorStore(self.settings)
            dense_store.ensure_collection()
            dense_store.replace_document(dense_points)
            published.append("text")
            representation_store = QdrantRepresentationStore(self.settings)
            representation_store.ensure_collection()
            representation_store.replace_document(
                task.document_id,
                task.knowledge_base_id,
                representation_points,
            )
            published.append("multimodal")
            sparse_store = SparseIndexStore(
                self.settings.sparse_index_path,
                config=SparseIndexConfig(
                    bm25_k1=self.settings.sparse_bm25_k1,
                    bm25_b=self.settings.sparse_bm25_b,
                ),
            )
            sparse_store.replace_document(
                task.knowledge_base_id,
                task.document_id,
                sparse_records,
            )
            published.append("bm25")
            if self.settings.visual_embedding_enabled:
                visual_encoder = VisualEmbeddingEncoder(self.settings)
                visual_store = QdrantVisualStore(self.settings, visual_encoder)
                builder = VisualIndexBuilder(self.settings, visual_store)
                published_visual = builder.publish_document(
                    task.document_id,
                    visual_entries,
                )
                published.append("visual")
            else:
                published_visual = 0
        except Exception as error:
            self._compensate_publish(
                task,
                published=published,
                dense_store=dense_store,
                representation_store=representation_store,
                sparse_store=sparse_store,
                visual_store=visual_store,
                evidence_written=evidence_written,
            )
            raise IngestionStepError(
                "publish_failed", f"索引发布失败，已回滚本次发布的索引：{error}"
            ) from error
        finally:
            if visual_store is not None:
                visual_store.close()
            if visual_encoder is not None:
                visual_encoder.close()
            if sparse_store is not None:
                sparse_store.close()
            _release_torch_memory()
        return {
            "elapsed_ms": round((perf_counter() - started) * 1_000, 3),
            "published_indexes": published,
            "text_points": len(dense_points),
            "representation_points": len(representation_points),
            "sparse_records": len(sparse_records),
            "visual_points": published_visual,
        }

    def _compensate_publish(
        self,
        task: IngestionTask,
        *,
        published: list[str],
        dense_store: QdrantVectorStore | None,
        representation_store: QdrantRepresentationStore | None,
        sparse_store: SparseIndexStore | None,
        visual_store: QdrantVisualStore | None,
        evidence_written: bool,
    ) -> None:
        try:
            if dense_store is not None:
                dense_store.delete_document(task.document_id)
            if representation_store is not None:
                representation_store.delete_document(
                    task.document_id,
                    knowledge_base_id=task.knowledge_base_id,
                )
            if sparse_store is not None:
                sparse_store.delete_document(task.knowledge_base_id, task.document_id)
            if visual_store is not None:
                visual_store.delete_document(task.document_id)
            if evidence_written:
                remove_evidence_artifact(self.settings.data_dir, task.document_id)
        except Exception:  # pragma: no cover - best effort compensation
            logger.exception("ingestion publish compensation failed document=%s", task.document_id)

    # -- verify ---------------------------------------------------------

    def verify(self, task: IngestionTask) -> dict[str, object]:
        bundle = _read_bundle(self.task_store, task.document_id)
        expected_chunks = int(bundle.get("chunk_count") or 0)
        expected_representations = len(bundle.get("representation_points") or [])
        expected_visual = len(bundle.get("visual_entries") or [])
        chunked = self._load_chunked(task.document_id)
        if len(chunked.chunks) != expected_chunks:
            raise IngestionStepError("verify_chunks", "分块数量与索引计划不一致")
        dense_store = QdrantVectorStore(self.settings)
        representation_store = QdrantRepresentationStore(self.settings)
        sparse_store = SparseIndexStore(
            self.settings.sparse_index_path,
            config=SparseIndexConfig(
                bm25_k1=self.settings.sparse_bm25_k1,
                bm25_b=self.settings.sparse_bm25_b,
            ),
        )
        visual_encoder: VisualEmbeddingEncoder | None = None
        visual_store: QdrantVisualStore | None = None
        try:
            dense_count = len(
                dense_store.list_document_chunk_payloads(
                    knowledge_base_id=task.knowledge_base_id,
                    document_id=task.document_id,
                )
            )
            representation_count = representation_store.document_point_count(
                task.document_id,
                knowledge_base_id=task.knowledge_base_id,
            )
            first_text = chunked.chunks[0].text or "PDF 文档"
            sparse_hits = sparse_store.search(
                first_text[:60],
                knowledge_base_id=task.knowledge_base_id,
                top_k=1,
                document_id=task.document_id,
            )
            sparse_ok = (
                not sparse_hits.items or sparse_hits.items[0].document_id == task.document_id
            )
            if self.settings.visual_embedding_enabled:
                visual_encoder = VisualEmbeddingEncoder(self.settings)
                visual_store = QdrantVisualStore(self.settings, visual_encoder)
                visual_count = len(visual_store._scroll_ids(document_id=task.document_id))
            else:
                visual_count = 0
        finally:
            if visual_store is not None:
                visual_store.close()
            if visual_encoder is not None:
                visual_encoder.close()
            sparse_store.close()
        if dense_count != expected_chunks:
            raise IngestionStepError("verify_text_index", "文本索引数量校验失败")
        if representation_count != expected_representations:
            raise IngestionStepError("verify_multimodal_index", "多模态索引数量校验失败")
        if not sparse_ok:
            raise IngestionStepError("verify_bm25", "BM25 索引校验失败")
        if expected_visual and visual_count == 0:
            raise IngestionStepError("verify_visual_index", "视觉索引缺少向量")
        if not expected_visual and visual_count != 0:
            raise IngestionStepError("verify_visual_index", "视觉索引应清空但仍有残留")
        return {
            "text_points": dense_count,
            "representation_points": representation_count,
            "sparse_ok": sparse_ok,
            "visual_points": visual_count,
        }

    # -- graph (explicitly out of scope) --------------------------------

    def graph(self, task: IngestionTask) -> dict[str, object]:
        return {
            "skipped": True,
            "reason": "graph_index_not_part_of_auto_ingest",
        }


__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "DocumentSnapshot",
    "IngestionStepError",
    "IngestionSteps",
]
