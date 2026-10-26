# ruff: noqa: RUF001
"""Focused tests for the M8 single-host PDF auto-ingestion pipeline."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient

from knowledge_scope.evidence.lifecycle import write_evidence_artifact
from knowledge_scope.evidence.service import build_evidence_document
from knowledge_scope.ingestion.models import IngestionTask
from knowledge_scope.ingestion.steps import (
    BUNDLE_SCHEMA_VERSION,
    INGESTION_PIPELINE_VERSION,
    IngestionStepError,
    IngestionSteps,
)
from knowledge_scope.ingestion.store import IngestionStoreError, IngestionTaskStore
from knowledge_scope.ingestion.worker import IngestionError, IngestionWorker
from knowledge_scope.parsing.models import CanonicalDocument, ImageBlock, Page, TextBlock
from knowledge_scope.shared.config import Settings
from test_document_assets import (
    JPEG_BYTES,
    PNG_BYTES,
    create_knowledge_base,
    upload_document,
    write_parsing_artifact,
)


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "visual_embedding_enabled": True,
        **overrides,
    }
    return Settings(
        _env_file=None,
        environment="test",
        data_dir=tmp_path,
        **values,  # type: ignore[arg-type]
    )


IMAGE_ASSET_A = "two-images/auto/images/a.png"
IMAGE_ASSET_B = "two-images/auto/images/b.jpg"


def _canonical_with_images(document_id: UUID, image_count: int) -> CanonicalDocument:
    blocks: list[TextBlock | ImageBlock] = [
        TextBlock(block_id="p1-b1", reading_order=0, text="页面正文内容")
    ]
    assets = [IMAGE_ASSET_A, IMAGE_ASSET_B]
    for index in range(image_count):
        blocks.append(
            ImageBlock(
                block_id=f"p1-b{index + 2}",
                reading_order=index + 1,
                asset_ref=assets[index],
                caption=f"示意图 {index + 1}",
            )
        )
    return CanonicalDocument(
        document_id=document_id,
        pages=[Page(page_number=1, blocks=blocks)],
    )


class FakeSteps:
    """Deterministic in-memory replacement for the real pipeline steps."""

    def __init__(
        self,
        *,
        fail_at: str | None = None,
        images: int = 2,
        indexed: int | None = None,
        missing: int = 0,
        invalid: int = 0,
        skipped_policy: int = 0,
        publish_visual: bool = True,
    ) -> None:
        self.fail_at = fail_at
        self.images = images
        self.indexed = images if indexed is None else indexed
        self.missing = missing
        self.invalid = invalid
        self.skipped_policy = skipped_policy
        self.publish_visual = publish_visual
        self.calls: list[str] = []

    def _stage(self, name: str, details: dict[str, object]) -> dict[str, object]:
        self.calls.append(name)
        if self.fail_at == name:
            raise RuntimeError(f"{name} exploded")
        return {"elapsed_ms": 1.5, **details}

    def parse(self, task: IngestionTask) -> dict[str, object]:
        return self._stage("parse", {"canonical_ref": "parsing/x/canonical.json"})

    def chunk(self, task: IngestionTask) -> dict[str, object]:
        return self._stage("chunk", {"chunks": 3})

    def text_index(self, task: IngestionTask) -> dict[str, object]:
        return self._stage("text_index", {"vectors": 3})

    def sparse_index(self, task: IngestionTask) -> dict[str, object]:
        return self._stage("sparse_index", {})

    def multimodal_index(self, task: IngestionTask) -> dict[str, object]:
        return self._stage("multimodal_index", {"evidence_image_count": self.images})

    def visual_index(self, task: IngestionTask) -> dict[str, object]:
        if self.images == 0:
            return self._stage(
                "visual_index",
                {"skipped": True, "reason": "no_images", "evidence_image_count": 0},
            )
        details = {
            "evidence_image_count": self.images,
            "visual_indexed_count": self.indexed,
            "visual_missing_count": self.missing,
            "visual_invalid_count": self.invalid,
            "visual_skipped_by_policy": self.skipped_policy,
            "visual_selected_count": self.indexed + self.missing + self.invalid,
        }
        if self.missing or self.invalid:
            raise IngestionStepError(
                "visual_assets_missing",
                "视觉素材缺失或无法解码，请恢复素材后重试",
                details=details,
            )
        return self._stage("visual_index", details)

    def publish(self, task: IngestionTask) -> dict[str, object]:
        published = ["text", "multimodal", "bm25"]
        if self.publish_visual:
            published.append("visual")
        return self._stage(
            "publish",
            {
                "published_indexes": published,
                "text_points": 3,
                "representation_points": 4,
                "sparse_records": 3,
                "visual_points": self.images,
            },
        )

    def verify(self, task: IngestionTask) -> dict[str, object]:
        visual_ready = self.missing == 0 and self.invalid == 0
        return self._stage(
            "verify",
            {
                "text_points": 3,
                "representation_points": 4,
                "sparse_ok": True,
                "sparse_records": 3,
                "visual_points": self.indexed,
                "visual_expected": self.indexed,
                "visual_active": self.indexed,
                "visual_ready": visual_ready,
                "evidence_image_count": self.images,
                "visual_skipped_by_policy": self.skipped_policy,
            },
        )

    def graph(self, task: IngestionTask) -> dict[str, object]:
        return self._stage("graph", {"skipped": True, "ready": False})


def _worker(
    tmp_path: Path,
    steps: FakeSteps,
    **settings_overrides: object,
) -> tuple[IngestionWorker, IngestionTaskStore]:
    settings = _settings(tmp_path, **settings_overrides)
    store = IngestionTaskStore(settings)
    worker = IngestionWorker(settings, store, steps=steps)
    return worker, store


def test_task_store_roundtrip_and_corruption(tmp_path: Path) -> None:
    store = IngestionTaskStore(_settings(tmp_path))
    task = IngestionTask(document_id=uuid4(), knowledge_base_id=uuid4())
    store.write(task)
    loaded = store.read(task.document_id)
    assert loaded is not None
    assert loaded.status == "queued"
    assert store.list_tasks() == [loaded]
    store.write_bundle(task.document_id, {"dense_points": []})
    assert store.read_bundle(task.document_id) == {"dense_points": []}
    store.remove_bundle(task.document_id)
    assert store.read_bundle(task.document_id) is None
    store.task_path(task.document_id).write_text("{not json")
    with pytest.raises(IngestionStoreError):
        store.read(task.document_id)
    store.delete(task.document_id)
    assert store.read(task.document_id) is None


def test_worker_full_pipeline_completes(tmp_path: Path) -> None:
    steps = FakeSteps()
    worker, store = _worker(tmp_path, steps)
    task = worker.enqueue(uuid4(), uuid4())
    worker._run_document(task.document_id)
    finished = store.read(task.document_id)
    assert finished is not None
    assert finished.status == "completed"
    assert finished.attempts == 1
    assert finished.chunk_count == 3
    assert finished.image_count == 2
    assert finished.visual_indexed_count == 2
    assert finished.visual_missing_count == 0
    assert finished.visual_skipped_by_policy == 0
    assert finished.indexes["text"] is True
    assert finished.indexes["bm25"] is True
    assert finished.indexes["multimodal"] is True
    assert finished.indexes["visual"] is True
    assert finished.indexes["graph"] is False
    assert finished.visual_status == "ready"
    assert finished.error_stage is None
    assert steps.calls == [
        "parse",
        "chunk",
        "text_index",
        "sparse_index",
        "multimodal_index",
        "visual_index",
        "publish",
        "verify",
        "graph",
    ]


def test_worker_visual_ready_counts_three_images(tmp_path: Path) -> None:
    steps = FakeSteps(images=3)
    worker, store = _worker(tmp_path, steps)
    task = worker.enqueue(uuid4(), uuid4())
    worker._run_document(task.document_id)
    finished = store.read(task.document_id)
    assert finished is not None
    assert finished.status == "completed"
    assert finished.image_count == 3
    assert finished.visual_indexed_count == 3
    assert finished.visual_missing_count == 0
    assert finished.visual_status == "ready"


def test_worker_partial_missing_assets_is_not_ready_and_retry_recovers(
    tmp_path: Path,
) -> None:
    steps = FakeSteps(images=3, indexed=2, missing=1)
    worker, store = _worker(tmp_path, steps)
    task = worker.enqueue(uuid4(), uuid4())
    worker._run_document(task.document_id)
    failed = store.read(task.document_id)
    assert failed is not None
    assert failed.status == "failed"
    assert failed.error_stage == "visual_index"
    assert failed.error_code == "visual_assets_missing"
    assert failed.image_count == 3
    assert failed.visual_indexed_count == 2
    assert failed.visual_missing_count == 1
    assert failed.visual_status == "assets_missing"
    assert failed.indexes["visual"] is False

    steps.missing = 0
    steps.indexed = 3
    assert worker.retry(task.document_id).status == "queued"
    worker._run_document(task.document_id)
    finished = store.read(task.document_id)
    assert finished is not None
    assert finished.status == "completed"
    assert finished.image_count == 3
    assert finished.visual_indexed_count == 3
    assert finished.visual_missing_count == 0
    assert finished.visual_status == "ready"


def test_worker_all_assets_missing_is_not_reported_as_no_images(tmp_path: Path) -> None:
    steps = FakeSteps(images=3, indexed=0, missing=3)
    worker, store = _worker(tmp_path, steps)
    task = worker.enqueue(uuid4(), uuid4())
    worker._run_document(task.document_id)
    failed = store.read(task.document_id)
    assert failed is not None
    assert failed.status == "failed"
    assert failed.error_code == "visual_assets_missing"
    assert failed.image_count == 3
    assert failed.visual_indexed_count == 0
    assert failed.visual_missing_count == 3
    assert failed.visual_status == "assets_missing"
    assert failed.visual_status != "skipped_no_images"


def test_worker_visual_disabled_with_images_reports_not_enabled(tmp_path: Path) -> None:
    steps = FakeSteps(images=2, publish_visual=False)
    worker, store = _worker(tmp_path, steps, visual_embedding_enabled=False)
    task = worker.enqueue(uuid4(), uuid4())
    worker._run_document(task.document_id)
    finished = store.read(task.document_id)
    assert finished is not None
    assert finished.status == "completed"
    assert finished.image_count == 2
    assert finished.indexes["visual"] is False
    assert finished.visual_status == "not_enabled"


def test_worker_selective_policy_skips_are_distinct_from_missing(tmp_path: Path) -> None:
    steps = FakeSteps(images=3, indexed=2, skipped_policy=1)
    worker, store = _worker(tmp_path, steps, ingestion_visual_mode="selective")
    task = worker.enqueue(uuid4(), uuid4())
    worker._run_document(task.document_id)
    finished = store.read(task.document_id)
    assert finished is not None
    assert finished.status == "completed"
    assert finished.image_count == 3
    assert finished.visual_indexed_count == 2
    assert finished.visual_skipped_by_policy == 1
    assert finished.visual_missing_count == 0
    assert finished.visual_status == "ready"


def test_worker_failure_then_retry_recovers(tmp_path: Path) -> None:
    failing = FakeSteps(fail_at="chunk")
    worker, store = _worker(tmp_path, failing)
    task = worker.enqueue(uuid4(), uuid4())
    worker._run_document(task.document_id)
    failed = store.read(task.document_id)
    assert failed is not None
    assert failed.status == "failed"
    assert failed.error_stage == "chunk"
    assert failed.error_code == "stage_failed"
    assert "exploded" in (failed.error_message or "")
    retried = worker.retry(task.document_id)
    assert retried.status == "queued"
    assert retried.attempts == 0
    worker._steps = FakeSteps()  # type: ignore[assignment]
    worker._run_document(task.document_id)
    finished = store.read(task.document_id)
    assert finished is not None
    assert finished.status == "completed"


def test_worker_no_image_document_still_completes(tmp_path: Path) -> None:
    steps = FakeSteps(images=0)
    worker, store = _worker(tmp_path, steps)
    task = worker.enqueue(uuid4(), uuid4())
    worker._run_document(task.document_id)
    finished = store.read(task.document_id)
    assert finished is not None
    assert finished.status == "completed"
    assert finished.image_count == 0
    assert finished.indexes["visual"] is True
    assert finished.visual_status == "skipped_no_images"


def test_worker_recovery_requeues_interrupted_then_exhausts(tmp_path: Path) -> None:
    store = IngestionTaskStore(_settings(tmp_path))
    document_id = uuid4()
    task = IngestionTask(document_id=document_id, knowledge_base_id=uuid4())
    task.ensure_stages()
    task.status = "running"
    task.attempts = 1
    task.current_stage = "chunk"
    task.stages["chunk"] = task.stages["chunk"].model_copy(update={"status": "running"})
    store.write(task)
    worker = IngestionWorker(_settings(tmp_path), store, steps=FakeSteps())
    recovered = worker._recover_interrupted()
    assert recovered == [document_id]
    resumed = store.read(document_id)
    assert resumed is not None
    assert resumed.status == "queued"
    assert resumed.interruptions == 1
    assert resumed.stages["chunk"].status == "interrupted"

    exhausted = store.read(document_id)
    assert exhausted is not None
    exhausted.status = "running"
    exhausted.attempts = 3
    store.write(exhausted)
    assert worker._recover_interrupted() == []
    final = store.read(document_id)
    assert final is not None
    assert final.status == "failed"
    assert final.error_code == "ingestion_interrupted"


def test_real_visual_step_keeps_evidence_image_count_when_disabled(tmp_path: Path) -> None:
    settings = _settings(tmp_path, visual_embedding_enabled=False)
    store = IngestionTaskStore(settings)
    steps = IngestionSteps(settings, store)
    document_id = uuid4()
    knowledge_base_id = uuid4()
    canonical = _canonical_with_images(document_id, 2)
    write_parsing_artifact(
        tmp_path,
        document_id,
        canonical=canonical,
        assets={IMAGE_ASSET_A: PNG_BYTES, IMAGE_ASSET_B: JPEG_BYTES},
    )
    artifact = build_evidence_document(canonical, knowledge_base_id)
    write_evidence_artifact(tmp_path, artifact)
    store.write_bundle(
        document_id,
        {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "pipeline_version": INGESTION_PIPELINE_VERSION,
            "evidence": artifact.model_dump(mode="json"),
            "image_count": 2,
        },
    )
    task = IngestionTask(document_id=document_id, knowledge_base_id=knowledge_base_id)
    details = steps.visual_index(task)
    assert details["skipped"] is True
    assert details["reason"] == "visual_embedding_disabled"
    assert details["evidence_image_count"] == 2
    bundle = store.read_bundle(document_id)
    assert bundle is not None
    assert bundle["visual_evidence_count"] == 2

    worker = IngestionWorker(settings, store, steps=steps)
    worker._apply_details(task, "multimodal_index", {"evidence_image_count": 2})
    worker._apply_details(task, "visual_index", details)
    assert task.image_count == 2
    assert task.visual_status == "not_enabled"
    assert task.indexes["visual"] is False


def test_worker_enqueue_is_idempotent(tmp_path: Path) -> None:
    worker, _store = _worker(tmp_path, FakeSteps())
    document_id = uuid4()
    kb = uuid4()
    first = worker.enqueue(document_id, kb)
    second = worker.enqueue(document_id, kb)
    assert first.document_id == second.document_id
    worker._run_document(document_id)
    completed = worker.enqueue(document_id, kb)
    assert completed.status == "completed"
    with pytest.raises(IngestionError) as rebuild_error:
        worker.enqueue(document_id, kb, force=True)
    assert rebuild_error.value.code == "rebuild_not_supported"
    with pytest.raises(IngestionError) as retry_error:
        worker.retry(document_id)
    assert retry_error.value.code == "rebuild_not_supported"


class FakeApiWorker:
    """Minimal worker facade used by API tests."""

    def __init__(self, store: IngestionTaskStore) -> None:
        self.store = store
        self.enqueued: list[UUID] = []

    def enqueue(self, document_id: UUID, knowledge_base_id: UUID, **_: object) -> IngestionTask:
        self.enqueued.append(document_id)
        task = IngestionTask(document_id=document_id, knowledge_base_id=knowledge_base_id)
        self.store.write(task)
        return task

    def status(self, document_id: UUID) -> IngestionTask | None:
        return self.store.read(document_id)

    def retry(self, document_id: UUID) -> IngestionTask:
        task = self.store.read(document_id)
        assert task is not None
        task.status = "queued"
        task.attempts = 0
        self.store.write(task)
        return task

    def list_tasks(self, knowledge_base_id: UUID | None = None) -> list[IngestionTask]:
        tasks = self.store.list_tasks()
        if knowledge_base_id is not None:
            tasks = [task for task in tasks if task.knowledge_base_id == knowledge_base_id]
        return tasks


@pytest.mark.anyio
async def test_ingestion_api_status_retry_and_list(
    ingestion_client_factory: object,
    test_data_dir: Path,
) -> None:
    store = IngestionTaskStore(_settings(test_data_dir))
    fake = FakeApiWorker(store)
    factory = ingestion_client_factory
    async with factory(fake) as client:  # type: ignore[operator]
        kb_id = await create_knowledge_base(client, "自动入库测试")
        document = await upload_document(client, kb_id)
        document_id = UUID(str(document["id"]))
        assert fake.enqueued == [document_id]

        status_response = await client.get(
            f"/api/v1/knowledge-bases/{kb_id}/documents/{document_id}/ingestion"
        )
        assert status_response.status_code == 200
        body = status_response.json()
        assert body["task"]["document_id"] == str(document_id)
        assert body["task"]["status"] == "queued"

        retry_response = await client.post(
            f"/api/v1/knowledge-bases/{kb_id}/documents/{document_id}/ingestion/retry"
        )
        assert retry_response.status_code == 202
        assert retry_response.json()["task"]["status"] == "queued"

        list_response = await client.get(f"/api/v1/knowledge-bases/{kb_id}/ingestion")
        assert list_response.status_code == 200
        assert len(list_response.json()["tasks"]) == 1


@pytest.mark.anyio
async def test_ingestion_api_disabled_without_worker(client: AsyncClient) -> None:
    kb_id = await create_knowledge_base(client, "无 worker 测试")
    document = await upload_document(client, kb_id)
    response = await client.get(
        f"/api/v1/knowledge-bases/{kb_id}/documents/{document['id']}/ingestion"
    )
    assert response.status_code == 503
    assert response.json()["detail"] == "自动入库任务未启用"


def test_sparse_document_chunk_count_tracks_active_generation(tmp_path):
    from uuid import uuid4

    from knowledge_scope.retrieval.sparse import SparseChunkRecord, SparseIndexStore

    store = SparseIndexStore(tmp_path / "sparse.sqlite3")
    kb_id = uuid4()
    document_id = uuid4()
    record = SparseChunkRecord(
        knowledge_base_id=kb_id,
        chunking_config_fingerprint="a" * 64,
        chunk_id="chunk-1",
        document_id=document_id,
        ordinal=0,
        text="hello",
        page_start=1,
        page_end=1,
        source_block_ids=["p1-b1"],
        section_path=[],
        content_types=["text"],
        asset_refs=[],
    )
    store.replace_document(kb_id, document_id, [record])
    assert store.document_chunk_count(kb_id, document_id) == 1
    assert store.delete_document(kb_id, document_id) == 1
    assert store.document_chunk_count(kb_id, document_id) == 0
    store.close()
