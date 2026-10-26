# ruff: noqa: RUF001
"""Single-host background worker that runs the PDF auto-ingestion pipeline.

One process-local thread consumes a FIFO queue.  CPU/GPU heavy stages therefore
never overlap with each other, which keeps memory bounded on a 12 GB workstation
GPU.  State is written to the file-backed task store after every stage so the
UI can poll progress and a restarted process can recover interrupted runs.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from time import perf_counter
from typing import Any
from uuid import UUID

from knowledge_scope.shared.config import Settings

from .models import STAGE_LABELS, STAGE_ORDER, IngestionTask, status_entry
from .store import IngestionTaskStore

logger = logging.getLogger(__name__)

_HEAVY_STAGES = {
    "parse",
    "text_index",
    "multimodal_index",
    "visual_index",
    "publish",
    "verify",
}


class IngestionError(RuntimeError):
    """User-facing error raised by the ingestion worker facade."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _bundle_has(bundle: dict[str, Any] | None, *keys: str) -> bool:
    if not bundle:
        return False
    return all(key in bundle for key in keys)


class IngestionWorker:
    """Queue-based, one-heavy-task-at-a-time ingestion runner."""

    def __init__(
        self,
        settings: Settings,
        task_store: IngestionTaskStore,
        *,
        steps: object | None = None,
        steps_factory: Callable[[], object] | None = None,
    ) -> None:
        self.settings = settings
        self.task_store = task_store
        if steps is None and steps_factory is None:
            from .steps import IngestionSteps

            steps = IngestionSteps(settings, task_store)
        self._steps = steps if steps is not None else None
        self._steps_factory = steps_factory
        self._queue: queue.Queue[UUID] = queue.Queue()
        self._lock = threading.Lock()
        self._enqueued: set[UUID] = set()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._current_document_id: UUID | None = None

    # -- lifecycle ------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        recovered = self._recover_interrupted()
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="ingestion-worker",
            daemon=True,
        )
        self._thread.start()
        for document_id in recovered:
            self._enqueue_document(document_id)

    def stop(self, *, timeout: float = 10.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _steps_impl(self) -> object:
        if self._steps is None:
            assert self._steps_factory is not None
            self._steps = self._steps_factory()
        return self._steps

    # -- public facade --------------------------------------------------

    def status(self, document_id: UUID) -> IngestionTask | None:
        return self.task_store.read(document_id)

    def list_tasks(self, knowledge_base_id: UUID | None = None) -> list[IngestionTask]:
        tasks = self.task_store.list_tasks()
        if knowledge_base_id is not None:
            tasks = [task for task in tasks if task.knowledge_base_id == knowledge_base_id]
        tasks.sort(key=lambda task: task.updated_at, reverse=True)
        return tasks

    def enqueue(
        self,
        document_id: UUID,
        knowledge_base_id: UUID,
        *,
        source_sha256: str | None = None,
        force: bool = False,
    ) -> IngestionTask:
        """Queue one document; idempotent while a run is pending or running."""

        with self._lock:
            existing = self.task_store.read(document_id)
            if existing is not None and existing.knowledge_base_id != knowledge_base_id:
                raise IngestionError("ingestion_task_conflict", "文档归属知识库不一致")
            if existing is not None:
                if existing.status in ("queued", "running"):
                    return existing
                if existing.status == "completed":
                    if not force:
                        return existing
                    # Completed documents are immutable through this pipeline.
                    # Re-parsing invalidates chunks/evidence and a failed publish
                    # could destroy the previous valid index, so forced rebuilds
                    # are intentionally rejected until safe rollback exists.
                    raise IngestionError(
                        "rebuild_not_supported",
                        "该文档已完成自动入库；当前版本不支持强制重建，"
                        "以避免破坏既有索引。如需重建，请删除文档后重新上传。",
                    )
                if existing.status == "failed" and not force:
                    return existing
            if existing is None:
                task = IngestionTask(
                    document_id=document_id,
                    knowledge_base_id=knowledge_base_id,
                    source_sha256=source_sha256,
                )
            else:
                task = existing
            self._prepare_task(task)
            self.task_store.write(task)
            self._enqueue_document(document_id)
            return task

    def retry(self, document_id: UUID) -> IngestionTask:
        with self._lock:
            task = self.task_store.read(document_id)
            if task is None:
                raise IngestionError("ingestion_task_not_found", "没有找到该文档的入库任务")
            if task.status in ("queued", "running"):
                return task
            if task.status == "completed":
                raise IngestionError(
                    "rebuild_not_supported",
                    "该文档已完成自动入库；当前版本不支持重新处理，"
                    "以避免破坏既有索引。如需重建，请删除文档后重新上传。",
                )
            self._prepare_task(task, reset_attempts=True)
            self.task_store.write(task)
            self._enqueue_document(document_id)
            return task

    def _prepare_task(self, task: IngestionTask, *, reset_attempts: bool = False) -> None:
        task.ensure_stages()
        for name in STAGE_ORDER:
            task.stages[name] = task.stages[name].model_copy(
                update={
                    "status": "pending",
                    "started_at": None,
                    "finished_at": None,
                    "message": None,
                    "error_code": None,
                    "details": {},
                }
            )
        task.status = "queued"
        task.current_stage = None
        task.finished_at = None
        task.error_stage = None
        task.error_code = None
        task.error_message = None
        task.chunk_count = 0
        task.image_count = 0
        task.visual_status = "pending"
        task.timings_ms = {}
        task.vram_peak_mb = None
        task.indexes = {
            "text": False,
            "bm25": False,
            "multimodal": False,
            "visual": False,
            "graph": False,
        }
        if reset_attempts:
            task.attempts = 0
        task.updated_at = _utc_now()

    # -- recovery -------------------------------------------------------

    def _recover_interrupted(self) -> list[UUID]:
        """Resume tasks left queued/running by a previous process."""

        recovered: list[UUID] = []
        for task in self.task_store.list_tasks():
            if task.status not in ("queued", "running"):
                continue
            task.interruptions += 1
            if task.current_stage is not None:
                stage = task.stages.get(task.current_stage)
                if stage is not None and stage.status == "running":
                    stage.status = "interrupted"
                    stage.message = "进程中断，正在恢复"
                    task.stages[task.current_stage] = stage
            if task.attempts >= self.settings.ingestion_max_attempts:
                task.status = "failed"
                task.error_code = "ingestion_interrupted"
                task.error_message = "入库进程多次中断，请手动重试"
                task.finished_at = _utc_now()
                logger.warning(
                    "ingestion recovery exhausted attempts document=%s",
                    task.document_id,
                )
            else:
                task.status = "queued"
                task.current_stage = None
                recovered.append(task.document_id)
            task.updated_at = _utc_now()
            self.task_store.write(task)
        return recovered

    # -- queue loop -----------------------------------------------------

    def _enqueue_document(self, document_id: UUID) -> None:
        if document_id in self._enqueued:
            return
        self._enqueued.add(document_id)
        self._queue.put(document_id)

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                document_id = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._run_document(document_id)
            except Exception:  # pragma: no cover - defensive boundary
                logger.exception("ingestion worker crashed document=%s", document_id)
            finally:
                with self._lock:
                    self._enqueued.discard(document_id)
                self._queue.task_done()

    def _run_document(self, document_id: UUID) -> None:
        task = self.task_store.read(document_id)
        if task is None or task.status != "queued":
            return
        with self._lock:
            self._current_document_id = document_id
        task.status = "running"
        task.attempts += 1
        task.updated_at = _utc_now()
        self.task_store.write(task)
        steps = self._steps_impl()
        for name in STAGE_ORDER:
            if task.status != "running":
                break
            handler = getattr(steps, name)
            self._run_stage(task, name, handler)
        with self._lock:
            self._current_document_id = None

    def _run_stage(
        self,
        task: IngestionTask,
        name: str,
        handler: Callable[[IngestionTask], dict[str, object]],
    ) -> None:
        if self._stage_is_ready(task, name):
            stage = task.stages[name]
            stage.status = "skipped"
            stage.message = "已完成的阶段不再重复执行"
            stage.finished_at = _utc_now()
            task.timings_ms.setdefault(name, 0.0)
            task.updated_at = _utc_now()
            self.task_store.write(task)
            return
        stage = task.stages[name]
        if stage.status == "pending":
            stage.started_at = stage.started_at or _utc_now()
        stage.status = "running"
        stage.message = STAGE_LABELS.get(name)
        task.current_stage = name
        task.updated_at = _utc_now()
        self.task_store.write(task)
        started = perf_counter()
        try:
            details = handler(task)
        except Exception as error:
            elapsed = (perf_counter() - started) * 1_000
            code = getattr(error, "code", "stage_failed")
            message = str(error) or "入库阶段失败"
            stage.status = "failed"
            stage.finished_at = _utc_now()
            stage.error_code = str(code)
            stage.message = message[:2_000]
            task.status = "failed"
            task.current_stage = name
            task.error_stage = name
            task.error_code = str(code)
            task.error_message = message[:2_000]
            task.finished_at = _utc_now()
            task.timings_ms[name] = round(elapsed, 3)
            if name == "visual_index":
                task.visual_status = "failed"
            task.updated_at = _utc_now()
            self.task_store.write(task)
            logger.warning(
                "ingestion stage failed document=%s stage=%s code=%s message=%s",
                task.document_id,
                name,
                code,
                message,
            )
            return
        elapsed = (perf_counter() - started) * 1_000
        stage.status = "completed"
        stage.finished_at = _utc_now()
        stage.details = dict(details or {})
        stage.message = None
        task.timings_ms[name] = round(float((details or {}).get("elapsed_ms") or elapsed), 3)
        self._apply_details(task, name, details or {})
        task.updated_at = _utc_now()
        self.task_store.write(task)
        if task.status == "running" and name == STAGE_ORDER[-1]:
            self._finish_task(task)

    def _apply_details(
        self,
        task: IngestionTask,
        name: str,
        details: dict[str, object],
    ) -> None:
        if "chunks" in details:
            task.chunk_count = int(details["chunks"] or 0)  # type: ignore[arg-type]
        if "images" in details:
            task.image_count = int(details["images"] or 0)  # type: ignore[arg-type]
        vram = details.get("vram_peak_mb")
        if isinstance(vram, (int, float)) and float(vram) > (task.vram_peak_mb or 0):
            task.vram_peak_mb = float(vram)
        if name == "visual_index":
            if bool(details.get("skipped")):
                reason = str(details.get("reason") or "")
                task.visual_status = (
                    "not_enabled" if reason == "visual_embedding_disabled" else "skipped_no_images"
                )
            else:
                task.visual_status = (
                    "assets_missing" if int(details.get("images") or 0) == 0 else "pending"
                )
        if name == "publish":
            published = details.get("published_indexes")
            if isinstance(published, list):
                for key in ("text", "multimodal", "bm25"):
                    if key in published:
                        task.indexes[key] = True
                if "visual" in published:
                    task.indexes["visual"] = True
        if name == "verify":
            if bool(details.get("sparse_ok")):
                task.indexes["bm25"] = True
            if int(details.get("text_points") or 0) > 0:  # type: ignore[arg-type]
                task.indexes["text"] = True
            if int(details.get("representation_points") or 0) > 0:  # type: ignore[arg-type]
                task.indexes["multimodal"] = True
            if self.settings.visual_embedding_enabled:
                visual_expected = int(details.get("visual_expected") or 0)
                task.indexes["visual"] = bool(details.get("visual_ready")) or (
                    visual_expected == 0 and task.image_count == 0
                )
                if task.image_count == 0:
                    task.visual_status = "skipped_no_images"
                elif task.indexes["visual"]:
                    task.visual_status = "ready"
                else:
                    task.visual_status = "assets_missing"
            elif task.image_count == 0:
                task.visual_status = "skipped_no_images"
            else:
                task.visual_status = "not_enabled"
        if name == "graph":
            task.indexes["graph"] = bool(details.get("ready", False))

    def _finish_task(self, task: IngestionTask) -> None:
        task.status = "completed"
        task.current_stage = None
        task.finished_at = _utc_now()
        task.updated_at = _utc_now()
        self.task_store.write(task)

    def _stage_is_ready(self, task: IngestionTask, name: str) -> bool:
        stage = task.stages.get(name)
        if stage is None or stage.status != "completed":
            return False
        bundle = self.task_store.read_bundle(task.document_id)
        required: dict[str, tuple[str, ...]] = {
            "text_index": ("dense_points", "chunk_count"),
            "sparse_index": ("sparse_records",),
            "multimodal_index": ("evidence", "representation_points"),
            "visual_index": ("visual_entries",),
        }
        keys = required.get(name)
        if keys is None:
            return name not in _HEAVY_STAGES
        return _bundle_has(bundle, *keys)


def ingestion_status_payload(task: IngestionTask) -> dict[str, object]:
    return {
        "task": task.model_dump(mode="json"),
        "summary": status_entry(task).model_dump(mode="json"),
    }


__all__ = [
    "IngestionError",
    "IngestionWorker",
    "ingestion_status_payload",
]
