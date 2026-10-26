# ruff: noqa: RUF001
"""File-backed contracts for the single-host PDF auto-ingestion pipeline."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

INGESTION_TASK_SCHEMA_VERSION = "1.0"
INGESTION_PIPELINE_VERSION = "m8-auto-ingest-v1"

IngestionStatus = Literal["queued", "running", "completed", "failed"]
StageStatus = Literal["pending", "running", "completed", "failed", "skipped", "interrupted"]
VisualIndexStatus = Literal[
    "pending",
    "not_enabled",
    "skipped_no_images",
    "assets_missing",
    "ready",
    "failed",
]

STAGE_ORDER: tuple[str, ...] = (
    "parse",
    "chunk",
    "text_index",
    "sparse_index",
    "multimodal_index",
    "visual_index",
    "publish",
    "verify",
    "graph",
)

STAGE_LABELS: dict[str, str] = {
    "parse": "正在解析",
    "chunk": "正在分块",
    "text_index": "正在建立文本索引",
    "sparse_index": "正在建立 BM25 索引",
    "multimodal_index": "正在建立多模态索引",
    "visual_index": "正在建立视觉索引",
    "publish": "正在发布索引",
    "verify": "正在验证索引",
    "graph": "知识图谱（独立流程）",
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class IngestionStageState(BaseModel):
    """One stage inside an ingestion task."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    name: str
    status: StageStatus = "pending"
    started_at: str | None = None
    finished_at: str | None = None
    message: str | None = Field(default=None, max_length=2_000)
    error_code: str | None = Field(default=None, max_length=100)
    details: dict[str, object] = Field(default_factory=dict)


class IngestionTask(BaseModel):
    """Durable state of one document ingestion run."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    schema_version: str = INGESTION_TASK_SCHEMA_VERSION
    document_id: UUID
    knowledge_base_id: UUID
    pipeline_version: str = INGESTION_PIPELINE_VERSION
    status: IngestionStatus = "queued"
    current_stage: str | None = None
    attempts: int = 0
    interruptions: int = 0
    created_at: str = Field(default_factory=_utc_now)
    updated_at: str = Field(default_factory=_utc_now)
    finished_at: str | None = None
    stages: dict[str, IngestionStageState] = Field(default_factory=dict)
    indexes: dict[str, bool] = Field(
        default_factory=lambda: {
            "text": False,
            "bm25": False,
            "multimodal": False,
            "visual": False,
            "graph": False,
        }
    )
    visual_status: VisualIndexStatus = "pending"
    source_sha256: str | None = None
    chunk_count: int = 0
    image_count: int = 0
    timings_ms: dict[str, float] = Field(default_factory=dict)
    vram_peak_mb: float | None = None
    error_stage: str | None = None
    error_code: str | None = None
    error_message: str | None = Field(default=None, max_length=2_000)

    def ensure_stages(self) -> None:
        for name in STAGE_ORDER:
            self.stages.setdefault(name, IngestionStageState(name=name))

    def reset_stage(self, name: str) -> IngestionStageState:
        self.stages[name] = IngestionStageState(name=name)
        return self.stages[name]


class IngestionStatusEntry(BaseModel):
    """Compact per-document status used by the polling endpoint."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    document_id: UUID
    knowledge_base_id: UUID
    status: IngestionStatus
    current_stage: str | None
    stage_label: str | None
    stage_status: StageStatus | None
    stage_message: str | None
    error_stage: str | None
    error_code: str | None
    error_message: str | None
    chunk_count: int
    image_count: int
    attempts: int
    updated_at: str
    visual_status: VisualIndexStatus


def status_entry(task: IngestionTask) -> IngestionStatusEntry:
    stage = task.stages.get(task.current_stage or "")
    return IngestionStatusEntry(
        document_id=task.document_id,
        knowledge_base_id=task.knowledge_base_id,
        status=task.status,
        current_stage=task.current_stage,
        stage_label=STAGE_LABELS.get(task.current_stage or ""),
        stage_status=stage.status if stage is not None else None,
        stage_message=stage.message if stage is not None else None,
        error_stage=task.error_stage,
        error_code=task.error_code,
        error_message=task.error_message,
        chunk_count=task.chunk_count,
        image_count=task.image_count,
        attempts=task.attempts,
        updated_at=task.updated_at,
        visual_status=task.visual_status,
    )


__all__ = [
    "INGESTION_PIPELINE_VERSION",
    "INGESTION_TASK_SCHEMA_VERSION",
    "STAGE_LABELS",
    "STAGE_ORDER",
    "IngestionStageState",
    "IngestionStatus",
    "IngestionStatusEntry",
    "IngestionTask",
    "StageStatus",
    "VisualIndexStatus",
    "status_entry",
]
