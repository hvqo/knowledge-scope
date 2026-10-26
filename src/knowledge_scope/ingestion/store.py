"""Atomic file-backed ingestion task and publish-bundle storage."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any
from uuid import UUID

from knowledge_scope.shared.config import Settings

from .models import INGESTION_TASK_SCHEMA_VERSION, IngestionTask

INGESTION_DIRECTORY_NAME = "ingestion"
TASKS_DIRECTORY_NAME = "tasks"
BUNDLE_FILENAME = "bundle.json"


class IngestionStoreError(RuntimeError):
    """Raised when ingestion state cannot be read or persisted safely."""


def _ingestion_root(settings: Settings) -> Path:
    root = Path(settings.data_dir) / INGESTION_DIRECTORY_NAME
    if root.is_symlink():
        raise IngestionStoreError("ingestion state directory must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _tasks_root(settings: Settings) -> Path:
    root = _ingestion_root(settings) / TASKS_DIRECTORY_NAME
    if root.is_symlink():
        raise IngestionStoreError("ingestion task directory must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() and path.is_symlink():
        raise IngestionStoreError("refusing to overwrite a symlinked ingestion state file")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}-",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


class IngestionTaskStore:
    """Durable task state kept outside PostgreSQL to avoid a schema migration."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def task_path(self, document_id: UUID) -> Path:
        return _tasks_root(self.settings) / f"{document_id}.json"

    def read(self, document_id: UUID) -> IngestionTask | None:
        path = self.task_path(document_id)
        if path.is_symlink():
            raise IngestionStoreError("ingestion task file must not be a symlink")
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            task = IngestionTask.model_validate(payload)
        except Exception as error:
            raise IngestionStoreError("ingestion task state is invalid") from error
        if task.schema_version != INGESTION_TASK_SCHEMA_VERSION:
            raise IngestionStoreError("unsupported ingestion task schema version")
        task.ensure_stages()
        return task

    def write(self, task: IngestionTask) -> IngestionTask:
        task.ensure_stages()
        _atomic_write_json(self.task_path(task.document_id), task.model_dump(mode="json"))
        return task

    def delete(self, document_id: UUID) -> None:
        path = self.task_path(document_id)
        if path.is_symlink():
            raise IngestionStoreError("ingestion task file must not be a symlink")
        path.unlink(missing_ok=True)
        bundle_path = self.bundle_path(document_id)
        if bundle_path.is_symlink():
            raise IngestionStoreError("ingestion bundle must not be a symlink")
        bundle_path.unlink(missing_ok=True)
        bundle_dir = bundle_path.parent
        if bundle_dir.is_dir() and not any(bundle_dir.iterdir()):
            bundle_dir.rmdir()

    def list_tasks(self) -> list[IngestionTask]:
        tasks: list[IngestionTask] = []
        for path in sorted(_tasks_root(self.settings).glob("*.json")):
            if path.name.startswith("."):
                continue
            try:
                task = self.read(UUID(path.stem))
            except (ValueError, IngestionStoreError):
                continue
            if task is not None:
                tasks.append(task)
        return tasks

    # -- publish bundle -------------------------------------------------

    def bundle_path(self, document_id: UUID) -> Path:
        return _ingestion_root(self.settings) / str(document_id) / BUNDLE_FILENAME

    def read_bundle(self, document_id: UUID) -> dict[str, Any] | None:
        path = self.bundle_path(document_id)
        if path.is_symlink():
            raise IngestionStoreError("ingestion bundle must not be a symlink")
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            raise IngestionStoreError("ingestion bundle is invalid") from error
        if not isinstance(payload, dict):
            raise IngestionStoreError("ingestion bundle is invalid")
        return payload

    def write_bundle(self, document_id: UUID, payload: dict[str, Any]) -> None:
        self.bundle_path(document_id).unlink(missing_ok=True)
        _atomic_write_json(self.bundle_path(document_id), payload)

    def remove_bundle(self, document_id: UUID) -> None:
        path = self.bundle_path(document_id)
        if path.is_symlink():
            raise IngestionStoreError("ingestion bundle must not be a symlink")
        path.unlink(missing_ok=True)
        if path.parent.is_dir() and not any(path.parent.iterdir()):
            path.parent.rmdir()


__all__ = [
    "BUNDLE_FILENAME",
    "INGESTION_DIRECTORY_NAME",
    "TASKS_DIRECTORY_NAME",
    "IngestionStoreError",
    "IngestionTaskStore",
]
