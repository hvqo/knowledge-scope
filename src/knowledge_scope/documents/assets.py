"""Resolve and read original multimodal assets referenced by evidence lineage.

The asset bytes produced by MinerU live in the parsing artifact directory:
``<data_dir>/parsing/<document_id>/mineru/<asset_ref>``.  ``asset_ref`` is an
opaque slash separated path owned by the parsing stage, so resolving it always
requires the document id and must never trust the client supplied value.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final
from uuid import UUID

from pydantic import ValidationError

from knowledge_scope.documents.models import Document
from knowledge_scope.evidence.lifecycle import (
    EvidenceArtifactError,
    assert_evidence_artifact_current,
)
from knowledge_scope.evidence.models import MultimodalEvidence, MultimodalEvidenceDocument
from knowledge_scope.parsing.models import CanonicalDocument
from knowledge_scope.parsing.service import PARSING_DIRECTORY_NAME
from knowledge_scope.shared.config import Settings

CANONICAL_ARTIFACT_FILENAME: Final = "canonical.json"
MINERU_OUTPUT_DIRECTORY_NAME: Final = "mineru"
MAX_ASSET_BYTES: Final = 20 * 1024 * 1024

ASSET_MEDIA_TYPES: Final[dict[str, str]] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


class AssetAccessError(RuntimeError):
    """Base class for controlled evidence asset access failures."""

    code: str = "asset_unavailable"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class AssetNotFoundError(AssetAccessError):
    """The evidence, its parsing artifacts, or the requested asset is unavailable."""

    code = "asset_not_found"


class AssetUnsupportedError(AssetAccessError):
    """The resolved bytes are not a supported previewable image format."""

    code = "asset_unsupported"


class AssetTooLargeError(AssetAccessError):
    """The resolved asset exceeds the safe preview size budget."""

    code = "asset_too_large"


@dataclass(frozen=True, slots=True)
class EvidenceAsset:
    """One resolvable asset owned by an evidence item."""

    index: int
    asset_ref: str
    path: Path
    media_type: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class EvidenceAssetPreview:
    """Public, path-free facts about one evidence asset."""

    index: int
    media_type: str | None
    size_bytes: int | None
    available: bool


@dataclass(frozen=True, slots=True)
class EvidenceSnapshot:
    """One validated evidence item of the current canonical document."""

    document_id: UUID
    knowledge_base_id: UUID
    evidence: MultimodalEvidence


def _data_root(data_dir: Path) -> Path:
    configured_root = Path(data_dir)
    if configured_root.exists() and not configured_root.is_dir():
        raise AssetNotFoundError("数据目录不可用")
    try:
        return configured_root.resolve()
    except OSError as error:
        raise AssetNotFoundError("数据目录不可用") from error


def _validate_directory(path: Path, data_root: Path, label: str) -> None:
    """Reject symlinked or out-of-root parsing directory components."""
    if path.is_symlink():
        raise AssetNotFoundError(f"{label}不可用")
    if path.exists() and not path.is_dir():
        raise AssetNotFoundError(f"{label}不可用")
    try:
        resolved = path.resolve()
    except OSError as error:
        raise AssetNotFoundError(f"{label}不可用") from error
    if not resolved.is_relative_to(data_root):
        raise AssetNotFoundError(f"{label}不可用")


def _parsing_document_root(settings: Settings, document_id: UUID) -> Path:
    """Return the controlled parsing artifact directory of one document."""
    data_root = _data_root(settings.data_dir)
    parsing_root = data_root / PARSING_DIRECTORY_NAME
    _validate_directory(parsing_root, data_root, "解析产物目录")
    document_root = parsing_root / str(document_id)
    _validate_directory(document_root, data_root, "文档解析产物")
    if not document_root.is_dir():
        raise AssetNotFoundError("文档尚未完成解析")
    return document_root


def _mineru_output_root(settings: Settings, document_id: UUID) -> Path:
    """Return the MinerU output root that all ``asset_ref`` values are relative to."""
    document_root = _parsing_document_root(settings, document_id)
    data_root = _data_root(settings.data_dir)
    mineru_root = document_root / MINERU_OUTPUT_DIRECTORY_NAME
    _validate_directory(mineru_root, data_root, "文档素材目录")
    if not mineru_root.is_dir():
        raise AssetNotFoundError("文档尚未完成解析")
    return mineru_root


def load_current_canonical_document(settings: Settings, document_id: UUID) -> CanonicalDocument:
    """Load the canonical document that pairs with the current evidence artifact."""
    document_root = _parsing_document_root(settings, document_id)
    canonical_path = document_root / CANONICAL_ARTIFACT_FILENAME
    if canonical_path.is_symlink() or not canonical_path.is_file():
        raise AssetNotFoundError("文档解析产物不可用")
    try:
        payload = canonical_path.read_bytes()
    except OSError as error:
        raise AssetNotFoundError("文档解析产物不可用") from error
    try:
        canonical = CanonicalDocument.model_validate_json(payload)
    except (ValidationError, UnicodeDecodeError, ValueError) as error:
        raise AssetNotFoundError("文档解析产物不可用") from error
    if canonical.document_id != document_id:
        raise AssetNotFoundError("文档解析产物不可用")
    return canonical


def load_current_evidence(
    settings: Settings,
    document: Document,
) -> tuple[CanonicalDocument, MultimodalEvidenceDocument]:
    """Load the current canonical document and its validated evidence artifact."""
    canonical = load_current_canonical_document(settings, document.id)
    try:
        artifact = assert_evidence_artifact_current(
            settings.data_dir,
            canonical,
            document.knowledge_base_id,
        )
    except EvidenceArtifactError as error:
        raise AssetNotFoundError("证据产物不存在、已过期或不可用") from error
    if artifact.document_id != document.id:
        raise AssetNotFoundError("证据产物不属于该文档")
    return canonical, artifact


def get_evidence(
    settings: Settings,
    document: Document,
    evidence_id: str,
) -> EvidenceSnapshot:
    """Resolve one evidence item without exposing artifact internals."""
    _, artifact = load_current_evidence(settings, document)
    evidence = next((item for item in artifact.evidence if item.evidence_id == evidence_id), None)
    if evidence is None:
        raise AssetNotFoundError("证据不存在")
    if evidence.lineage.knowledge_base_id != document.knowledge_base_id:
        raise AssetNotFoundError("证据不存在")
    if evidence.lineage.document_id != document.id:
        raise AssetNotFoundError("证据不存在")
    return EvidenceSnapshot(
        document_id=document.id,
        knowledge_base_id=document.knowledge_base_id,
        evidence=evidence,
    )


def _safe_asset_parts(asset_ref: str) -> tuple[str, ...]:
    """Split a validated opaque asset reference into safe POSIX path parts."""
    if not asset_ref or "\\" in asset_ref or "//" in asset_ref:
        raise AssetNotFoundError("素材路径不合法")
    parts = tuple(asset_ref.split("/"))
    if any(part in {"", ".", ".."} for part in parts):
        raise AssetNotFoundError("素材路径不合法")
    return parts


def _detect_media_type(path: Path, expected: str) -> str:
    try:
        with path.open("rb") as asset_file:
            header = asset_file.read(16)
    except OSError as error:
        raise AssetNotFoundError("素材文件不可读") from error
    if header.startswith(b"\xff\xd8\xff"):
        detected = "image/jpeg"
    elif header.startswith(b"\x89PNG\r\n\x1a\n"):
        detected = "image/png"
    elif len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        detected = "image/webp"
    else:
        raise AssetUnsupportedError("素材格式不受支持")
    if detected != expected:
        raise AssetUnsupportedError("素材格式与文件扩展名不一致")
    return detected


def _resolve_asset_path(
    settings: Settings,
    document_id: UUID,
    asset_ref: str,
) -> tuple[Path, str, int]:
    """Resolve one validated asset reference inside the document's MinerU output."""
    mineru_root = _mineru_output_root(settings, document_id)
    parts = _safe_asset_parts(asset_ref)
    candidate = mineru_root.joinpath(*parts)
    if candidate.is_symlink():
        raise AssetNotFoundError("素材文件不可用")
    current = candidate
    while current != mineru_root:
        if current.is_symlink():
            raise AssetNotFoundError("素材文件不可用")
        current = current.parent
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise AssetNotFoundError("素材文件不存在") from error
    if not resolved.is_relative_to(mineru_root.resolve()):
        raise AssetNotFoundError("素材路径不合法")
    if not resolved.is_file():
        raise AssetNotFoundError("素材文件不存在")
    suffix = resolved.suffix.lower()
    expected_media_type = ASSET_MEDIA_TYPES.get(suffix)
    if expected_media_type is None:
        raise AssetUnsupportedError("素材格式不受支持")
    try:
        size_bytes = resolved.stat().st_size
    except OSError as error:
        raise AssetNotFoundError("素材文件不可用") from error
    if size_bytes <= 0:
        raise AssetNotFoundError("素材文件为空")
    if size_bytes > MAX_ASSET_BYTES:
        raise AssetTooLargeError("素材文件超过预览大小限制")
    media_type = _detect_media_type(resolved, expected_media_type)
    return resolved, media_type, size_bytes


def resolve_evidence_asset(
    settings: Settings,
    document: Document,
    evidence_id: str,
    asset_index: int,
) -> EvidenceAsset:
    """Resolve the asset at ``asset_index`` of one evidence item after full validation."""
    snapshot = get_evidence(settings, document, evidence_id)
    asset_refs = snapshot.evidence.lineage.asset_refs
    if asset_index < 0 or asset_index >= len(asset_refs):
        raise AssetNotFoundError("证据素材不存在")
    asset_ref = asset_refs[asset_index]
    path, media_type, size_bytes = _resolve_asset_path(settings, document.id, asset_ref)
    return EvidenceAsset(
        index=asset_index,
        asset_ref=asset_ref,
        path=path,
        media_type=media_type,
        size_bytes=size_bytes,
    )


def list_evidence_asset_previews(
    settings: Settings,
    document: Document,
    evidence_id: str,
) -> tuple[EvidenceAssetPreview, ...]:
    """Describe every declared asset, marking historical files as unavailable."""
    snapshot = get_evidence(settings, document, evidence_id)
    previews: list[EvidenceAssetPreview] = []
    for index, asset_ref in enumerate(snapshot.evidence.lineage.asset_refs):
        try:
            _, media_type, size_bytes = _resolve_asset_path(settings, document.id, asset_ref)
        except AssetAccessError:
            previews.append(
                EvidenceAssetPreview(
                    index=index,
                    media_type=None,
                    size_bytes=None,
                    available=False,
                )
            )
        else:
            previews.append(
                EvidenceAssetPreview(
                    index=index,
                    media_type=media_type,
                    size_bytes=size_bytes,
                    available=True,
                )
            )
    return tuple(previews)


def read_evidence_asset_bytes(asset: EvidenceAsset) -> bytes:
    """Read one resolved asset for future in-process consumers such as a VLM."""
    try:
        payload = asset.path.read_bytes()
    except OSError as error:
        raise AssetNotFoundError("素材文件不可读") from error
    if len(payload) != asset.size_bytes:
        raise AssetNotFoundError("素材文件已变化, 请重新打开预览")
    return payload


__all__ = [
    "ASSET_MEDIA_TYPES",
    "CANONICAL_ARTIFACT_FILENAME",
    "MAX_ASSET_BYTES",
    "MINERU_OUTPUT_DIRECTORY_NAME",
    "AssetAccessError",
    "AssetNotFoundError",
    "AssetTooLargeError",
    "AssetUnsupportedError",
    "EvidenceAsset",
    "EvidenceAssetPreview",
    "EvidenceSnapshot",
    "get_evidence",
    "list_evidence_asset_previews",
    "load_current_canonical_document",
    "load_current_evidence",
    "read_evidence_asset_bytes",
    "resolve_evidence_asset",
]
