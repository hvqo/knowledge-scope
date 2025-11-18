"""Build native visual vectors from authoritative Evidence and parsing assets."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any, Literal
from uuid import UUID

from knowledge_scope.evidence.lifecycle import (
    EvidenceArtifactError,
    load_evidence_artifact,
)
from knowledge_scope.evidence.models import MultimodalEvidence, MultimodalEvidenceDocument
from knowledge_scope.parsing.models import CanonicalDocument
from knowledge_scope.parsing.service import PARSING_DIRECTORY_NAME
from knowledge_scope.shared.config import Settings

from .visual_index import (
    QdrantVisualStore,
    VisualRepresentationRef,
    VisualVectorPayload,
    visual_encoder_fingerprint,
)
from .visual_policy import decide_image_index

VISUAL_ASSET_DIRECTORY = "mineru"

BuildMode = Literal["all", "selective"]


class VisualBuildError(RuntimeError):
    """Raised when the visual index cannot be built safely."""


@dataclass
class VisualBuildStats:
    knowledge_base_id: UUID
    mode: BuildMode
    collection_name: str
    model_id: str
    documents_scanned: int = 0
    documents_indexed: int = 0
    evidences_scanned: int = 0
    image_evidences: int = 0
    images_total: int = 0
    images_selected: int = 0
    images_indexed: int = 0
    images_missing: int = 0
    images_invalid: int = 0
    images_skipped_by_policy: int = 0
    skipped_reasons: dict[str, int] = field(default_factory=dict)
    embedding_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    vram_peak_mb: float | None = None
    document_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "knowledge_base_id": str(self.knowledge_base_id),
            "mode": self.mode,
            "collection_name": self.collection_name,
            "model_id": self.model_id,
            "documents_scanned": self.documents_scanned,
            "documents_indexed": self.documents_indexed,
            "evidences_scanned": self.evidences_scanned,
            "image_evidences": self.image_evidences,
            "images_total": self.images_total,
            "images_selected": self.images_selected,
            "images_indexed": self.images_indexed,
            "images_missing": self.images_missing,
            "images_invalid": self.images_invalid,
            "images_skipped_by_policy": self.images_skipped_by_policy,
            "skipped_reasons": dict(sorted(self.skipped_reasons.items())),
            "embedding_latency_ms": round(self.embedding_latency_ms, 3),
            "total_latency_ms": round(self.total_latency_ms, 3),
            "vram_peak_mb": self.vram_peak_mb,
            "document_ids": self.document_ids,
        }


@dataclass(frozen=True)
class _AssetCandidate:
    payload: VisualVectorPayload
    image_bytes: bytes


def _iter_evidence_documents(settings: Settings) -> list[MultimodalEvidenceDocument]:
    root = Path(settings.data_dir) / "evidence"
    if not root.is_dir():
        return []
    documents: list[MultimodalEvidenceDocument] = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or directory.name.startswith("."):
            continue
        try:
            document_id = UUID(directory.name)
        except ValueError:
            continue
        try:
            documents.append(load_evidence_artifact(settings.data_dir, document_id))
        except EvidenceArtifactError:
            continue
    return documents


def _load_canonical(settings: Settings, document_id: UUID) -> CanonicalDocument | None:
    path = Path(settings.data_dir) / PARSING_DIRECTORY_NAME / str(document_id) / "canonical.json"
    if not path.is_file():
        return None
    try:
        return CanonicalDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _page_signals(
    canonical: CanonicalDocument | None,
    page_number: int,
) -> tuple[int | None, bool]:
    if canonical is None:
        return None, False
    for page in canonical.pages:
        if page.page_number != page_number:
            continue
        image_count = sum(1 for block in page.blocks if block.type == "image")
        has_table = any(block.type == "table" for block in page.blocks)
        return image_count, has_table
    return None, False


def _representation_text(evidence: MultimodalEvidence, representation_type: str) -> str | None:
    for representation in evidence.representations:
        if representation.representation_type != representation_type:
            continue
        if representation.content and representation.content.strip():
            return representation.content.strip()
    return None


def _representation_ids(evidence: MultimodalEvidence) -> list[str]:
    return sorted({representation.representation_id for representation in evidence.representations})


def _representation_refs(evidence: MultimodalEvidence) -> list[VisualRepresentationRef]:
    refs: dict[str, VisualRepresentationRef] = {}
    for representation in evidence.representations:
        refs.setdefault(
            representation.representation_id,
            VisualRepresentationRef(
                representation_id=representation.representation_id,
                representation_type=representation.representation_type,
                content=representation.content,
                reference=representation.reference,
            ),
        )
    return [refs[key] for key in sorted(refs)]


def _asset_path(settings: Settings, document_id: UUID, asset_ref: str) -> Path:
    root = (
        Path(settings.data_dir) / PARSING_DIRECTORY_NAME / str(document_id) / VISUAL_ASSET_DIRECTORY
    )
    candidate = (root / asset_ref).resolve()
    root_resolved = root.resolve()
    if root_resolved not in candidate.parents:
        raise VisualBuildError("asset path escapes the parsing artifact root")
    return candidate


def _image_meta(image_bytes: bytes) -> tuple[int | None, int | None, str]:
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover
        raise VisualBuildError("缺少 Pillow 依赖") from error
    try:
        image = Image.open(BytesIO(image_bytes))
        image.load()
    except Exception:
        return None, None, "image/invalid"
    mime = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
    }.get(image.format or "", "application/octet-stream")
    return image.width, image.height, mime


class VisualIndexBuilder:
    """Build or rebuild visual vectors for one knowledge base."""

    def __init__(self, settings: Settings, store: QdrantVisualStore) -> None:
        self.settings = settings
        self.store = store

    def build(
        self,
        knowledge_base_id: UUID,
        *,
        document_ids: list[UUID] | None = None,
        mode: BuildMode = "all",
    ) -> VisualBuildStats:
        if mode not in {"all", "selective"}:
            raise ValueError("mode must be all or selective")
        if self.settings.visual_model_id and not self.settings.visual_embedding_enabled:
            # Building is an explicit operator action; it does not silently
            # enable query-time visual retrieval for everyone.
            pass
        started = perf_counter()
        self.store.ensure_collection()
        model_fingerprint = visual_encoder_fingerprint(self.store.encoder.contract)
        selected_document_ids = set(document_ids or [])
        stats = VisualBuildStats(
            knowledge_base_id=knowledge_base_id,
            mode=mode,
            collection_name=self.store.collection_name,
            model_id=self.store.model_id,
        )
        for artifact in _iter_evidence_documents(self.settings):
            if artifact.knowledge_base_id != knowledge_base_id:
                continue
            if selected_document_ids and artifact.document_id not in selected_document_ids:
                continue
            stats.documents_scanned += 1
            canonical = _load_canonical(self.settings, artifact.document_id)
            entries: list[_AssetCandidate] = []
            for evidence in artifact.evidence:
                stats.evidences_scanned += 1
                if evidence.modality != "image":
                    continue
                stats.image_evidences += 1
                page_number = evidence.lineage.page_start
                page_image_count, page_has_table = _page_signals(canonical, page_number)
                caption = _representation_text(evidence, "caption")
                context = _representation_text(evidence, "context")
                representation_ids = _representation_ids(evidence)
                for asset_index, asset_ref in enumerate(evidence.lineage.asset_refs):
                    stats.images_total += 1
                    try:
                        path = _asset_path(self.settings, artifact.document_id, asset_ref)
                    except VisualBuildError:
                        stats.images_invalid += 1
                        continue
                    if not path.is_file():
                        stats.images_missing += 1
                        stats.skipped_reasons["asset_missing"] = (
                            stats.skipped_reasons.get("asset_missing", 0) + 1
                        )
                        continue
                    try:
                        image_bytes = path.read_bytes()
                    except OSError:
                        stats.images_invalid += 1
                        continue
                    width, height, mime = _image_meta(image_bytes)
                    if mime == "image/invalid" or not image_bytes:
                        stats.images_invalid += 1
                        continue
                    decision = decide_image_index(
                        caption=caption,
                        context=context,
                        image_width=width,
                        image_height=height,
                        page_image_count=page_image_count,
                        page_has_table=page_has_table,
                        asset_size_bytes=len(image_bytes),
                    )
                    if mode == "selective" and decision.mode != "text_and_visual":
                        stats.images_skipped_by_policy += 1
                        stats.skipped_reasons["policy_text_only"] = (
                            stats.skipped_reasons.get("policy_text_only", 0) + 1
                        )
                        continue
                    stats.images_selected += 1
                    payload = VisualVectorPayload(
                        collection_fingerprint=model_fingerprint,
                        knowledge_base_id=knowledge_base_id,
                        document_id=artifact.document_id,
                        evidence_id=evidence.evidence_id,
                        page_number=max(1, page_number),
                        section_path=list(evidence.lineage.section_path),
                        source_block_ids=list(evidence.lineage.source_block_ids),
                        asset_index=asset_index,
                        asset_ref=asset_ref,
                        asset_sha256=hashlib.sha256(image_bytes).hexdigest(),
                        asset_mime=mime,
                        image_width=width,
                        image_height=height,
                        caption=caption,
                        context=context,
                        representation_ids=representation_ids,
                        representations=_representation_refs(evidence),
                        policy_mode=decision.mode,
                        policy_reason=decision.reason,
                        policy_version=decision.policy_version,
                        model_id=self.store.model_id,
                        model_fingerprint=model_fingerprint,
                        embedding_dim=self.store.encoder.contract.dimension,
                    )
                    entries.append(_AssetCandidate(payload=payload, image_bytes=image_bytes))
            if not entries:
                continue
            embed_started = perf_counter()
            indexed = self.store.replace_document(
                artifact.document_id,
                [(entry.payload, entry.image_bytes) for entry in entries],
            )
            stats.embedding_latency_ms += (perf_counter() - embed_started) * 1_000
            stats.images_indexed += indexed
            stats.documents_indexed += 1
            stats.document_ids.append(str(artifact.document_id))
        stats.total_latency_ms = (perf_counter() - started) * 1_000
        vram_probe = getattr(self.store.encoder, "vram_peak_mb", None)
        stats.vram_peak_mb = vram_probe() if callable(vram_probe) else vram_probe
        return stats


__all__ = [
    "BuildMode",
    "VisualBuildError",
    "VisualBuildStats",
    "VisualIndexBuilder",
]
