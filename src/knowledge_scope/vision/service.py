# ruff: noqa: RUF001
"""Vision analysis orchestration: evidence-safe image load, provider call, normalization."""

from __future__ import annotations

from knowledge_scope.documents.assets import (
    get_evidence,
    read_evidence_asset_bytes,
    resolve_evidence_asset,
)
from knowledge_scope.documents.models import Document
from knowledge_scope.shared.config import Settings

from .errors import (
    VisionError,
    VisionImageTooLargeError,
    VisionRequestError,
)
from .provider import VLMProvider
from .schemas import (
    VisionAnalysisResult,
    VisionSource,
    VisionTaskType,
    normalize_chart,
    normalize_description,
    normalize_table,
)

_IMAGE_TASKS = frozenset({"describe_image", "analyze_chart"})
_SUPPORTED_IMAGE_MODALITIES = frozenset({"image", "table"})


class VisionAnalysisService:
    """Run one explicitly requested vision task against one evidence asset."""

    def __init__(self, settings: Settings, provider: VLMProvider | None) -> None:
        self._settings = settings
        self._provider = provider

    async def analyze(
        self,
        document: Document,
        *,
        evidence_id: str,
        asset_index: int,
        task_type: VisionTaskType,
        question: str | None = None,
    ) -> VisionAnalysisResult:
        snapshot = get_evidence(self._settings, document, evidence_id)
        evidence = snapshot.evidence
        self._validate_modality(task_type, evidence.modality)
        asset = resolve_evidence_asset(self._settings, document, evidence_id, asset_index)
        if asset.size_bytes > self._settings.vlm_max_image_bytes:
            raise VisionImageTooLargeError(
                f"素材大小超过视觉模型输入上限（{self._settings.vlm_max_image_bytes} 字节）"
            )
        if not asset.media_type.startswith("image/"):
            raise VisionRequestError("该证据素材不是受支持的图片格式")
        source = VisionSource(
            knowledge_base_id=evidence.lineage.knowledge_base_id,
            document_id=evidence.lineage.document_id,
            evidence_id=evidence.evidence_id,
            page_start=evidence.lineage.page_start,
            page_end=evidence.lineage.page_end,
            source_block_ids=list(evidence.lineage.source_block_ids),
        )
        if self._provider is None:
            return VisionAnalysisResult(
                status="not_configured",
                task_type=task_type,
                sources=[source],
                error_code="vlm_not_configured",
                message="视觉模型未启用或未配置，无法执行图片理解",
            )

        image_bytes = read_evidence_asset_bytes(asset)
        try:
            raw = await self._provider.complete(
                image_bytes=image_bytes,
                media_type=asset.media_type,
                task_type=task_type,
                question=question,
            )
        except VisionError as error:
            return self._failure(
                task_type=task_type,
                code=error.code,
                message=error.message,
                source=source,
            )

        try:
            return self._normalize(
                task_type=task_type,
                content=raw.content,
                source=source,
                provider=raw.provider,
                model=raw.model,
                latency_ms=raw.latency_ms,
            )
        except VisionError as error:
            return self._failure(
                task_type=task_type,
                code=error.code,
                message=error.message,
                source=source,
                provider=raw.provider,
                model=raw.model,
                latency_ms=raw.latency_ms,
            )

    @staticmethod
    def _validate_modality(task_type: VisionTaskType, modality: str) -> None:
        if task_type in _IMAGE_TASKS and modality != "image":
            raise VisionRequestError("该任务仅支持图片证据")
        if task_type == "extract_table" and modality not in _SUPPORTED_IMAGE_MODALITIES:
            raise VisionRequestError("该任务仅支持图片或表格证据")

    def _normalize(
        self,
        *,
        task_type: VisionTaskType,
        content: str,
        source: VisionSource,
        provider: str,
        model: str,
        latency_ms: float,
    ) -> VisionAnalysisResult:
        if task_type == "describe_image":
            answer, warnings = normalize_description(content)
            return VisionAnalysisResult(
                status="needs_review" if warnings else "ok",
                task_type=task_type,
                answer=answer,
                provider=provider,
                model=model,
                latency_ms=latency_ms,
                warnings=warnings,
                sources=[source],
            )
        if task_type == "analyze_chart":
            answer, chart, warnings = normalize_chart(content)
            return VisionAnalysisResult(
                status="needs_review" if warnings else "ok",
                task_type=task_type,
                answer=answer,
                chart_analysis=chart,
                provider=provider,
                model=model,
                latency_ms=latency_ms,
                warnings=warnings,
                sources=[source],
            )
        table, warnings = normalize_table(content)
        return VisionAnalysisResult(
            status="needs_review",
            task_type=task_type,
            extracted_table=table,
            extraction_method="vlm",
            provider=provider,
            model=model,
            latency_ms=latency_ms,
            warnings=warnings,
            sources=[source],
        )

    @staticmethod
    def _failure(
        *,
        task_type: VisionTaskType,
        code: str,
        message: str,
        source: VisionSource,
        provider: str | None = None,
        model: str | None = None,
        latency_ms: float | None = None,
    ) -> VisionAnalysisResult:
        return VisionAnalysisResult(
            status="failed",
            task_type=task_type,
            provider=provider,
            model=model,
            latency_ms=latency_ms,
            sources=[source],
            error_code=code,
            message=message,
        )


async def analyze_evidence_image(
    settings: Settings,
    provider: VLMProvider | None,
    document: Document,
    *,
    evidence_id: str,
    asset_index: int,
    task_type: VisionTaskType,
    question: str | None = None,
) -> VisionAnalysisResult:
    """Convenience entry point for callers that do not need to keep a service instance."""
    return await VisionAnalysisService(settings, provider).analyze(
        document,
        evidence_id=evidence_id,
        asset_index=asset_index,
        task_type=task_type,
        question=question,
    )


__all__ = ["VisionAnalysisService", "analyze_evidence_image"]
