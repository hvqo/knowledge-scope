"""Isolated vision understanding adapter and evidence-scoped analysis service."""

from __future__ import annotations

from .errors import (
    VisionConfigurationError,
    VisionConnectionError,
    VisionError,
    VisionImageTooLargeError,
    VisionProviderError,
    VisionRefusalError,
    VisionRequestError,
    VisionResponseFormatError,
    VisionTimeoutError,
)
from .provider import (
    VLM_PROVIDER_NAME,
    OpenAICompatibleVLMProvider,
    VLMProvider,
    VLMResult,
    create_vlm_provider,
)
from .schemas import (
    ChartAnalysis,
    ExtractedTable,
    VisionAnalysisRequest,
    VisionAnalysisResult,
    VisionSource,
    VisionStatus,
    VisionTaskType,
)
from .service import VisionAnalysisService, analyze_evidence_image

__all__ = [
    "VLM_PROVIDER_NAME",
    "ChartAnalysis",
    "ExtractedTable",
    "OpenAICompatibleVLMProvider",
    "VLMProvider",
    "VLMResult",
    "VisionAnalysisRequest",
    "VisionAnalysisResult",
    "VisionAnalysisService",
    "VisionConfigurationError",
    "VisionConnectionError",
    "VisionError",
    "VisionImageTooLargeError",
    "VisionProviderError",
    "VisionRefusalError",
    "VisionRequestError",
    "VisionResponseFormatError",
    "VisionSource",
    "VisionStatus",
    "VisionTaskType",
    "VisionTimeoutError",
    "analyze_evidence_image",
    "create_vlm_provider",
]
