"""Controlled error categories for the isolated vision integration."""

from __future__ import annotations


class VisionError(RuntimeError):
    """Base class for controlled vision failures with a stable error code."""

    code: str = "vision_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class VisionConfigurationError(VisionError):
    """The vision provider is disabled or incompletely configured."""

    code = "vlm_not_configured"


class VisionRequestError(VisionError):
    """The requested vision task is not valid for the resolved evidence asset."""

    code = "invalid_request"


class VisionImageTooLargeError(VisionRequestError):
    """The evidence image exceeds the configured vision input size limit."""

    code = "image_too_large"


class VisionProviderError(VisionError):
    """The provider returned a non-success HTTP status or an unusable payload."""

    code = "provider_error"

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class VisionTimeoutError(VisionProviderError):
    """The provider did not answer within the configured timeout."""

    code = "provider_timeout"


class VisionConnectionError(VisionProviderError):
    """The provider endpoint could not be reached."""

    code = "provider_connection"


class VisionResponseFormatError(VisionError):
    """The provider answered but the content is not the expected JSON contract."""

    code = "invalid_response"


class VisionRefusalError(VisionError):
    """The model refused to analyse the image."""

    code = "model_refusal"


__all__ = [
    "VisionConfigurationError",
    "VisionConnectionError",
    "VisionError",
    "VisionImageTooLargeError",
    "VisionProviderError",
    "VisionRefusalError",
    "VisionRequestError",
    "VisionResponseFormatError",
    "VisionTimeoutError",
]
