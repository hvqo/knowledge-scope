# ruff: noqa: RUF001
"""OpenAI-compatible vision provider adapter, independent from the text LLM gateway."""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from ..shared.config import Settings
from .errors import (
    VisionConfigurationError,
    VisionConnectionError,
    VisionProviderError,
    VisionRefusalError,
    VisionResponseFormatError,
    VisionTimeoutError,
)
from .prompts import build_prompt
from .schemas import VisionTaskType

VLM_PROVIDER_NAME = "openai-compatible"


@dataclass(frozen=True, slots=True)
class VLMResult:
    """One raw vision model answer plus the metadata worth recording."""

    content: str
    provider: str
    model: str
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class VLMProvider(Protocol):
    """Minimal async vision interface so tests and future providers can swap in."""

    provider_name: str
    model: str

    async def complete(
        self,
        *,
        image_bytes: bytes,
        media_type: str,
        task_type: VisionTaskType,
        question: str | None = None,
    ) -> VLMResult: ...

    async def aclose(self) -> None: ...


class OpenAICompatibleVLMProvider:
    """Call an OpenAI-compatible chat/completions endpoint with one image.

    ``trust_env`` is disabled on purpose: local vision endpoints are reached
    directly and must not inherit host proxy variables. API keys, image bytes and
    prompts are never logged here.
    """

    provider_name = VLM_PROVIDER_NAME

    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        base_url = settings.vlm_base_url.strip()
        model = settings.vlm_model.strip()
        if not base_url or not model:
            raise VisionConfigurationError("视觉模型地址或模型名称未配置")
        self._settings = settings
        self._client = client
        self._owns_client = client is None
        self.model = model
        self._endpoint = base_url.rstrip("/") + "/"

    def _http_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._endpoint,
                timeout=httpx.Timeout(self._settings.vlm_timeout_seconds),
                trust_env=False,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    def _headers(self) -> dict[str, str]:
        api_key = self._settings.vlm_api_key
        if api_key is None:
            return {}
        return {"Authorization": f"Bearer {api_key.get_secret_value()}"}

    @staticmethod
    def _image_content(*, image_bytes: bytes, media_type: str) -> dict[str, Any]:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{encoded}"}}

    @staticmethod
    def _message_text(message: dict[str, Any]) -> str:
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "\n".join(parts).strip()
        return ""

    async def complete(
        self,
        *,
        image_bytes: bytes,
        media_type: str,
        task_type: VisionTaskType,
        question: str | None = None,
    ) -> VLMResult:
        client = self._http_client()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": build_prompt(task_type, question)},
                        self._image_content(image_bytes=image_bytes, media_type=media_type),
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": self._settings.vlm_max_output_tokens,
            "stream": False,
        }
        reasoning_effort = self._settings.vlm_reasoning_effort.strip()
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        started = time.perf_counter()
        try:
            response = await client.post(
                "chat/completions",
                json=payload,
                headers=self._headers(),
            )
        except httpx.TimeoutException as error:
            raise VisionTimeoutError("视觉模型调用超时") from error
        except httpx.HTTPError as error:
            raise VisionConnectionError("无法连接视觉模型服务") from error
        latency_ms = round((time.perf_counter() - started) * 1_000, 2)

        if response.status_code >= 400:
            retryable = response.status_code == 429 or response.status_code >= 500
            raise VisionProviderError(
                f"视觉模型返回 HTTP {response.status_code}",
                status_code=response.status_code,
                retryable=retryable,
            )
        try:
            data = response.json()
        except ValueError as error:
            raise VisionResponseFormatError("视觉模型返回的不是有效 JSON") from error
        if not isinstance(data, dict):
            raise VisionResponseFormatError("视觉模型返回结构无效")
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise VisionResponseFormatError("视觉模型未返回候选结果")
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict):
            raise VisionResponseFormatError("视觉模型未返回消息内容")
        if choice.get("finish_reason") == "content_filter" or message.get("refusal"):
            raise VisionRefusalError("视觉模型拒绝分析该图片")
        content = self._message_text(message)
        if not content:
            reasoning = message.get("reasoning")
            if (
                isinstance(reasoning, str)
                and reasoning.strip()
                and choice.get("finish_reason") == "length"
            ):
                raise VisionResponseFormatError(
                    "视觉模型输出被截断：仅返回了推理内容，请设置推理强度为 none/low 或提高输出上限"
                )
            raise VisionResponseFormatError("视觉模型返回了空内容")
        usage = data.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        return VLMResult(
            content=content,
            provider=self.provider_name,
            model=str(data.get("model") or self.model),
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
            completion_tokens=completion_tokens if isinstance(completion_tokens, int) else None,
        )


def create_vlm_provider(settings: Settings) -> VLMProvider | None:
    """Create the configured vision provider, or ``None`` when it is disabled."""
    if not settings.vlm_enabled:
        return None
    return OpenAICompatibleVLMProvider(settings)


__all__ = [
    "VLM_PROVIDER_NAME",
    "OpenAICompatibleVLMProvider",
    "VLMProvider",
    "VLMResult",
    "create_vlm_provider",
]
