# Chinese conversation text intentionally keeps its full-width punctuation.
# ruff: noqa: RUF001

from __future__ import annotations

import json

import pytest

from knowledge_scope.chat.memory import (
    MemoryExtractionService,
    filter_new_memories,
    parse_extracted_memories,
)
from knowledge_scope.llm.schemas import LLMRequest, LLMResult
from knowledge_scope.shared.config import Settings


class _Gateway:
    def __init__(self, text: str = "", error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResult:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return LLMResult(text=self.text, provider="fake", model="fake-model", latency_ms=1.0)


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, environment="test", **overrides)  # type: ignore[arg-type]


def _service(gateway: _Gateway, **overrides: object) -> MemoryExtractionService:
    return MemoryExtractionService(gateway, _settings(**overrides))  # type: ignore[arg-type]


def test_parse_extracts_normalized_sentences_from_json() -> None:
    text = '前缀 ["用户在准备中考化学复习", "  偏好要点式回答  ", "重复", "重复"] 后缀'

    memories = parse_extracted_memories(text)

    # One copy of a repeated memory survives; the duplicate does not.
    assert memories == ("用户在准备中考化学复习", "偏好要点式回答", "重复")


def test_parse_rejects_non_json_and_oversized_output() -> None:
    assert parse_extracted_memories("没有数组") == ()
    assert parse_extracted_memories('{"不是数组": true}') == ()
    assert parse_extracted_memories("[ broken") == ()


def test_filter_drops_duplicates_and_restatements() -> None:
    kept = filter_new_memories(
        ("用户在准备中考化学复习", "全新的记忆", "用户在准备中考化学复习"),
        ("用户在准备 中考化学复习。",),
    )

    assert kept == ("全新的记忆",)


@pytest.mark.anyio
async def test_extraction_requires_two_messages_and_an_enabled_flag() -> None:
    gateway = _Gateway(text='["用户在准备中考化学复习"]')

    assert await _service(gateway).extract((("user", "第一问"),), ()) == ()
    assert gateway.requests == []
    assert (
        await _service(gateway, chat_memory_enabled=False).extract(
            (("user", "第一问"), ("assistant", "第一答")),
            (),
        )
        == ()
    )
    assert gateway.requests == []


@pytest.mark.anyio
async def test_extraction_returns_deduplicated_memories() -> None:
    payload = json.dumps(
        ["用户在准备中考化学复习", "用户在准备中考化学复习", "偏好要点式回答"],
        ensure_ascii=False,
    )
    gateway = _Gateway(text=payload)

    memories = await _service(gateway).extract(
        (("user", "我想复习化学"), ("assistant", "好的，我们开始")),
        ("用户在准备中考化学复习。",),
    )

    assert memories == ("偏好要点式回答",)
    request = gateway.requests[0]
    assert request.task_type == "agent"
    # Non-streaming calls to reasoning models must disable thinking.
    assert request.reasoning == "disabled"
    assert "用户：我想复习化学" in request.messages[1].content


@pytest.mark.anyio
async def test_extraction_never_breaks_the_conversation() -> None:
    gateway = _Gateway(error=RuntimeError("provider down"))

    assert await _service(gateway).extract((("user", "问"), ("assistant", "答")), ()) == ()
