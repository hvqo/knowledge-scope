# Chinese prompt text intentionally uses Chinese punctuation.
# ruff: noqa: RUF001

"""Long-term memory: extraction, deduplication, and prompt injection.

Memories are durable, reusable facts about the person using the workspace —
goals, standing instructions, preferences — not facts copied out of documents.
They are scoped to one knowledge base and injected into later questions so the
assistant keeps context across conversations.
"""

from __future__ import annotations

import json
import logging
from typing import Final

from knowledge_scope.llm.gateway import LLMGateway
from knowledge_scope.llm.schemas import LLMMessage, LLMRequest
from knowledge_scope.shared.config import Settings

logger = logging.getLogger(__name__)

MEMORY_PROMPT_VERSION: Final = "chat-memory-v1"
MEMORY_EXTRACTION_MAX_TOKENS: Final = 400
MEMORY_ITEM_MAX_CHARACTERS: Final = 200
MEMORY_MAX_ITEMS_PER_EXTRACTION: Final = 5

_EXTRACTION_SYSTEM_PROMPT = (
    "你负责维护一个长期记忆库。从对话里找出值得长期记住、且以后还会有用的信息，"
    "例如用户的目标、背景、偏好、固定要求或持续进行的任务。\n"
    "规则：\n"
    "1. 只记录与用户本人或其任务有关的信息；文档里的知识点、答案内容、临时细节都不要记。\n"
    "2. 每条记忆写成一句独立的中文短句（不超过 120 字），不带编号、引号或解释。\n"
    "3. 最多输出 5 条；没有值得记的就输出空数组。\n"
    '4. 只输出一个 JSON 数组，例如 ["用户在准备中考化学复习"]，不要输出其他文字。\n'
)


def _normalize_memory(value: str) -> str:
    return " ".join(value.split())


def _memory_key(value: str) -> str:
    """Return a comparison key tolerant of punctuation and case."""

    normalized = _normalize_memory(value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def parse_extracted_memories(text: str) -> tuple[str, ...]:
    """Parse the model output into normalized memory strings."""

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end <= start:
        return ()
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return ()
    if not isinstance(parsed, list):
        return ()
    memories: list[str] = []
    seen: set[str] = set()
    for item in parsed:
        if not isinstance(item, str):
            continue
        normalized = _normalize_memory(item)
        key = _memory_key(normalized)
        if not key or key in seen:
            continue
        seen.add(key)
        memories.append(normalized[:MEMORY_ITEM_MAX_CHARACTERS])
        if len(memories) >= MEMORY_MAX_ITEMS_PER_EXTRACTION:
            break
    return tuple(memories)


def filter_new_memories(
    extracted: tuple[str, ...],
    existing: tuple[str, ...],
) -> tuple[str, ...]:
    """Drop memories that duplicate or repeat something already remembered."""

    existing_keys = {_memory_key(value) for value in existing}
    kept: list[str] = []
    kept_keys: set[str] = set()
    for value in extracted:
        key = _memory_key(value)
        if not key or key in existing_keys or key in kept_keys:
            continue
        # A memory that merely restates a longer existing one adds noise.
        if any(key in _memory_key(item) or _memory_key(item) in key for item in existing):
            continue
        kept_keys.add(key)
        kept.append(value)
    return tuple(kept)


class MemoryExtractionService:
    """Extract durable memories from one finished conversation."""

    def __init__(self, gateway: LLMGateway, settings: Settings) -> None:
        self._gateway = gateway
        self._settings = settings

    @property
    def enabled(self) -> bool:
        return self._settings.chat_memory_enabled

    async def extract(
        self,
        messages: tuple[tuple[str, str], ...],
        existing: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Return new memories worth keeping, or an empty tuple.

        Any failure is swallowed: memory extraction must never break the
        conversation it observes.
        """

        if not self.enabled or len(messages) < 2:
            return ()
        rendered: list[str] = []
        budget = self._settings.chat_memory_extract_chars
        for role, content in messages:
            line = f"{'用户' if role == 'user' else '助手'}：{' '.join(content.split())}"
            rendered.append(line[:budget])
            budget -= min(len(line), budget)
            if budget <= 0:
                break
        try:
            result = await self._gateway.complete(
                LLMRequest(
                    messages=[
                        LLMMessage(
                            role="system",
                            content=f"提示版本：{MEMORY_PROMPT_VERSION}\n{_EXTRACTION_SYSTEM_PROMPT}",
                        ),
                        LLMMessage(role="user", content="\n".join(rendered)),
                    ],
                    task_type="agent",
                    temperature=0,
                    max_tokens=MEMORY_EXTRACTION_MAX_TOKENS,
                    reasoning="disabled",
                )
            )
        except Exception as error:
            logger.debug("memory extraction skipped (%s)", type(error).__name__)
            return ()
        extracted = parse_extracted_memories(result.text)
        return filter_new_memories(extracted, existing)


__all__ = [
    "MEMORY_ITEM_MAX_CHARACTERS",
    "MEMORY_MAX_ITEMS_PER_EXTRACTION",
    "MEMORY_PROMPT_VERSION",
    "MemoryExtractionService",
    "filter_new_memories",
    "parse_extracted_memories",
]
