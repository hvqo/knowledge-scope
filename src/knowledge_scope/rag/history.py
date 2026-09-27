# Chinese prompt text intentionally uses Chinese punctuation.
# ruff: noqa: RUF001

"""History windowing and rolling compression for multi-turn questions.

Long conversations must not be cut off mid-topic: the newest turns stay
verbatim, and everything older is compressed into a short rolling summary that
rides along with the prompt.  The summary is cached by the content it was
compressed from, so follow-ups that leave the old turns untouched skip the
compression call entirely.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Final

from knowledge_scope.llm.gateway import LLMGateway
from knowledge_scope.llm.schemas import LLMMessage, LLMRequest
from knowledge_scope.shared.config import Settings

from .cache import RetrievalCache
from .schemas import RAG_HISTORY_TURN_MAX_CHARACTERS, RAGHistoryTurn

logger = logging.getLogger(__name__)

HISTORY_SUMMARY_PROMPT_VERSION: Final = "chat-history-summary-v1"
HISTORY_SUMMARY_CACHE_VERSION: Final = "v1"
HISTORY_SUMMARY_MAX_CHARACTERS: Final = 800
HISTORY_SUMMARY_MAX_TOKENS: Final = 400

_SUMMARY_SYSTEM_PROMPT = (
    "你负责把一段较早的对话压缩成要点摘要，供后续对话作为背景使用。\n"
    "规则：\n"
    "1. 保留用户的目标、已确认的事实、做出的决定、偏好和未解决的问题；"
    "寒暄、重复内容和与任务无关的细节都不要。\n"
    "2. 只输出摘要本身，用中文短句，可用“-”开头分条，总长不超过 400 字。\n"
    "3. 不要添加资料里没有的推断，也不要评论这段对话。\n"
)


def _render_turn(role: str, content: str) -> str:
    speaker = "用户" if role == "user" else "助手"
    collapsed = " ".join(content.split())
    return f"{speaker}：{collapsed[:RAG_HISTORY_TURN_MAX_CHARACTERS]}"


def _summary_cache_key(older: tuple[RAGHistoryTurn, ...]) -> str:
    payload = "\n".join(f"{turn.role}:{turn.content}" for turn in older)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"chat:summary:{HISTORY_SUMMARY_CACHE_VERSION}:{digest}"


@dataclass(frozen=True, slots=True)
class HistoryPlan:
    """How the conversation history will be presented to the model."""

    summary: str | None
    recent_turns: tuple[RAGHistoryTurn, ...]
    compressed_turns: int
    cache_hit: bool


class HistoryCompressor:
    """Split the history window and compress the older turns once per content."""

    def __init__(
        self,
        gateway: LLMGateway,
        settings: Settings,
        *,
        cache: RetrievalCache | None = None,
    ) -> None:
        self._gateway = gateway
        self._settings = settings
        self._cache = cache

    def split(
        self,
        history: tuple[RAGHistoryTurn, ...],
    ) -> tuple[tuple[RAGHistoryTurn, ...], tuple[RAGHistoryTurn, ...]]:
        """Return the turns to compress and the turns to keep verbatim."""

        limit = self._settings.rag_history_recent_turns
        if len(history) <= limit:
            return (), history
        boundary = len(history) - limit
        older = history[:boundary]
        recent = history[boundary:]
        # Keep the verbatim window inside its own character budget by dropping
        # the oldest recent turns first.
        total = sum(len(turn.content) for turn in recent)
        while recent and total > self._settings.rag_history_recent_chars:
            total -= len(recent[0].content)
            older = (*older, recent[0])
            recent = recent[1:]
        return older, recent

    async def summarize(
        self,
        older: tuple[RAGHistoryTurn, ...],
    ) -> tuple[str | None, bool]:
        """Return the summary for the older turns and whether it came from cache."""

        if not older:
            return None, False
        key = _summary_cache_key(older)
        if self._cache is not None:
            try:
                cached = self._cache.get(key)
            except Exception as error:  # pragma: no cover - defensive
                logger.debug("history summary cache read failed (%s)", type(error).__name__)
                cached = None
            if cached is not None:
                return (cached or None), True
        rendered = "\n".join(_render_turn(turn.role, turn.content) for turn in older)
        try:
            result = await self._gateway.complete(
                LLMRequest(
                    messages=[
                        LLMMessage(
                            role="system",
                            content=f"提示版本：{HISTORY_SUMMARY_PROMPT_VERSION}\n{_SUMMARY_SYSTEM_PROMPT}",
                        ),
                        LLMMessage(role="user", content=rendered),
                    ],
                    task_type="rag_answer",
                    temperature=0,
                    max_tokens=HISTORY_SUMMARY_MAX_TOKENS,
                    reasoning="disabled",
                )
            )
        except Exception as error:
            logger.debug("history compression skipped (%s)", type(error).__name__)
            return None, False
        summary = " ".join(result.text.split())
        if not summary or len(summary) > HISTORY_SUMMARY_MAX_CHARACTERS:
            return None, False
        if self._cache is not None:
            try:
                self._cache.set(
                    key,
                    summary,
                    self._settings.rag_history_summary_ttl_seconds,
                )
            except Exception as error:  # pragma: no cover - defensive
                logger.debug("history summary cache write failed (%s)", type(error).__name__)
        return summary, False

    async def prepare(self, history: tuple[RAGHistoryTurn, ...]) -> HistoryPlan:
        """Return the plan for presenting one conversation history."""

        older, recent = self.split(history)
        summary, cache_hit = await self.summarize(older)
        return HistoryPlan(
            summary=summary,
            recent_turns=recent,
            compressed_turns=len(older),
            cache_hit=cache_hit,
        )


__all__ = [
    "HISTORY_SUMMARY_MAX_CHARACTERS",
    "HISTORY_SUMMARY_PROMPT_VERSION",
    "HistoryCompressor",
    "HistoryPlan",
]
