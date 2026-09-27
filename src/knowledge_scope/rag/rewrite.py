# Chinese prompt text intentionally uses Chinese punctuation.
# ruff: noqa: RUF001, RUF002

"""Rewrite follow-up questions into standalone retrieval queries.

A question like "它有哪些限制？" has no usable search terms on its own; the
retrieved material is then unrelated and the answer degrades to "资料不足".
Rewriting costs one short non-streaming model call and only runs when the
question actually looks like a follow-up, so standalone questions keep their
current latency.
"""

from __future__ import annotations

import logging
from typing import Final

from knowledge_scope.llm.gateway import LLMGateway
from knowledge_scope.llm.schemas import LLMMessage, LLMRequest
from knowledge_scope.shared.config import Settings

from .prompt import normalize_history_content
from .routing import strip_query_filler
from .schemas import RAG_HISTORY_MAX_CHARACTERS, RAGHistoryTurn

logger = logging.getLogger(__name__)

REWRITE_PROMPT_VERSION: Final = "rag-rewrite-v1"
REWRITE_QUERY_MAX_CHARACTERS: Final = 300
REWRITE_MAX_TOKENS: Final = 160
# A short question, or one that points back at the previous turn.
FOLLOW_UP_MAX_CHARACTERS: Final = 8
_FOLLOW_UP_MARKERS: Final[tuple[str, ...]] = (
    "它",
    "他",
    "她",
    "这个",
    "那个",
    "这些",
    "那些",
    "上述",
    "上面",
    "刚才",
    "之前",
    "前面",
    "上一条",
    "继续",
    "接着",
    "再",
    "进一步",
    "展开",
    "详细",
    "具体",
    "为什么",
    "呢",
    "this",
    "that",
    "these",
    "those",
    "it",
    "they",
    "above",
    "previous",
    "more",
    "why",
)

_REWRITE_SYSTEM_PROMPT = (
    "你是检索查询改写器。把用户的最新问题改写成一个可以独立检索的完整问题。\n"
    "规则：\n"
    "1. 只补全最新问题里省略的主语、对象和指代，不要回答它，也不要补充事实。\n"
    "2. 对话里可能有多个话题，优先采用最近一轮用户问题的主题。\n"
    "3. 保留原问题的语言与提问意图；如果原问题已经可以独立检索，原样输出。\n"
    "4. 只输出改写后的问题本身，不要引号、编号、解释或多余文字。\n"
)


def looks_like_follow_up(query: str, *, has_history: bool) -> bool:
    """Return whether the question depends on the previous turns."""

    if not has_history:
        return False
    normalized = strip_query_filler(query)
    if not normalized:
        return False
    if len(normalized) <= FOLLOW_UP_MAX_CHARACTERS:
        return True
    return any(marker in normalized for marker in _FOLLOW_UP_MARKERS)


class QueryRewriteService:
    """Small wrapper around the LLM gateway for follow-up rewriting."""

    def __init__(self, gateway: LLMGateway, settings: Settings) -> None:
        self._gateway = gateway
        self._settings = settings

    @property
    def enabled(self) -> bool:
        return self._settings.rag_followup_rewrite_enabled

    async def rewrite(
        self,
        query: str,
        history: tuple[RAGHistoryTurn, ...],
    ) -> str | None:
        """Return a standalone query, or ``None`` to keep the original one."""

        if not self.enabled or not looks_like_follow_up(query, has_history=bool(history)):
            return None
        messages: list[LLMMessage] = [
            LLMMessage(
                role="system",
                content=f"提示版本：{REWRITE_PROMPT_VERSION}\n{_REWRITE_SYSTEM_PROMPT}",
            )
        ]
        budget = RAG_HISTORY_MAX_CHARACTERS
        for turn in history:
            content = normalize_history_content(turn.content)
            if not content:
                continue
            messages.append(LLMMessage(role=turn.role, content=content[:budget]))
            budget -= min(len(content), budget)
            if budget <= 0:
                break
        messages.append(LLMMessage(role="user", content=f"最新问题：{query}\n\n改写后的检索问题："))
        try:
            result = await self._gateway.complete(
                LLMRequest(
                    messages=messages,
                    # Rewriting is a bounded sub-step of the RAG flow, so it
                    # reuses the RAG task routing instead of inventing a task.
                    task_type="rag_answer",
                    temperature=0,
                    max_tokens=REWRITE_MAX_TOKENS,
                    # Non-streaming calls to reasoning models must disable
                    # thinking, otherwise the whole budget is spent on
                    # reasoning and the rewritten text comes back empty.
                    reasoning="disabled",
                )
            )
        except Exception as error:
            # Rewriting is an optimization: any failure falls back to the
            # original question instead of failing the whole answer.
            logger.debug("query rewrite skipped (%s)", type(error).__name__)
            return None
        rewritten = " ".join(result.text.split()).strip("\"'“”")
        if not rewritten or len(rewritten) > REWRITE_QUERY_MAX_CHARACTERS:
            return None
        if strip_query_filler(rewritten) == strip_query_filler(query):
            return None
        return rewritten


__all__ = [
    "FOLLOW_UP_MAX_CHARACTERS",
    "REWRITE_PROMPT_VERSION",
    "REWRITE_QUERY_MAX_CHARACTERS",
    "QueryRewriteService",
    "looks_like_follow_up",
]
