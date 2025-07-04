# Chinese conversation text intentionally keeps its full-width punctuation.

from __future__ import annotations

import asyncio

from knowledge_scope.llm.schemas import LLMRequest, LLMResult
from knowledge_scope.rag.cache import MemoryRetrievalCache
from knowledge_scope.rag.history import HistoryCompressor
from knowledge_scope.rag.schemas import RAGHistoryTurn
from knowledge_scope.shared.config import Settings


class _Gateway:
    def __init__(self, summary: str = "", error: Exception | None = None) -> None:
        self.summary = summary
        self.error = error
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResult:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return LLMResult(text=self.summary, provider="fake", model="fake-model", latency_ms=1.0)


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, environment="test", **overrides)  # type: ignore[arg-type]


def _turns(count: int, *, content: str = "内容") -> tuple[RAGHistoryTurn, ...]:
    return tuple(
        RAGHistoryTurn(role="user" if index % 2 == 0 else "assistant", content=f"{content}{index}")
        for index in range(count)
    )


def test_window_keeps_recent_turns_verbatim() -> None:
    compressor = HistoryCompressor(_Gateway(), _settings(rag_history_recent_turns=3))

    older, recent = compressor.split(_turns(5))

    assert len(older) == 2
    assert len(recent) == 3
    assert [turn.content for turn in recent] == ["内容2", "内容3", "内容4"]


def test_window_drops_recent_turns_that_overflow_the_character_budget() -> None:
    compressor = HistoryCompressor(
        _Gateway(),
        _settings(rag_history_recent_turns=4, rag_history_recent_chars=250),
    )

    older, recent = compressor.split(_turns(6, content="很长的一段对话内容"))

    assert older, "oversized recent turns must fall back into the compressed part"
    assert sum(len(turn.content) for turn in recent) <= 250


def test_short_histories_never_touch_the_compressor() -> None:
    gateway = _Gateway()
    compressor = HistoryCompressor(gateway, _settings())

    plan = asyncio.run(compressor.prepare(_turns(3)))

    assert plan.summary is None
    assert plan.cache_hit is False
    assert plan.compressed_turns == 0
    assert gateway.requests == []


def test_compression_is_cached_by_the_compressed_content() -> None:
    gateway = _Gateway(summary="- 用户在准备中考化学复习\n- 已确认使用要点式回答")
    cache = MemoryRetrievalCache(max_entries=8)
    compressor = HistoryCompressor(gateway, _settings(rag_history_recent_turns=2), cache=cache)
    history = _turns(6)

    first = asyncio.run(compressor.prepare(history))
    second = asyncio.run(compressor.prepare(history))

    expected = "- 用户在准备中考化学复习\n- 已确认使用要点式回答".replace("\n", " ")
    assert first.summary == expected and first.cache_hit is False
    assert second.summary == expected and second.cache_hit is True
    # One compression call covers every follow-up until the old turns change.
    assert len(gateway.requests) == 1


def test_compression_failure_leaves_the_history_untouched() -> None:
    gateway = _Gateway(error=RuntimeError("provider down"))
    compressor = HistoryCompressor(gateway, _settings(rag_history_recent_turns=2))

    plan = asyncio.run(compressor.prepare(_turns(5)))

    assert plan.summary is None
    assert plan.cache_hit is False
    assert plan.compressed_turns == 3


def test_off_topic_compression_output_is_dropped() -> None:
    gateway = _Gateway(summary="")
    compressor = HistoryCompressor(gateway, _settings(rag_history_recent_turns=2))

    plan = asyncio.run(compressor.prepare(_turns(4)))

    assert plan.summary is None
