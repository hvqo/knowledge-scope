# Chinese questions intentionally keep their full-width punctuation.
# ruff: noqa: RUF001

from __future__ import annotations

from collections.abc import Sequence

import pytest

from knowledge_scope.llm.schemas import LLMRequest, LLMResult, LLMStreamEvent
from knowledge_scope.rag.rewrite import (
    REWRITE_PROMPT_VERSION,
    QueryRewriteService,
    looks_like_follow_up,
)
from knowledge_scope.rag.schemas import RAGHistoryTurn
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
        return LLMResult(
            text=self.text,
            provider="fake",
            model="fake-model",
            latency_ms=3.0,
        )

    def stream(self, request: LLMRequest) -> Sequence[LLMStreamEvent]:  # pragma: no cover
        raise AssertionError("rewriting must never stream")


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, environment="test", **overrides)  # type: ignore[arg-type]


def _history() -> tuple[RAGHistoryTurn, ...]:
    return (
        RAGHistoryTurn(role="user", content="二次沉淀池的作用是什么？"),
        RAGHistoryTurn(role="assistant", content="它负责泥水分离 [C1]"),
    )


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("它有哪些限制？", True),
        ("为什么？", True),
        ("再展开讲讲", True),
        ("这些数据说明了什么", True),
        ("上述流程的第一步是什么", True),
        ("曝气池的溶解氧标准是多少", False),
        ("水土保持方案里的监测频次是多少？", False),
    ],
)
def test_follow_up_detection(query: str, expected: bool) -> None:
    assert looks_like_follow_up(query, has_history=True) is expected


def test_first_question_is_never_a_follow_up() -> None:
    assert looks_like_follow_up("它有哪些限制？", has_history=False) is False
    assert looks_like_follow_up("", has_history=True) is False


@pytest.mark.anyio
async def test_rewrite_returns_the_standalone_question() -> None:
    gateway = _Gateway(text="  二次沉淀池有哪些限制？  ")
    service = QueryRewriteService(gateway, _settings())  # type: ignore[arg-type]

    rewritten = await service.rewrite("它有哪些限制？", _history())

    assert rewritten == "二次沉淀池有哪些限制？"
    request = gateway.requests[0]
    assert request.task_type == "rag_answer"
    # Reasoning models return empty text unless thinking is disabled.
    assert request.reasoning == "disabled"
    assert request.max_tokens == 160
    assert request.temperature == 0
    contents = [message.content for message in request.messages]
    assert REWRITE_PROMPT_VERSION in contents[0]
    assert "它负责泥水分离" in contents[2]
    # Citation markers from earlier turns must not be reusable search text.
    assert "[C1]" not in contents[2]
    assert "二次沉淀池的作用是什么" in contents[1]


@pytest.mark.anyio
async def test_standalone_questions_skip_the_rewrite_call() -> None:
    gateway = _Gateway(text="改写结果")
    service = QueryRewriteService(gateway, _settings())  # type: ignore[arg-type]

    assert await service.rewrite("曝气池的溶解氧标准是多少", _history()) is None
    assert gateway.requests == []


@pytest.mark.anyio
async def test_rewrite_can_be_disabled() -> None:
    gateway = _Gateway(text="改写结果")
    service = QueryRewriteService(  # type: ignore[arg-type]
        gateway,
        _settings(rag_followup_rewrite_enabled=False),
    )

    assert await service.rewrite("它有哪些限制？", _history()) is None
    assert gateway.requests == []


@pytest.mark.anyio
async def test_rewrite_falls_back_when_the_model_fails() -> None:
    gateway = _Gateway(error=RuntimeError("provider down"))
    service = QueryRewriteService(gateway, _settings())  # type: ignore[arg-type]

    assert await service.rewrite("它有哪些限制？", _history()) is None


@pytest.mark.anyio
async def test_rewrite_is_ignored_when_it_changes_nothing_useful() -> None:
    for text in ("", "   ", "它有哪些限制？", "长" * 400):
        gateway = _Gateway(text=text)
        service = QueryRewriteService(gateway, _settings())  # type: ignore[arg-type]

        assert await service.rewrite("它有哪些限制？", _history()) is None
