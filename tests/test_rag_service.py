# Chinese questions intentionally keep their full-width punctuation.
# ruff: noqa: RUF001

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from uuid import UUID, uuid4

import pytest

from knowledge_scope.llm.errors import LLMProviderError
from knowledge_scope.llm.schemas import LLMRequest, LLMResult, LLMStreamEvent
from knowledge_scope.rag.cache import MemoryRetrievalCache
from knowledge_scope.rag.rewrite import QueryRewriteService
from knowledge_scope.rag.schemas import RAGHistoryTurn, RAGQueryRequest
from knowledge_scope.rag.service import RAG_INSUFFICIENT_EVIDENCE, RAGService
from knowledge_scope.retrieval.qdrant import (
    QDRANT_COLLECTION_SCHEMA_VERSION,
    ChunkVectorPayload,
    RetrievedChunk,
)
from knowledge_scope.retrieval.reranking import RerankedChunk
from knowledge_scope.retrieval.service import RetrievalResult
from knowledge_scope.retrieval.unified import (
    UnifiedBranchContribution,
    UnifiedBranchExecution,
    UnifiedCandidate,
    UnifiedRetrievalResult,
    chunk_candidate_id,
)
from knowledge_scope.shared.config import Settings

KNOWLEDGE_BASE_ID = UUID("44444444-4444-4444-8444-444444444444")
DOCUMENT_ID = UUID("33333333-3333-4333-8333-333333333333")


def _chunk(chunk_id: str, text: str, block_id: str, rank: int) -> RetrievedChunk:
    return RetrievedChunk(
        point_id=uuid4(),
        score=1.0 / rank,
        payload=ChunkVectorPayload(
            collection_schema_version=QDRANT_COLLECTION_SCHEMA_VERSION,
            chunk_id=chunk_id,
            document_id=DOCUMENT_ID,
            page_start=rank,
            page_end=rank,
            source_block_ids=[block_id],
            section_path=["章节"],
            content_types=["text"],
            asset_refs=[],
            text=text,
            chunking_config_fingerprint="a" * 64,
            embedding_model="Qwen/Qwen3-Embedding-0.6B",
            embedding_model_revision="revision",
            embedding_config_fingerprint="b" * 64,
        ),
    )


class _Retrieval:
    def __init__(
        self,
        items: Sequence[RetrievedChunk] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.items = tuple(items or ())
        self.error = error
        self.calls: list[dict[str, object]] = []

    def search(self, query: str, **filters: object) -> RetrievalResult:
        self.calls.append({"query": query, **filters})
        if self.error is not None:
            raise self.error
        return RetrievalResult(
            query=query,
            limit=10,
            model_id="Qwen/Qwen3-Embedding-0.6B",
            collection_name="test_chunks",
            items=self.items,
        )


class _Reranking:
    def __init__(self, items: Sequence[RetrievedChunk] | None = None) -> None:
        self.items = tuple(items) if items is not None else None
        self.calls: list[tuple[str, tuple[RetrievedChunk, ...], int]] = []

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        *,
        limit: int,
    ) -> tuple[RerankedChunk, ...]:
        candidate_tuple = tuple(candidates)
        self.calls.append((query, candidate_tuple, limit))
        ranked_items = self.items if self.items is not None else candidate_tuple
        return tuple(
            RerankedChunk(chunk=item, dense_rank=index, reranker_score=float(-index))
            for index, item in enumerate(ranked_items, start=1)
        )[:limit]


class _Gateway:
    def __init__(
        self,
        events: Sequence[LLMStreamEvent] = (),
        error: BaseException | None = None,
        rewrite: str = "",
    ) -> None:
        self.events = tuple(events)
        self.error = error
        self.rewrite = rewrite
        self.requests: list[LLMRequest] = []
        self.complete_requests: list[LLMRequest] = []
        self.closed = False

    async def complete(self, request: LLMRequest) -> LLMResult:
        self.complete_requests.append(request)
        return LLMResult(
            text=self.rewrite,
            provider="fake",
            model="fake-model",
            latency_ms=2.0,
        )

    async def _stream(self) -> AsyncIterator[LLMStreamEvent]:
        try:
            if self.error is not None:
                raise self.error
            for event in self.events:
                yield event
        finally:
            self.closed = True

    def stream(self, request: LLMRequest) -> AsyncIterator[LLMStreamEvent]:
        self.requests.append(request)
        return self._stream()


class _UnifiedRetrieval:
    """Minimal unified retrieval stub: one chunk per call, in fused order."""

    def __init__(self, texts: Sequence[str] = ("统一检索结果",)) -> None:
        self.texts = tuple(texts)
        self.calls: list[str] = []

    async def search(self, query: str, knowledge_base_id: UUID, **kwargs: object) -> object:
        self.calls.append(query)
        items = tuple(
            UnifiedCandidate(
                candidate_id=chunk_candidate_id(KNOWLEDGE_BASE_ID, DOCUMENT_ID, f"unified-{index}"),
                candidate_kind="chunk",
                source="dense",
                knowledge_base_id=KNOWLEDGE_BASE_ID,
                document_id=DOCUMENT_ID,
                chunk_id=f"unified-{index}",
                page_start=index,
                page_end=index,
                source_block_ids=[f"unified-block-{index}"],
                section_path=["章节"],
                content_types=["text"],
                source_text=text,
                rerank_text=text,
                branches=[UnifiedBranchContribution(branch="dense", rank=index, score=0.9)],
                final_rank=index,
                final_reranker_score=0.9,
            )
            for index, text in enumerate(self.texts, start=1)
        )
        return UnifiedRetrievalResult(
            query=query,
            knowledge_base_id=KNOWLEDGE_BASE_ID,
            config_fingerprint="f" * 64,
            reranker_model_id="BAAI/bge-reranker-v2-m3",
            branches=[
                UnifiedBranchExecution(
                    branch="dense",
                    status="success",
                    candidate_count=len(items),
                    latency_ms=1,
                )
            ],
            candidate_pool_size=len(items),
            normalization_latency_ms=0,
            rerank_latency_ms=0,
            total_latency_ms=2,
            degraded=False,
            items=list(items),
        )


def _service(
    retrieval: _Retrieval,
    reranking: _Reranking,
    gateway: _Gateway,
    *,
    cache: MemoryRetrievalCache | None = None,
    rewrite: QueryRewriteService | None = None,
    unified: _UnifiedRetrieval | None = None,
) -> RAGService:
    settings = Settings(_env_file=None, environment="test", rag_context_budget_chars=100)
    return RAGService(  # type: ignore[arg-type]
        retrieval,
        reranking,
        gateway,
        settings,
        unified_retrieval=unified,
        cache=cache,
        rewrite=rewrite,
    )


async def _events(service: RAGService) -> list[object]:
    return [event async for event in service.stream(RAGQueryRequest(query="问题"))]


@pytest.mark.anyio
async def test_rag_stream_orders_answer_citations_and_completion() -> None:
    retrieval = _Retrieval(
        [
            _chunk("chunk-1", "第一条答案", "block-1", 1),
            _chunk("chunk-2", "第二条答案", "block-2", 2),
        ]
    )
    reranking = _Reranking()
    gateway = _Gateway(
        [
            LLMStreamEvent(delta="答案", provider="fake", model="fake-model"),
            LLMStreamEvent(
                delta=" [C1]",
                provider="fake",
                model="fake-model",
                input_tokens=18,
                output_tokens=3,
                finish_reason="stop",
            ),
        ]
    )

    events = await _events(_service(retrieval, reranking, gateway))

    assert [event.event for event in events] == [
        "answer_delta",
        "answer_delta",
        "citations",
        "complete",
    ]
    assert events[2].data["items"][0]["marker"] == "C1"
    assert events[2].data["items"][0]["document_id"] == str(DOCUMENT_ID)
    assert events[3].data["status"] == "completed"
    assert events[3].data["input_tokens"] == 18
    assert events[3].data["retrieval_latency_ms"] >= 0
    assert events[3].data["llm_latency_ms"] >= 0
    assert events[3].data["latency_ms"] >= events[3].data["llm_latency_ms"]
    assert gateway.requests[0].task_type == "rag_answer"
    assert "[C1]" in gateway.requests[0].messages[1].content
    assert retrieval.calls[0]["knowledge_base_id"] is None


@pytest.mark.anyio
async def test_auto_mode_keeps_small_lookups_on_the_vector_path() -> None:
    retrieval = _Retrieval([_chunk("chunk-1", "答案", "block-1", 1)])
    gateway = _Gateway([LLMStreamEvent(delta="答案", provider="fake", model="fake-model")])
    service = _service(retrieval, _Reranking(), gateway)

    events = [
        event
        async for event in service.stream(
            RAGQueryRequest(query="曝气池的溶解氧标准是多少", retrieval_mode="auto")
        )
    ]

    assert events[-1].data["route"] == "dense"
    assert retrieval.calls[0]["query"] == "曝气池的溶解氧标准是多少"
    assert gateway.complete_requests == []


@pytest.mark.anyio
async def test_repeated_questions_are_served_from_the_retrieval_cache() -> None:
    retrieval = _Retrieval([_chunk("chunk-1", "答案", "block-1", 1)])
    gateway = _Gateway([LLMStreamEvent(delta="答案", provider="fake", model="fake-model")])
    service = _service(
        retrieval,
        _Reranking(),
        gateway,
        cache=MemoryRetrievalCache(max_entries=4),
    )
    request = RAGQueryRequest(query="曝气池的溶解氧标准是多少", retrieval_mode="dense")

    first = [event async for event in service.stream(request)]
    second = [event async for event in service.stream(request)]

    assert first[-1].data["retrieval_cached"] is False
    assert second[-1].data["retrieval_cached"] is True
    assert second[-1].data["route"] == "dense"
    assert len(retrieval.calls) == 1
    assert len(second[1].data["items"]) == 1


@pytest.mark.anyio
async def test_follow_up_questions_retrieve_with_the_rewritten_query() -> None:
    retrieval = _Retrieval([_chunk("chunk-1", "答案", "block-1", 1)])
    gateway = _Gateway(
        [LLMStreamEvent(delta="答案", provider="fake", model="fake-model")],
        rewrite="二次沉淀池有哪些限制",
    )
    service = _service(
        retrieval,
        _Reranking(),
        gateway,
        rewrite=QueryRewriteService(gateway, Settings(_env_file=None, environment="test")),  # type: ignore[arg-type]
    )
    history = (
        RAGHistoryTurn(role="user", content="二次沉淀池的作用是什么？"),
        RAGHistoryTurn(role="assistant", content="它负责泥水分离 [C1]"),
    )

    events = [
        event
        async for event in service.stream(
            RAGQueryRequest(query="它有哪些限制？", retrieval_mode="dense", history=list(history))
        )
    ]

    assert retrieval.calls[0]["query"] == "二次沉淀池有哪些限制"
    assert events[-1].data["rewritten_query"] == "二次沉淀池有哪些限制"
    assert len(gateway.complete_requests) == 1
    # The answer still targets the question the user actually asked.
    assert "它有哪些限制？" in gateway.requests[0].messages[-1].content


@pytest.mark.anyio
async def test_auto_lookup_escalates_when_the_fast_path_misses_the_subject() -> None:
    # The vector path returns material about dissolved oxygen in fish ponds,
    # but nothing about the aeration tank the question asks about.
    retrieval = _Retrieval([_chunk("chunk-1", "养鱼池增氧与气体溶解度实验", "block-1", 1)])
    unified = _UnifiedRetrieval(["曝气池溶解氧控制范围为 2-4 mg/L"])
    gateway = _Gateway([LLMStreamEvent(delta="答案", provider="fake", model="fake-model")])
    service = _service(retrieval, _Reranking(), gateway, unified=unified)

    events = [
        event
        async for event in service.stream(
            RAGQueryRequest(
                query="曝气池的溶解氧控制在多少",
                knowledge_base_id=KNOWLEDGE_BASE_ID,
                retrieval_mode="auto",
            )
        )
    ]

    assert unified.calls == ["曝气池的溶解氧控制在多少"]
    assert events[-1].data["route"] == "unified"
    assert events[-1].data["retrieval_escalated"] is True
    assert "曝气池溶解氧控制范围" in events[1].data["items"][0]["snippet"]


@pytest.mark.anyio
async def test_auto_lookup_keeps_the_fast_path_when_the_context_covers_it() -> None:
    retrieval = _Retrieval([_chunk("chunk-1", "曝气池的溶解氧控制范围为 2-4 mg/L", "block-1", 1)])
    unified = _UnifiedRetrieval()
    gateway = _Gateway([LLMStreamEvent(delta="答案", provider="fake", model="fake-model")])
    service = _service(retrieval, _Reranking(), gateway, unified=unified)

    events = [
        event
        async for event in service.stream(
            RAGQueryRequest(
                query="曝气池的溶解氧控制在多少",
                knowledge_base_id=KNOWLEDGE_BASE_ID,
                retrieval_mode="auto",
            )
        )
    ]

    assert unified.calls == []
    assert events[-1].data["route"] == "dense"
    assert events[-1].data["retrieval_escalated"] is False


@pytest.mark.anyio
async def test_an_explicit_fast_choice_is_never_escalated() -> None:
    retrieval = _Retrieval([_chunk("chunk-1", "养鱼池增氧与气体溶解度实验", "block-1", 1)])
    unified = _UnifiedRetrieval()
    gateway = _Gateway([LLMStreamEvent(delta="答案", provider="fake", model="fake-model")])
    service = _service(retrieval, _Reranking(), gateway, unified=unified)

    events = [
        event
        async for event in service.stream(
            RAGQueryRequest(
                query="曝气池的溶解氧控制在多少",
                knowledge_base_id=KNOWLEDGE_BASE_ID,
                retrieval_mode="dense",
            )
        )
    ]

    assert unified.calls == []
    assert events[-1].data["retrieval_escalated"] is False


@pytest.mark.anyio
async def test_unanswered_questions_are_cached_too() -> None:
    retrieval = _Retrieval([_chunk("image", "", "image-block", 1)])
    service = _service(
        retrieval,
        _Reranking(),
        _Gateway(),
        cache=MemoryRetrievalCache(max_entries=4),
    )
    request = RAGQueryRequest(
        query="曝气池的溶解氧控制在多少",
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        retrieval_mode="auto",
    )

    first = [event async for event in service.stream(request)]
    second = [event async for event in service.stream(request)]

    assert first[0].data["text"] == RAG_INSUFFICIENT_EVIDENCE
    assert first[-1].data["retrieval_cached"] is False
    assert second[-1].data["retrieval_cached"] is True
    assert len(retrieval.calls) == 1


@pytest.mark.anyio
async def test_repeated_follow_ups_reuse_the_rewrite() -> None:
    retrieval = _Retrieval([_chunk("chunk-1", "曝气池溶解氧控制范围 2-4 mg/L", "block-1", 1)])
    gateway = _Gateway(
        [LLMStreamEvent(delta="答案", provider="fake", model="fake-model")],
        rewrite="曝气池溶解氧控制有哪些限制",
    )
    service = _service(
        retrieval,
        _Reranking(),
        gateway,
        cache=MemoryRetrievalCache(max_entries=4),
        rewrite=QueryRewriteService(gateway, Settings(_env_file=None, environment="test")),  # type: ignore[arg-type]
    )
    history = [
        RAGHistoryTurn(role="user", content="曝气池的溶解氧控制在多少"),
        RAGHistoryTurn(role="assistant", content="需要结合工艺要求判断 [C1]"),
    ]

    first = [
        event
        async for event in service.stream(
            RAGQueryRequest(
                query="它有哪些限制？",
                knowledge_base_id=KNOWLEDGE_BASE_ID,
                retrieval_mode="dense",
                history=history,
            )
        )
    ]
    second = [
        event
        async for event in service.stream(
            RAGQueryRequest(
                query="它有哪些限制？",
                knowledge_base_id=KNOWLEDGE_BASE_ID,
                retrieval_mode="dense",
                history=history,
            )
        )
    ]

    assert len(gateway.complete_requests) == 1
    assert first[-1].data["rewrite_latency_ms"] is not None
    assert second[-1].data["rewrite_latency_ms"] == 0
    assert second[-1].data["rewritten_query"] == "曝气池溶解氧控制有哪些限制"


@pytest.mark.anyio
async def test_the_rewritten_question_decides_the_route() -> None:
    retrieval = _Retrieval()
    unified = _UnifiedRetrieval()
    gateway = _Gateway(
        [LLMStreamEvent(delta="答案", provider="fake", model="fake-model")],
        rewrite="曝气池溶解氧控制有哪些限制",
    )
    service = _service(
        retrieval,
        _Reranking(),
        gateway,
        unified=unified,
        rewrite=QueryRewriteService(gateway, Settings(_env_file=None, environment="test")),  # type: ignore[arg-type]
    )

    events = [
        event
        async for event in service.stream(
            RAGQueryRequest(
                query="它呢",
                knowledge_base_id=KNOWLEDGE_BASE_ID,
                retrieval_mode="auto",
                history=[
                    RAGHistoryTurn(role="user", content="曝气池的溶解氧控制在多少"),
                    RAGHistoryTurn(role="assistant", content="2-4 mg/L [C1]"),
                ],
            )
        )
    ]

    # "它呢" alone would look like a lookup; the rewritten question needs the
    # full retrieval path, and the vector path is never even tried.
    assert retrieval.calls == []
    assert unified.calls == ["曝气池溶解氧控制有哪些限制"]
    assert events[-1].data["route"] == "unified"


@pytest.mark.anyio
async def test_rag_returns_controlled_insufficient_evidence_without_llm_call() -> None:
    retrieval = _Retrieval([_chunk("image", "", "image-block", 1)])
    gateway = _Gateway()

    events = await _events(_service(retrieval, _Reranking(), gateway))

    assert [event.event for event in events] == ["answer_delta", "citations", "complete"]
    assert events[0].data["text"] == RAG_INSUFFICIENT_EVIDENCE
    assert events[1].data["items"] == []
    assert events[2].data["status"] == "insufficient_evidence"
    assert gateway.requests == []


@pytest.mark.anyio
async def test_rag_converts_retrieval_failure_to_sse_error_events() -> None:
    service = _service(_Retrieval(error=RuntimeError("private failure")), _Reranking(), _Gateway())

    events = await _events(service)

    assert [event.event for event in events] == ["error", "complete"]
    assert events[0].data == {
        "category": "retrieval",
        "message": "retrieval pipeline failed",
    }
    assert events[1].data["status"] == "error"


@pytest.mark.anyio
async def test_rag_converts_provider_failure_without_exposing_provider_details() -> None:
    retrieval = _Retrieval([_chunk("chunk-1", "答案", "block-1", 1)])
    gateway = _Gateway(error=LLMProviderError("api", "private provider detail"))

    events = await _events(_service(retrieval, _Reranking(), gateway))

    assert [event.event for event in events] == ["error", "complete"]
    assert events[0].data == {
        "category": "api",
        "message": "LLM provider rejected the request",
    }
    assert "private provider detail" not in str(events)


@pytest.mark.anyio
async def test_model_markers_never_change_application_citation_metadata() -> None:
    retrieval = _Retrieval(
        [
            _chunk("chunk-1", "答案一", "block-1", 1),
            _chunk("chunk-2", "答案二", "block-2", 2),
        ]
    )
    gateway = _Gateway(
        [LLMStreamEvent(delta="模型输出 [C999] [C1] [C1]", provider="fake", model="fake")]
    )

    events = await _events(_service(retrieval, _Reranking(), gateway))

    citation_items = events[-2].data["items"]
    assert [item["marker"] for item in citation_items] == ["C1", "C2"]
    assert "C999" not in {item["marker"] for item in citation_items}


@pytest.mark.anyio
async def test_rag_closes_gateway_stream_when_consumer_stops() -> None:
    gateway = _Gateway([LLMStreamEvent(delta="答案", provider="fake", model="fake")])
    stream = _service(
        _Retrieval([_chunk("chunk-1", "答案", "block-1", 1)]),
        _Reranking(),
        gateway,
    ).stream(RAGQueryRequest(query="问题"))

    first = await anext(stream)
    assert first.event == "answer_delta"
    assert gateway.closed is False

    await stream.aclose()

    assert gateway.closed is True


@pytest.mark.anyio
async def test_rag_propagates_cancellation() -> None:
    retrieval = _Retrieval([_chunk("chunk-1", "答案", "block-1", 1)])
    service = _service(
        retrieval,
        _Reranking(),
        _Gateway(error=asyncio.CancelledError()),
    )

    with pytest.raises(asyncio.CancelledError):
        await _events(service)


@pytest.mark.anyio
async def test_direct_route_answers_small_talk_without_any_retrieval() -> None:
    retrieval = _Retrieval()
    reranking = _Reranking()
    gateway = _Gateway()
    service = _service(retrieval, reranking, gateway)

    events = [event async for event in service.stream(RAGQueryRequest(query="你好，你能做什么？"))]

    assert [event.event for event in events] == ["answer_delta", "citations", "complete"]
    assert "资料问答助手" in events[0].data["text"]
    assert events[1].data["items"] == []
    assert events[2].data["status"] == "completed"
    assert events[2].data["route"] == "direct"
    assert events[2].data["retrieval_latency_ms"] == 0.0
    assert events[2].data["llm_latency_ms"] is None
    assert retrieval.calls == []
    assert reranking.calls == []
    assert gateway.requests == []


@pytest.mark.anyio
async def test_content_question_still_retrieves_and_reports_its_route() -> None:
    retrieval = _Retrieval([_chunk("chunk-1", "第一条答案", "block-1", 1)])
    gateway = _Gateway([LLMStreamEvent(delta="答案", provider="fake", model="fake-model")])
    service = _service(retrieval, _Reranking(), gateway)

    events = [
        event async for event in service.stream(RAGQueryRequest(query="这份资料的核心结论是什么？"))
    ]

    assert events[-1].data["route"] == "dense"
    assert retrieval.calls != []
    assert gateway.requests != []


@pytest.mark.anyio
async def test_memories_and_history_compression_reach_the_prompt() -> None:
    retrieval = _Retrieval([_chunk("chunk-1", "答案", "block-1", 1)])
    gateway = _Gateway([LLMStreamEvent(delta="答案", provider="fake", model="fake-model")])
    history = [RAGHistoryTurn(role="user", content=f"旧问题{index}") for index in range(12)]
    service = _service(retrieval, _Reranking(), gateway)

    events = [
        event
        async for event in service.stream(
            RAGQueryRequest(
                query="现在呢",
                knowledge_base_id=KNOWLEDGE_BASE_ID,
                retrieval_mode="dense",
                history=history,
                memories=("用户在准备中考化学复习",),
            )
        )
    ]

    complete = events[-1].data
    assert complete["memories_used"] == 1
    assert complete["history_compressed_turns"] >= 1
    # The compressed turn count is stable even without a usable summary.
    assert complete["history_summary_used"] is False
    prompt = gateway.requests[0].messages[0].content + gateway.requests[0].messages[1].content
    assert "长期记忆" in prompt
    assert "用户在准备中考化学复习" in prompt
