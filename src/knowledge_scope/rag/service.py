"""Small async RAG orchestration over the existing retrieval and LLM services."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import dataclass, replace
from time import perf_counter

from knowledge_scope.llm.errors import LLMError
from knowledge_scope.llm.gateway import LLMGateway
from knowledge_scope.llm.schemas import LLMRequest
from knowledge_scope.retrieval.embedding import EmbeddingModelError
from knowledge_scope.retrieval.qdrant import VectorStoreError
from knowledge_scope.retrieval.reranking import RerankedChunk, RerankerError, RerankingService
from knowledge_scope.retrieval.service import DenseRetrievalService, RetrievalError, RetrievalResult
from knowledge_scope.retrieval.unified import (
    UnifiedRetrievalError,
    UnifiedRetrievalResult,
    UnifiedRetrievalService,
)
from knowledge_scope.shared.config import Settings

from .cache import (
    CachedRetrieval,
    RetrievalCache,
    build_retrieval_cache_key,
    build_rewrite_cache_key,
    decode_retrieval,
    encode_retrieval,
)
from .context import ContextSelection, assemble_context
from .history import HistoryCompressor
from .prompt import RAG_PROMPT_VERSION, build_rag_messages
from .rewrite import QueryRewriteService
from .routing import covers_subject, direct_answer, resolve_route, term_coverage
from .schemas import (
    RAGQueryRequest,
    RAGRequestedRetrievalMode,
    RAGRetrievalMode,
    RAGStreamEvent,
)

logger = logging.getLogger(__name__)

RAG_INSUFFICIENT_EVIDENCE = "当前检索到的资料不足以回答该问题。"


@dataclass(frozen=True, slots=True)
class RAGSelection:
    """Selected context plus the retrieval result that produced it."""

    retrieval_mode: RAGRetrievalMode
    context: ContextSelection
    dense_result: RetrievalResult | None = None
    reranked: tuple[RerankedChunk, ...] = ()
    unified_result: UnifiedRetrievalResult | None = None
    from_cache: bool = False
    # True when an automatic fast lookup was upgraded to unified retrieval.
    escalated: bool = False


class RAGService:
    """Coordinate synchronous local retrieval with the async LLM gateway."""

    def __init__(
        self,
        retrieval: DenseRetrievalService,
        reranking: RerankingService,
        gateway: LLMGateway,
        settings: Settings,
        *,
        unified_retrieval: UnifiedRetrievalService | None = None,
        cache: RetrievalCache | None = None,
        rewrite: QueryRewriteService | None = None,
    ) -> None:
        self._retrieval = retrieval
        self._reranking = reranking
        self._gateway = gateway
        self._settings = settings
        self._unified_retrieval = unified_retrieval
        self._cache = cache
        self._rewrite = rewrite
        self._history = HistoryCompressor(gateway, settings, cache=cache)
        # Avoid queueing cancelled requests into the thread pool while keeping
        # synchronous local model work off the event loop.
        self._selection_gate = asyncio.Semaphore(1)

    def _cached_retrieval(
        self,
        request: RAGQueryRequest,
        requested_mode: RAGRequestedRetrievalMode,
    ) -> CachedRetrieval | None:
        """Return the cached result for the requested scope, if any.

        The key uses the scope the caller asked for: an ``auto`` question is
        cached once, whichever path ended up producing the context, and the
        stored payload records the path that was used.
        """

        if self._cache is None:
            return None
        payload = self._safe_cache_get(
            build_retrieval_cache_key(
                knowledge_base_id=request.knowledge_base_id,
                document_id=request.document_id,
                retrieval_mode=requested_mode,
                query=request.query,
            )
        )
        if payload is None:
            return None
        try:
            return decode_retrieval(payload)
        except (ValueError, KeyError, TypeError):
            logger.debug("discarding an unreadable cached retrieval result")
            return None

    def _safe_cache_get(self, key: str) -> str | None:
        try:
            return self._cache.get(key) if self._cache is not None else None
        except Exception as error:  # pragma: no cover - defensive, cache is optional
            logger.debug("retrieval cache read failed (%s)", type(error).__name__)
            return None

    def _store_retrieval(
        self,
        request: RAGQueryRequest,
        requested_mode: RAGRequestedRetrievalMode,
        *,
        retrieval_mode: RAGRetrievalMode,
        escalated: bool,
        context: ContextSelection,
    ) -> None:
        if self._cache is None:
            return
        # Misses are cached too, with a shorter ttl: a question the corpus
        # cannot answer yet is the most expensive one to repeat.
        ttl_seconds = (
            self._settings.rag_cache_ttl_seconds
            if context.items
            else self._settings.rag_cache_empty_ttl_seconds
        )
        try:
            self._cache.set(
                build_retrieval_cache_key(
                    knowledge_base_id=request.knowledge_base_id,
                    document_id=request.document_id,
                    retrieval_mode=requested_mode,
                    query=request.query,
                ),
                encode_retrieval(
                    CachedRetrieval(
                        retrieval_mode=retrieval_mode,
                        escalated=escalated,
                        context=context,
                    )
                ),
                ttl_seconds,
            )
        except Exception as error:  # pragma: no cover - defensive, cache is optional
            logger.debug("retrieval cache write failed (%s)", type(error).__name__)

    @staticmethod
    def _context_text(context: ContextSelection) -> str:
        return " ".join(
            " ".join((" ".join(item.citation.section_path), item.text)) for item in context.items
        )

    def _needs_escalation(self, query: str, context: ContextSelection) -> bool:
        """Return whether the fast path clearly missed the question's subject.

        The vector path is good enough when its context talks about the thing
        the question is about (the leading content term) and still covers at
        least `rag_auto_escalate_min_coverage` of the question's other content
        terms.  Otherwise the full retrieval path gets a chance to find the
        material the lookup needs.
        """

        if not context.items:
            return True
        text = self._context_text(context)
        if not covers_subject(query, text):
            return True
        minimum = self._settings.rag_auto_escalate_min_coverage
        return minimum > 0 and term_coverage(query, text) < minimum

    def _select_dense_context(self, request: RAGQueryRequest) -> RAGSelection:
        dense_result = self._retrieval.search(
            request.query,
            limit=self._settings.rag_candidate_limit,
            knowledge_base_id=request.knowledge_base_id,
            document_id=request.document_id,
        )
        text_candidates = tuple(item for item in dense_result.items if item.payload.text.strip())
        reranked = self._reranking.rerank(
            request.query,
            text_candidates,
            limit=self._settings.rag_rerank_limit,
        )
        context = assemble_context(
            reranked,
            budget_chars=self._settings.rag_context_budget_chars,
        )
        return RAGSelection(
            retrieval_mode="dense",
            context=context,
            dense_result=dense_result,
            reranked=reranked,
        )

    async def _select_unified_context(self, request: RAGQueryRequest) -> RAGSelection:
        if self._unified_retrieval is None:
            raise UnifiedRetrievalError("unified retrieval is not configured")
        if request.knowledge_base_id is None:
            raise ValueError("knowledge_base_id is required for unified retrieval")
        unified_result = await self._unified_retrieval.search(
            request.query,
            request.knowledge_base_id,
            document_id=request.document_id,
        )
        context = assemble_context(
            unified_result.items,
            budget_chars=self._settings.rag_context_budget_chars,
        )
        return RAGSelection(
            retrieval_mode="unified",
            context=context,
            unified_result=unified_result,
        )

    async def _resolve_selection(
        self,
        request: RAGQueryRequest,
        *,
        requested_mode: RAGRequestedRetrievalMode,
        search_query: str,
        effective_mode: RAGRetrievalMode,
    ) -> RAGSelection:
        """Run the retrieval path, upgrading an automatic lookup when needed.

        With ``auto`` the cheap vector path is tried first; if its context does
        not even mention the subject the question asked about, the full
        retrieval path runs on top of it.  An explicit ``dense`` choice is
        never upgraded.
        """

        cached = self._cached_retrieval(request, requested_mode)
        if cached is not None:
            return RAGSelection(
                retrieval_mode=cached.retrieval_mode,
                context=cached.context,
                from_cache=True,
                escalated=cached.escalated,
            )

        retrieval_request = request.model_copy(
            update={"query": search_query, "retrieval_mode": effective_mode}
        )
        if effective_mode == "unified":
            selection = await self._select_unified_context(retrieval_request)
        else:
            selection = await asyncio.to_thread(self._select_dense_context, retrieval_request)

        escalated = False
        if (
            requested_mode == "auto"
            and effective_mode == "dense"
            and self._unified_retrieval is not None
            and request.knowledge_base_id is not None
            and self._needs_escalation(search_query, selection.context)
        ):
            selection = await self._select_unified_context(retrieval_request)
            escalated = True

        self._store_retrieval(
            request,
            requested_mode,
            retrieval_mode=selection.retrieval_mode,
            escalated=escalated,
            context=selection.context,
        )
        return replace(selection, escalated=escalated)

    @staticmethod
    def _selection_metadata(selection: RAGSelection) -> dict[str, object]:
        metadata: dict[str, object] = {
            "route": selection.retrieval_mode,
            "retrieval_mode": selection.retrieval_mode,
            "retrieval_cached": selection.from_cache,
            "retrieval_escalated": selection.escalated,
        }
        if selection.unified_result is None:
            return metadata
        # Cached contexts keep no branch detail, so degraded stays unknown.
        metadata["retrieval_degraded"] = (
            None if selection.from_cache else selection.unified_result.degraded
        )
        metadata["retrieval_branch_statuses"] = (
            None
            if selection.from_cache
            else {branch.branch: branch.status for branch in selection.unified_result.branches}
        )
        return metadata

    @staticmethod
    def _retrieval_error_category(error: BaseException) -> str:
        if isinstance(error, ValueError):
            return "request"
        if isinstance(
            error,
            (
                EmbeddingModelError,
                RetrievalError,
                VectorStoreError,
                RerankerError,
                UnifiedRetrievalError,
            ),
        ):
            return "retrieval"
        return "retrieval"

    @staticmethod
    def _llm_error_message(error: LLMError) -> str:
        messages = {
            "configuration": "LLM provider is not configured",
            "timeout": "LLM provider timed out",
            "connection": "LLM provider is unavailable",
            "api": "LLM provider rejected the request",
            "malformed_response": "LLM provider returned an invalid response",
            "provider": "LLM generation failed",
            "usage_persistence": "LLM usage could not be recorded",
            "cancelled": "LLM generation was cancelled",
        }
        return messages.get(error.category, "LLM generation failed")

    @staticmethod
    def _error_events(
        *,
        category: str,
        message: str,
        started: float,
        retrieval_latency_ms: float | None = None,
        llm_latency_ms: float | None = None,
        provider: str | None = None,
        model: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        finish_reason: str | None = None,
        retrieval_mode: RAGRetrievalMode = "dense",
        retrieval_degraded: bool | None = None,
        retrieval_branch_statuses: dict[str, str] | None = None,
    ) -> tuple[RAGStreamEvent, RAGStreamEvent]:
        return (
            RAGStreamEvent(
                event="error",
                data={"category": category, "message": message},
            ),
            RAGStreamEvent(
                event="complete",
                data={
                    "status": "error",
                    "route": retrieval_mode,
                    "prompt_version": RAG_PROMPT_VERSION,
                    "provider": provider,
                    "model": model,
                    "retrieval_mode": retrieval_mode,
                    "retrieval_degraded": retrieval_degraded,
                    "retrieval_branch_statuses": retrieval_branch_statuses,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "finish_reason": finish_reason,
                    "retrieval_latency_ms": retrieval_latency_ms,
                    "llm_latency_ms": llm_latency_ms,
                    "latency_ms": (perf_counter() - started) * 1000,
                },
            ),
        )

    async def _rewrite_query(
        self,
        request: RAGQueryRequest,
    ) -> tuple[str | None, float | None]:
        """Return the standalone search query and the rewrite latency, if any."""

        if self._rewrite is None or not request.history:
            return None, None
        history = tuple(request.history)
        cache_key = build_rewrite_cache_key(
            query=request.query,
            history_digest=hashlib.sha256(
                "\n".join(f"{turn.role}:{turn.content}" for turn in history).encode("utf-8")
            ).hexdigest(),
        )
        cached = self._safe_cache_get(cache_key)
        if cached is not None:
            return (cached or None), 0.0
        started = perf_counter()
        try:
            rewritten = await asyncio.wait_for(
                self._rewrite.rewrite(request.query, history),
                timeout=self._settings.rag_rewrite_timeout_seconds,
            )
        except TimeoutError:
            logger.debug("query rewrite exceeded its budget")
            return None, None
        latency_ms = (perf_counter() - started) * 1000
        if self._cache is not None:
            try:
                self._cache.set(
                    cache_key,
                    rewritten or "",
                    self._settings.rag_cache_ttl_seconds,
                )
            except Exception as error:  # pragma: no cover - defensive
                logger.debug("rewrite cache write failed (%s)", type(error).__name__)
        return rewritten, latency_ms

    async def stream(self, request: RAGQueryRequest) -> AsyncIterator[RAGStreamEvent]:
        """Yield answer, citation, and terminal status events for one question."""
        started = perf_counter()
        route, intent = resolve_route(request.query, request.retrieval_mode)
        if route == "direct":
            answer = direct_answer(intent) if intent is not None else ""
            yield RAGStreamEvent(event="answer_delta", data={"text": answer})
            yield RAGStreamEvent(
                event="citations",
                data={"prompt_version": RAG_PROMPT_VERSION, "items": []},
            )
            yield RAGStreamEvent(
                event="complete",
                data={
                    "status": "completed",
                    "route": "direct",
                    "prompt_version": RAG_PROMPT_VERSION,
                    "provider": None,
                    "model": None,
                    # Nothing was retrieved, so the mode is reported for context only.
                    "retrieval_mode": (
                        "dense" if request.retrieval_mode == "auto" else request.retrieval_mode
                    ),
                    "retrieval_degraded": None,
                    "retrieval_branch_statuses": None,
                    "input_tokens": None,
                    "output_tokens": None,
                    "finish_reason": None,
                    "retrieval_latency_ms": 0.0,
                    "llm_latency_ms": None,
                    "latency_ms": (perf_counter() - started) * 1000,
                },
            )
            return

        # Older turns are compressed first so both the rewrite and the answer
        # work on a bounded window, with the summary riding along as background.
        history_plan = await self._history.prepare(tuple(request.history))
        # A follow-up is rewritten first, so both the routing decision and the
        # retrieval itself work on a standalone question.
        rewritten_query, rewrite_latency_ms = await self._rewrite_query(
            request.model_copy(update={"history": list(history_plan.recent_turns)}),
        )
        search_query = rewritten_query if rewritten_query is not None else request.query
        search_route, _ = resolve_route(search_query, request.retrieval_mode)
        effective_mode: RAGRetrievalMode = "dense" if search_route == "direct" else search_route
        if effective_mode == "unified" and request.knowledge_base_id is None:
            effective_mode = "dense"

        selection_started = perf_counter()
        try:
            async with self._selection_gate:
                selection = await self._resolve_selection(
                    request,
                    requested_mode=request.retrieval_mode,
                    search_query=search_query,
                    effective_mode=effective_mode,
                )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            retrieval_latency_ms = (perf_counter() - selection_started) * 1000
            events = self._error_events(
                category=self._retrieval_error_category(error),
                message=(
                    "query is invalid"
                    if isinstance(error, ValueError)
                    else "retrieval pipeline failed"
                ),
                started=started,
                retrieval_latency_ms=retrieval_latency_ms,
                retrieval_mode=effective_mode,
            )
            for event in events:
                yield event
            return

        retrieval_latency_ms = (perf_counter() - selection_started) * 1000
        selection_metadata = {
            **self._selection_metadata(selection),
            "rewritten_query": rewritten_query,
            "rewrite_latency_ms": rewrite_latency_ms,
            "history_compressed_turns": history_plan.compressed_turns,
            "history_cache_hit": history_plan.cache_hit,
            "history_summary_used": history_plan.summary is not None,
            "memories_used": len(request.memories),
        }
        if not selection.context.items:
            yield RAGStreamEvent(
                event="answer_delta",
                data={"text": RAG_INSUFFICIENT_EVIDENCE},
            )
            yield RAGStreamEvent(
                event="citations",
                data={"prompt_version": RAG_PROMPT_VERSION, "items": []},
            )
            yield RAGStreamEvent(
                event="complete",
                data={
                    "status": "insufficient_evidence",
                    "prompt_version": RAG_PROMPT_VERSION,
                    "provider": None,
                    "model": None,
                    **selection_metadata,
                    "input_tokens": None,
                    "output_tokens": None,
                    "finish_reason": None,
                    "retrieval_latency_ms": retrieval_latency_ms,
                    "llm_latency_ms": None,
                    "latency_ms": (perf_counter() - started) * 1000,
                },
            )
            return

        llm_request = LLMRequest(
            messages=build_rag_messages(
                request.query,
                selection.context,
                history_plan.recent_turns,
                summary=history_plan.summary,
                memories=request.memories,
            ),
            task_type="rag_answer",
            temperature=0.0,
            max_tokens=self._settings.rag_max_tokens,
            reasoning="disabled",
        )
        provider: str | None = None
        model: str | None = None
        input_tokens: int | None = None
        output_tokens: int | None = None
        finish_reason: str | None = None
        llm_started = perf_counter()
        has_answer_text = False
        try:
            async with aclosing(self._gateway.stream(llm_request)) as gateway_stream:
                async for event in gateway_stream:
                    provider = event.provider
                    model = event.model
                    if event.input_tokens is not None:
                        input_tokens = event.input_tokens
                    if event.output_tokens is not None:
                        output_tokens = event.output_tokens
                    if event.finish_reason is not None:
                        finish_reason = event.finish_reason
                    if event.delta:
                        has_answer_text = has_answer_text or bool(event.delta.strip())
                        yield RAGStreamEvent(
                            event="answer_delta",
                            data={"text": event.delta},
                        )
        except asyncio.CancelledError:
            raise
        except LLMError as error:
            llm_latency_ms = (perf_counter() - llm_started) * 1000
            events = self._error_events(
                category=error.category,
                message=self._llm_error_message(error),
                started=started,
                retrieval_latency_ms=retrieval_latency_ms,
                llm_latency_ms=llm_latency_ms,
                provider=provider,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                finish_reason=finish_reason,
                retrieval_mode=selection.retrieval_mode,
                retrieval_degraded=selection_metadata.get("retrieval_degraded"),
                retrieval_branch_statuses=selection_metadata.get("retrieval_branch_statuses"),
            )
            for event in events:
                yield event
            return
        except Exception:
            llm_latency_ms = (perf_counter() - llm_started) * 1000
            events = self._error_events(
                category="provider",
                message="LLM generation failed",
                started=started,
                retrieval_latency_ms=retrieval_latency_ms,
                llm_latency_ms=llm_latency_ms,
                provider=provider,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                finish_reason=finish_reason,
                retrieval_mode=selection.retrieval_mode,
                retrieval_degraded=selection_metadata.get("retrieval_degraded"),
                retrieval_branch_statuses=selection_metadata.get("retrieval_branch_statuses"),
            )
            for event in events:
                yield event
            return

        llm_latency_ms = (perf_counter() - llm_started) * 1000
        if not has_answer_text:
            truncated = finish_reason == "length"
            events = self._error_events(
                category="truncated_output" if truncated else "empty_output",
                message=(
                    "LLM provider exhausted the output token budget without visible text"
                    if truncated
                    else "LLM provider returned an empty response"
                ),
                started=started,
                retrieval_latency_ms=retrieval_latency_ms,
                llm_latency_ms=llm_latency_ms,
                provider=provider,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                finish_reason=finish_reason,
                retrieval_mode=selection.retrieval_mode,
                retrieval_degraded=selection_metadata.get("retrieval_degraded"),
                retrieval_branch_statuses=selection_metadata.get("retrieval_branch_statuses"),
            )
            for event in events:
                yield event
            return

        yield RAGStreamEvent(
            event="citations",
            data={
                "prompt_version": RAG_PROMPT_VERSION,
                "items": [
                    citation.model_dump(mode="json") for citation in selection.context.citations
                ],
            },
        )
        yield RAGStreamEvent(
            event="complete",
            data={
                "status": "completed",
                "prompt_version": RAG_PROMPT_VERSION,
                "provider": provider or self._settings.llm_provider,
                "model": model or self._settings.llm_model,
                **selection_metadata,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "finish_reason": finish_reason,
                "retrieval_latency_ms": retrieval_latency_ms,
                "llm_latency_ms": llm_latency_ms,
                "latency_ms": (perf_counter() - started) * 1000,
            },
        )


__all__ = ["RAG_INSUFFICIENT_EVIDENCE", "RAGSelection", "RAGService"]
