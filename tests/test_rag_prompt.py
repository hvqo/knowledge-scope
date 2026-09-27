# Chinese question text intentionally keeps its full-width punctuation.
# ruff: noqa: RUF001

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from knowledge_scope.rag.context import assemble_context
from knowledge_scope.rag.prompt import RAG_PROMPT_VERSION, build_rag_messages
from knowledge_scope.rag.schemas import (
    RAG_HISTORY_MAX_CHARACTERS,
    RAG_HISTORY_MAX_TURNS,
    RAGHistoryTurn,
    RAGQueryRequest,
)
from knowledge_scope.retrieval.qdrant import (
    QDRANT_COLLECTION_SCHEMA_VERSION,
    ChunkVectorPayload,
    RetrievedChunk,
)
from knowledge_scope.retrieval.reranking import RerankedChunk


def _context(text: str) -> object:
    chunk = RerankedChunk(
        chunk=RetrievedChunk(
            point_id=uuid4(),
            score=1.0,
            payload=ChunkVectorPayload(
                collection_schema_version=QDRANT_COLLECTION_SCHEMA_VERSION,
                chunk_id="chunk-1",
                document_id=UUID("22222222-2222-4222-8222-222222222222"),
                page_start=1,
                page_end=1,
                source_block_ids=["block-1"],
                section_path=["第一章", "概念"],
                content_types=["text"],
                asset_refs=[],
                text=text,
                chunking_config_fingerprint="a" * 64,
                embedding_model="Qwen/Qwen3-Embedding-0.6B",
                embedding_model_revision="revision",
                embedding_config_fingerprint="b" * 64,
            ),
        ),
        dense_rank=1,
        reranker_score=1.0,
    )
    return assemble_context([chunk], budget_chars=100)


def test_rag_prompt_is_versioned_and_contains_only_application_markers() -> None:
    messages = build_rag_messages("什么是概念?", _context("概念是对事物本质特征的概括。"))

    assert len(messages) == 2
    assert RAG_PROMPT_VERSION in messages[0].content
    assert "只依据本轮用户消息中提供的检索资料作答" in messages[0].content
    assert "[C1]" in messages[1].content
    assert "document_id=22222222-2222-4222-8222-222222222222" in messages[1].content
    assert "什么是概念?" in messages[1].content


def test_empty_context_prompt_requires_insufficient_evidence_response() -> None:
    messages = build_rag_messages("问题", _context(""))

    assert "没有检索到可用的文本资料" in messages[1].content
    assert "资料完全无法回答该问题时" in messages[0].content


def test_prompt_handles_questions_about_the_assistant_itself() -> None:
    messages = build_rag_messages("你能做什么？", _context("教材中的句型练习。"))

    system_prompt = messages[0].content

    assert "询问你的身份、能力或使用方法" in system_prompt
    assert "不要添加引用标记" in system_prompt
    assert "不要在本次回答中复用" in system_prompt
    assert RAG_PROMPT_VERSION == "rag-qa-v6"
    # The answer must open with the answer itself and keep citations tidy.
    assert "第一句就给出结论" in system_prompt
    assert "句末标点之前" in system_prompt
    assert "不要单独成行" in system_prompt
    assert "同一位置最多放两个标记" in system_prompt
    # It must read as a reply to the user, not as a summary of the material.
    assert "资料中提到" in system_prompt
    assert "用“你”称呼提问者" in system_prompt
    assert "不要用“资料不足”开头" in system_prompt
    assert "用 Markdown 组织排版" in system_prompt
    # Formulas must reach the client as LaTeX, never as bold plain text.
    assert "数学内容一律用 LaTeX" in system_prompt
    assert "$a^x=N$" in system_prompt
    assert "$$x=\\log_a N$$" in system_prompt


def test_history_turns_precede_the_current_question_and_lose_their_markers() -> None:
    history = [
        RAGHistoryTurn(role="user", content="第一个问题"),
        RAGHistoryTurn(role="assistant", content="第一个回答 [C1] [C2]"),
    ]

    messages = build_rag_messages("再展开讲讲", _context("补充资料。"), history)

    assert [message.role for message in messages] == ["system", "user", "assistant", "user"]
    assert messages[1].content == "第一个问题"
    assert messages[2].content == "第一个回答"
    assert "[C1]" not in messages[2].content
    assert "再展开讲讲" in messages[3].content
    assert "[C1]" in messages[3].content


def test_blank_history_turns_are_dropped() -> None:
    history = [
        RAGHistoryTurn(role="assistant", content="[C1]"),
        RAGHistoryTurn(role="user", content="保留这一条"),
    ]

    messages = build_rag_messages("问题", _context("资料"), history)

    assert [message.role for message in messages] == ["system", "user", "user"]
    assert messages[1].content == "保留这一条"


def test_history_budget_is_bounded() -> None:
    turns = [{"role": "user", "content": "问题"} for _ in range(RAG_HISTORY_MAX_TURNS + 1)]

    with pytest.raises(ValidationError):
        RAGQueryRequest(
            query="问题",
            knowledge_base_id=uuid4(),
            retrieval_mode="unified",
            history=turns,
        )

    # The budget is generous now: long conversations are compressed by the
    # service instead of being rejected, but a hard cap still guards abuse.
    oversized = [{"role": "user", "content": "长" * 4_000} for _ in range(16)]
    assert sum(len(item["content"]) for item in oversized) > RAG_HISTORY_MAX_CHARACTERS
    with pytest.raises(ValidationError):
        RAGQueryRequest(
            query="问题",
            knowledge_base_id=uuid4(),
            retrieval_mode="unified",
            history=oversized,
        )

    blank = [{"role": "user", "content": "   "}]
    with pytest.raises(ValidationError):
        RAGQueryRequest(
            query="问题",
            knowledge_base_id=uuid4(),
            retrieval_mode="unified",
            history=blank,
        )
