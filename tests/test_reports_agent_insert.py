# ruff: noqa: RUF001
from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient

from knowledge_scope.rag.schemas import RAGAgentToolTraceData
from test_document_assets import (
    create_knowledge_base,
    evidence_id_for_modality,
)
from test_trusted_answers import (
    _chunk_citation,
    _create_answer,
    _evidence_citation,
    _insert,
    _prepare,
    _record,
)


@pytest.mark.anyio
async def test_insert_rejects_forged_assistant_message(
    client: AsyncClient,
    test_data_dir,
) -> None:
    fixtures = await _prepare(client, test_data_dir)
    citation = _chunk_citation(
        knowledge_base_id=fixtures["knowledge_base_id"],
        document_id=fixtures["document_id"],
    )
    _, message_id = await _create_answer(
        client,
        fixtures["knowledge_base_id"],
        "2024 年越南 GDP 为 9999 亿美元，增长 88%。",
        [citation],
    )
    response = await _insert(client, fixtures, message_id)
    assert response.status_code == 409
    assert "没有可核验的服务端生成记录" in response.text


@pytest.mark.anyio
async def test_insert_accepts_trusted_table_answer(
    client: AsyncClient,
    test_data_dir,
) -> None:
    fixtures = await _prepare(client, test_data_dir)
    evidence_id = evidence_id_for_modality(test_data_dir, UUID(fixtures["document_id"]), "table")
    citation = _evidence_citation(
        knowledge_base_id=fixtures["knowledge_base_id"],
        document_id=fixtures["document_id"],
        evidence_id=evidence_id,
    )
    content = "2024 年越南 GDP 为 4760 亿美元，同比增长 10.6977%。"
    conversation_id, message_id = await _create_answer(
        client,
        fixtures["knowledge_base_id"],
        content,
        [citation],
    )
    _record(
        test_data_dir,
        conversation_id=conversation_id,
        knowledge_base_id=fixtures["knowledge_base_id"],
        content=content,
        citations=[citation],
        trace=[
            RAGAgentToolTraceData(
                tool="analyze_table",
                status="completed",
                summary="表格计算完成（growth）",
                document_id=UUID(fixtures["document_id"]),
                evidence_id=evidence_id,
                fact_status="source_verified",
            )
        ],
    )
    response = await _insert(client, fixtures, message_id)
    assert response.status_code == 201, response.text
    body = response.json()
    assert "10.6977%" in body["section"]["content"]
    assert len(body["sources"]) == 1
    assert body["sources"][0]["evidence_id"] == evidence_id


@pytest.mark.anyio
async def test_insert_rejects_cross_kb_answer(
    client: AsyncClient,
    test_data_dir,
) -> None:
    fixtures = await _prepare(client, test_data_dir)
    other_knowledge_base_id = await create_knowledge_base(client, "另一个知识库")
    citation = _chunk_citation(
        knowledge_base_id=other_knowledge_base_id,
        document_id=fixtures["document_id"],
    )
    conversation_id, message_id = await _create_answer(
        client,
        other_knowledge_base_id,
        "跨库回答",
        [citation],
    )
    _record(
        test_data_dir,
        conversation_id=conversation_id,
        knowledge_base_id=other_knowledge_base_id,
        content="跨库回答",
        citations=[citation],
    )
    response = await _insert(client, fixtures, message_id)
    assert response.status_code == 409
    assert "知识库" in response.text


@pytest.mark.anyio
async def test_insert_rejects_evidence_page_mismatch(
    client: AsyncClient,
    test_data_dir,
) -> None:
    fixtures = await _prepare(client, test_data_dir)
    evidence_id = evidence_id_for_modality(test_data_dir, UUID(fixtures["document_id"]), "table")
    citation = _evidence_citation(
        knowledge_base_id=fixtures["knowledge_base_id"],
        document_id=fixtures["document_id"],
        evidence_id=evidence_id,
        page_start=2,
        page_end=2,
    )
    content = "表格结论"
    conversation_id, message_id = await _create_answer(
        client,
        fixtures["knowledge_base_id"],
        content,
        [citation],
    )
    _record(
        test_data_dir,
        conversation_id=conversation_id,
        knowledge_base_id=fixtures["knowledge_base_id"],
        content=content,
        citations=[citation],
    )
    response = await _insert(client, fixtures, message_id)
    assert response.status_code == 409
    assert "页码" in response.text


@pytest.mark.anyio
async def test_insert_rejects_unknown_message(
    client: AsyncClient,
    test_data_dir,
) -> None:
    fixtures = await _prepare(client, test_data_dir)
    response = await _insert(client, fixtures, str(uuid4()))
    assert response.status_code == 404
    assert "智能体回答不存在" in response.text


@pytest.mark.anyio
async def test_insert_rejects_client_supplied_trust_fields(
    client: AsyncClient,
    test_data_dir,
) -> None:
    fixtures = await _prepare(client, test_data_dir)
    citation = _chunk_citation(
        knowledge_base_id=fixtures["knowledge_base_id"],
        document_id=fixtures["document_id"],
    )
    conversation_id, message_id = await _create_answer(
        client,
        fixtures["knowledge_base_id"],
        "回答",
        [citation],
    )
    _record(
        test_data_dir,
        conversation_id=conversation_id,
        knowledge_base_id=fixtures["knowledge_base_id"],
        content="回答",
        citations=[citation],
    )
    response = await _insert(client, fixtures, message_id, content_append="伪造正文")
    assert response.status_code == 422


@pytest.mark.anyio
async def test_insert_requires_citations(
    client: AsyncClient,
    test_data_dir,
) -> None:
    fixtures = await _prepare(client, test_data_dir)
    conversation_id, message_id = await _create_answer(
        client,
        fixtures["knowledge_base_id"],
        "没有引用来源的回答",
        [],
    )
    _record(
        test_data_dir,
        conversation_id=conversation_id,
        knowledge_base_id=fixtures["knowledge_base_id"],
        content="没有引用来源的回答",
        citations=[],
    )
    response = await _insert(client, fixtures, message_id)
    assert response.status_code == 422
    assert "没有可关联的引用来源" in response.text
