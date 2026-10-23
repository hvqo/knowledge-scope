# ruff: noqa: RUF001
from __future__ import annotations

from uuid import UUID

import pytest
from httpx import AsyncClient

from knowledge_scope.evidence.lifecycle import evidence_artifact_path
from test_document_assets import (
    build_canonical,
    create_knowledge_base,
    evidence_id_for_modality,
    upload_document,
    write_evidence,
    write_parsing_artifact,
)


def _citation(
    knowledge_base_id: str,
    document: dict[str, object],
    *,
    marker: str = "C1",
    evidence_id: str | None = None,
    chunk_id: str | None = None,
    modality: str | None = None,
    page_start: int = 1,
    page_end: int = 1,
    source_block_ids: list[str] | None = None,
) -> dict[str, object]:
    citation: dict[str, object] = {
        "marker": marker,
        "document_id": document["id"],
        "knowledge_base_id": knowledge_base_id,
        "page_start": page_start,
        "page_end": page_end,
        "source_block_ids": source_block_ids or ["p1-b2"],
        "section_path": ["第 5 章", "光合作用"],
        "section_title": "图 5-14 光合作用过程的示意图",
        "snippet": "图 5-14 光合作用过程的示意图",
        "snippet_kind": "representation",
    }
    if evidence_id is not None:
        citation.update(
            {
                "candidate_kind": "evidence",
                "evidence_id": evidence_id,
                "modality": modality or "image",
                "representation_ids": ["representation_v1_" + "a" * 64],
                "asset_refs": [],
            }
        )
    else:
        citation.update(
            {
                "candidate_kind": "chunk",
                "chunk_id": chunk_id or "chunk-000012",
                "snippet_kind": "source",
            }
        )
    return citation


async def _create_chat_answer(
    client: AsyncClient,
    knowledge_base_id: str,
    content: str,
    citations: list[dict[str, object]],
) -> str:
    conversation = await client.post(
        "/api/v1/chat/conversations",
        json={"knowledge_base_id": knowledge_base_id},
    )
    assert conversation.status_code == 201, conversation.text
    conversation_id = conversation.json()["id"]
    message = await client.post(
        f"/api/v1/chat/conversations/{conversation_id}/messages",
        json={"role": "assistant", "content": content, "citations": citations},
    )
    assert message.status_code == 201, message.text
    return str(message.json()["id"])


async def _prepare_report(client: AsyncClient, test_data_dir: object) -> dict[str, object]:
    knowledge_base_id = await create_knowledge_base(client, "报告插入验收")
    document = await upload_document(client, knowledge_base_id, filename="report-source.pdf")
    canonical = build_canonical(UUID(str(document["id"])))
    write_parsing_artifact(test_data_dir, UUID(str(document["id"])), canonical=canonical)  # type: ignore[arg-type]
    write_evidence(test_data_dir, canonical, knowledge_base_id)  # type: ignore[arg-type]
    report_response = await client.post(
        "/api/v1/reports",
        json={"title": "M5 智能体报告", "knowledge_base_id": knowledge_base_id},
    )
    assert report_response.status_code == 201, report_response.text
    report = report_response.json()
    return {
        "knowledge_base_id": knowledge_base_id,
        "document": document,
        "report": report,
        "test_data_dir": test_data_dir,
    }


async def _insert(
    client: AsyncClient,
    report: dict[str, object],
    chat_message_id: str,
    **extra: object,
) -> object:
    section_id = report["sections"][0]["id"]  # type: ignore[index]
    return await client.post(
        f"/api/v1/reports/{report['id']}/sections/{section_id}/agent-answers/insert",
        json={"chat_message_id": chat_message_id, **extra},
    )


@pytest.mark.anyio
async def test_insert_agent_answer_uses_server_side_message_body(
    client: AsyncClient,
    test_data_dir: object,
) -> None:
    state = await _prepare_report(client, test_data_dir)
    evidence_id = evidence_id_for_modality(
        test_data_dir,  # type: ignore[arg-type]
        UUID(str(state["document"]["id"])),
        "image",
    )
    message_id = await _create_chat_answer(
        client,
        str(state["knowledge_base_id"]),
        "根据视觉分析，图片展示了光合作用的结构 [C1]。",
        [_citation(str(state["knowledge_base_id"]), state["document"], evidence_id=evidence_id)],
    )
    response = await _insert(client, state["report"], message_id)
    assert response.status_code == 201, response.text  # type: ignore[attr-defined]
    body = response.json()  # type: ignore[attr-defined]
    assert "视觉分析草稿" in body["section"]["content"]
    assert body["section"]["content"].endswith("图片展示了光合作用的结构 [C1]。")
    assert body["warnings"] == ["已插入视觉分析草稿，正文已标记为待核对"]
    assert body["sources"][0]["evidence_id"] == evidence_id


@pytest.mark.anyio
async def test_insert_agent_answer_blocks_unverified_visual_numbers(
    client: AsyncClient,
    test_data_dir: object,
) -> None:
    state = await _prepare_report(client, test_data_dir)
    evidence_id = evidence_id_for_modality(
        test_data_dir,  # type: ignore[arg-type]
        UUID(str(state["document"]["id"])),
        "image",
    )
    message_id = await _create_chat_answer(
        client,
        str(state["knowledge_base_id"]),
        "图 1 中最低季度为 Q1，销售额为 120 万元 [C1]。",
        [_citation(str(state["knowledge_base_id"]), state["document"], evidence_id=evidence_id)],
    )
    response = await _insert(client, state["report"], message_id)
    assert response.status_code == 409, response.text  # type: ignore[attr-defined]
    assert "未经核验的视觉数值" in response.json()["detail"]  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_insert_agent_answer_allows_verified_table_answer(
    client: AsyncClient,
    test_data_dir: object,
) -> None:
    state = await _prepare_report(client, test_data_dir)
    message_id = await _create_chat_answer(
        client,
        str(state["knowledge_base_id"]),
        "表 1 中越南 2024 年比 2023 年增长 10.6977%，绝对变化 460 亿美元 [C1]。",
        [
            _citation(
                str(state["knowledge_base_id"]),
                state["document"],
                evidence_id=evidence_id_for_modality(
                    test_data_dir,  # type: ignore[arg-type]
                    UUID(str(state["document"]["id"])),
                    "table",
                ),
                modality="table",
                source_block_ids=["p1-b3"],
            )
        ],
    )
    response = await _insert(client, state["report"], message_id)
    assert response.status_code == 201, response.text  # type: ignore[attr-defined]
    assert response.json()["section"]["content"].endswith("绝对变化 460 亿美元 [C1]。")  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_insert_agent_answer_rejects_cross_knowledge_base_message(
    client: AsyncClient,
    test_data_dir: object,
) -> None:
    state = await _prepare_report(client, test_data_dir)
    other_kb = await create_knowledge_base(client, "另一个知识库")
    other_document = await upload_document(client, other_kb, filename="other.pdf")
    message_id = await _create_chat_answer(
        client,
        other_kb,
        "跨库回答 [C1]。",
        [_citation(other_kb, other_document, chunk_id="chunk-other")],
    )
    response = await _insert(client, state["report"], message_id)
    assert response.status_code == 409, response.text  # type: ignore[attr-defined]
    assert "一致的知识库" in response.json()["detail"]  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_insert_agent_answer_rejects_mismatched_evidence_page(
    client: AsyncClient,
    test_data_dir: object,
) -> None:
    state = await _prepare_report(client, test_data_dir)
    evidence_id = evidence_id_for_modality(
        test_data_dir,  # type: ignore[arg-type]
        UUID(str(state["document"]["id"])),
        "table",
    )
    message_id = await _create_chat_answer(
        client,
        str(state["knowledge_base_id"]),
        "表格数据 [C1]。",
        [
            _citation(
                str(state["knowledge_base_id"]),
                state["document"],
                evidence_id=evidence_id,
                modality="table",
                page_start=9,
                page_end=9,
                source_block_ids=["p1-b3"],
            )
        ],
    )
    response = await _insert(client, state["report"], message_id)
    assert response.status_code == 409, response.text  # type: ignore[attr-defined]
    assert response.json()["detail"] == "证据页码与服务器记录不一致"  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_insert_agent_answer_ignores_untrusted_client_body(
    client: AsyncClient,
    test_data_dir: object,
) -> None:
    state = await _prepare_report(client, test_data_dir)
    other_kb = await create_knowledge_base(client, "伪造知识库")
    message_id = await _create_chat_answer(
        client,
        str(state["knowledge_base_id"]),
        "真实回答 [C1]。",
        [_citation(str(state["knowledge_base_id"]), state["document"], chunk_id="chunk-000012")],
    )
    response = await _insert(
        client,
        state["report"],
        message_id,
        content_append="伪造内容（包含伪造结论）",
        sources=[{"kind": "rag_evidence", "knowledge_base_id": other_kb}],
        verified=True,
    )
    assert response.status_code == 422, response.text  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_insert_agent_answer_rejects_unknown_message(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/reports/00000000-0000-0000-0000-000000000000"
        "/sections/00000000-0000-0000-0000-000000000000/agent-answers/insert",
        json={"chat_message_id": "00000000-0000-0000-0000-000000000000"},
    )
    assert response.status_code == 404


@pytest.mark.anyio
async def test_insert_agent_answer_degrades_for_historical_missing_artifacts(
    client: AsyncClient,
    test_data_dir: object,
) -> None:
    state = await _prepare_report(client, test_data_dir)
    evidence_id = evidence_id_for_modality(
        test_data_dir,  # type: ignore[arg-type]
        UUID(str(state["document"]["id"])),
        "table",
    )
    message_id = await _create_chat_answer(
        client,
        str(state["knowledge_base_id"]),
        "历史表格引用 [C1]。",
        [
            _citation(
                str(state["knowledge_base_id"]),
                state["document"],
                evidence_id=evidence_id,
                modality="table",
                source_block_ids=["p1-b3"],
            )
        ],
    )
    evidence_artifact_path(test_data_dir, UUID(str(state["document"]["id"]))).unlink()  # type: ignore[arg-type]
    (test_data_dir / "parsing" / str(state["document"]["id"])).rename(  # type: ignore[operator]
        test_data_dir / "parsing" / "removed"  # type: ignore[operator]
    )
    response = await _insert(client, state["report"], message_id)
    assert response.status_code == 201, response.text  # type: ignore[attr-defined]
    assert "历史文档缺少解析产物" in response.json()["warnings"][0]  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_insert_agent_answer_requires_citations(
    client: AsyncClient,
    test_data_dir: object,
) -> None:
    state = await _prepare_report(client, test_data_dir)
    message_id = await _create_chat_answer(
        client,
        str(state["knowledge_base_id"]),
        "没有引用的回答。",
        [],
    )
    response = await _insert(client, state["report"], message_id)
    assert response.status_code == 422, response.text  # type: ignore[attr-defined]
