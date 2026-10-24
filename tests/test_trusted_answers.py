# ruff: noqa: RUF001
from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient

from knowledge_scope.chat.trust import (
    record_streamed_answer,
    verify_streamed_answer,
)
from knowledge_scope.rag.schemas import (
    RAGAgentToolTraceData,
    RAGCitation,
    RAGCompleteData,
    RAGVisualSource,
)
from knowledge_scope.shared.config import Settings
from test_document_assets import (
    build_canonical,
    create_knowledge_base,
    evidence_id_for_modality,
    upload_document,
    write_evidence,
    write_parsing_artifact,
)

DOC_UUID = uuid4()
EVIDENCE_ID = "evidence_v1_" + "a" * 64


def _settings(test_data_dir: Path) -> Settings:
    return Settings(_env_file=None, environment="test", data_dir=str(test_data_dir))


def _evidence_citation(
    *,
    knowledge_base_id: str,
    document_id: str,
    evidence_id: str = EVIDENCE_ID,
    modality: str = "table",
    page_start: int = 1,
    page_end: int = 1,
    source_block_ids: list[str] | None = None,
    marker: str = "C1",
) -> RAGCitation:
    return RAGCitation(
        marker=marker,
        document_id=UUID(document_id),
        knowledge_base_id=UUID(knowledge_base_id),
        candidate_kind="evidence",
        evidence_id=evidence_id,
        modality=modality,  # type: ignore[arg-type]
        representation_ids=["representation_v1_" + "b" * 64],
        asset_refs=[],
        page_start=page_start,
        page_end=page_end,
        source_block_ids=source_block_ids or ["p1-b3"],
        section_path=["第 1 章"],
        snippet="表 1 各国 GDP",
        snippet_kind="representation",
    )


def _chunk_citation(*, knowledge_base_id: str, document_id: str) -> RAGCitation:
    return RAGCitation(
        marker="C2",
        document_id=UUID(document_id),
        candidate_kind="chunk",
        chunk_id="chunk-000001",
        page_start=1,
        page_end=1,
        source_block_ids=["p1-b1"],
        section_path=["第 1 章"],
        snippet="各国 GDP 数据",
        snippet_kind="source",
    )


def _complete(
    *,
    safety_gate: str = "none",
    fact_status: str | None = "source_verified",
    trace: list[RAGAgentToolTraceData] | None = None,
    notice: str | None = None,
    visual_sources: list[RAGVisualSource] | None = None,
) -> RAGCompleteData:
    return RAGCompleteData(
        status="completed",
        prompt_version="agent-rag-v1",
        provider="deepseek",
        model="deepseek-flash",
        retrieval_mode="unified",
        route="unified",
        answer_mode="agent",
        tool_calls=len(trace or []),
        latency_ms=1.0,
        fact_status=fact_status,
        safety_gate=safety_gate,  # type: ignore[arg-type]
        safety_notice=notice,
        agent_tool_trace=trace or [],
        visual_sources=visual_sources or [],
    )


def _record(
    test_data_dir: Path,
    *,
    conversation_id: str,
    knowledge_base_id: str,
    content: str,
    citations: list[RAGCitation],
    safety_gate: str = "none",
    fact_status: str | None = "source_verified",
    trace: list[RAGAgentToolTraceData] | None = None,
    notice: str | None = None,
) -> str:
    trust_id = record_streamed_answer(
        _settings(test_data_dir),
        conversation_id=UUID(conversation_id),
        knowledge_base_id=UUID(knowledge_base_id),
        content=content,
        citations=citations,
        answer_mode="agent",
        complete=_complete(
            safety_gate=safety_gate,
            fact_status=fact_status,
            trace=trace,
            notice=notice,
        ),
    )
    assert trust_id is not None
    return trust_id


async def _prepare(client: AsyncClient, test_data_dir: Path) -> dict[str, str]:
    knowledge_base_id = await create_knowledge_base(client, "可信回答测试")
    document = await upload_document(client, knowledge_base_id, filename="trusted-table.pdf")
    document_id = str(document["id"])
    canonical = build_canonical(UUID(document_id))
    write_parsing_artifact(test_data_dir, UUID(document_id), canonical=canonical)
    write_evidence(test_data_dir, canonical, knowledge_base_id)
    report = await client.post(
        "/api/v1/reports",
        json={"title": "可信插入报告", "knowledge_base_id": knowledge_base_id},
    )
    assert report.status_code == 201, report.text
    report_body = report.json()
    return {
        "knowledge_base_id": knowledge_base_id,
        "document_id": document_id,
        "report_id": str(report_body["id"]),
        "section_id": str(report_body["sections"][0]["id"]),
    }


async def _create_answer(
    client: AsyncClient,
    knowledge_base_id: str,
    content: str,
    citations: list[RAGCitation],
) -> tuple[str, str]:
    conversation = await client.post(
        "/api/v1/chat/conversations",
        json={"knowledge_base_id": knowledge_base_id},
    )
    assert conversation.status_code == 201, conversation.text
    conversation_id = str(conversation.json()["id"])
    message = await client.post(
        f"/api/v1/chat/conversations/{conversation_id}/messages",
        json={
            "role": "assistant",
            "content": content,
            "citations": [item.model_dump(mode="json") for item in citations],
        },
    )
    assert message.status_code == 201, message.text
    return conversation_id, str(message.json()["id"])


async def _insert(
    client: AsyncClient,
    fixtures: dict[str, str],
    chat_message_id: str,
    **extra: object,
):
    return await client.post(
        f"/api/v1/reports/{fixtures['report_id']}/sections/"
        f"{fixtures['section_id']}/agent-answers/insert",
        json={"chat_message_id": chat_message_id, **extra},
    )


def test_record_and_verify_roundtrip(test_data_dir: Path) -> None:
    settings = _settings(test_data_dir)
    conversation_id = uuid4()
    knowledge_base_id = uuid4()
    citations = [
        _evidence_citation(
            knowledge_base_id=str(knowledge_base_id),
            document_id=str(DOC_UUID),
            evidence_id=evidence_id_for_modality(test_data_dir, DOC_UUID, "table")
            if (test_data_dir / "parsing" / str(DOC_UUID)).exists()
            else EVIDENCE_ID,
        )
    ]
    content = "2024 年越南 GDP 为 4760 亿美元。"
    trust_id = record_streamed_answer(
        settings,
        conversation_id=conversation_id,
        knowledge_base_id=knowledge_base_id,
        content=content,
        citations=citations,
        answer_mode="agent",
        complete=_complete(),
    )
    assert trust_id is not None
    trusted = verify_streamed_answer(
        settings,
        conversation_id=conversation_id,
        knowledge_base_id=knowledge_base_id,
        content=content,
        citations=citations,
    )
    assert trusted is not None
    assert trusted.trust_id == trust_id
    assert len(trusted.citations) == 1
    secret = test_data_dir / "agent-trust" / ".trust-secret"
    assert secret.is_file()
    assert secret.stat().st_mode & 0o777 == 0o600


def test_verify_rejects_tampered_content(test_data_dir: Path) -> None:
    settings = _settings(test_data_dir)
    conversation_id = uuid4()
    knowledge_base_id = uuid4()
    citations = [
        _chunk_citation(knowledge_base_id=str(knowledge_base_id), document_id=str(DOC_UUID))
    ]
    assert record_streamed_answer(
        settings,
        conversation_id=conversation_id,
        knowledge_base_id=knowledge_base_id,
        content="可信回答",
        citations=citations,
        answer_mode="agent",
        complete=_complete(),
    )
    assert (
        verify_streamed_answer(
            settings,
            conversation_id=conversation_id,
            knowledge_base_id=knowledge_base_id,
            content="可信回答 120 万元",
            citations=citations,
        )
        is None
    )


def test_verify_rejects_tampered_citations(test_data_dir: Path) -> None:
    settings = _settings(test_data_dir)
    conversation_id = uuid4()
    knowledge_base_id = uuid4()
    citations = [
        _chunk_citation(knowledge_base_id=str(knowledge_base_id), document_id=str(DOC_UUID))
    ]
    assert record_streamed_answer(
        settings,
        conversation_id=conversation_id,
        knowledge_base_id=knowledge_base_id,
        content="可信回答",
        citations=citations,
        answer_mode="agent",
        complete=_complete(),
    )
    tampered = [
        *citations,
        _evidence_citation(
            knowledge_base_id=str(knowledge_base_id),
            document_id=str(DOC_UUID),
            marker="C2",
        ),
    ]
    assert (
        verify_streamed_answer(
            settings,
            conversation_id=conversation_id,
            knowledge_base_id=knowledge_base_id,
            content="可信回答",
            citations=tampered,
        )
        is None
    )


def test_verify_rejects_wrong_scope_and_signature(test_data_dir: Path) -> None:
    settings = _settings(test_data_dir)
    conversation_id = uuid4()
    knowledge_base_id = uuid4()
    citations = [
        _chunk_citation(knowledge_base_id=str(knowledge_base_id), document_id=str(DOC_UUID))
    ]
    trust_id = record_streamed_answer(
        settings,
        conversation_id=conversation_id,
        knowledge_base_id=knowledge_base_id,
        content="可信回答",
        citations=citations,
        answer_mode="agent",
        complete=_complete(),
    )
    assert trust_id
    assert (
        verify_streamed_answer(
            settings,
            conversation_id=uuid4(),
            knowledge_base_id=knowledge_base_id,
            content="可信回答",
            citations=citations,
        )
        is None
    )
    record_path = test_data_dir / "agent-trust" / f"{trust_id}.json"
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    payload["payload"]["content_sha256"] = "f" * 64
    record_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    assert (
        verify_streamed_answer(
            settings,
            conversation_id=conversation_id,
            knowledge_base_id=knowledge_base_id,
            content="可信回答",
            citations=citations,
        )
        is None
    )


@pytest.mark.anyio
async def test_forged_assistant_message_is_rejected(
    client: AsyncClient,
    test_data_dir: Path,
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
async def test_trusted_table_answer_can_be_inserted(
    client: AsyncClient,
    test_data_dir: Path,
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
    assert body["sources"][0]["evidence_id"] == evidence_id


@pytest.mark.anyio
async def test_trusted_visual_numeric_answer_is_rejected(
    client: AsyncClient,
    test_data_dir: Path,
) -> None:
    fixtures = await _prepare(client, test_data_dir)
    citation = _evidence_citation(
        knowledge_base_id=fixtures["knowledge_base_id"],
        document_id=fixtures["document_id"],
        modality="image",
        source_block_ids=["p1-b2"],
    )
    content = "Q1 销售额为 120 万元。"
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
        safety_gate="visual_numeric_withheld",
        fact_status="visual_estimate",
        trace=[
            RAGAgentToolTraceData(
                tool="analyze_image",
                status="completed",
                summary="图片理解完成",
                document_id=UUID(fixtures["document_id"]),
                fact_status="visual_estimate",
            )
        ],
        notice="视觉数值未经核验",
    )
    response = await _insert(client, fixtures, message_id)
    assert response.status_code == 409
    assert "未经核验的视觉数值" in response.text


@pytest.mark.anyio
async def test_image_table_extraction_cannot_bypass_gate(
    client: AsyncClient,
    test_data_dir: Path,
) -> None:
    fixtures = await _prepare(client, test_data_dir)
    evidence_id = evidence_id_for_modality(test_data_dir, UUID(fixtures["document_id"]), "table")
    citation = _evidence_citation(
        knowledge_base_id=fixtures["knowledge_base_id"],
        document_id=fixtures["document_id"],
        evidence_id=evidence_id,
        modality="table",
    )
    content = "扫描表格显示 2024 年 GDP 为 9999 亿美元。"
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
        safety_gate="unverified_numbers_withheld",
        fact_status="unverified",
        trace=[
            RAGAgentToolTraceData(
                tool="analyze_image",
                status="completed",
                summary="图片表格提取完成，结果需人工核对",
                document_id=UUID(fixtures["document_id"]),
                evidence_id=evidence_id,
                fact_status="unverified",
            )
        ],
    )
    response = await _insert(client, fixtures, message_id)
    assert response.status_code == 409


@pytest.mark.anyio
async def test_trusted_visual_draft_without_numbers_inserts_with_warning(
    client: AsyncClient,
    test_data_dir: Path,
) -> None:
    fixtures = await _prepare(client, test_data_dir)
    citation = _evidence_citation(
        knowledge_base_id=fixtures["knowledge_base_id"],
        document_id=fixtures["document_id"],
        evidence_id=evidence_id_for_modality(test_data_dir, UUID(fixtures["document_id"]), "image"),
        modality="image",
        source_block_ids=["p1-b2"],
    )
    content = "图中展示了光反应和暗反应两个阶段，未给出具体数值。"
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
        safety_gate="visual_unverified",
        fact_status="visual_estimate",
        trace=[
            RAGAgentToolTraceData(
                tool="analyze_image",
                status="completed",
                summary="图片理解完成",
                document_id=UUID(fixtures["document_id"]),
                fact_status="visual_estimate",
            )
        ],
    )
    response = await _insert(client, fixtures, message_id)
    assert response.status_code == 201, response.text
    body = response.json()
    assert "视觉分析草稿" in body["section"]["content"]
    assert any("待核对" in warning for warning in body["warnings"])


@pytest.mark.anyio
async def test_unverified_visual_answer_with_chunk_source_is_rejected(
    client: AsyncClient,
    test_data_dir: Path,
) -> None:
    fixtures = await _prepare(client, test_data_dir)
    citation = _chunk_citation(
        knowledge_base_id=fixtures["knowledge_base_id"],
        document_id=fixtures["document_id"],
    )
    content = "图中 Q1 销售额约为 120 万元。"
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
        safety_gate="vision_unavailable",
        fact_status="not_available",
        trace=[
            RAGAgentToolTraceData(
                tool="analyze_image",
                status="failed",
                summary="无法连接视觉模型服务",
                document_id=UUID(fixtures["document_id"]),
                fact_status="not_available",
                error_code="provider_connection",
            )
        ],
    )
    response = await _insert(client, fixtures, message_id)
    assert response.status_code == 409


def test_trusted_record_survives_process_restart(test_data_dir: Path) -> None:
    conversation_id = uuid4()
    knowledge_base_id = uuid4()
    citations = [
        _chunk_citation(knowledge_base_id=str(knowledge_base_id), document_id=str(DOC_UUID))
    ]
    content = "2024 年越南 GDP 为 4760 亿美元。"
    trust_id = _record(
        test_data_dir,
        conversation_id=str(conversation_id),
        knowledge_base_id=str(knowledge_base_id),
        content=content,
        citations=citations,
    )
    restarted_settings = Settings(
        _env_file=None,
        environment="test",
        data_dir=str(test_data_dir),
    )
    verified = verify_streamed_answer(
        restarted_settings,
        conversation_id=conversation_id,
        knowledge_base_id=knowledge_base_id,
        content=content,
        citations=citations,
    )
    assert verified is not None
    assert verified.trust_id == trust_id


def test_concurrent_workers_share_the_same_trust_key(test_data_dir: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    conversation_id = uuid4()
    knowledge_base_id = uuid4()
    citations = [
        _chunk_citation(knowledge_base_id=str(knowledge_base_id), document_id=str(DOC_UUID))
    ]

    def _writer(index: int) -> str:
        settings = Settings(
            _env_file=None,
            environment="test",
            data_dir=str(test_data_dir),
        )
        content = f"并发写入的回答 {index}"
        trust_id = record_streamed_answer(
            settings,
            conversation_id=conversation_id,
            knowledge_base_id=knowledge_base_id,
            content=content,
            citations=citations,
            answer_mode="agent",
            complete=_complete(),
        )
        assert trust_id is not None
        return content

    with ThreadPoolExecutor(max_workers=8) as pool:
        contents = list(pool.map(_writer, range(8)))

    # The key is created exactly once and every record is verifiable afterwards.
    secret_path = test_data_dir / "agent-trust" / ".trust-secret"
    assert len(secret_path.read_text(encoding="ascii").strip()) == 64
    for content in contents:
        verified = verify_streamed_answer(
            _settings(test_data_dir),
            conversation_id=conversation_id,
            knowledge_base_id=knowledge_base_id,
            content=content,
            citations=citations,
        )
        assert verified is not None


def test_secret_file_reader_waits_for_a_complete_value(test_data_dir: Path) -> None:
    import time
    from concurrent.futures import ThreadPoolExecutor

    from knowledge_scope.chat.trust import _read_secret_file

    trust_root = test_data_dir / "agent-trust"
    trust_root.mkdir(parents=True, exist_ok=True)
    secret_path = trust_root / ".trust-secret"
    secret_path.write_text("", encoding="ascii")
    secret = bytes(range(32))
    expected = secret.hex()

    def _writer() -> None:
        time.sleep(0.12)
        secret_path.write_text(expected, encoding="ascii")

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_writer)
        read = _read_secret_file(secret_path)
        future.result()
    assert read == secret


def test_missing_secret_fails_closed(test_data_dir: Path) -> None:
    conversation_id = uuid4()
    knowledge_base_id = uuid4()
    citations = [
        _chunk_citation(knowledge_base_id=str(knowledge_base_id), document_id=str(DOC_UUID))
    ]
    content = "2024 年越南 GDP 为 4760 亿美元。"
    _record(
        test_data_dir,
        conversation_id=str(conversation_id),
        knowledge_base_id=str(knowledge_base_id),
        content=content,
        citations=citations,
    )
    (test_data_dir / "agent-trust" / ".trust-secret").unlink()
    verified = verify_streamed_answer(
        _settings(test_data_dir),
        conversation_id=conversation_id,
        knowledge_base_id=knowledge_base_id,
        content=content,
        citations=citations,
    )
    assert verified is None


def test_repeated_identical_answer_does_not_downgrade_safety_state(test_data_dir: Path) -> None:
    conversation_id = uuid4()
    knowledge_base_id = uuid4()
    citations = [
        _evidence_citation(
            knowledge_base_id=str(knowledge_base_id),
            document_id=str(DOC_UUID),
            modality="image",
        )
    ]
    content = "图表中 Q1 是销售额最低的季度。"
    trace = [
        RAGAgentToolTraceData(
            tool="analyze_image",
            status="completed",
            summary="图片理解完成",
            document_id=DOC_UUID,
            evidence_id=EVIDENCE_ID,
            fact_status="visual_estimate",
        )
    ]
    trust_id = _record(
        test_data_dir,
        conversation_id=str(conversation_id),
        knowledge_base_id=str(knowledge_base_id),
        content=content,
        citations=citations,
        safety_gate="visual_numeric_withheld",
        fact_status="visual_estimate",
        trace=trace,
    )
    # A later generation with the same text/citations but a "verified" state must
    # not replace the more conservative record.
    _record(
        test_data_dir,
        conversation_id=str(conversation_id),
        knowledge_base_id=str(knowledge_base_id),
        content=content,
        citations=citations,
        safety_gate="none",
        fact_status="source_verified",
    )
    verified = verify_streamed_answer(
        _settings(test_data_dir),
        conversation_id=conversation_id,
        knowledge_base_id=knowledge_base_id,
        content=content,
        citations=citations,
    )
    assert verified is not None
    assert verified.trust_id == trust_id
    assert verified.safety_gate == "visual_numeric_withheld"
