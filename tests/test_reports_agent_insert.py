# ruff: noqa: RUF001
from __future__ import annotations

from uuid import UUID

import pytest
from httpx import AsyncClient

from knowledge_scope.evidence.lifecycle import evidence_artifact_path
from knowledge_scope.parsing.models import CanonicalDocument
from test_document_assets import (
    build_canonical,
    create_knowledge_base,
    evidence_id_for_modality,
    upload_document,
    write_evidence,
    write_parsing_artifact,
)


def _source_payload(
    knowledge_base_id: str,
    document: dict[str, object],
    evidence_id: str,
    *,
    page_start: int = 1,
    page_end: int = 1,
    source_block_ids: list[str] | None = None,
) -> dict[str, object]:
    return {
        "kind": "rag_evidence",
        "title": "图 5-14 光合作用过程的示意图",
        "knowledge_base_id": knowledge_base_id,
        "document_id": document["id"],
        "document_title": document["original_filename"],
        "evidence_id": evidence_id,
        "chunk_id": None,
        "page_start": page_start,
        "page_end": page_end,
        "evidence_type": "image",
        "snippet": "图 5-14 光合作用过程的示意图",
        "section_path": ["第 5 章", "光合作用"],
        "source_block_ids": source_block_ids or ["p1-b2"],
    }


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


@pytest.mark.anyio
async def test_insert_agent_answer_persists_content_and_validated_sources(
    client: AsyncClient,
    test_data_dir: object,
) -> None:
    state = await _prepare_report(client, test_data_dir)
    report = state["report"]
    report_id = report["id"]
    section_id = report["sections"][0]["id"]
    evidence_id = evidence_id_for_modality(
        test_data_dir,  # type: ignore[arg-type]
        UUID(str(state["document"]["id"])),
        "image",
    )
    response = await client.post(
        f"/api/v1/reports/{report_id}/sections/{section_id}/agent-answers/insert",
        json={
            "content_append": "根据视觉分析，图片展示了光合作用的结构 [C1]。",
            "sources": [
                _source_payload(str(state["knowledge_base_id"]), state["document"], evidence_id)
            ],
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["warnings"] == []
    assert body["section"]["content"].endswith("图片展示了光合作用的结构 [C1]。")
    assert len(body["sources"]) == 1
    assert body["sources"][0]["evidence_id"] == evidence_id
    assert body["sources"][0]["source_block_ids"] == ["p1-b2"]
    detail = await client.get(f"/api/v1/reports/{report_id}")
    assert detail.status_code == 200
    sources = detail.json()["sections"][0]["sources"]
    assert [source["evidence_id"] for source in sources] == [evidence_id]


@pytest.mark.anyio
async def test_insert_agent_answer_rejects_cross_knowledge_base_source(
    client: AsyncClient,
    test_data_dir: object,
) -> None:
    state = await _prepare_report(client, test_data_dir)
    other_kb = await create_knowledge_base(client, "另一个知识库")
    other_document = await upload_document(client, other_kb, filename="other.pdf")
    evidence_id = evidence_id_for_modality(
        test_data_dir,  # type: ignore[arg-type]
        UUID(str(state["document"]["id"])),
        "image",
    )
    report = state["report"]
    response = await client.post(
        f"/api/v1/reports/{report['id']}/sections/{report['sections'][0]['id']}"
        "/agent-answers/insert",
        json={
            "content_append": "跨库引用",
            "sources": [_source_payload(other_kb, other_document, evidence_id)],
        },
    )
    assert response.status_code == 409
    assert response.json()["detail"] == "请先为报告选择与证据一致的知识库"


@pytest.mark.anyio
async def test_insert_agent_answer_rejects_mismatched_evidence_page(
    client: AsyncClient,
    test_data_dir: object,
) -> None:
    state = await _prepare_report(client, test_data_dir)
    report = state["report"]
    evidence_id = evidence_id_for_modality(
        test_data_dir,  # type: ignore[arg-type]
        UUID(str(state["document"]["id"])),
        "image",
    )
    response = await client.post(
        f"/api/v1/reports/{report['id']}/sections/{report['sections'][0]['id']}"
        "/agent-answers/insert",
        json={
            "content_append": "页码被篡改",
            "sources": [
                _source_payload(
                    str(state["knowledge_base_id"]),
                    state["document"],
                    evidence_id,
                    page_start=2,
                    page_end=2,
                )
            ],
        },
    )
    assert response.status_code == 409
    assert response.json()["detail"] == "证据页码与服务器记录不一致"


@pytest.mark.anyio
async def test_insert_agent_answer_rejects_stale_evidence(
    client: AsyncClient,
    test_data_dir: object,
) -> None:
    state = await _prepare_report(client, test_data_dir)
    report = state["report"]
    document_id = UUID(str(state["document"]["id"]))
    evidence_id = evidence_id_for_modality(test_data_dir, document_id, "image")  # type: ignore[arg-type]
    canonical_path = (
        test_data_dir  # type: ignore[operator]
        / "parsing"
        / str(document_id)
        / "canonical.json"
    )
    stale = CanonicalDocument.model_validate_json(canonical_path.read_bytes())
    changed_text = stale.pages[0].blocks[0].model_copy(update={"text": "内容已被修改"})  # type: ignore[attr-defined]
    changed_blocks = (changed_text, *stale.pages[0].blocks[1:])
    changed = stale.model_copy(
        update={"pages": [stale.pages[0].model_copy(update={"blocks": list(changed_blocks)})]}
    )
    canonical_path.write_text(changed.model_dump_json(indent=2) + "\n")
    response = await client.post(
        f"/api/v1/reports/{report['id']}/sections/{report['sections'][0]['id']}"
        "/agent-answers/insert",
        json={
            "content_append": "过期证据",
            "sources": [
                _source_payload(str(state["knowledge_base_id"]), state["document"], evidence_id)
            ],
        },
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "证据来源无效或已过期，已拒绝插入"


@pytest.mark.anyio
async def test_insert_agent_answer_degrades_for_historical_missing_artifacts(
    client: AsyncClient,
    test_data_dir: object,
) -> None:
    state = await _prepare_report(client, test_data_dir)
    report = state["report"]
    evidence_id = evidence_id_for_modality(
        test_data_dir,  # type: ignore[arg-type]
        UUID(str(state["document"]["id"])),
        "image",
    )
    # Remove parsing/evidence artifacts to emulate a historical document whose
    # source metadata is no longer verifiable.
    evidence_artifact_path(test_data_dir, UUID(str(state["document"]["id"]))).unlink()  # type: ignore[arg-type]
    (test_data_dir / "parsing" / str(state["document"]["id"])).rename(  # type: ignore[operator]
        test_data_dir / "parsing" / "removed"  # type: ignore[operator]
    )
    response = await client.post(
        f"/api/v1/reports/{report['id']}/sections/{report['sections'][0]['id']}"
        "/agent-answers/insert",
        json={
            "content_append": "历史引用",
            "sources": [
                _source_payload(str(state["knowledge_base_id"]), state["document"], evidence_id)
            ],
        },
    )
    assert response.status_code == 201, response.text
    assert len(response.json()["warnings"]) == 1
    assert "历史文档缺少解析产物" in response.json()["warnings"][0]


@pytest.mark.anyio
async def test_insert_agent_answer_requires_at_least_one_source(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/reports/not-a-uuid/sections/not-a-uuid/agent-answers/insert",
        json={"content_append": "没有来源", "sources": []},
    )
    assert response.status_code == 422


@pytest.mark.anyio
async def test_insert_agent_answer_accepts_chunk_source_without_evidence_claim(
    client: AsyncClient,
    test_data_dir: object,
) -> None:
    """Chunk-only sources are allowed, but must not claim Evidence-level validation."""
    state = await _prepare_report(client, test_data_dir)
    report = state["report"]
    response = await client.post(
        f"/api/v1/reports/{report['id']}/sections/{report['sections'][0]['id']}/agent-answers/insert",
        json={
            "content_append": "该结论来自文本片段 [C1]。",
            "sources": [
                {
                    "kind": "rag_evidence",
                    "title": "文本片段 #12",
                    "knowledge_base_id": state["knowledge_base_id"],
                    "document_id": state["document"]["id"],
                    "document_title": state["document"]["original_filename"],
                    "evidence_id": None,
                    "chunk_id": "chunk-000012",
                    "page_start": 1,
                    "page_end": 1,
                    "evidence_type": "text",
                    "snippet": "来自文本片段的引用。",
                    "section_path": ["第 5 章"],
                    "source_block_ids": [],
                }
            ],
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["warnings"] == []
    assert body["sources"][0]["evidence_id"] is None
    assert body["sources"][0]["chunk_id"] == "chunk-000012"
