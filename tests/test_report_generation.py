# Chinese API messages are intentionally asserted here.
# ruff: noqa: RUF001

from __future__ import annotations

import json
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pytest
from httpx import AsyncClient

from conftest import _test_client
from knowledge_scope.llm.schemas import LLMRequest, LLMResult
from knowledge_scope.reports.export import (
    REPORT_EXPORT_MAX_BODY_CHARS,
    REPORT_EXPORT_MAX_SECTIONS,
    REPORT_EXPORT_MAX_SOURCE_REFERENCES,
    ReportExportError,
    _report_lines,
    build_docx,
    build_pdf,
)
from knowledge_scope.reports.schemas import ReportDetailResponse


class _FakeGateway:
    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = list(responses)
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResult:
        self.requests.append(request)
        return LLMResult(
            text=self._responses.pop(0),
            provider="fake",
            model="fake-model",
            input_tokens=11,
            output_tokens=7,
            latency_ms=1.0,
            finish_reason="stop",
        )


async def _create_report(client: AsyncClient, *, with_source: bool = False) -> dict[str, Any]:
    datasource_id: str | None = None
    if with_source:
        datasource = await client.post(
            "/api/v1/chatbi/data-sources",
            json={
                "display_name": "报告数据",
                "connection_ref": "env:CHATBI_DEMO_DATABASE_URL",
            },
        )
        assert datasource.status_code == 201
        datasource_id = datasource.json()["id"]
    response = await client.post(
        "/api/v1/reports",
        json={"title": "季度报告", "datasource_id": datasource_id},
    )
    assert response.status_code == 201
    report = response.json()
    if with_source:
        section_id = report["sections"][0]["id"]
        source = await client.post(
            f"/api/v1/reports/{report['id']}/sections/{section_id}/sources",
            json={
                "kind": "chatbi_result",
                "title": "地区客户数",
                "datasource_id": datasource_id,
                "datasource_name": "报告数据",
                "question": "统计地区客户数",
                "columns": [{"name": "地区", "data_type": "text", "nullable": False, "ordinal": 0}],
                "rows": [["华东"]],
                "row_count": 1,
                "truncated": False,
                "truncation_reason": None,
            },
        )
        assert source.status_code == 201
        report = (await client.get(f"/api/v1/reports/{report['id']}")).json()
    return report


@pytest.mark.anyio
async def test_report_ai_preview_is_unpersisted_and_citations_are_mapped(
    postgres_test_database: str,
    postgres_test_engine: Any,
    test_data_dir: Path,
) -> None:
    gateway = _FakeGateway(
        [
            '{"items":[{"title":"经营概览","summary":"整理当前报告的主要事实"}]}',
            '{"content":"本章节引用了 [R1]。","citations":["R1"]}',
        ]
    )
    async with _test_client(
        postgres_test_database,
        postgres_test_engine,
        test_data_dir,
        llm_gateway=gateway,
    ) as client:
        report = await _create_report(client, with_source=True)
        section_id = report["sections"][0]["id"]
        outline = await client.post(
            f"/api/v1/reports/{report['id']}/ai/outline",
            json={"topic": "季度报告", "instruction": "保持简洁"},
        )
        assert outline.status_code == 200
        assert outline.json()["items"][0]["title"] == "经营概览"

        draft = await client.post(
            f"/api/v1/reports/{report['id']}/sections/{section_id}/ai/generate",
            json={},
        )
        assert draft.status_code == 200
        source_id = report["sections"][0]["sources"][0]["id"]
        stable_marker = f"S-{source_id.replace('-', '')[:12].upper()}"
        assert draft.json()["content"] == f"本章节引用了 [{stable_marker}]。"
        assert draft.json()["citations"][0]["marker"].startswith("S-")

        detail = await client.get(f"/api/v1/reports/{report['id']}")
        assert detail.json()["sections"][0]["content"] == ""
        assert [request.task_type for request in gateway.requests] == [
            "report_generation",
            "report_generation",
        ]
        assert all(request.response_format is not None for request in gateway.requests)
        assert all(request.reasoning == "disabled" for request in gateway.requests)
        assert all(request.max_tokens == 2_048 for request in gateway.requests)


@pytest.mark.anyio
async def test_report_ai_rejects_invalid_output_without_leaking_provider_text(
    postgres_test_database: str,
    postgres_test_engine: Any,
    test_data_dir: Path,
) -> None:
    gateway = _FakeGateway(['{"content":"bad [R9]","citations":["R9"]}'])
    async with _test_client(
        postgres_test_database,
        postgres_test_engine,
        test_data_dir,
        llm_gateway=gateway,
    ) as client:
        report = await _create_report(client)
        section_id = report["sections"][0]["id"]
        response = await client.post(
            f"/api/v1/reports/{report['id']}/sections/{section_id}/ai/generate",
            json={},
        )
        assert response.status_code == 502
        assert response.json() == {"detail": "报告生成结果未通过格式校验，请重试"}
        assert "R9" not in response.text


@pytest.mark.anyio
@pytest.mark.parametrize(
    "content",
    ["bad [R13]", "bad [R999]", "bad [Rabc]", "bad [R1"],
)
async def test_report_ai_rejects_unknown_or_malformed_citation_markers(
    postgres_test_database: str,
    postgres_test_engine: Any,
    test_data_dir: Path,
    content: str,
) -> None:
    gateway = _FakeGateway([json.dumps({"content": content, "citations": []})])
    async with _test_client(
        postgres_test_database,
        postgres_test_engine,
        test_data_dir,
        llm_gateway=gateway,
    ) as client:
        report = await _create_report(client)
        section_id = report["sections"][0]["id"]
        response = await client.post(
            f"/api/v1/reports/{report['id']}/sections/{section_id}/ai/generate",
            json={},
        )
        assert response.status_code == 502
        assert "R" not in response.text


@pytest.mark.anyio
async def test_report_exports_are_server_generated_with_safe_download_headers(
    client: AsyncClient,
) -> None:
    report = await _create_report(client, with_source=True)
    for extension, media_type in (
        ("docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        ("pdf", "application/pdf"),
    ):
        response = await client.get(f"/api/v1/reports/{report['id']}/export/{extension}")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith(media_type)
        assert "filename*=" in response.headers["content-disposition"]
        assert "data/" not in response.headers["content-disposition"]
        assert len(response.content) > 100
        if extension == "docx":
            source_id = report["sections"][0]["sources"][0]["id"]
            document_xml = ZipFile(BytesIO(response.content)).read("word/document.xml")
            assert source_id.encode() not in document_xml
            assert b"S-" in document_xml

    detail = ReportDetailResponse.model_validate(
        (await client.get(f"/api/v1/reports/{report['id']}")).json()
    )
    lines = list(_report_lines(detail))
    header_index = lines.index("地区")
    row_index = lines.index("华东")
    assert header_index < row_index


def _report_detail_for_export_bounds(
    *,
    section_count: int = 1,
    references_per_section: int = 0,
    content: str = "正文",
    table_columns: int = 0,
    table_rows: int = 0,
) -> ReportDetailResponse:
    from uuid import uuid4

    sections = []
    for _ in range(section_count):
        section_id = uuid4()
        sources = []
        for _ in range(references_per_section):
            sources.append(
                {
                    "id": uuid4(),
                    "section_id": section_id,
                    "kind": "chatbi_result",
                    "title": "来源",
                    "knowledge_base_id": None,
                    "datasource_id": None,
                    "document_id": None,
                    "document_title": "数据源",
                    "evidence_id": None,
                    "chunk_id": None,
                    "page_start": None,
                    "page_end": None,
                    "evidence_type": None,
                    "snippet": None,
                    "section_path": [],
                    "source_block_ids": [],
                    "question": "问题",
                    "answer": "回答",
                    "columns": [
                        {
                            "name": f"列{index}",
                            "data_type": "text",
                            "nullable": True,
                            "ordinal": index,
                        }
                        for index in range(table_columns)
                    ],
                    "rows": [["值"] * table_columns for _ in range(table_rows)],
                    "row_count": table_rows,
                    "truncated": False,
                    "truncation_reason": None,
                    "redacted_sql": None,
                    "chart_spec": None,
                    "created_at": "2026-01-01T00:00:00Z",
                }
            )
        sections.append(
            {
                "id": section_id,
                "report_id": uuid4(),
                "title": "章节",
                "content": content,
                "position": len(sections),
                "sources": sources,
                "created_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
            }
        )
    return ReportDetailResponse.model_validate(
        {
            "id": uuid4(),
            "title": "报告",
            "knowledge_base_id": None,
            "datasource_id": None,
            "sections": sections,
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
        }
    )


@pytest.mark.parametrize(
    "builder, detail",
    [
        (
            build_docx,
            _report_detail_for_export_bounds(section_count=REPORT_EXPORT_MAX_SECTIONS + 1),
        ),
        (
            build_pdf,
            _report_detail_for_export_bounds(section_count=REPORT_EXPORT_MAX_SECTIONS + 1),
        ),
        (
            build_docx,
            _report_detail_for_export_bounds(
                references_per_section=REPORT_EXPORT_MAX_SOURCE_REFERENCES + 1,
            ),
        ),
        (
            build_pdf,
            _report_detail_for_export_bounds(
                references_per_section=REPORT_EXPORT_MAX_SOURCE_REFERENCES + 1,
            ),
        ),
        (
            build_docx,
            _report_detail_for_export_bounds(
                section_count=11,
                content="正文" * (REPORT_EXPORT_MAX_BODY_CHARS // 10),
            ),
        ),
        (
            build_pdf,
            _report_detail_for_export_bounds(
                section_count=11,
                content="正文" * (REPORT_EXPORT_MAX_BODY_CHARS // 10),
            ),
        ),
        (
            build_docx,
            _report_detail_for_export_bounds(
                references_per_section=11,
                table_columns=50,
                table_rows=100,
            ),
        ),
        (
            build_pdf,
            _report_detail_for_export_bounds(
                references_per_section=11,
                table_columns=50,
                table_rows=100,
            ),
        ),
    ],
)
def test_report_export_rejects_aggregate_bounds(builder: Any, detail: ReportDetailResponse) -> None:
    with pytest.raises(ReportExportError, match="aggregate_bounds"):
        builder(detail)
