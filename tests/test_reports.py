from uuid import uuid4

import pytest
from httpx import AsyncClient


async def _create_knowledge_base(client: AsyncClient, name: str) -> dict[str, object]:
    response = await client.post("/api/v1/knowledge-bases", json={"name": name})
    assert response.status_code == 201
    return response.json()


async def _create_datasource(client: AsyncClient, name: str) -> dict[str, object]:
    response = await client.post(
        "/api/v1/chatbi/data-sources",
        json={
            "display_name": name,
            "connection_ref": "env:CHATBI_DEMO_DATABASE_URL",
        },
    )
    assert response.status_code == 201
    return response.json()


@pytest.mark.anyio
async def test_report_and_section_lifecycle_is_persistent_and_ordered(client: AsyncClient) -> None:
    knowledge_base = await _create_knowledge_base(client, "报告资料")
    datasource = await _create_datasource(client, "业务数据")

    created_response = await client.post(
        "/api/v1/reports",
        json={
            "title": "  月度经营报告  ",
            "knowledge_base_id": knowledge_base["id"],
            "datasource_id": datasource["id"],
        },
    )
    assert created_response.status_code == 201
    created = created_response.json()
    assert created["title"] == "月度经营报告"
    assert [section["title"] for section in created["sections"]] == ["概览"]

    section_response = await client.post(
        f"/api/v1/reports/{created['id']}/sections",
        json={"title": "结论", "content": "初稿", "position": 0},
    )
    assert section_response.status_code == 201
    section = section_response.json()
    assert section["position"] == 0

    update_section_response = await client.patch(
        f"/api/v1/reports/{created['id']}/sections/{section['id']}",
        json={"title": "执行摘要", "content": "已保存内容", "position": 1},
    )
    assert update_section_response.status_code == 200
    assert update_section_response.json()["title"] == "执行摘要"
    assert update_section_response.json()["position"] == 1

    update_report_response = await client.patch(
        f"/api/v1/reports/{created['id']}",
        json={"title": "月度经营报告 - 更新"},
    )
    assert update_report_response.status_code == 200
    assert update_report_response.json()["title"] == "月度经营报告 - 更新"

    detail_response = await client.get(f"/api/v1/reports/{created['id']}")
    assert detail_response.status_code == 200
    assert [section["title"] for section in detail_response.json()["sections"]] == [
        "概览",
        "执行摘要",
    ]

    source_response = await client.post(
        f"/api/v1/reports/{created['id']}/sections/{created['sections'][0]['id']}/sources",
        json={
            "kind": "chatbi_result",
            "title": "报告删除级联来源",
            "datasource_id": datasource["id"],
            "datasource_name": "业务数据",
            "question": "统计数量",
            "answer": "3",
            "columns": [],
            "rows": [],
            "row_count": 1,
            "truncated": False,
            "truncation_reason": None,
        },
    )
    assert source_response.status_code == 201
    source_id = source_response.json()["id"]

    delete_section_response = await client.delete(
        f"/api/v1/reports/{created['id']}/sections/{section['id']}"
    )
    assert delete_section_response.status_code == 204

    knowledge_base_delete_response = await client.delete(
        f"/api/v1/knowledge-bases/{knowledge_base['id']}"
    )
    assert knowledge_base_delete_response.status_code == 409
    datasource_delete_response = await client.delete(
        f"/api/v1/chatbi/data-sources/{datasource['id']}"
    )
    assert datasource_delete_response.status_code == 409
    assert (await client.get(f"/api/v1/reports/{created['id']}")).status_code == 200

    delete_report_response = await client.delete(f"/api/v1/reports/{created['id']}")
    assert delete_report_response.status_code == 204
    assert (await client.get(f"/api/v1/reports/{created['id']}")).status_code == 404
    assert (await client.get("/api/v1/reports")).json()["items"] == []
    assert (
        await client.patch(
            f"/api/v1/reports/{created['id']}/sections/{created['sections'][0]['id']}",
            json={"content": "不应存在"},
        )
    ).status_code == 404
    assert (await client.get(f"/api/v1/knowledge-bases/{knowledge_base['id']}")).status_code == 200
    assert (await client.get(f"/api/v1/chatbi/data-sources/{datasource['id']}")).status_code == 200
    # A non-null report_id source can only be removed by the database CASCADE.
    assert (
        await client.delete(f"/api/v1/reports/{created['id']}/sources/{source_id}")
    ).status_code == 404
    assert (
        await client.delete(f"/api/v1/knowledge-bases/{knowledge_base['id']}")
    ).status_code == 204
    assert (
        await client.delete(f"/api/v1/chatbi/data-sources/{datasource['id']}")
    ).status_code == 204


@pytest.mark.anyio
async def test_report_sources_are_typed_scoped_and_bounded(client: AsyncClient) -> None:
    knowledge_base = await _create_knowledge_base(client, "证据资料")
    datasource = await _create_datasource(client, "分析数据")
    report = (
        await client.post(
            "/api/v1/reports",
            json={
                "title": "来源报告",
                "knowledge_base_id": knowledge_base["id"],
                "datasource_id": datasource["id"],
            },
        )
    ).json()
    section_id = report["sections"][0]["id"]

    source_response = await client.post(
        f"/api/v1/reports/{report['id']}/sections/{section_id}/sources",
        json={
            "kind": "chatbi_result",
            "title": "地区客户数",
            "datasource_id": datasource["id"],
            "datasource_name": "分析数据",
            "question": "统计各地区客户数",
            "answer": "华东最多",
            "columns": [
                {"name": "地区", "data_type": "text", "nullable": False, "ordinal": 0},
                {"name": "数量", "data_type": "bigint", "nullable": False, "ordinal": 1},
            ],
            "rows": [["华东", 3], ["华南", 2]],
            "row_count": 2,
            "truncated": False,
            "truncation_reason": None,
            "redacted_sql": (
                "SELECT region, COUNT(*) FROM public.sales "
                "WHERE customer_name = 'secret' GROUP BY region"
            ),
            "chart_spec": None,
        },
    )
    assert source_response.status_code == 201
    source = source_response.json()
    assert source["kind"] == "chatbi_result"
    assert source["rows"] == [["华东", 3], ["华南", 2]]
    assert source["redacted_sql"].startswith("SELECT")
    assert "secret" not in source["redacted_sql"]

    insert_response = await client.post(
        f"/api/v1/reports/{report['id']}/sections/{section_id}/sources/insert",
        json={
            "source": {
                "kind": "chatbi_result",
                "title": "地区客户数插入",
                "datasource_id": datasource["id"],
                "datasource_name": "分析数据",
                "question": "统计各地区客户数",
                "answer": "华东最多",
                "columns": [],
                "rows": [],
                "row_count": 0,
                "truncated": False,
                "truncation_reason": None,
            },
            "content_append": "【地区客户数插入】\n华东最多",
        },
    )
    assert insert_response.status_code == 201

    oversized_insert_response = await client.post(
        f"/api/v1/reports/{report['id']}/sections/{section_id}/sources/insert",
        json={
            "source": {
                "kind": "chatbi_result",
                "title": "不应保存",
                "datasource_id": datasource["id"],
                "datasource_name": "分析数据",
                "question": "统计各地区客户数",
                "columns": [],
                "rows": [],
                "row_count": 0,
                "truncated": False,
                "truncation_reason": None,
            },
            "content_append": "x" * 100_000,
        },
    )
    assert oversized_insert_response.status_code == 422

    mismatched_datasource = await client.post(
        f"/api/v1/reports/{report['id']}/sections/{section_id}/sources",
        json={
            "kind": "chatbi_result",
            "title": "错误数据源",
            "datasource_id": str(uuid4()),
            "datasource_name": "其他数据源",
            "question": "统计数量",
            "columns": [],
            "rows": [],
            "row_count": 0,
            "truncated": False,
            "truncation_reason": None,
        },
    )
    assert mismatched_datasource.status_code == 409

    missing_document = await client.post(
        f"/api/v1/reports/{report['id']}/sections/{section_id}/sources",
        json={
            "kind": "rag_evidence",
            "title": "不存在的证据",
            "knowledge_base_id": knowledge_base["id"],
            "document_id": str(uuid4()),
            "chunk_id": "chunk-1",
            "document_title": "不存在.pdf",
            "page_start": 1,
            "page_end": 1,
            "evidence_type": "text",
            "snippet": "不应保存",
        },
    )
    assert missing_document.status_code == 404

    document_response = await client.post(
        f"/api/v1/knowledge-bases/{knowledge_base['id']}/documents",
        files={
            "file": (
                "质量手册.pdf",
                b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<<>>\n%%EOF\n",
                "application/pdf",
            )
        },
    )
    assert document_response.status_code == 201
    document_id = document_response.json()["id"]
    rag_response = await client.post(
        f"/api/v1/reports/{report['id']}/sections/{section_id}/sources",
        json={
            "kind": "rag_evidence",
            "title": "质量控制证据",
            "knowledge_base_id": knowledge_base["id"],
            "document_id": document_id,
            "evidence_id": "evidence-1",
            "document_title": "质量手册.pdf",
            "page_start": 1,
            "page_end": 1,
            "evidence_type": "text",
            "snippet": "保留来源块链路",
            "section_path": ["质量控制"],
            "source_block_ids": ["block-1"],
        },
    )
    assert rag_response.status_code == 201
    assert rag_response.json()["source_block_ids"] == ["block-1"]

    oversized_section_path = await client.post(
        f"/api/v1/reports/{report['id']}/sections/{section_id}/sources",
        json={
            "kind": "rag_evidence",
            "title": "过长章节路径",
            "knowledge_base_id": knowledge_base["id"],
            "document_id": document_id,
            "chunk_id": "chunk-2",
            "document_title": "质量手册.pdf",
            "page_start": 1,
            "page_end": 1,
            "section_path": ["x" * 256],
        },
    )
    assert oversized_section_path.status_code == 422

    oversized_block_id = await client.post(
        f"/api/v1/reports/{report['id']}/sections/{section_id}/sources",
        json={
            "kind": "rag_evidence",
            "title": "过长来源块",
            "knowledge_base_id": knowledge_base["id"],
            "document_id": document_id,
            "chunk_id": "chunk-3",
            "document_title": "质量手册.pdf",
            "page_start": 1,
            "page_end": 1,
            "source_block_ids": ["x" * 256],
        },
    )
    assert oversized_block_id.status_code == 422

    detail = (await client.get(f"/api/v1/reports/{report['id']}")).json()
    assert len(detail["sections"][0]["sources"]) == 3
    sources = detail["sections"][0]["sources"]
    assert sources == sorted(sources, key=lambda source: (source["created_at"], source["id"]))
    repeated_detail = (await client.get(f"/api/v1/reports/{report['id']}")).json()
    assert [source["id"] for source in repeated_detail["sections"][0]["sources"]] == [
        source["id"] for source in sources
    ]
    rag_source = next(source for source in sources if source["title"] == "质量控制证据")
    assert rag_source["source_block_ids"] == ["block-1"]
    assert "地区客户数插入" in detail["sections"][0]["content"]

    delete_source_response = await client.delete(
        f"/api/v1/reports/{report['id']}/sources/{source['id']}"
    )
    assert delete_source_response.status_code == 204
