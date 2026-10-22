"""Deterministic table parsing, calculation and HTTP contract tests for M2."""

# ruff: noqa: RUF001

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from httpx import AsyncClient

from knowledge_scope.parsing.models import (
    CanonicalDocument,
    Page,
    TableBlock,
)
from knowledge_scope.parsing.service import PARSING_DIRECTORY_NAME
from knowledge_scope.table_analysis.models import (
    TABLE_ANALYSIS_SCHEMA_VERSION,
    AggregateParameters,
    CompareParameters,
    FilterParameters,
    GrowthParameters,
    SelectParameters,
    SortParameters,
    StructuredTable,
    TableNotComputableError,
    TableOperationError,
)
from knowledge_scope.table_analysis.operations import (
    run_aggregate,
    run_compare,
    run_filter,
    run_growth,
    run_select,
    run_sort,
)
from knowledge_scope.table_analysis.parsing import (
    detect_unit_hint,
    parse_html_table,
    parse_markdown_table,
)
from test_document_assets import (
    SECOND_PDF,
    VALID_PDF,
    create_knowledge_base,
    upload_document,
    write_evidence,
)

KNOWLEDGE_BASE_ID = UUID("11111111-1111-4111-8111-111111111111")
DOCUMENT_ID = UUID("22222222-2222-4222-8222-222222222222")
SOURCE_FINGERPRINT = "a" * 64
EVIDENCE_ID = "evidence_v1_" + "b" * 64

UNIT_CAPTION = "表 1 各国 GDP 单位：亿美元"

COUNTRY_TABLE_HTML = """
<table>
  <tr><th>国家</th><th>2023年GDP</th><th>2024年GDP</th></tr>
  <tr><td>越南</td><td>4300</td><td>4760</td></tr>
  <tr><td>泰国</td><td>5000</td><td>5200</td></tr>
  <tr><td>文莱</td><td>0</td><td>50</td></tr>
</table>
"""

COUNTRY_TABLE_MARKDOWN = """| 国家 | 2023年GDP | 2024年GDP |
| --- | --- | --- |
| 越南 | 4300 | 4760 |
| 泰国 | 5000 | 5200 |
| 文莱 | 0 | 50 |"""

COMPARE_TABLE_MARKDOWN = """| 国家 | 2023年GDP | 2024年GDP |
| --- | --- | --- |
| 越南 | 4400 | 4700 |
| 泰国 | 5100 | 5300 |
| 文莱 | 10 | 60 |"""

PERCENT_TABLE_MARKDOWN = """| 指标 | 2022年 | 2023年 |
| --- | --- | --- |
| 增长率 | 10% | 12.5% |
| 通胀率 | - | 0 |"""


def wrap_table(parsed: Any) -> StructuredTable:
    """Wrap already-parsed cell data into the standalone structured-table model."""
    return StructuredTable(
        schema_version=TABLE_ANALYSIS_SCHEMA_VERSION,
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        document_id=DOCUMENT_ID,
        evidence_id=EVIDENCE_ID,
        page_start=2,
        page_end=2,
        source_block_ids=["p2-b1"],
        section_path=["第三章", "经济数据"],
        source_fingerprint=SOURCE_FINGERPRINT,
        caption=parsed.caption,
        unit_hint=parsed.unit_hint,
        columns=parsed.columns,
        rows=parsed.rows,
        row_labels=parsed.row_labels,
        warnings=parsed.warnings,
        computable=True,
    )


def country_table() -> StructuredTable:
    return wrap_table(parse_html_table(COUNTRY_TABLE_HTML, unit_hint="亿美元"))


def percent_table() -> StructuredTable:
    return wrap_table(parse_markdown_table(PERCENT_TABLE_MARKDOWN))


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_html_table_identifies_headers_units_and_values() -> None:
    parsed = parse_html_table(COUNTRY_TABLE_HTML, caption=UNIT_CAPTION, unit_hint="亿美元")

    assert [column.header for column in parsed.columns] == [
        "国家",
        "2023年GDP",
        "2024年GDP",
    ]
    assert parsed.columns[1].kind == "number"
    assert parsed.columns[1].unit == "亿美元"
    assert parsed.rows[0][0].text == "越南"
    assert parsed.rows[0][1].value == "4300"
    assert parsed.rows[0][2].value == "4760"
    assert parsed.row_labels == ["越南", "泰国", "文莱"]


def test_parse_html_table_normalizes_colspan_and_rowspan_headers() -> None:
    html = """
    <table>
      <tr><th rowspan="2">国家</th><th colspan="2">GDP（亿美元）</th></tr>
      <tr><th>2023年</th><th>2024年</th></tr>
      <tr><td>越南</td><td>4300</td><td>4760</td></tr>
    </table>
    """
    parsed = parse_html_table(html)

    assert [column.header for column in parsed.columns] == [
        "国家",
        "GDP（亿美元） / 2023年",
        "GDP（亿美元） / 2024年",
    ]
    assert "2023年" in parsed.columns[1].header
    assert parsed.rows[0][1].value == "4300"
    assert parsed.rows[0][2].value == "4760"


def test_parse_html_table_rejects_unreconstructable_data_merge() -> None:
    html = """
    <table>
      <tr><th>国家</th><th>2023年GDP</th></tr>
      <tr><td>越南</td><td rowspan="2">4300</td></tr>
      <tr><td>泰国</td></tr>
    </table>
    """
    with pytest.raises(TableNotComputableError) as error:
        parse_html_table(html)

    assert error.value.code == "complex_merge"


def test_parse_markdown_table_supports_percent_missing_and_zero() -> None:
    parsed = parse_markdown_table(PERCENT_TABLE_MARKDOWN)

    assert parsed.columns[1].kind == "percent"
    assert parsed.rows[0][1].value == "10"
    assert parsed.rows[0][2].value == "12.5"
    assert parsed.rows[1][1].value is None
    assert parsed.rows[1][2].value == "0"
    assert parsed.row_labels == ["增长率", "通胀率"]
    assert detect_unit_hint("表 2 单位：%") == "%"


def test_parse_number_handles_thousands_separators_and_accounting_negatives() -> None:
    html = """
    <table>
      <tr><th>项目</th><th>金额</th></tr>
      <tr><td>收入</td><td>1,234.5</td></tr>
      <tr><td>亏损</td><td>(1,234)</td></tr>
    </table>
    """
    parsed = parse_html_table(html)

    assert parsed.rows[0][1].value == "1234.5"
    assert parsed.rows[1][1].value == "-1234"


def test_parse_table_without_numeric_data_is_not_computable() -> None:
    html = "<table><tr><th>名称</th></tr><tr><td>说明文字</td></tr></table>"
    with pytest.raises(TableNotComputableError) as error:
        parse_html_table(html)

    assert error.value.code == "no_numeric_data"


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


def test_select_returns_parsed_numeric_values() -> None:
    outcome = run_select(country_table(), SelectParameters(columns=["国家", "2024年GDP"], rows=[0]))

    assert outcome.status == "ok"
    assert [column.name for column in outcome.columns] == ["国家", "2024年GDP"]
    assert outcome.rows[0].cells == ["越南", "4760"]
    assert outcome.units == ["亿美元"]


def test_filter_and_sort_are_deterministic() -> None:
    filtered = run_filter(
        country_table(),
        FilterParameters(column="2024年GDP", operator="gte", value="5000"),
    )
    assert [row.cells[0] for row in filtered.rows] == ["泰国"]

    sorted_outcome = run_sort(
        country_table(),
        SortParameters(column="2024年GDP", direction="desc", limit=2),
    )
    assert [row.cells[0] for row in sorted_outcome.rows] == ["泰国", "越南"]


def test_aggregate_sum_average_min_max() -> None:
    outcome = run_aggregate(
        country_table(),
        AggregateParameters(column="2024年GDP", functions=["sum", "avg", "min", "max"]),
    )

    assert outcome.status == "ok"
    assert outcome.rows[0].cells == ["10010", "3336.666666666666666666666667", "50", "5200"]
    assert "sum(2024年GDP) = 10010" in (outcome.formula or "")


def test_growth_matches_expected_year_over_year_rate() -> None:
    outcome = run_growth(
        country_table(),
        GrowthParameters(
            axis="columns",
            row="越南",
            base_column="2023年GDP",
            target_column="2024年GDP",
        ),
    )

    assert outcome.status == "ok"
    assert outcome.rows[0].cells == ["越南", "4300", "4760", "460", "10.6977"]
    assert "10.6977%" in (outcome.formula or "")


def test_growth_rejects_zero_base() -> None:
    with pytest.raises(TableOperationError) as error:
        run_growth(
            country_table(),
            GrowthParameters(
                axis="columns",
                row="文莱",
                base_column="2023年GDP",
                target_column="2024年GDP",
            ),
        )

    assert error.value.code == "division_by_zero"


def test_growth_on_percent_column_is_reported_in_percentage_points() -> None:
    percent_growth = parse_markdown_table(
        "| 年份 | 增长率 |\n| --- | --- |\n| 2022年 | 10% |\n| 2023年 | 12.5% |"
    )
    outcome = run_growth(
        wrap_table(percent_growth),
        GrowthParameters(axis="rows", column="增长率", base_row="2022年", target_row="2023年"),
    )

    assert outcome.rows[0].cells == ["增长率", "10", "12.5", "2.5", "25.0000"]
    assert outcome.inputs["change_kind"] == "percentage_point"


def test_compare_aligns_keys_and_computes_differences() -> None:
    outcome = run_compare(
        country_table(),
        wrap_table(parse_markdown_table(COMPARE_TABLE_MARKDOWN, unit_hint="亿美元")),
        CompareParameters(
            key_column="国家",
            value_column="2024年GDP",
            keys=["越南", "泰国"],
        ),
    )

    assert outcome.status == "ok"
    rows = {row.cells[0]: row.cells[1:] for row in outcome.rows}
    assert rows["越南"] == ["4760", "4700", "60", "1.2766"]
    assert rows["泰国"] == ["5200", "5300", "-100", "-1.8868"]


def test_compare_rejects_duplicate_keys_and_unit_mismatch() -> None:
    duplicate = country_table().model_copy(
        update={
            "rows": [
                *country_table().rows,
                country_table().rows[0].copy(),
            ],
            "row_labels": [*country_table().row_labels, "越南"],
        }
    )
    with pytest.raises(TableNotComputableError) as error:
        run_compare(
            duplicate,
            wrap_table(parse_markdown_table(COMPARE_TABLE_MARKDOWN, unit_hint="亿美元")),
            CompareParameters(key_column="国家", value_column="2024年GDP"),
        )
    assert error.value.code == "duplicate_key"

    with pytest.raises(TableNotComputableError) as mismatch:
        run_compare(
            country_table(),
            wrap_table(parse_markdown_table(COMPARE_TABLE_MARKDOWN)),
            CompareParameters(key_column="国家", value_column="2024年GDP"),
        )
    assert mismatch.value.code == "unit_missing"


def test_compare_handles_zero_denominator_with_warning() -> None:
    # Build a table whose B value is zero to exercise the controlled zero divisor.
    zero_markdown = """| 国家 | 2024年GDP |
| --- | --- |
| 越南 | 0 |"""
    outcome = run_compare(
        country_table(),
        wrap_table(parse_markdown_table(zero_markdown, unit_hint="亿美元")),
        CompareParameters(key_column="国家", value_column="2024年GDP", keys=["越南"]),
    )

    assert outcome.status == "ok"
    assert any("相对变化不可计算" in warning for warning in outcome.warnings)


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------


def build_table_canonical(
    document_id: UUID,
    *,
    markdown: str | None = None,
    html: str | None = None,
    caption: str | None = None,
    asset_ref: str | None = None,
) -> CanonicalDocument:
    block = TableBlock(
        block_id="p1-b1",
        reading_order=0,
        markdown=markdown,
        html=html,
        caption=caption,
        asset_ref=asset_ref,
    )
    return CanonicalDocument(
        document_id=document_id,
        pages=[Page(page_number=1, blocks=[block])],
    )


def write_table_artifact(
    test_data_dir: Path,
    document_id: UUID,
    canonical: CanonicalDocument,
) -> None:
    document_root = test_data_dir / PARSING_DIRECTORY_NAME / str(document_id)
    document_root.mkdir(parents=True, exist_ok=True)
    (document_root / "canonical.json").write_text(
        canonical.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    if canonical.pages and canonical.pages[0].blocks:
        block = canonical.pages[0].blocks[0]
        if block.type == "table" and block.asset_ref:
            asset_path = document_root / "mineru" / block.asset_ref
            asset_path.parent.mkdir(parents=True, exist_ok=True)
            asset_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 24)


async def prepare_table_document(
    client: AsyncClient,
    test_data_dir: Path,
    *,
    knowledge_base_id: str | None = None,
    content: bytes = VALID_PDF,
    filename: str = "gdp-table.pdf",
    markdown: str | None = COUNTRY_TABLE_MARKDOWN,
    html: str | None = None,
    caption: str | None = UNIT_CAPTION,
    asset_ref: str | None = None,
) -> tuple[str, str, str]:
    kb_id = knowledge_base_id or await create_knowledge_base(client, name="表格分析知识库")
    document = await upload_document(client, kb_id, filename=filename, content=content)
    document_id = uuid_from(document["id"])
    canonical = build_table_canonical(
        document_id,
        markdown=markdown,
        html=html,
        caption=caption,
        asset_ref=asset_ref,
    )
    write_table_artifact(test_data_dir, document_id, canonical)
    write_evidence(test_data_dir, canonical, kb_id)
    evidence_id = table_evidence_id(test_data_dir, document_id)
    return kb_id, str(document_id), evidence_id


def uuid_from(value: object) -> UUID:
    return UUID(str(value))


def table_evidence_id(test_data_dir: Path, document_id: UUID) -> str:
    from knowledge_scope.evidence.lifecycle import evidence_artifact_path
    from knowledge_scope.evidence.models import MultimodalEvidenceDocument

    artifact = MultimodalEvidenceDocument.model_validate_json(
        evidence_artifact_path(test_data_dir, document_id).read_bytes()
    )
    return next(item.evidence_id for item in artifact.evidence if item.modality == "table")


def analyze_payload(
    kb_id: str,
    document_id: str,
    evidence_id: str,
    operation: str,
    parameters: dict[str, Any],
    compare: tuple[str, str] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "knowledge_base_id": kb_id,
        "operation": operation,
        "table": {"document_id": document_id, "evidence_id": evidence_id},
        "parameters": parameters,
    }
    if compare is not None:
        payload["compare_table"] = {
            "document_id": compare[0],
            "evidence_id": compare[1],
        }
    return payload


async def post_analyze(client: AsyncClient, payload: dict[str, Any]) -> Any:
    return await client.post("/api/v1/table-analysis/analyze", json=payload)


@pytest.mark.anyio
async def test_api_select_returns_traceable_values(
    client: AsyncClient,
    test_data_dir: Path,
) -> None:
    kb_id, document_id, evidence_id = await prepare_table_document(client, test_data_dir)
    payload = analyze_payload(
        kb_id,
        document_id,
        evidence_id,
        "select",
        {"columns": ["国家", "2024年GDP"], "rows": [0]},
    )

    response = await post_analyze(client, payload)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ok"
    assert body["rows"][0]["cells"] == ["越南", "4760"]
    assert body["sources"][0]["evidence_id"] == evidence_id
    assert body["sources"][0]["page_start"] == 1
    assert body["tables"][0]["computable"] is True


@pytest.mark.anyio
async def test_api_growth_uses_decimal_computation(
    client: AsyncClient,
    test_data_dir: Path,
) -> None:
    kb_id, document_id, evidence_id = await prepare_table_document(client, test_data_dir)
    payload = analyze_payload(
        kb_id,
        document_id,
        evidence_id,
        "growth",
        {
            "axis": "columns",
            "row": "越南",
            "base_column": "2023年GDP",
            "target_column": "2024年GDP",
        },
    )

    response = await post_analyze(client, payload)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["rows"][0]["cells"][4] == "10.6977"
    assert body["formula"].endswith("= 10.6977%")


@pytest.mark.anyio
async def test_api_cross_document_compare_returns_both_sources(
    client: AsyncClient,
    test_data_dir: Path,
) -> None:
    kb_id, document_a, evidence_a = await prepare_table_document(client, test_data_dir)
    _, document_b, evidence_b = await prepare_table_document(
        client,
        test_data_dir,
        knowledge_base_id=kb_id,
        content=SECOND_PDF,
        filename="gdp-table-b.pdf",
        markdown=COMPARE_TABLE_MARKDOWN,
    )
    payload = analyze_payload(
        kb_id,
        document_a,
        evidence_a,
        "compare",
        {"key_column": "国家", "value_column": "2024年GDP", "keys": ["越南"]},
        compare=(document_b, evidence_b),
    )

    response = await post_analyze(client, payload)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["rows"][0]["cells"] == ["越南", "4760", "4700", "60", "1.2766"]
    assert [source["evidence_id"] for source in body["sources"]] == [evidence_a, evidence_b]
    assert {table["table_role"] for table in body["tables"]} == {"table_a", "table_b"}


@pytest.mark.anyio
async def test_api_rejects_cross_knowledge_base_document(
    client: AsyncClient,
    test_data_dir: Path,
) -> None:
    _kb_a, document_a, evidence_a = await prepare_table_document(client, test_data_dir)
    kb_b = await create_knowledge_base(client, name="另一知识库")
    payload = analyze_payload(
        kb_b,
        document_a,
        evidence_a,
        "select",
        {"columns": ["国家"], "rows": [0]},
    )

    response = await post_analyze(client, payload)

    assert response.status_code == 404


@pytest.mark.anyio
async def test_api_reports_stale_evidence_as_controlled_404(
    client: AsyncClient,
    test_data_dir: Path,
) -> None:
    kb_id, document_id, evidence_id = await prepare_table_document(client, test_data_dir)
    stale = build_table_canonical(
        uuid_from(document_id),
        markdown="| 国家 | 2024年GDP |\n| --- | --- |\n| 越南 | 4800 |",
        caption=UNIT_CAPTION,
    )
    write_table_artifact(test_data_dir, uuid_from(document_id), stale)
    payload = analyze_payload(
        kb_id,
        document_id,
        evidence_id,
        "select",
        {"columns": ["国家"], "rows": [0]},
    )

    response = await post_analyze(client, payload)

    assert response.status_code == 404
    assert "证据产物不存在、已过期或不可用" in response.json()["detail"]


@pytest.mark.anyio
async def test_api_image_only_table_is_not_computable(
    client: AsyncClient,
    test_data_dir: Path,
) -> None:
    kb_id, document_id, evidence_id = await prepare_table_document(
        client,
        test_data_dir,
        markdown=None,
        caption="表 3 图片表格",
        asset_ref="gdp/auto/images/table.png",
    )
    payload = analyze_payload(
        kb_id,
        document_id,
        evidence_id,
        "select",
        {"columns": [0], "rows": [0]},
    )

    response = await post_analyze(client, payload)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "not_computable"
    assert body["error_code"] == "table_only_image"


@pytest.mark.anyio
async def test_api_unknown_evidence_and_invalid_parameters(
    client: AsyncClient,
    test_data_dir: Path,
) -> None:
    kb_id, document_id, evidence_id = await prepare_table_document(client, test_data_dir)

    missing = await post_analyze(
        client,
        analyze_payload(
            kb_id,
            document_id,
            "evidence_v1_" + "0" * 64,
            "select",
            {"columns": [0], "rows": [0]},
        ),
    )
    assert missing.status_code == 404

    invalid = await post_analyze(
        client,
        analyze_payload(
            kb_id,
            document_id,
            evidence_id,
            "growth",
            {"axis": "rows", "column": "2024年GDP"},
        ),
    )
    assert invalid.status_code == 422
