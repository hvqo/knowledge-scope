# PDF 表格结构化分析（M2）

M2 在现有 CanonicalDocument / Evidence 体系之上，新增一条可复用、确定性的 PDF 表格计算链路：

```
PDF 表格 Evidence
  → HTML / Markdown 表示
  → StructuredTable（列、原始单元格、解析后数值、单位、警告）
  → 确定性操作（select / filter / sort / aggregate / growth / compare）
  → 带来源引用的 TableAnalysisResult（Tool Result）
```

本阶段不引入视觉模型，也不引入 Agent 意图路由。计算只由显式结构化参数驱动；模型不能生成代码、SQL 或表达式参与计算。

## 模块与职责

| 文件 | 职责 |
| --- | --- |
| `src/knowledge_scope/table_analysis/models.py` | 稳定契约：StructuredTable、操作参数、结果、错误类型、大小限制 |
| `src/knowledge_scope/table_analysis/parsing.py` | 仅用标准库解析 HTML / Markdown 表格，识别表头、单位、年份、百分比、缺失值、合并单元格 |
| `src/knowledge_scope/table_analysis/operations.py` | 使用 `Decimal` 的确定性计算，返回实际公式与关键输入 |
| `src/knowledge_scope/table_analysis/service.py` | TableAnalysisService：Evidence 读取 → 解析 → 执行操作 → 组装来源/元数据 |
| `src/knowledge_scope/api/table_analysis.py` | 只读 HTTP 入口 `POST /api/v1/table-analysis/analyze` |

Evidence 读取复用 M1 的 `knowledge_scope.documents.assets.get_evidence`，因此自动继承“文档属于知识库、Evidence 属于文档、Evidence 未过期”的校验。

## StructuredTable

一条表格 Evidence 解析为：

- 来源：`knowledge_base_id`、`document_id`、`evidence_id`、`page_start/page_end`、`source_block_ids`、`section_path`、`source_fingerprint`；
- 结构：`columns`（表头、原始表头、单位、年份、数值/百分比/文本类型、数值与缺失计数）、`rows`、`row_labels`；
- 单元格：原始文本 `text`、解析后数值 `value`（十进制字符串）、类型、合并跨度、是否为跨度派生的副本；
- 质量信息：`warnings`、`computable`、`not_computable_reason`；
- 表级线索：`caption`、`unit_hint`。

数值统一解析为十进制字符串（`Decimal`），保留精确性；百分比按百分点存储，例如 `12.5%` → `12.5`。

## 支持的操作

| operation | 参数 | 说明 |
| --- | --- | --- |
| `select` | `columns`、`rows` | 按列/行读取解析后的数值或文本 |
| `filter` | `column`、`operator`、`value`、`columns` | 数值比较或文本 eq/ne/contains |
| `sort` | `column`、`direction`、`limit`、`columns` | 数值或文本排序，缺失值排在末尾 |
| `aggregate` | `column`、`functions`、`group_by` | sum / avg / min / max，可分组 |
| `growth` | `axis=rows/columns` + 基期、报告期 | 绝对增量 + 相对变化；百分比列输出百分点变化 |
| `compare` | `key_column`、`value_column`、`keys` | 两张表按共同键对齐，输出 A/B、绝对差异、相对差异 |

列/行既可用 0 开始的下标，也可用唯一表头/行标签；重名时返回 `ambiguous_column` / `ambiguous_row`，不猜测。

`growth` 与 `compare` 的公式、关键输入值、单位会写入结果的 `formula` / `inputs` / `units`。所有计算使用 `Decimal`，相对变化统一量化到 4 位小数（`ROUND_HALF_UP`）。

## 能力边界（显式不可计算）

以下情况返回 `status=not_computable` 与稳定 `error_code`，绝不猜测数据：

- `table_only_image`：只有图片素材、没有 HTML/Markdown 文本的表格（M3 视觉模型接入后可扩展）；
- `table_missing` / `table_content_missing`：Evidence 不是表格或缺少可解析内容；
- `complex_merge`：数据区存在无法可靠还原的合并单元格；
- `complex_header`：表头层级过多；
- `no_numeric_data`：表格中没有可计算的数值；
- `table_too_large`：超过行/列/长度上限；
- `duplicate_key`、`unit_missing`、`unit_mismatch`、`no_matching_keys`：跨表对比时无法确认对齐口径。

操作参数或数据状态问题返回 `status=error` 与错误码，例如 `division_by_zero`、`column_not_numeric`、`unknown_column`、`ambiguous_column`、`invalid_parameter`、`result_too_large`。

`compare` 在计算前强制检查：同一指标列、两表单位均存在且一致；不同货币/单位直接拒绝；键值重复时拒绝；对比表数值为 0 时相对变化留空并给出警告。

## HTTP 契约

```http
POST /api/v1/table-analysis/analyze
Content-Type: application/json

{
  "knowledge_base_id": "<kb-uuid>",
  "operation": "growth",
  "table": {
    "document_id": "<document-uuid>",
    "evidence_id": "evidence_v1_<sha256>"
  },
  "parameters": {
    "axis": "columns",
    "row": "越南",
    "base_column": "2023年GDP",
    "target_column": "2024年GDP"
  }
}
```

成功或表格本身不可计算时均返回 200，响应体是 `TableAnalysisResult`：

```json
{
  "schema_version": "1.0",
  "operation": "growth",
  "status": "ok",
  "columns": [{"name": "指标", "kind": "text", "unit": null}, "..."],
  "rows": [{"cells": ["GDP", "4300", "4760", "460", "10.6977"]}],
  "formula": "((4760 - 4300) / 4300) × 100% = 10.6977%",
  "inputs": {"base_value": "4300", "target_value": "4760", "change_kind": "absolute"},
  "units": ["亿美元"],
  "sources": [{"table_role": "table_a", "document_id": "...", "evidence_id": "...", "page_start": 1, "page_end": 1, "source_block_ids": ["p1-b1"]}],
  "warnings": [],
  "tables": [{"table_role": "table_a", "row_count": 3, "computable": true, "columns": ["..."]}]
}
```

- 文档不属于该知识库、Evidence 不存在/过期、跨知识库访问：404（不泄露内部路径）。
- 请求参数结构错误：422。
- 接口只接受结构化参数，不接受 SQL、Python 或任意表达式。

`compare` 的请求额外包含 `compare_table`，响应 `sources` 按 `table_a` / `table_b` 返回两张表的 Evidence ID、页码与来源 block。

## M4 Agent Tool 复用

`TableAnalysisService` 与 `TableAnalysisResult` 即未来的 Agent Tool 契约：输入是显式选中的知识库、1–2 条 Evidence、操作类型与参数；输出包含状态、结果、实际公式与输入、单位、来源、警告和失败错误码，全部 JSON 可序列化。M4 只需在工具路由层完成意图识别与 Evidence 选择，不需要改变计算实现。

## 限制与测试

- 表格数据量：≤ 500 行、≤ 50 列、HTML ≤ 2,000,000 字符；结果行 ≤ 200；警告 ≤ 30 条。
- 不支持：图片表格识别、跨表不同单位/币种换算、复杂多层表头、无法还原的合并单元格、任意表达式。
- 测试：`tests/test_table_analysis.py` 覆盖解析、Decimal 计算、零分母、重复键、单位不一致、跨文档对比、跨知识库 404、Evidence 过期 404、图片表格不可计算等场景。
