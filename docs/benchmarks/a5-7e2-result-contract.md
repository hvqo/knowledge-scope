# A5.7e2：ChatBI 结构化结果契约

本阶段把 NL2SQL 的响应从单一 SQL 字段扩展为一次结构化响应：

```text
问题 + 受信任 Schema
  → ResultContract + SQL（一次 NL2SQL 调用）
  → ResultContract 结构校验
  → SQL/ResultContract 一致性校验
  → A5.3 trusted SQL 校验
  → A5.4 只读执行
```

## 契约范围

当前契约版本为 `1.0`，生成提示版本为 `a5.3-v4`。契约包含：

- `row_grain`：`scalar`、`detail` 或 `grouped`；
- `grain_keys`：描述行或实体稳定身份的规范来源列；不要求出现在 SELECT 中；
- `output_columns`：按返回顺序排列的 source、aggregate 或 derived 输出；
- `group_by`：规范分组来源列；
- `order_by`：规范来源/输出键及 `asc` 或 `desc`，可表达二级稳定排序；
- `limit`：问题明确要求时使用的非负整数上限，否则为 `null`。

`grain_keys` 与 `output_columns` 是不同的概念。例如逐笔销售可以用
`sales.sale_id` 作为 `grain_keys`，同时只输出客户名称和金额。分组结果必须在
`group_by` 中保留所有稳定身份键。来源列由当前受信任 schema discovery 解析，
不能只依赖输出别名。

机器可读结构固定为以下字段。`schema.relation.column` 是单个字符串引用，不是对象；
`output_columns` 使用 `kind` 作为 discriminator：`source` 只使用 `source`，`aggregate`
使用 `function` 和可选 `source`，`derived` 使用 `expression` 和 `source_columns`。所有
数组字段都可以为空，但 `output_columns` 至少有一项；`limit` 没有上限要求时使用显式
`null`。排序项只能使用 `key` 与 `direction`：

```json
{
  "result_contract": {
    "contract_version": "1.0",
    "row_grain": "grouped",
    "grain_keys": ["public.example.category"],
    "output_columns": [
      {"kind": "source", "source": "public.example.category", "alias": "category"},
      {"kind": "aggregate", "function": "sum", "source": "public.example.amount", "alias": "total_amount"},
      {"kind": "derived", "expression": "amount * 1.0", "source_columns": ["public.example.amount"], "alias": "normalized_amount"}
    ],
    "group_by": ["public.example.category"],
    "order_by": [
      {"key": "public.example.category", "direction": "asc"},
      {"key": "total_amount", "direction": "desc"}
    ],
    "limit": null
  },
  "sql": "SELECT ..."
}
```

`contract_version` 当前值为 `"1.0"`，`row_grain` 只能是 `scalar`、`detail`、`grouped`；
没有排序时 `order_by` 为 `[]`，没有分组时 `group_by` 为 `[]`，没有稳定行键时
`grain_keys` 为 `[]`。这些是 prompt 的结构示例，不代表允许使用示例中的未发现对象；
实际 schema 与 SQL 仍必须来自受信任的 discovery 结果。

## 校验边界

应用先检查契约的字段、枚举、唯一性、来源列、分组、排序和上限，再检查 SQL 的
有序投影、聚合/派生表达式、DISTINCT、GROUP BY、ORDER BY 与 LIMIT 是否一致。
别名变化只要底层语义来源不变即可通过；缺列、多列、列顺序、排序方向或二级排序
不一致会进入现有的一次 repair。契约无效或不一致不会被当作空结果，也不会绕过
A5.3 的 PostgreSQL AST/只读/对象授权校验。

ResultContract 只约束结果结构，不能证明 SQL 的业务正确性；它不替代 A5.3 安全
校验、不替代执行结果评测，也不解决不支持或含糊问题的资格判断。负向请求安全仍
是独立的后续工作，本阶段没有改变拒答策略。

## Repair 与兼容性

初次生成和有界 repair 都使用一个 `a5.3-v4` 结构化响应，并沿用现有的
`reasoning="disabled"`、`max_tokens=1024` 和一次 repair 上限。契约不通过时，
repair 可以重新生成完整的契约与 SQL；契约有效但 SQL 不一致时，repair prompt
会携带受控的契约和验证错误，在允许的长度内提供上一条 SQL。不会提供评测 oracle、
参考 SQL 或隐藏推理。

评测 provenance 对新运行记录 `nl2sql_result_contract_enabled=true` 和
`nl2sql_prompt_version=a5.3-v4`。旧的 Run #5–#9 文件不改写；缺少该字段的历史
配置仍按旧格式读取，并不会被误标记为已使用新契约。

## A5.7e2a：结构化输出边界诊断

`a5.7e2a` 只观察 Layer 0：provider 返回内容、JSON 解码、顶层结构、`ResultContract`
字段结构和 SQL 字段的形状。它不把这些检查与 Layer 1 的问题到契约语义、Layer 2 的
契约到 SQL 一致性或 Layer 3 的 SQL 业务语义混为一谈。

诊断产物只保留应用生成的安全元数据：阶段、JSON/结构状态、排序后的顶层和契约 key、
`sql` key 是否存在、是否为字符串、字符长度、受限字段路径和规范化错误码。它不保留
provider 原文、SQL、业务值、推理内容或凭据。诊断阶段使用以下固定分类：

- `provider_content_missing`、`json_decode_failed`、`top_level_shape_invalid`；
- `top_level_schema_invalid`、`required_top_level_field_missing`、
  `result_contract_schema_invalid`、`sql_field_invalid`；
- `result_contract_semantic_validation_failed`、`sql_contract_consistency_failed`。

当前 provider 请求仍使用 DeepSeek Chat Completions 的
`response_format={"type":"json_object"}`。JSON mode 只约束 provider 返回 JSON 对象的
形式，不等价于对嵌套 `ResultContract` 执行 JSON Schema 校验；本地 JSON、Pydantic、语义、
一致性和 A5.3 校验仍是权威边界。请求 prompt 明确要求 JSON，并描述
`{"result_contract":{...},"sql":"..."}` 目标形状；a5.7e2a 首次 probe 不修改 prompt，
后续 e2b 只补充当前生产模型的精确嵌套结构示例。

Run #10 的原始产物保持不可变。它报告的“结构化输出失败”应解释为 Layer 0 观察结果，
不能解释成 47 个问题已经完成了 ResultContract 语义评测；其中负例的生成失败也不能被
当作已经证明拒答策略正确。`a5.7e2a` provider probe 只允许冻结 DEV 的
`aggregation-01`、`join-02`、`group-by-02` 三例，TEST 调用数为零，结果写入被 Git 忽略的
诊断运行时目录。

### a5.7e2a 首次三例诊断结果

2026-09-16 的首次 probe 共完成 6 次 provider 调用：每个 case 各有一次初次生成和一次
正常 repair。3/3 初次响应和 3/3 repair 响应都完成了 JSON 解码和顶层对象检查，顶层 key
均为 `result_contract`、`sql`；但 6 次均在 `result_contract_schema_invalid` 停止。
因此 `semantic_contract_validation`、SQL/契约一致性检查和 A5.3 校验均未到达，三例都
没有可用 SQL candidate。所有调用的 `finish_reason` 为 `stop`，`token_limit_status` 为
`not_reached`，所以这次结果不是输出截断。

安全诊断暴露的共同结构问题是 `output_columns` 过短或形状不符合当前模型；join/grouped
响应还在 `order_by` 与 output-column 内使用了当前模型不接受的 `term`、`reference`、
`name` 等字段。记录只包含字段路径和规范化 Pydantic 错误码，不包含 provider 原文或 SQL。
本次 aggregate 统计为：JSON 解码失败 0、ResultContract schema 失败 6、语义契约失败 0、
SQL/契约一致性失败 0、到达 A5.3 校验 0。该证据支持“当前 prompt 对嵌套契约表示的约束不够
明确或与实现模型不一致”的应用/schema 兼容性诊断，但不能单独证明 Chat Completions JSON
mode 不具备完成该结构的能力。

### A5.7e2b：对齐后的三例兼容性 probe

在共享结构示例与 `a5.3-v4` prompt 对齐后，2026-09-16 的第二次 probe 仍只使用同一三例
DEV controls，共完成 6 次 provider 调用：3 次初始 NL2SQL 生成和 3 次结果分析，没有
触发 repair。三个初始响应均为 `finish_reason=stop`、`token_limit_status=not_reached`，
JSON 解码、顶层结构、ResultContract schema、语义校验、SQL/契约一致性和 A5.3 校验均
通过；每例都得到可用 candidate、执行成功并完成分析。统计为：初始 parseable 3/3、
JSON 解码失败 0、ResultContract schema 失败 0、语义契约失败 0、SQL/契约一致性失败 0、
到达 A5.3 校验 3/3、repair 0 次。该结果支持确定性的 prompt/schema 表示不一致已修复，
也表明当前 Chat Completions `json_object` 路径对这三个控制例仍可用；它不是完整 DEV/TEST
质量结论，Run #10 的历史解释保持不变。基于结构边界的强成功标准，下一步可受控地安排
完整 Run #11，但仍不得把三例 probe 当作总体质量指标。
