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
质量结论，Run #10 的历史解释保持不变。完整 Run #11 已作为历史产物保留；本节的
三例 probe 不能替代完整评估，也不能被当作总体质量指标。

## A5.7e2c：ResultContract 语义可观测性

这次诊断只增加评测端的安全摘要，不改变 `a5.3-v4` prompt、NL2SQL 生成、repair、
验证器或执行器。摘要由应用在 `ResultContract` 已通过 Pydantic/语义边界后生成，包含：
契约版本、行粒度、规范 grain keys、按顺序排列的输出列、分组、排序和 limit。输出列只
记录 `kind`、规范来源、聚合函数、source columns，以及由 AST 解析得到的表达式类别；不
记录原始派生表达式、别名、provider 原文、推理内容、SQL 或业务字面量。表达式类别是
有限的结构标签，`COALESCE` 这类函数按 `function` 记录，不代表业务语义已经正确。

比较只在评测端进行，并使用冻结 reference SQL 派生的 11 条语义意图作为 oracle；比较
不会进入 prompt、repair、SQL 验证或执行。六个组件分别为 `row_grain`、`grain_keys`、
有序 `output_columns`、`group_by`、`order_by`、`limit`，整体结果为 `exact`、`partial`、
`incorrect` 或 `unavailable`。`Execution Accuracy` 仍是主要指标，本诊断不声称精度、
召回率或准确率。

### 固定 DEV 诊断运行

运行 ID 为 `6002bdb4-1bba-4438-a83c-e3f8160f4011`，只调用以下 11 条 DEV case，TEST
调用数为 0：`join-04`、`join-05`、`empty-02`、`simple-04`、`group-by-01`、
`top-k-02`、`top-k-03`、`null-02`、`join-06`、`multi-join-01`、`multi-join-03`。
配置为 DeepSeek model `deepseek-v4-flash-vision-exp`、`a5.3-v4`、JSON mode、
`reasoning=disabled`、NL2SQL/analysis 均为 1024 tokens；provider 产生 23 次调用（11
次初始生成、1 次 repair、11 次分析）。

下表的“生成摘要”只使用安全结构标签。`↑`/`↓` 表示排序方向；`—` 表示没有可用契约。
“最早层”是本 case 在这次运行中最早暴露的问题层：L0 是传输/结构边界，L1 是契约与
冻结语义意图不一致，L2 是契约正确但 SQL 没有实现契约，L3 是契约和 SQL 结构对齐但
SQL 业务语义仍不一致。`null-02` 的 L2 发生在初始生成，repair 后恢复。

| case | 期望语义意图 | 生成摘要（最终一次） | 契约比较 | 不一致组件 | SQL/契约 | 执行等价 | 最早层 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `join-04` | 逐笔销售：销售编号、客户名称、销售日期，按销售编号升序 | detail; grain=`sales.sale_id`; cols=`sales.sale_id, customers.customer_name, sales.sold_on`; order=`sales.sale_id↑`; limit=`null` | exact | — | passed | true | — |
| `join-05` | 每位客户的最早销售日期，包含无销售客户 | grouped; grain=`customers.customer_id`; cols=`customers.customer_id, customers.customer_name, min(sales.sold_on)`; group=`customers.customer_id, customers.customer_name`; order=`customers.customer_id↑` | exact | — | passed | true | — |
| `empty-02` | 客户名称以指定前缀开头的客户 | detail; grain=`customers.customer_id`; cols=`customers.customer_id, customers.customer_name`; order=`customers.customer_id↑`; limit=`1000` | partial | order_by, limit | passed | true | L1 |
| `simple-04` | 客户编号为 1 的客户名称 | 无可用契约 | unavailable | all unavailable | schema not reached | — | L0 |
| `group-by-01` | 按地区汇总销售笔数和金额 | grouped; grain=`regions.region_code`; cols=`regions.region_code, count(sales.sale_id), sum(sales.amount)`; group=`regions.region_code`; order=`regions.region_code↑` | partial | output_columns, group_by | passed | false | L1 |
| `top-k-02` | 最早发生的销售及其日期、金额 | detail; grain=`sales.sale_id`; cols=`sales.sale_id, sales.customer_id, sales.region_code, sales.sold_on, sales.amount`; order=`sales.sold_on↑`; limit=`1` | partial | output_columns, order_by, limit | passed | false | L1 |
| `top-k-03` | 最近发生的销售及其日期、金额 | detail; grain=`sales.sale_id`; cols=`sales.sale_id, sales.customer_id, sales.region_code, sales.sold_on, sales.amount`; order=`sales.sold_on↓`; limit=`1` | partial | output_columns, order_by, limit | passed | false | L1 |
| `null-02` | 每位客户的销售笔数和总额，无销售客户总额为 0 | grouped; grain=`customers.customer_id`; cols=`customers.customer_id, customers.customer_name, count(sales.sale_id), derived(function; source=`sales.amount`)`; group=`customers.customer_id, customers.customer_name`; order=`customers.customer_id↑` | exact | 初始 L2 已由 repair 恢复 | passed（初始失败，repair 通过） | true | L2 → recovered |
| `join-06` | 金额超过阈值的销售及客户 | detail; grain=`sales.sale_id`; cols=`customers.customer_name, sales.amount`; order=`[]`; limit=`null` | partial | output_columns, order_by | passed | false | L1 |
| `multi-join-01` | 按客户和地区汇总销售笔数及总额 | grouped; grain=`customers.customer_id, regions.region_code`; cols=`customers.customer_id, customers.customer_name, regions.region_code, regions.region_name, count(sales.sale_id), sum(sales.amount)`; group=`customers.customer_id, customers.customer_name, regions.region_code, regions.region_name`; order=`customers.customer_id↑, regions.region_code↑` | partial | output_columns | passed | false | L1 |
| `multi-join-03` | 列出每个地区有销售记录的客户及销售笔数 | grouped; grain=`regions.region_name, customers.customer_id`; cols=`regions.region_name, customers.customer_id, count(sales.sale_id)`; group=`regions.region_name, customers.customer_id`; order=`regions.region_name↑, customers.customer_id↑` | partial | grain_keys, output_columns, group_by, order_by | passed | false | L1 |

#### 结果汇总与解释

11 条中有 10 条生成了至少一个安全契约摘要：最终比较为 `exact=3`（`join-04`、
`join-05`、`null-02`）、`partial=7`、`incorrect=0`、`unavailable=1`（`simple-04`）。
`row_grain` 正确 10/10 个可比较 case；`grain_keys` 正确 9、错误 1；
`output_columns` 正确 4、错误 6；`group_by` 正确 8、错误 2；`order_by` 正确 5、错误 5；
`limit` 正确 7、错误 3。契约摘要是结构诊断，不把这些数量解释成质量指标。

`simple-04` 是唯一的 L0 终态：初始与 repair 的 JSON 和顶层形状通过，但嵌套
`ResultContract` schema 均失败，没有可用契约。`null-02` 初始契约通过语义校验但没有
实现 SQL/契约一致性，第一次 repair 生成完整契约和 SQL 后通过；它是一次可见的 L2 →
recovered 路径，而不是静默接受不一致结果。其 `COALESCE` 在安全摘要中按 AST function
类别记录，避免保存原始表达式。

其余 7 个 partial case 均在 `ResultContract` 和 SQL 一致性边界通过后才执行；它们的
契约与冻结语义意图的差异被标为 L1。没有 case 在本次最终结果中进入 L3。该表支持的
下一步只有一个：先把更高频的投影、排序和 limit 契约差异作为下一次独立诊断问题，不能
据此修改或声称当前 prompt 已经解决了语义正确性。

安全运行产物写入被 Git 忽略的 `data/evaluation/a5-7/provider/`，文件为
`chatbi-eval-v2-result-contract-diagnostic.json`；旧 Run #5–#11 产物和冻结 dataset
没有改写。该运行没有改变评测数据、fixture、provider 配置或 ChatBI 业务路径。
