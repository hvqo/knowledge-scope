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

本阶段没有调用 provider、没有运行 DEV/TEST，也没有修改冻结数据集或 fixture。
