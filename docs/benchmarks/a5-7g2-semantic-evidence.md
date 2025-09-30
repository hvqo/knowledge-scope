# A5.7g2：有界语义证据契约

## 定位

本契约是 ChatBI 评测专用的观测旁路，版本为
`a5.7g2-semantic-evidence-v1`。它描述已经通过现有 SQL 校验并完成执行的
候选结果，不参与 NL2SQL 生成、修复、授权校验、SQL 执行、Agent、MCP 或
正常 API/CLI 请求。

证据采集发生在评测 runner 已经拿到内存中的 `ValidatedSQL` 和有界
`QueryExecutionResult` 之后。证据不会反馈给前面的生成、校验或执行阶段，
也不是安全能力或执行凭证。

## 与既有评测的关系

需要区分四种概念：

- Formal Execution Accuracy：沿用冻结 A5.7f2 的正式结果比较口径。
- Semantic Proof：沿用冻结的语义证明规则；本阶段不增加证明定理。
- Output Shape：现有结果形状评估。
- Semantic Evidence：候选 SQL 和已归一化结果的有界、可复现描述。

证据采集本身不会提高模型质量，也不会改变语义分数。当前采集器只把
`SemanticEvidenceRecord` 放入新的 provider artifact 旁路；旧的
`a5.7b-provider-run-v1`（包括 Runs #5–#11）仍按旧路径读取，缺少证据时
仍保持原有 `UNASSESSABLE` 行为。新 artifact 使用明确的
`a5.7g2-provider-run-v2` 外层版本，并在 provenance 中声明证据版本。

## 记录内容

每条记录绑定以下 provenance：

- case、split、datasource
- 数据集、fixture、schema fingerprint
- SQL dialect（当前为 PostgreSQL）
- SQL parser 版本、canonical JSON 版本、collector 版本
- 证据模式和确定性 evidence digest

生成侧只保存有限的 provider/model 标识、prompt 版本、reasoning 模式、
finish reason、parse 状态、output budget、token 数和 generation/repair
尝试摘要。不会保存原始 provider response、思维链、凭据、DSN 或不受限的
prompt/SQL 文本。

### Projection、predicate 与 literal

投影使用有标签的结构表达式，区分 `source_column`、`aggregate`、
`derived`、`literal` 和 `unknown`。关系、列名和别名是独立字段；别名只
是展示信息，不改变表达式身份。

谓词记录有限的 `AND`、`OR`、`NOT`、比较、`IS NULL`、`IN` 和 `BETWEEN`
结构。业务 literal 不写入原文，只保留带类型、带域分离的 identity。当前
synthetic fixture 使用：

```text
SHA-256("a5.7g2-literal-v1|" + canonical_type + "|" + canonical_value)
```

`NULL` 使用符号 identity。该 hash 只用于冻结 synthetic benchmark 的可
复现比较，不是防字典攻击的保密机制；未来敏感数据需要新的版本化契约，
例如由外部密钥管理的 keyed HMAC。`sensitive_future` 模式在本阶段明确
拒绝，不会降级为 synthetic 模式。

### Join、grain、aggregate 与 set operation

记录有限的 source relation、join 类型、左右 key lineage 和 join predicate。
主键、唯一性、外键、非空等约束继续由共享 schema fingerprint 契约负责，
不会在每条记录复制整份 schema。

记录 `DISTINCT`、`DISTINCT ON` 及其有界表达式、GROUP BY 语义键和
`grain_status`（只有可确定推导时才为 `proven`，否则为 `unknown`）。不支持的
`DISTINCT ON` 表达式会令 SQL 证据变为 `partial`。支持既有评测需要的 COUNT、SUM、AVG、MIN、
MAX、有限算术/函数结构；不支持的结构写成带有限 reason code 的
`unknown`，并将 SQL 证据标记为 `completeness=partial`，不会伪装成完整语义。
UNION、INTERSECT、EXCEPT 记录操作类型、ALL 语义和子树 digest。`UNION BY NAME`
以及无法安全传播的根级 WITH/set-operation 作用域明确标记为 `partial`；递归或
嵌套 CTE 也不会被错误地宣称为完整。顶层
ORDER BY、LIMIT、OFFSET、FETCH 也会单独记录。无法安全建模的顶层修饰符只
产生 `partial` 和 reason code，不写入原始 SQL，也不扩展 A5.7f2 的证明规则。

### ORDER、LIMIT 与结果

ORDER BY 保存表达式 lineage、方向及显式 NULLS FIRST/LAST。LIMIT、OFFSET、
FETCH 分别保存 `absent`、有界 numeric `value`、`all`、`unavailable` 或
`unsupported` 状态；分页控制值与业务 literal 属于不同隐私类别。

结果证据只接受已经由现有执行器限制过的归一化结果，保存执行状态、row count、
有界列元数据、截断状态/原因，以及保留 NULL 和重复行多重性的 ordered/multiset
fingerprint。行 fingerprint 使用版本化的单遍算法
`a5.7g2-streaming-rows-v1`；它逐行处理输入，不把全部 rows 复制到新的列表中。
只有 `synthetic_exact_v1` 且执行成功时，才可能保存不超过上限的 exact rows；当
行数或字节预算耗尽时，保存的是有明确 reason code 的 bounded prefix，而不是把
prefix 冒充为完整结果。它不是语义评估必须依赖的字段。

## 有界和失败策略

单条记录的硬上限为：

| 项目 | 上限 |
| --- | ---: |
| semantic AST nodes | 512 |
| expression depth | 32 |
| 每类 list entries | 128 |
| identifier UTF-8 bytes | 512 |
| result columns | 64 |
| exact result rows | 256 |
| exact normalized result JSON | 64 KiB |
| SemanticEvidenceRecord JSON | 128 KiB |

超过上限的记录不会部分写入不完整结构，而是按固定顺序降级：先去掉 exact
rows，再去掉 provider attempt metadata，再去掉可选的结果列/排序与 hash
metadata，最后返回只含 provenance、状态和 reason code 的静态最小记录。每一
步都会重新计算 digest，最终记录仍必须不超过 128 KiB；不会递归尝试任意完整
记录或抛出第二个同类大小异常。结果 hash 只处理已经在执行阶段有界的
normalized rows，不为证据采集重新物化完整查询结果。

生成失败、没有 provider response、未尝试解析、结构化解析失败、没有 SQL
candidate、未到达 validation 或未到达 execution，分别保留为安全的 lifecycle
status/reason code。证据采集发生在权威 `ChatBIResult` 已产生之后；旁路采集
异常只会生成 `unavailable` 证据，不会替换、修改或使有效的权威结果失败。
50 个 case 的最大记录体积约为 6.4 MiB；这是文档预算，不要求一次性预分配。

Generation/repair attempt metadata 同样逐个迭代，最多保留 128 次尝试。超过该
上限不会抛出或物化全部输入，而是设置 `attempts_truncated=true`、
`completeness=partial` 和 `attempt_metadata_truncated`。

## Digest 与读取约束

digest 对排除自身 digest 字段后的 canonical JSON 计算 SHA-256；canonical JSON
使用 UTF-8、排序 key 和固定 separators。它用于完整性和可复现性，不是认证
机制。构建器会填入 digest，但读取序列化的 g2 evidence 时必须要求恰好 64 位
小写十六进制 digest，并重新计算比对；缺失、空值、大小写错误、非十六进制或
旧 digest 都 fail closed。读取新 artifact 时还必须验证 evidence version、
dataset、fixture、schema fingerprint；任何 provenance 不匹配都 fail closed。

冻结的 A5.7f2 semantic evaluator 可以在未来读取已经验证的旁路证据，但本
阶段没有把证据接回生成、校验或执行路径，也没有扩大 proof rules。采集器不
读取 semantic policy，不保存 expected SQL、expected rows、reference
projection 或 required fields。候选证据与问题要求保持分离。

## 当前范围

本阶段不运行 provider、DEV 或 TEST，不重写冻结数据和历史 artifact，也不
声明质量提升。`SemanticEvidenceRecord` 是为后续在证据契约冻结后进行的
离线语义分析准备的、可校验且有界的 evaluation-only 数据。
