# A5.7g3：DEV 语义策略冻结

## 范围与来源

`a5.7g3-semantic-policy-v1` 是冻结 ChatBI v2 DEV 正向问题的人工策略
sidecar，文件为 `a5-7g3-semantic-policy-v1.json`。策略只依据 DEV 问题、
受信的 `chatbi-demo-v2` schema 和 Projection Policy D 编写。它不读取候选
SQL、结果行、正式 EA、证明结果、provider 输出或 repair 输出，也不会进入
prompt、SQL 校验、SQL 执行或 Agent 生产链路。

当前 artifact 明确覆盖 47 个 DEV 正向 case；TEST 的问题内容不进入该策略，
TEST 仍保持封存。旧的 `a5.7f2-semantic-policy-v1` artifact 和历史读取路径
保持不变。f2 与 g3 loader 共用重复键拒绝解析器；有效的 f2 artifact 语义和
fingerprint 不变，历史 artifact 本身未被改写。

`SemanticEvidence` 表示候选实际产生了什么；`SemanticPolicy` 表示题目要求
什么。两者不能互相生成或替代。

## 可执行字段与行粒度

策略字段使用封闭词汇，不接受 SQL 表达式或通用表达式语言。词汇包括：

- 直接字段：`customer_id`、`customer_name`、`sale_id`、`region_name`、
  `region_code`、`market`、`amount`、`sold_on`；
- 聚合字段：`aggregate:sum:amount`、`aggregate:count:sale_id`、
  `aggregate:count:customer_id`、`aggregate:count:distinct:customer_id`、
  `aggregate:avg:amount`、`aggregate:max:amount`、
  `aggregate:min:amount`、`aggregate:min:sold_on`、
  `aggregate:max:sold_on`；
- 派生语义字段：`derived:amount_minus_overall_average`、
  `derived:region_sales_share`。

每个 case 的 `row_grain` 只能是 `scalar`、`detail`、`entity`、`grouped` 或
`pair`。`pair` 只表示题目要求的两个身份之间的一条关系，例如客户-销售或
客户-地区；它不是通用图元组或 GraphRAG 规则。

稳定 ID 只在题目明确要求、实体标签可能重复，或事实/关系身份确实需要时
进入必需语义；辅助 JOIN、过滤、分组和排序字段不会自动变成可见字段。区域
名称由受信 schema 的唯一约束支持时可作为区域身份。列顺序默认属于展示层，
只有问题明确要求排序语义时才设置 `semantic_order_required`。安全额外字段
默认不允许；可选字段只能提供展示上下文，不能改变身份、粒度或聚合。

## 审阅元数据与指纹

每个策略记录 `authoring_confidence`（`explicit`、`schema_required`、
`product_rule_derived`、`ambiguous`）和不超过 512 字符的
`human_rationale`。这些字段仅用于审阅记录，不参与语义通过/失败、必需字段、
身份证明、粒度、顺序或正式比较。策略的确定性 SHA-256 fingerprint 会绑定
版本、数据集/fixture/schema 指纹、完整可执行字段和这些审阅元数据；任何策略
或审阅记录变更都会使 artifact 失效。

artifact 只允许上述字段和受信 schema 元数据，禁止候选 SQL、expected SQL、
结果行、证明结论、正式 EA 和 provider 响应。未知字段、未知语义字段、超长或
包含执行/模型输出文本的 rationale 都会 fail closed。

此文件只描述 DEV 策略冻结，不代表 provider benchmark，也不改变正式
`compare_normalized_result()` 或历史 Run 报告。
