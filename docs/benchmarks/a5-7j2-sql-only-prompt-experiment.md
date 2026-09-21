# A5.7j2：SQL-only 投影与粒度提示词实验

## 假设与边界

A5.7i6 已经确认当前生成链路具备稳定的结构化输出：正例初次 generation
`47/47` 可解析、没有截断或 repair。当前主要问题集中在结果形状，包括请求字段投影、
稳定实体身份、分组粒度、指标保留以及显示字段和内部代码之间的选择。

A5.7j2 只增加一段通用的结果形状约束到 SQL-only NL2SQL 提示词。它不增加规划器、
ResultContract、第二次模型调用、self-check、重试或输出预算，也不改变 SQL 校验和执行。

生产输出仍是单个 JSON 对象：

```json
{"sql":"SELECT ..."}
```

本实验使用独立提示词版本 `a5.7j2-v1`。历史 `a5.3-v3` provider 产物保持可读取且不回写；
`a5.3-v4` 仍然只用于显式 ResultContract 诊断。

## 提示词改动

提示词要求模型在生成 SQL 前，从用户问题和受信任的结构化 schema context 中确定：

- 明确请求的可见字段、维度和指标；
- 一行总计、每个实体、每个分组、明细记录或实体对等结果粒度；
- 稳定身份是否需要参与分组以避免合并不同实体；
- 请求的筛选、日期边界、排序和 top-k 语义；
- 字段级 count、sum、avg 和派生指标的原始语义。

内部 ID 不会因为内部分组而自动进入明细结果；仅用于 join、filter、group 或 order 的辅助列
默认不展示。提示词没有加入 case ID、benchmark 问题、reference SQL、期望结果、SemanticPolicy
字段或已知失败标签。schema comments 仍不进入 provider prompt。

## 未加入的方案

本实验没有加入 planner JSON、ResultContract、chain-of-thought、第二次模型校验、语义策略
生成或额外 provider call。原因是 A5.7d 的 reasoning/预算实验没有带来 Execution Accuracy
改善，而本轮首先需要验证一个低成本、可回滚的 SQL-only 结果形状约束。

## 冻结契约

以下内容保持不变：

- Query Eligibility 及其 prompt/schema；
- SemanticEvidence、SemanticPolicy、semantic proof 和正式 comparator；
- A5.3 validator、A5.4 executor、analysis 和 ResultContract 实现；
- DEV/TEST 数据集、fixture、oracle 和 TEST 封存状态；
- provider/model、reasoning disabled、NL2SQL/repair 1024 tokens。

## 后续 DEV 评测

实现后才能运行一次新的 DEV 评测，记录：

- Formal Execution Accuracy；
- 请求投影、身份、粒度和指标诊断；
- initial parseability、截断、validation failure 和 repair；
- eligibility/negative-gate 安全结果；
- 输入/输出 tokens、调用次数和延迟。

不得把当前 DEV 观察写成 TEST 结论或模型质量保证。回滚条件是结构化输出稳定性、负例安全、
校验可靠性或延迟/token 成本发生预先不可接受的回退；该实验不以修改冻结 TEST 为手段追求指标。

## 状态

当前文件和提示词改动只定义实验，尚未运行 provider、DEV 或 TEST。
