# A5.7e3：ResultContract 单独规划 Probe

## 目的与边界

本 probe 只测量一个独立的结构化 `ResultContract` 规划调用，目的是观察
ResultContract 与 SQL 生成任务分开后是否更稳定。它不是生产路径，也不是模型质量
评测；不会生成 SQL、调用 repair、执行 SQL 或调用结果分析模型。生产 ChatBI 路径、
`a5.3-v4` prompt、ResultContract schema、验证器、推理设置和冻结数据均未改变。

请求边界为：

```text
registered datasource
  → trusted schema discovery
  → comment-free structural schema context
  → one contract-only nl2sql gateway call
  → ResultContract JSON/schema validation
  → ResultContract semantic validation
```

fixture bootstrap 和 schema discovery 仍会使用已有的本地只读元数据检查；本 probe 不把
模型返回的任何 SQL 交给 SQL validator 或执行器。provider 原文、推理内容、业务值和
凭据不写入产物。

## 固定配置

- 版本：`a5.7e3`；prompt：`a5.7e3-contract-only-v1`；artifact schema：
  `a5.7e3-provider-contract-only-v1`；
- 数据：冻结 `a5.7-v2` DEV，固定 11 例；TEST 调用数为 0；
- 数据源：注册的 `chatbi-demo-v2-authoritative`，datasource ID 为
  `a0e1eff3-45ad-512a-8dc4-71ab7cbe2125`；
- provider/model：`deepseek` / `deepseek-flash`；temperature `0`；
  `response_format=json_object`；`reasoning=disabled`；`max_tokens=1024`；
  gateway retries `0`；
- schema context：由同一注册数据源的 trusted discovery 生成；结构信息使用确定性
  JSON，评论不进入 prompt；
- 运行 ID：`67bd1e4f-7198-4bf9-ae10-cc2707b48af9`。

该运行使用时工作树有未提交变更，因此 provenance 中的 `git_dirty=true`；这只描述
运行时状态，不改变数据和配置身份。

## 与 A5.7e2c 的比较

“同一调用”列取自此前相同 11 例的 `a5.7e2c` 最终安全契约摘要；contract-only 列是
本次单独规划调用的结果。比较只在评测端进行，不进入 prompt，也不作为 SQL 或 repair
输入。`unavailable` 表示没有可比较的语义契约。

| case | 同一调用 | contract-only 解析 | contract-only 比较 | 变化 | 输入/输出 tokens | 生成 ms |
| --- | --- | --- | --- | --- | ---: | ---: |
| `join-04` | exact | schema failure | unavailable | exact → unavailable | 793 / 99 | 631.0 |
| `join-05` | exact | parsed | partial（`output_columns`, `group_by`） | exact → partial | 798 / 133 | 849.0 |
| `empty-02` | partial（`order_by`, `limit`） | schema failure | unavailable | partial → unavailable | 794 / 115 | 719.7 |
| `simple-04` | unavailable | schema failure | unavailable | unavailable → unavailable | 794 / 76 | 947.0 |
| `group-by-01` | partial（`output_columns`, `group_by`） | parsed | partial（`output_columns`, `group_by`） | partial → partial | 791 / 161 | 722.0 |
| `top-k-02` | partial（`output_columns`, `order_by`, `limit`） | schema failure | unavailable | partial → unavailable | 789 / 180 | 1131.8 |
| `top-k-03` | partial（`output_columns`, `order_by`, `limit`） | schema failure | unavailable | partial → unavailable | 789 / 180 | 1269.9 |
| `null-02` | exact | parsed | partial（`output_columns`） | exact → partial | 801 / 189 | 1234.0 |
| `join-06` | partial（`output_columns`, `order_by`） | schema failure | unavailable | partial → unavailable | 796 / 96 | 1006.8 |
| `multi-join-01` | partial（`output_columns`） | parsed | partial（`output_columns`） | partial → partial | 794 / 286 | 1388.9 |
| `multi-join-03` | partial（`grain_keys`, `output_columns`, `group_by`, `order_by`） | parsed | partial（`output_columns`, `group_by`） | partial → partial | 795 / 194 | 955.1 |

### 汇总

contract-only 共 11 次 provider invocation，11/11 provider 成功，无 JSON decode failure，
无语义验证失败；5/11 通过 `ResultContract` 结构和语义验证，6/11 在
`result_contract_schema_invalid` 终止。最终安全比较为：

| overall | count |
| --- | ---: |
| exact | 0 |
| partial | 5 |
| incorrect | 0 |
| unavailable | 6 |

可比较的五例组件状态为：`row_grain` 正确 5、`grain_keys` 正确 5、`order_by` 正确 5、
`limit` 正确 5；`output_columns` 错误 5；`group_by` 正确 2、错误 3。六例 schema failure
的组件均为 `unavailable`。本摘要只描述契约结构，不是 precision、recall、accuracy 或
Execution Accuracy。

此前 e2c 的同一 11 例最终结果为 `exact=3`、`partial=7`、`incorrect=0`、
`unavailable=1`，但 e2c 允许正常生产协议和一次 repair；因此这不是严格的单变量因果
对照。contract-only 没有表现出比同一调用更高的稳定性，也没有消除嵌套 schema 失败。

## 性能与 usage

- generation latency：P50 `955.1 ms`，P95 `1,388.9 ms`，11 例总计 `10,855.3 ms`；
- input tokens：总计 `8,734`，均值 `794.0`，P50 `794`，P95 `801`；
- output tokens：总计 `1,709`，均值 `155.4`，P50 `161`，P95 `286`；
- provider success `11/11`，每例恰好一次 invocation；配置未提供价格，因此 cost 为
  `null`，没有用零值代替未知成本。

这些是 11 例诊断运行的观测值，不是生产延迟或总体质量指标。

## 结论与后续

这组结果**不支持**“把 SQL 与 ResultContract 分开就能解决当前问题”的结论。单独
契约 prompt 仍有 6/11 schema failure，剩余五例也没有 exact；由于 prompt、repair
行为和任务输出边界同时变化，不能把差异归因于某一个因素。两阶段规划在当前证据下
不应进入生产路径，也不应据此改变冻结 benchmark。

推荐的下一项独立工作为 **A5.7e4 — ResultContract contract-only schema stability**：
只针对这 11 个 DEV 控制例复核并稳定嵌套契约的 provider 输出边界，仍保持 TEST=0，
不改变生产 NL2SQL 语义、不引入 SQL 执行或结果分析，也不把本 probe 当成质量指标。

安全运行产物写入 Git 忽略的：
`data/evaluation/a5-7/provider/chatbi-eval-v2-contract-only-probe.json`。
