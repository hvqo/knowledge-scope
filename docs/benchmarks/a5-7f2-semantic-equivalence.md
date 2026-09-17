# A5.7f2：语义等价诊断器

## 范围

A5.7f2 只增加离线评估诊断，不改变 ChatBI 生产链路、SQL 生成、A5.3
校验、结果归一化、冻结数据集或正式 Execution Accuracy。

正式指标仍由 `compare_normalized_result()` 计算。它比较完整的结果形状，
包括列数、列位置、行值、行数和冻结的截断语义。这个指标没有被替换或重写。

本阶段另外提供两个诊断维度：

- **Diagnostic Semantic Equivalence**：按产品语义比较必需字段、值、多重性、
  行粒度、实体身份、NULL/数值容差和必要时的行顺序。
- **Output Shape Compliance**：记录列顺序、别名、额外字段和可选展示字段缺失，
  并明确区分 `compliant`、`non_compliant`、`unassessable`。

语义等价可以为真而输出形状不完全合规。两者都不能被解释为新的生产约束，
也不能把诊断指标称为正式 Accuracy。

## 冻结的投影政策

策略版本为 `a5.7f2-semantic-policy-v1`，存放在
`docs/benchmarks/a5-7-semantic-policy-v1.json`。它是独立的、版本化的评估
sidecar，来源是对冻结 DEV 问题进行的人类产品语义评审，不是自动生成的
ground truth，也不会进入 prompt、repair、SQL validator 或 Agent。

政策规则如下：

1. 用户明确请求的字段或明确的 schema 同义字段必须存在。
2. 主键/业务键可以为了行粒度、分组、实体身份和确定性保留在内部；它们不会
   因此自动成为可见列。
3. 用户明确请求 ID 时展示 ID。
4. entity-level 结果中，如果重复标签不能区分不同实体，ID 是语义必需列。
5. detail/fact 结果不会仅因为内部需要 `sale_id` 就把它暴露给用户。
6. 辅助 JOIN、filter、group、order 字段默认不是可见输出要求。
7. 列顺序属于展示层，只有用户明确要求或独立 API contract 要求时才是语义要求。
8. 不改变行粒度、`DISTINCT`、聚合、数据访问边界的安全额外字段单独记录为
   presentation 差异，不自动判定为语义错误。

比较器不能从结果列名安全推断业务含义时，会使用已验证 SQL 的 AST；如果没有
实际 SQL lineage，或仍然无法建立可靠的投影 lineage，就返回 `unassessable`，
而不是猜测别名含义。

空结果不会仅因为两边都是零行就通过。只有在当前查询的可信 AST lineage 能够
证明来源、谓词、分组和必要投影的查询结构一致时，空结果才可诊断为等价；无法
证明时标记为 `unassessable`。额外展示字段只有在分组键、`DISTINCT`、聚合和
行粒度没有变化时才可视为 presentation-only 差异。无序结果按必需字段的多重集
比较，保留重复行；明确要求顺序的 case 仍按位置比较。

默认 policy sidecar 从源码文件相对于仓库根目录的确定路径加载，不依赖进程当前
工作目录。它仍可以通过显式 `policy_path` 替换，且不会搜索任意父目录。

sidecar 同时绑定冻结的 `chatbi-demo-v2` fixture 文件、fixture 数据指纹和
权威 catalog schema 指纹。schema sidecar 保存完整的表/视图、列顺序、类型、可空性、
主键、唯一约束、外键和 CHECK 定义；provider preflight 与 sidecar 都通过同一个
共享的确定性 JSON/SHA-256 canonical contract 计算 `fixture_schema_fingerprint`。
加载时会重新计算，并要求 recomputed sidecar fingerprint、sidecar 声明值和 provider
artifact 的 `fixture_schema_fingerprint` 三者完全相等。只修改一项约束、列或关系
身份而保留原声明指纹会 fail closed。证明器只使用 sidecar 中人工确认的约束元数据；
provider artifact 的数据集/fixture/schema 指纹不匹配时直接拒绝评估。未列出 case
只使用声明的默认 policy。当前默认 policy 不
声明必需语义字段，因此这类 case 会得到 `unassessable`，不会从 oracle SQL 或
Run #6 输出自动生成 policy。

`allow_safe_extra_fields` 仅为旧 sidecar 兼容保留，不再单独授权额外投影。额外
字段必须是显式 optional，或能由可信 schema 证明为同一分组键的函数依赖字段；
否则保持 `unassessable`。

## 五个不确定案例的处理

这些决定使用上述通用政策，而不是根据模型输出调分：

- `join-01`、`join-03`：每行仍是单笔销售，`sale_id` 用于内部 detail grain，
  但问题没有明确要求展示 ID，因此不是必需可见列。
- `null-03`、`derived-03`：结果是客户级实体结果；客户显示名存在重复，
  `customer_id` 必须保留以区分实体。
- `cte-set-01`：客户身份不能因为只显示标签而合并，`customer_id` 是必需列；
  问题涉及累计销售额阈值，因此 `total_amount` 也是回答所需的值。

## Run #6 离线诊断

Run #6 的安全 artifact 为本地运行产物
`data/evaluation/a5-7/provider/chatbi-eval-v2-dev-run-6.json`。它保存了脱敏
SQL、行数、状态和正式比较结果，但没有保存完整结果行及真实结果列元数据。
因此诊断器不会重建或猜测缺失的值；这也是历史运行兼容边界。

| 项目 | 结果 |
| --- | ---: |
| DEV positive cases | 47 |
| 正式 Execution Accuracy | 21/47 = 44.68% |
| 可独立诊断 cases | 14 |
| Diagnostic Semantic Equivalence | 6/14（可诊断集合） |
| Output Shape `compliant` | 13/47 |
| Output Shape `non_compliant` | 32/47 |
| Output Shape `unassessable` | 2/47 |
| 已证明的 presentation-only 差异 | 6 |
| 已证明的语义失败 | 8 |
| `unassessable` | 33 |
| 其中结构上可继续但缺少结果行的 cases | 2 |
| 其中执行/校验未产生结果的 cases | 2 |

这里的 33 个 `unassessable` 不是失败，也不是通过。多数 case 没有人工声明的
required semantic policy；另一个 case 虽有结构信息，但安全 artifact 没有足够的
结果行来验证值和多重性，两个 case 是执行/校验没有产生结果。`Output Shape
Compliance` 的分母仍是全部 47 个
positive cases，三种 shape 状态相加为 47；shape 状态与语义诊断分母不同。

已证明的语义等价包括：

- `join-01`、`join-04`：source/join、必需投影、粒度和结果行数在可信 SQL
  结构中一致；
- `group-by-04`、`group-by-05`：`regions.region_name` 的唯一约束证明了
  `region_code, region_name` 与 `region_name` 的分组依赖；
- `join-05`：`customers.customer_id` 主键证明了省略依赖显示列的分组等价；
- `multi-join-02`：`sales.customer_id` 非空外键指向 `customers.customer_id`
  主键，且被省略的 inner join 没有额外投影/过滤作用；地区分组也由唯一约束
  支持。

已证明的语义失败包括：

- `simple-02`：选择了 `region_code`，而题目/期望需要 `region_name`，不是纯展示差异；
- `group-by-01`：可信 schema 不能证明没有销售的地区被 inner join 保留，且
  `LEFT JOIN` 与 `INNER JOIN` 的差异是可证明的；
- `join-06`：`DISTINCT customer_name` 改变 detail grain 并折叠行；
- `multi-join-01`、`multi-join-03`：缺少客户稳定身份，可能合并同名客户；
- `null-03`、`derived-03`：缺少 policy 要求的实体身份/必需字段；
- `cte-set-01`：必需投影不一致。

本轮相对于早期 A5.7f2 离线报告的变化来自实现的保守性修正，不是重新运行
provider：证明器现在分别处理 source、join、predicate、grouping、DISTINCT、
projection、ordering 和 LIMIT，并使用 `PROVEN_EQUIVALENT`、
`PROVEN_DIFFERENT`、`UNKNOWN` 三态合并。`execution_equivalent=true` 不再
直接产生 semantic PASS；历史 artifact 中被脱敏的 predicate/LIMIT literal 会
导致 `unassessable`。外连接的 null-extension 边界不会复用内连接的列等价替换，
而 `group-by-04`/`group-by-05` 的诊断理由明确记录为 `UNIQUE NOT NULL` 依赖。
正式 comparator 没有改动，因此正式 EA 仍为 21/47。

本轮 `unassessable` case 为：
`simple-01`、`simple-03`、`simple-04`、`aggregation-01`、`aggregation-02`、
`aggregation-03`、`aggregation-04`、`aggregation-05`、`aggregation-06`、
`group-by-02`、`group-by-03`、`top-k-01`、`top-k-02`、`top-k-03`、`top-k-04`、
`predicate-01`、`predicate-02`、`predicate-03`、`predicate-04`、`join-02`、
`join-03`、`multi-join-04`、`date-01`、`date-02`、`date-03`、`null-01`、`null-02`、
`empty-01`、`empty-02`、`derived-01`、`derived-02`、`cte-set-02`、`cte-set-03`。

此前列为“可证明等价”的七个历史 case：`simple-03`、`simple-04`、
`top-k-02`、`top-k-03`、`top-k-04`、`date-01`、`date-03`，现在全部为
`unassessable`；它们的安全 artifact 只有脱敏 SQL，无法独立证明 predicate、
LIMIT 或日期 literal 的等价性。`join-03` 也从早期的语义失败改为
`unassessable`：其安全 artifact 没有结果行，而且脱敏 predicate/LIMIT literal
不能证明实际值不同。

此前 A5.7f1 的人工估计 `36/47` 来自对当时运行输出的人工复核；Run #6
repository-safe artifact 没有可重放的结果行/列元数据，不能由当前诊断器独立
复现该估计。当前报告因此只报告可证明的 6 个语义通过、8 个可证明失败和
33 个未可评估案例，不把人工估计改写成新的正式指标。Run #6 artifact 仍没有
完整结果行，所以 `empty-01`/`empty-02` 在离线重算中不能冒充空结果通过；空结果
规则由带完整结果的单元测试覆盖。

## 正式指标与历史兼容

`chatbi_evaluation.py` 中的 `compare_normalized_result()` 保持不变；新模块
`chatbi_semantic_evaluation.py` 只导入并读取它，不参与生产运行。历史 Runs
#6--#11 的 artifact 仍按原格式读取，新诊断字段是可选的，不回写历史文件。
Run #5 artifact 在当前 checkout 中不可用；这是环境/历史产物可用性限制，不是
本实现重新生成或推断 Run #5 的理由。

本阶段未调用 provider，也未运行 TEST。下一步建议为
**A5.7e5 — Negative Intent Gate**：生产 NL2SQL 配置已经稳定，Run #9 暴露的
negative safe success `0/3` 可以单独处理；不要把本诊断器的 `unassessable`
案例当成生产策略调参依据。
