# ChatBI 数据分析工作区

## 边界

数据分析工作区复用现有 ChatBI Agent 的生产链路，不直接调用 NL2SQL、评测接口或任意 SQL 接口：

```text
注册数据源
  → 可信 Schema Discovery
  → ChatBIEligibilityService
  → ChatBIAgentService
  → A5.3 校验
  → A5.4 只读执行与结果归一化
  → 有界产品结果
```

HTTP 接口为：

```text
POST /api/v1/chatbi/data-sources/{datasource_id}/ask
{
  "question": "统计每个地区的客户数量"
}
```

`datasource_id` 来自已注册数据源，问题是唯一的用户输入。服务端不会接受客户端提供的
`SchemaSnapshot`、`ValidatedSQL` 或 SQL 字符串，也不会为前端提供编辑或执行 SQL 的路径。

## 产品结果契约

接口只返回当前分析所需的有界字段：

- `answer`、`columns`、`rows`、`row_count`；
- `execution_status`、`truncated`、`truncation_reason`、`warnings`；
- `datasource_id`、耗时和已脱敏的 `redacted_sql`（默认折叠展示）；
- Eligibility 的产品状态和用户可读消息。

Provider、token、schema fingerprint、gate version、trace、SQL 尝试次数和其他评测诊断不会
进入产品响应。结果行沿用 A5.4 的有界归一化结果，表格始终是主展示；图表只根据返回的
类别列与数值列做确定性推断，不能替代原始表格。

`CLARIFY`、`REFUSE` 和 `UNAVAILABLE` 在 Agent 内部终止，前端只展示应用拥有的安全消息，
不会展示 SQL、结果行或诊断信息。

## 会话和生命周期

当前前端历史是页面会话内的 Pinia 状态，不写入后端，也不声称具备持久化历史。离开页面后
记录会消失。一次请求使用一个 `AbortController`；页面卸载或停止分析时取消请求，并通过
请求代次检查避免旧响应写入新会话。

数据源元数据使用现有 `/chatbi/data-sources` 接口，不在前端复制数据源 CRUD。只有启用且已
注册的数据源可以被选择；加载、空列表和请求失败均显示可操作的安全状态。

跨前端、API、外部业务数据库和 LLM provider 的操作不是全局原子事务。HTTP 层只负责调用
Agent 并投影结果；可信发现、SQL 校验、只读事务、结果资源限制和凭据解析仍由现有 ChatBI
服务负责。
