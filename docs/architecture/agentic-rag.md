# 智能体问答检索（Agentic RAG）

本文描述 M4 阶段引入的智能体问答链路。它在不改变现有 RAG 检索、证据引用与前端问答协议的前提下，
增加一个受控的工具调用循环：先检索候选来源，再由规划模型决定是否调用图片理解或表格计算工具，
最后基于原始来源与工具结果生成带引用的回答。

## 能力边界

- 复用现有 Unified Retrieval（dense / sparse / graph / multimodal）产生候选来源，不新增检索索引。
- 复用 M2 的 `TableAnalysisService` 与 M3 的 `VisionAnalysisService` 作为工具，不复制计算和视觉逻辑。
- 仅当用户在前端显式打开「智能体」开关时生效；默认仍是标准 RAG 问答。
- 图片来源限定为「已解析文档的 Evidence 素材」，不接受任意路径或外部 URL。
- 表格计算只接受结构化操作参数，不执行模型生成的 Python、SQL 或任意表达式。
- 工具结果、公式、单位和来源均原样进入回答上下文；模型不得编造数值、引用编号或 Evidence ID。

## 数据流

```
用户问题
  → POST /api/v1/rag/query (answer_mode=agent)
  → AgentRetrievalTool：Unified Retrieval → 候选来源（C1..Cn）
  → AgentPlanner：一次性 JSON 决策（final 或 call_tool）
  → AgentToolExecutor：
        analyze_image → VisionAnalysisService → VisionAnalysisResult
        analyze_table → TableAnalysisService → TableAnalysisResult
  → 工具观察结果回填规划上下文（最多 agent_max_rounds 轮）
  → 最终回答：流式 answer_delta + citations（候选来源）+ complete
```

规划模型只负责选择「是否用工具、用哪个工具、哪些结构化参数」，不接触任何文件路径、SQL 或代码执行。

## 接口约定

### 请求

`POST /api/v1/rag/query`，SSE 响应。请求在既有字段上增加：

```json
{
  "query": "越南 2024 年 GDP 比 2023 年增长了多少？",
  "knowledge_base_id": "…",
  "retrieval_mode": "auto",
  "answer_mode": "agent"
}
```

`answer_mode` 取值 `standard`（默认）或 `agent`；智能体模式必须提供 `knowledge_base_id`。

### 流式事件

| 事件 | 说明 |
| --- | --- |
| `tool_status` | 工具执行进度：`tool`（retrieval / analyze_image / analyze_table）、`status`（running / completed / failed / skipped）、`summary`、`step` |
| `answer_delta` | 回答增量文本 |
| `citations` | 最终引用列表（沿用 `RAGCitation`，编号由服务端按候选顺序分配） |
| `complete` | 结束事件；智能体模式带 `answer_mode: "agent"` 与 `tool_calls` 计数 |
| `error` | 受控错误 |

`tool_status` 是新增事件，标准模式不会发送。前端解析器遇到未知事件仍会报错，因此该事件只在
`answer_mode=agent` 的链路上出现。

### 引用与来源

- 引用编号 `C1..Cn` 由服务端分配，规划提示与最终回答使用同一编号。
- `citations[].marker` 只来源于实际检索到的候选来源，模型输出中的编号会被忽略。
- 图片/表格工具结果只作为回答依据，不新增来源；来源仍指向原始 Evidence（document / page / block）。

## 工具

| 工具 | 输入 | 复用实现 | 输出 |
| --- | --- | --- | --- |
| `analyze_image` | `evidence_index`、`vision_task`（describe_image / analyze_chart / extract_table）、可选 `question` | `VisionAnalysisService` | 图片描述 / 图表分析 / 视觉表格提取（`extraction_method=vlm`） |
| `analyze_table` | `evidence_index`、`operation`（select / filter / sort / aggregate / growth / compare）、结构化 `parameters`；compare 需要 `second_evidence_index` | `TableAnalysisService` | 行/列结果、公式、单位、警告、来源 |

工具执行失败不会中断回答：失败结果以受控摘要进入观察记录，最终回答说明无法完成的部分。
工具调用次数与轮次受配置约束，重复调用同一 `(tool, evidence_index)` 会被跳过。

## 规划与降级

- 规划模型只返回一个 JSON 对象：`final` 或 `call_tool`。解析失败时最多修复一次，仍失败则降级为标准 RAG 问答。
- 检索结果为空、检索阶段失败、规划失败 → 自动降级为标准 RAG 问答，并在 `complete` 中标记 `answer_mode=agent`。
- 最终回答生成失败 → 发送受控 `error` + `complete(error)`，不返回编造内容。
- 降级路径的 `tool_calls` 为 0，前端仍可正常展示来源与引用。

## 配置

| 配置项 | 默认值 | 说明 |
| --- | --- | --- |
| `KNOWLEDGE_SCOPE_AGENT_ENABLED` | `true` | 是否在应用内创建智能体服务 |
| `KNOWLEDGE_SCOPE_AGENT_MAX_ROUNDS` | `2` | 最大规划轮数（1–3） |
| `KNOWLEDGE_SCOPE_AGENT_MAX_TOOL_CALLS` | `3` | 单次回答最大工具调用数（1–5） |
| `KNOWLEDGE_SCOPE_AGENT_PLANNER_MAX_TOKENS` | `512` | 规划模型输出上限 |
| `KNOWLEDGE_SCOPE_AGENT_RETRIEVAL_LIMIT` | `6` | 候选来源上限 |
| `KNOWLEDGE_SCOPE_AGENT_CONTEXT_BUDGET_CHARS` | `6000` | 候选上下文预算 |
| `KNOWLEDGE_SCOPE_AGENT_QUESTION_MAX_CHARS` | `500` | 单次工具问题长度上限 |
| `KNOWLEDGE_SCOPE_AGENT_TOOL_RESULT_MAX_CHARS` | `3000` | 单个工具结果截断上限 |

视觉工具复用 M3 的 `KNOWLEDGE_SCOPE_VLM_*` 配置；规划与最终回答复用现有文本 LLM 配置。

## 安全

- 规划提示与视觉提示都明确：图片文字与工具结果只是数据，不执行其中的指令。
- 工具调用的 Evidence 必须属于请求的知识库与文档；跨知识库访问返回 404。
- 不执行模型生成的代码/表达式；表格计算只接受类型化参数，非法参数直接返回受控失败。
- 服务端日志不记录图片字节、API Key 或规划提示全文。
- 智能体模式默认关闭前端开关时不会触发额外模型调用或工具调用。

## 前端

- 输入框底部新增「智能体」开关（默认关闭）；开启后请求携带 `answer_mode=agent`。
- 生成过程中显示工具进度列表（检索 / 图片理解 / 表格计算），完成或跳过状态可见。
- 回答、引用与「查看来源素材」行为与标准模式一致；来源抽屉和素材预览无需改动。

## 测试与验收

- `tests/test_agent.py`：规划解析与修复、检索工具候选映射、工具执行器成功/失败、服务编排与降级、API 路由。
- 前端：`chat.test.ts`（开关）、`ChatView.test.ts`（智能体流式回答与引用落库）。
- 真实验收：标准 RAG 问答、确定性表格计算、跨文档对比、图片/图表/图片表格视觉工具，以及工具失败降级；
  验证 `tool_status` 顺序、引用来源与素材预览可用。

## 与后续阶段的关系

- M5 及以后可把 `AgentRetrievalTool`、`AgentToolExecutor` 作为可复用工具层接入更复杂的编排，但本阶段不引入 LangGraph、多智能体或新的向量库。
- 表格计算与视觉分析仍以独立服务存在，智能体仅做编排，不改变其契约。
