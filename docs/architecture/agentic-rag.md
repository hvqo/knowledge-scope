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

## M5 可靠性加固

- citation 只保留「被回答实际引用」或「工具实际使用」的候选：服务端按回答中的 `[Cx]` marker 过滤，
  模型自造或越界的 marker 在流式输出阶段即被剥离，未使用的候选不再出现在来源列表。
- 降级路径保留真实执行信息：`complete.agent_degraded=true` 时仍写入真实 `tool_calls`、
  `agent_tool_trace`，前端提示「未执行视觉分析或确定性计算」，不会把降级回答伪装成工具结果。
- 规划日志只记录失败类别、尝试次数和结构化校验错误，不写原始模型输出或用户问题。
  规划器在参数形状明确但 operation 标签错误时做一次安全归一化，避免无谓降级。
- 多轮问题复用现有 `HistoryCompressor` 与 `QueryRewriteService`：指代消解后的检索 query、压缩历史
  与改写依据进入规划与回答提示；`complete` 返回 `rewritten_query` 与历史压缩元数据。
- 单轮墙钟上限由 `KNOWLEDGE_SCOPE_AGENT_TIMEOUT_SECONDS` 控制，超时返回受控 `error` + `complete(error)`；
  每个请求独立持有流与规划状态，取消/断开连接不会污染并发会话。

## 智能体回答写入报告

`POST /api/v1/reports/{report_id}/sections/{section_id}/agent-answers/insert` 接收
`content_append` 与 1–12 条 `rag_evidence` 来源。服务端逐条校验：报告知识库与来源知识库一致、
文档属于该知识库、Evidence 页码与 source block 与服务器记录一致；历史文档缺少解析产物时
返回 warning 并保留引用信息，其他不一致直接 409/422 拒绝。校验通过后在单个事务中追加正文并持久化来源。
前端「插入报告」对话框复用该接口，不需要把回答重新交给模型生成。

## 与后续阶段的关系

- M5 及以后可把 `AgentRetrievalTool`、`AgentToolExecutor` 作为可复用工具层接入更复杂的编排，但本阶段不引入 LangGraph、多智能体或新的向量库。
- 表格计算与视觉分析仍以独立服务存在，智能体仅做编排，不改变其契约。

## M9 证据需求、充分性与单次补检索

M9 在原有「检索 → 规划 → 工具 → 回答」链路上增加一层**可检查的证据需求**，不引入新的编排框架：

1. `src/knowledge_scope/agent/evidence.py` 从问题中抽取 `EvidenceRequirement`（对象 / 年份或年份区间 /
   指标 / 是否需要图片特征 / 是否需要表格 / 是否比较），再对 ≤6 条候选做 `EvidenceAssessment`：
   逐项检查对象、年份、指标、图片、表格与比较对象覆盖，并检测候选表格之间的单位冲突。
   该检查是**词面匹配**：它只能证明候选「提到」了要求，不能证明语义上足以回答。
2. 若检查不通过且 `KNOWLEDGE_SCOPE_AGENT_EVIDENCE_SUFFICIENCY_ENABLED=true`，执行**至多一次**
   补充检索：`build_supplementary_query` 用缺失项（对象/年份/指标 + 图表/统计表）构造查询，保留
   知识库与文档范围，避免与原问题完全相同；结果按真实 `Document/Evidence/asset_index` 身份与
   初次候选去重合并，统一重编号 `C1..Cn`，并记录补检索原因、query、新增条数、覆盖前后差异与耗时。
   补检索失败时保留初次候选继续回答，并在 trace 中标记 failed。补检索与工具调用**分开计数**。
3. Planner 视觉兜底：当问题明确要求原生视觉特征（趋势/峰值/图中关系等）、候选含可读图片、
   视觉意图为 explicit 且模型直接返回 final 时，应用层强制构造一次 `analyze_image`；`auto_visual=True`
   写入 trace，避免只靠 Prompt 或模型自觉。同图不同任务/问题不判为重复调用。
4. Unified Retrieval 的 explicit 视觉配额从 2 提升到 4（`KNOWLEDGE_SCOPE_VISUAL_QUOTA_EXPLICIT`），
   修复「正确图片视觉排名第 3 却被文本候选挤出」的融合问题；配额仍只在视觉意图明确时生效。
5. `agent_max_rounds` 默认 3、`agent_max_tool_calls` 仍为 3；补检索不占工具预算，简单问题仍走短路径。
6. SSE `complete` 增加审计字段：`agent_evidence_required/missing/satisfied/coverage_before/after/conflicts`、
   `agent_supplementary_retrieval/query/added/latency_ms/error`、`agent_visual_auto_called`。
   证据不足时回答必须明确缺少什么，不拼凑；精确数值仍需确定性表格工具，视觉结论仍走 M6 安全门禁与
   可信台账/报告插入边界。

### M9.1 正确性修复与真实验收（10–15 题）

初次 M9 交付暴露了三类正确性问题：充分性判断按全局词面覆盖，可能把分别提到对象/年份/指标的无关候选拼成"充分"；
同一 Evidence 下的不同图片缺少 `asset_index`，合并/去重时会混淆；扫描表问题只调用结构化表格工具，未读取图片。
M9.1 在原分支做了最小修复：

1. **逐条 Evidence 绑定**：`assess_evidence()` 要求同一候选同时绑定对象 + 年份 + 指标（无对象时要求年份 + 指标共存）；
   "X 与 Y 的关系"会抽取 `relation_terms` 并要求直接证据；单位/统计口径冲突强制 `satisfied=false`；
   图片型问题需要图片证据绑定对象/关系，扫描表（`scanned_table`）允许图片自身满足视觉需求，但仍要求结构化表格绑定对象。
2. **多图 asset_index**：`AgentCandidate.asset_index` 通过 `get_evidence().lineage.asset_refs` 解析真实素材下标；
   `_candidate_identity` / `_merge_candidates` / `_renumber_candidates` / `_VisualExecution` / `_trace_parameters`
   全部保留 `asset_index`，`_effective_asset_index()` 用于工具调用与 trace。
3. **Planner 视觉兜底**：Planner 返回 `final` 但问题明确要求原生视觉特征时，应用层自动调用 `analyze_image`；
   比较/扫描表场景允许在表格工具之后补一次视觉调用；重复表格计划分支强制转视觉；仍受 `agent_max_tool_calls`
   与 auto_visual 一次约束，不改变 Planner 对文本/表格的自主判断。

真实验收（同一 15 题原始 gold；真实 DeepSeek Flash + Ollama `qwen3-vl:8b`；隔离 KB；
CPU 文本嵌入/reranker/视觉编码 + GPU VLM；Baseline 关闭 M9 机制）：

| 指标 | Baseline（关闭 M9） | M9.1（修复后） |
| --- | --- | --- |
| Tool Selection | 14/15 (93.3%) | 15/15 (100%) |
| Evidence Sufficiency | 8/9 (88.9%) | 9/9 (100%) |
| Supplementary Trigger | 6/9 (66.7%) | 9/9 (100%) |
| Key Evidence Coverage | 15/15 (100%) | 15/15 (100%) |
| Visual Tool Call | 14/15 (93.3%) | 15/15 (100%) |
| Answer Accuracy | 15/15 (100%) | 14/15 (93.3%) |
| E2E Success | 11/15 (73.3%) | 14/15 (93.3%) |
| 延迟 mean / median / p95 | 34.4s / 25.3s / 71.3s | 42.4s / 37.3s / 81.1s |

- 补检索：实际触发 3 次（q08/q09/q13，全部为 gold 期望不足的问题）；应补触发率 3/3，不应补避免率 12/12；
  初次充分性判断 15/15 与 gold 期望一致，补检索后最终充分性 9/9（q08/q09/q13 保持"证据不足"并明确说明缺什么）。
- 多图/扫描表：q14 同时调用 `analyze_image` + `analyze_table`，`visual_sources` 1 条、`asset_index` 与 Evidence 对齐；
  q08 先补检索再自动视觉 + 表格，最终引用两份文档。
- q03 失败为安全门禁：问题包含"峰值/精确数值"触发视觉数字门禁，回答被替换为确定性安全文本（英文趋势 + 数值打码），
  gold 中文短语不匹配；人工核对回答语义正确，属于安全策略与评测口径差异，不是新回归。
- 工具层仍有真实失败：q14 表格对比因中英文表头（国家 vs Country）报"未找到列"；q08/q14 的
  `analyze_image(extract_table)` 未提取出表格列；最终回答基于原始证据文本对照，已明确说明未使用工具计算。
- 默认开关决策：M9.1 在 E2E、充分性、补检索触发、视觉调用上均有明确收益且未发现重要回归（延迟增加约 8 秒，
  主要来自补检索与自动视觉），因此保持 `agent_evidence_sufficiency_enabled=true`、`agent_visual_tool_auto=true`；
  仍保留配置开关用于回退。
- 运行脚本：`scripts/m9_acceptance.py --fixtures <fixtures.json> --gold docs/benchmarks/m9-acceptance-gold.json --output result.json`
  （支持 `--only q01,q02` 单题复验）。

### 已知局限

- 证据充分性为逐条词面绑定，仍不能证明语义支撑：对「提到但语义不相关」「跨语言表头」「VLM 提取失败」等情况会误判；
  M9.1 通过保守缺失提示缓解，但未引入语义模型自评（避免以模型自评代替事实校验）。
- 自动视觉/补检索增加延迟（mean 34.4s → 42.4s）；扫描表跨语言表头仍是结构化表格工具的真实限制，回答只能降级为原文对照。
- 本数据集没有真实国别统计报告，使用真实技术/研究报告与已知表格 PDF 做技术验证，不构成国别业务结论。
- 视觉配额 / 自动视觉调用只覆盖“是否有图片与意图”的启发式判断，不保证视觉证据与问题语义匹配。
