# 多模态安全加固（P0）

本阶段的目标不是提升图片识别精度，而是阻止**未经核验的视觉数值**被当作已核实结论，进入智能体回答、SSE 流、报告正文或导出文档。视觉模型（VLM）只被视为“不可信的阅读辅助”，不是权威数据源。

## 1. 事实来源契约

应用层为每个结果标注事实状态（`fact_status`），模型自述的置信度不参与判定：

| fact_status | 含义 | 典型来源 |
| --- | --- | --- |
| `source_verified` | 来自结构化表格，并由确定性计算工具计算 | `TableAnalysisService` 成功执行后的结果 |
| `visual_estimate` | 视觉模型从图片中读出的估计值 | `analyze_chart` / `describe_image` |
| `unverified` | 来自图片型表格的 VLM 抽取，必须人工核对 | `extract_table` 任务 |
| `not_available` | 没有可用的视觉或结构化来源 | 视觉失败、未配置或未使用视觉工具 |

`VisionAnalysisResult` 增加 `fact_status` 与 `numeric_claims`；`AgentToolOutcome`、`RAGCompleteData`、`RAGCitation`/`RAGVisualSource` 逐层传递同一状态。`status=ok` 仅表示请求与 JSON 结构成功，不代表数字可信。

## 2. 服务端视觉数值输出闸门

- `src/knowledge_scope/vision/safety.py` 根据问题文本做可测试的风险分类：是否要求精确数值/增长率/排名/汇总，是否指向图片类素材；年份、季度标签、引用标记不会被误判为业务数值。
- 高风险且没有可信结构化结果时，智能体不再调用最终答案模型。应用直接生成确定性回答：说明“无法核验视觉数字”，可选保留去掉数字后的视觉描述，并附上实际分析的图片；`complete.safety_gate` 为 `visual_numeric_withheld` 或 `vision_unavailable`，`finish_reason="safety_gate"`。
- SSE 保护：模型**从未启动**，因此不会出现“先流式输出 120 万元、后补警告”的泄漏路径；高风险轮次只发送一个确定性的 `answer_delta`，随后是 `citations` 与 `complete`。
- 当表格确定性计算已成功（`source_verified`）时，闸门不会拦截；即使问题中包含“最低/增长/多少”等词。
- 视觉失败、未配置、超时或证据缺失时，不会回退到普通 RAG 继续“猜”数字；如需引用文本证据，必须明确说明视觉未完成。

## 3. 自动展示实际使用的图片

- 智能体实际执行 `analyze_image` 时，`complete` 事件携带 `visual_sources`（仅包含本次真实使用的图片 Evidence：知识库、文档、Evidence ID、页码、来源块、资产序号、事实状态）。这些字段全部由服务端从工具执行记录生成，模型不能提供 URL 或 Evidence ID。
- 前端在回答下方直接渲染“本次分析图片”卡片（`VisualSourcesCard.vue`）：缩略图通过既有 `GET /documents/{id}/evidence/{evidence_id}/assets/{index}` 读取；点击打开原尺寸预览；“打开原文 PDF”跳到对应页。
- 历史文档缺少解析产物时显示“原图不可用，无法核验”，不会用无关图片占位；视觉结果一律标注“视觉分析结果需核对”。
- 聊天历史持久化暂不包含这些附加元数据（需要数据库迁移）。当前前端会在同一会话内按回答正文缓存安全元数据，因此刚生成的回答（包括保存后重新渲染）仍能看到卡片；刷新页面后卡片不会恢复，原始来源抽屉与引用仍可追溯。

## 4. 报告插入安全边界

`POST /api/v1/reports/{report_id}/sections/{section_id}/agent-answers/insert` 不再接受客户端提交的正文与来源，只接受 `chat_message_id`：

- 服务端读取已持久化的智能体回答及其引用，校验会话与报告属于同一知识库；页码、来源块、Evidence 归属沿用原有校验。
- 正文与来源均由服务器从聊天记录重建，客户端无法伪造 `content_append`、`sources` 或 `verified=true`（多余字段返回 422）。
- 回答包含图片引用且正文含数值声明时拒绝插入（409）：未核验的视觉数值不能自动成为“已核实分析”。
- 不含高风险数值的视觉回答允许插入，但正文前自动加上“⚠️ 视觉分析草稿，需人工核对，不作为精确统计结论”，并在响应 `warnings` 中返回；DOCX/PDF 导出保留该提示。
- 结构化表格计算结果（`source_verified`）维持原有插入行为。

## 5. 相关测试

- `tests/test_multimodal_safety.py`：风险分类、数字抽取（忽略年份/季度）、闸门回答去除数字、SSE 不泄漏未核验数字、表格确定性计算不被拦截、视觉失败不回退 RAG。
- `tests/test_reports_agent_insert.py`：服务端正文重建、视觉数值 409、表格回答可插入、跨库/页码/伪造字段/历史缺失等边界。
- 前端 `VisualSourcesCard` 与聊天消息动作测试。

## 6. 已知限制

- 闸门针对“可识别的精确数值/图表数值问题”；无法识别所有语义变体，视觉模型仍可能在描述性文字中产生难以检测的错误。
- 图片型表格抽取固定标记为 `unverified`，不进入自动精确统计。
- 报告插入只验证来源元数据与关联关系，不能证明正文每句话都被来源支持；人工编辑内容不属于系统核实。
- 历史图片缺失、VLM 不稳定、跨语言表头对比等既有局限仍然存在。

## 7. 真实验收记录（2026-10-23）

环境：DeepSeek Flash（规划与最终回答）+ Ollama `qwen3-vl:8b`（视觉），临时知识库（4 份非敏感测试 PDF：图表、扫描表格、GDP 表格 A/B、数学教材），服务端 API 使用最新分支代码启动。

| 场景 | 实际结果 |
| --- | --- |
| 图表精确数值问题（“最低季度具体销售额”） | 真实 qwen3-vl:8b 返回含数字的描述；`complete.safety_gate="visual_numeric_withheld"`、`fact_status="visual_estimate"`、`finish_reason="safety_gate"`，回答中无具体数字（显示“〔数值未经核验〕”），`visual_sources=1`，前端展示告警横幅与“本次分析图片”缩略图（`/tmp/safety-gated-answer.png`）。 |
| 结构化表格增长（表 1，m2-pdf-a.pdf） | 表格工具实际执行，回答引用 10.6977%、公式与 460 亿美元；`safety_gate="none"`、`fact_status="source_verified"`，未被视觉闸门拦截。 |
| 报告插入（表格回答） | `agent-answers/insert` 仅提交 `chat_message_id`，服务端重建正文与 4 条来源，HTTP 201。 |
| 报告插入（视觉草稿，无精确数字） | HTTP 201，正文自动加“⚠️ 视觉分析草稿，需人工核对”提示，`warnings` 返回提示；导出保留该提示。 |
| 报告插入（伪造图片数值回答） | HTTP 409：“该回答包含未经核验的视觉数值，不能自动插入为已核实分析”。客户端伪造 `content_append/sources/verified` 字段返回 422。 |
| 标准 RAG 回归 | `/rag/query` 标准模式正常返回 588 个 SSE 事件，无错误。 |
| 浏览器端 | 浏览器中选择临时知识库、开启智能体提问图表数值问题：最终回答为确定性安全文本，回答下方直接出现图片卡片（缩略图可加载、可查看原图、可打开原文 PDF），无需先打开来源抽屉。 |

未完成项：P1 的有界二次检索与错误反馈入口本轮未实现（需要更多检索/前端改造，已记录为后续事项）；历史会话刷新后安全卡片不恢复；报告插入只校验来源元数据，不证明正文逐句被来源支持。
