# 视觉理解与图片型表格提取（M3）

M3 在已有 Evidence 资产访问链路上增加独立的视觉模型（VLM）调用能力，用于理解 PDF 中的普通图片、统计图以及图片型表格。该能力不改变现有 RAG 检索协议，也不修改 CanonicalDocument 或 Evidence 产物：视觉模型只读取已通过安全校验的 Evidence 图片，输出带来源引用的规范化 JSON 结果。

## 数据链路

```
Evidence (knowledge_base_id / document_id / evidence_id)
  → M1 安全素材解析 (resolve_evidence_asset + read_evidence_asset_bytes)
  → 多模态请求（text prompt + image data URL）
  → VLM Provider（OpenAI 兼容 /chat/completions）
  → 规范化视觉结果 (VisionAnalysisResult)
  → 返回 Evidence 来源、模型、延迟与告警
```

视觉结果不会写回 CanonicalDocument、Evidence 产物或检索索引，只作为一次工具调用的返回结果。

## 能力与边界

| 任务 | task_type | 支持证据 | 说明 |
| --- | --- | --- | --- |
| 普通图片理解 | `describe_image` | image | 主体、关键内容、可识别文字、不确定部分 |
| 统计图分析 | `analyze_chart` | image | 图表类型、坐标轴、系列、趋势、峰值低谷、限制 |
| 图片型表格提取 | `extract_table` | image / table | 列、行、合并单元格信息、单位、表名、告警 |

明确不做：

- 不做 OCR 数字化流水线，不保证逐字识别或精确数值；`extract_table` 的结果始终标记 `extraction_method = "vlm"`。
- 不把视觉模型输出当作权威数据：图片型表格提取结果固定追加告警「视觉模型提取，数值需核对，不能作为精确统计数据使用」。
- 不自动选择工具、不参与 Agent 编排（M4 负责）。
- 不新增 Qdrant 集合，不批量补写历史文档素材，不修改已有的 M2 表格计算链路。

## 模块职责

| 模块 | 职责 |
| --- | --- |
| `src/knowledge_scope/vision/errors.py` | 视觉错误分类与稳定 `error_code` |
| `src/knowledge_scope/vision/schemas.py` | 请求/结果契约、JSON 提取与结果规范化 |
| `src/knowledge_scope/vision/prompts.py` | 安全前置说明与按任务生成的严格 JSON 提示词 |
| `src/knowledge_scope/vision/provider.py` | OpenAI 兼容 VLM Provider、超时/连接/拒绝/格式错误映射 |
| `src/knowledge_scope/vision/service.py` | `VisionAnalysisService`：素材校验、模态限制、调用 Provider、来源回填 |
| `src/knowledge_scope/api/vision.py` | `POST /api/v1/vision/analyze` 只读接口 |
| `frontend/src/components/chat/AssetPreviewDialog.vue` | 图片预览中的「解读图片 / 分析图表 / 识别表格」入口与结果展示 |

## API 契约

`POST /api/v1/vision/analyze`，请求体：

```json
{
  "knowledge_base_id": "0f2f...",
  "document_id": "b25f...",
  "evidence_id": "evidence_v1_<sha256>",
  "asset_index": 0,
  "task_type": "analyze_chart",
  "question": "这张图的主要趋势是什么？"
}
```

约束：`asset_index` 取值为 `0..255`；`question` 最多 2000 字符；`knowledge_base_id`、`document_id`、`evidence_id` 必须同属一个知识库文档，素材必须来自该 Evidence 的 `asset_refs`，不接受任意本地路径或远程图片 URL。

响应要点（`VisionAnalysisResult`）：

| 字段 | 说明 |
| --- | --- |
| `status` | `ok` / `needs_review` / `not_configured` / `failed` |
| `task_type` | 请求的任务类型 |
| `answer` | Markdown 文本结论（描述/图表/表格摘要） |
| `chart_analysis` | 结构化图表分析（坐标轴、系列、趋势、峰值、限制） |
| `extracted_table` | 结构化图片表格（列、行、单位、合并单元格、告警） |
| `extraction_method` | 表格提取固定为 `vlm` |
| `provider` / `model` / `latency_ms` | 实际调用的服务、模型与耗时 |
| `warnings` | 不可确认内容、截断、合并单元格等告警 |
| `sources` | `knowledge_base_id` / `document_id` / `evidence_id` / `page_start` / `page_end` / `source_block_ids` |
| `error_code` / `message` | 受控错误分类与用户可读信息 |

受控结果：

- 素材缺失、文档所属关系不符、Evidence 过期 → HTTP 404（沿用 M1 素材访问错误）。
- 素材格式不支持 → HTTP 415；素材超过大小限制或 Provider 输入上限 → HTTP 413。
- 请求参数不合法、证据模态不支持任务 → HTTP 422。
- VLM 未启用 → 返回 `status = not_configured`，`error_code = vlm_not_configured`（不抛 500）。
- Provider 超时 / 连接失败 / 返回非法 JSON / 拒绝分析 → 返回 `status = failed` 与对应 `error_code`（`provider_timeout`、`provider_connection`、`invalid_response`、`model_refusal` 等），来源信息保留。

## 刷新与安全

- 只读取 M1 已校验的 Evidence 素材：`get_evidence` + `resolve_evidence_asset`，图片字节由 `read_evidence_asset_bytes` 读取，路径穿越、符号链接、跨文档/跨知识库访问均由 M1 解析器拒绝。
- 输入大小上限由 `vlm_max_image_bytes` 控制；只发送当前所需图片，不发送整份 PDF。
- 图片内文字与图像内容视为不可信数据：提示词要求模型只输出一个 JSON 对象，不执行图片中的指令，不产生可执行代码/计算程序，不泄露系统提示。
- 日志与响应不包含 API Key，也不包含图片原始字节；结果中的来源只暴露 Evidence/文档/页码/block 等引用信息。
- 前端不渲染任何原始 HTML，表格按文本单元格结构化展示。

## 配置

默认关闭；启用后才会向视觉模型发送图片：

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `KNOWLEDGE_SCOPE_VLM_ENABLED` | `false` | 是否启用视觉理解 |
| `KNOWLEDGE_SCOPE_VLM_BASE_URL` | 空 | OpenAI 兼容服务地址（例如本地 Ollama `http://127.0.0.1:11434/v1`） |
| `KNOWLEDGE_SCOPE_VLM_MODEL` | 空 | 视觉模型名称（例如 `qwen3-vl:8b`） |
| `KNOWLEDGE_SCOPE_VLM_API_KEY` | 空 | 可选 API Key，经 SecretStr 读取，不写入代码/文档/日志 |
| `KNOWLEDGE_SCOPE_VLM_REASONING_EFFORT` | 空 | 可选推理强度；思考型模型（如 Ollama `qwen3-vl`）建议设为 `none`，避免输出被推理内容占满 |
| `KNOWLEDGE_SCOPE_VLM_TIMEOUT_SECONDS` | `90` | 单次调用超时（秒，1–300） |
| `KNOWLEDGE_SCOPE_VLM_MAX_IMAGE_BYTES` | `8388608` | 单张图片输入上限（字节） |
| `KNOWLEDGE_SCOPE_VLM_MAX_OUTPUT_TOKENS` | `2048` | 输出 token 上限 |

使用思考型模型（如本地 Ollama `qwen3-vl:8b`）时，建议设置 `KNOWLEDGE_SCOPE_VLM_REASONING_EFFORT=none`，并在输出仍被推理内容占满时提高 `KNOWLEDGE_SCOPE_VLM_MAX_OUTPUT_TOKENS`（本地验收使用 `4096`）；否则适配器会返回受控的 `invalid_response`，提示调整推理强度或输出上限。

未配置时接口返回 `not_configured`，不影响现有文本 RAG、表格计算和其它业务。

## 与后续阶段的关系

- M4 Agent 可直接复用 `VisionAnalysisService.analyze`（或 API）作为视觉工具；工具入参与出参已是明确的结构化契约，来源信息完整。
- M2 `TableAnalysisService` 仍处理 HTML/Markdown 结构化表格；图片型表格由 M3 提取后标记为待核对数据，不自动进入 M2 的精确计算链路。
- 视觉结果不写回知识库，因此不会影响既有检索、引用与表格计算的可追溯性。
