# KnowledgeScope 多模态 Agentic RAG：真实指标与技术取舍

> 交接文档 B。基于冻结基线 `734fe651b6b0bb242166220ca30fea8ad1a0c7eb` 与仓库已归档评测结果，以及 2026-10-24 ~ 2026-10-26 的真实本机运行记录整理。
>
> **口径原则**：不同评测集、不同 gold、不同打分规则的数字不能互相比较；关键词/数字匹配不等同于语义准确率；未做人工逐项核验的数字不得写成“人工语义准确率”。

## 1. 评测环境

| 项 | 值 |
| --- | --- |
| 主机 | 单机，RTX 3060 12GB |
| Python / uv | 3.12.13 / uv 0.11.29 |
| PostgreSQL / Qdrant / Neo4j | 本地 Docker，5433 / 6333 / 7687 |
| 文本嵌入 | Qwen/Qwen3-Embedding-0.6B（1024 维，GPU/CPU） |
| Reranker | BGE-reranker-v2-m3 |
| 视觉嵌入 | Qwen/Qwen3-VL-Embedding-2B（2048 维，本地权重） |
| 生成模型 | DeepSeek Flash（真实 API） |
| 视觉理解 | Ollama qwen3-vl:8b（本地） |
| 基线 Qdrant | chunks 7567 / representations 79338 / 视觉集合按验收临时创建 |

12GB 显存约束下的稳定部署组合（真实验收使用）：

- 文本嵌入 / reranker / 视觉查询编码：CPU。
- VLM（qwen3-vl:8b）：GPU（Ollama，约 7.2GB）。
- 实测 GPU 峰值约 8.9GB，未 OOM；若把文本/视觉模型都放 GPU 会与 VLM 冲突 OOM（曾实测 `Cannot copy out of meta tensor` 的并发加载问题，已用全局模型加载锁修复）。

## 2. 文本 RAG 指标

来源：`docs/benchmarks/a2-5-retrieval-system-benchmark.md`、`docs/benchmarks/a3-7-hybrid-evaluation.md`。

评测集 A2.1：108 条人工查询（dev 72 / test 36），Qdrant + BGE Top-10。

| 方案（test 36） | H@1 | H@3 | H@5 | H@10 | MRR |
| --- | --- | --- | --- | --- | --- |
| Qdrant + BGE（主方案） | 0.7778 | 0.8333 | 0.8611 | 0.8611 | 0.8079 |
| dense 仅 BGE（早期） | 0.6389 | - | - | - | 0.7313 |
| hybrid dense_sparse_bge（A3.7） | 0.8056 | 0.9722 | 1.0 | 1.0 | 0.8852 |
| hybrid dense_graph_bge（A3.7） | 0.7778 | - | - | 0.9444 | 0.8449 |
| unified（chunk 级，A3.7） | 0.1944 | - | - | 1.0 | 0.4393 |

dev 72 上主方案：H@1 0.7917 / H@3 0.8750 / H@5 0.9167 / H@10 0.9722 / MRR 0.8494。

注意：

- unified 的 H@1 低是 chunk 级指标受多模态/图片 evidence 进入 Top10 影响，H@10 仍为 1.0。
- hybrid 数字来自独立评测（A3.7）的专用配置，不能当成线上默认路径的指标。

## 3. 多模态表示检索（A4.5）

来源：`docs/benchmarks/a4-5-retrieval-evaluation.md`。

测试集：v2 表示索引评测，gold 为 source-derived（非人工逐条标注），test 集：

| 方案 | Evidence H@10 | MRR |
| --- | --- | --- |
| multimodal_representation | 0.5833 | 0.1929 |
| unified | 0.6111 | 0.1909 |
| dense_bge | 0 | 0 |

说明：这是 2026-10 之前的表示检索结果；M7 之后新增的视觉向量分支是**独立实验**（见下），不能与 A4.5 混用。

## 4. M7 视觉向量实验（A/B/C）

来源：`docs/architecture/visual-retrieval.md`。数据：14 份 retained 教材文档（A1.5 原始 MinerU 产物），502 张图片；28 条人工视觉查询（15 条无 caption）；gold 为人工指定的图片 Evidence。

| 方案 | Recall@5 | Recall@10 | MRR |
| --- | --- | --- | --- |
| A. 文本 caption/context 向量 | 25.0% | 35.7% | 0.197 |
| B. selective 视觉索引（policy） | 92.9% | 92.9% | 0.866 |
| C. 全量视觉索引 | 92.9% | 96.4% | 0.871 |

- 无 caption 的 15 条：A 4/15，B/C 13/15。
- 索引策略：全量 502 张中 policy 仅判定 4 张 text_only，498 张 text_and_visual。
- 成本：全量编码 84.0s、原始向量 4.11MB；selective ≈83.4s、4.08MB（节省 <1%）。
- 结论：视觉向量相对文本表示有显著收益；本语料 caption 覆盖不足（28.6%），选择性索引没有有意义的成本节省，因此默认全量视觉索引，并在报告中如实记录该负向结论。

实验命令（仓库脚本）：`scripts/m7_prepare_eval_data.py`、`scripts/m7_retrieval_experiment.py`。

## 5. M7.1 工程验证（真实 Qdrant + 真实模型）

来源：`docs/architecture/visual-retrieval.md` §10–§14。

### 5.1 候选链诊断（10 条视觉查询，4 份真实报告 / 80 张图片）

| 环节 | 命中 |
| --- | --- |
| Visual Top-8 | 9/10 |
| Unified Top-8 | 8/10 |
| Agent Top-6 | 8/10 |

失败原因：q02 正确图片在视觉 rank3，但 unified 视觉配额 2 被其他视觉候选占满；q04 独立视觉检索 Top-8 未命中。

### 5.2 真实业务验收（12 条查询：10 视觉 + 2 文本）

| 指标 | 结果 |
| --- | --- |
| visual 命中 | 8/10 |
| analyze_image 调用 | 9/10 |
| 文本问答 | 2/2 |
| safety gate | 8 × visual_unverified / 1 × vision_unavailable / 1 × none |
| E2E 延迟 | mean 48.2s / median 46.3s / p95 79.0s |

### 5.3 性能实测（GPU/CPU 混合）

| 项 | P50 | P95 |
| --- | --- | --- |
| 视觉查询编码（GPU） | 20.69ms | 21.26ms |
| 视觉查询编码（CPU bf16） | 982.8ms | 1125.65ms |
| Qdrant 视觉查询 | 4.45ms | 5.64ms |
| BGE reranker | 18.63ms | 19.28ms |

模型加载 1.78s，GPU 显存峰值 4068.7MB；12GB 混合部署峰值约 8.9GB。

## 6. M8 自动入库指标

来源：`docs/architecture/auto-ingestion.md` 与 2026-10-26 M10 最终演示真实运行记录。

演示（本次冻结基线，网页上传、隔离 KB `002c82aa…`）：

| 文档 | 规模 | 关键阶段耗时（CPU 文本/视觉 + GPU VLM） |
| --- | --- | --- |
| m3-vision-acceptance.pdf（2 图） | chunks 1 / images 2 | parse 25.4s；text 2.6s；multimodal 3.3s；visual 87.4s；publish 80.6s；verify 0.24s |
| m2-pdf-a.pdf（无图） | chunks 1 / images 0 | parse 21.3s；text 0.87s；multimodal 2.59s；visual 0.007s；publish 62.5s；verify 0.26s |

结论与瓶颈：

- 全链路无需手工 CLI，网页上传后自动完成；两个文档从上传到 ready 约 3.5 分钟。
- 当前主要瓶颈是 **publish（60–80s）** 与 **CPU 视觉编码（2 图约 87s，含模型加载）**；GPU 视觉编码可显著加速（M7 实测 43ms/25 图预热后）。
- 失败恢复：解析失败后恢复权限并 retry → completed；BM25 记录缺失 → verify_bm25 失败；视觉活动代缺图片点 → verify_visual_index 失败；已完成文档 retry → 409 rebuild_not_supported。
- 图片计数语义：`image_count` 为原始图片数；`visual_indexed_count/missing/invalid/skipped_by_policy` 独立统计；视觉禁用时仍保留原始图片数，`visual_status=not_enabled`。

## 7. M9 证据充分性与补检索评测

来源：`docs/architecture/agentic-rag.md` M9 章节、`docs/benchmarks/m9-acceptance-gold.json`、`scripts/m9_acceptance.py`。15 道技术测试题（非业务国别研究集），未修改 gold 口径，真实 DeepSeek Flash + Ollama qwen3-vl:8b。

| 指标 | baseline（关闭 M9 机制） | M9.1（最终） |
| --- | --- | --- |
| Tool Selection | 14/15 | 15/15 |
| Evidence Sufficiency | 8/9 | 9/9 |
| Supplementary Trigger | 6/9 | 9/9 |
| Key Evidence Coverage | 15/15 | 15/15 |
| Visual Tool Call | 14/15 | 15/15 |
| Answer 匹配 | 15/15 | 14/15 |
| E2E | 11/15 | 14/15 |
| 延迟 mean / median / p95 | 34.4 / 25.3 / 71.3s | 42.4 / 37.3 / 81.1s |

补充检索实际触发 3 次（q08/q09/q13）；应补触发 3/3、不应补避免 12/12；工具调用 35 次，3 次执行失败（q08 视觉表格提取失败、q14 跨语言表头 + 视觉表格提取失败）。

失败与口径说明：

- q03 失败是安全门禁把回答替换为确定性安全文本（中文 gold 短语不匹配），人工核对语义正确。
- q08 回答中“相差约 440 亿美元”是模型归纳值，非表格工具输出；安全标记已归为未核验。
- q14 表格工具因中文“国家” vs 英文“Country”表头失败；回答降级为证据原文对照并说明限制。
- 14/15 E2E 是技术测试集匹配率，**不是整体语义准确率**。
- 默认开关：收益明确（Sufficiency/Supplementary/E2E 提升）且无重要回归，保持 `agent_evidence_sufficiency_enabled=true`、`agent_visual_tool_auto=true`；延迟增加约 8 秒。

## 8. M6 安全 / 可信插入验证

- 伪造 assistant 消息 + 报告插入 → HTTP 409“没有可核验的服务端生成记录”。
- 可信表格回答（10.6977%）插入 → 201，来源 9 条（8 Evidence + 1 chunk）。
- 视觉未核验数值回答（含精确数字）插入 → 409；无数字视觉草稿插入 → 201 + “⚠️ 视觉分析草稿，需人工核对”正文标记。
- 混合工具回答：表格结果不会把视觉数字提升为 `source_verified`。
- 高风险视觉数值问题：服务端确定性安全回答，SSE 不泄漏未核验数字。

## 9. 技术取舍

| 取舍 | 选择 | 理由 | 代价/限制 |
| --- | --- | --- | --- |
| 视觉编码器加载 | 直接用 transformers `Qwen3VLForConditionalGeneration` + 自定义 processor，不用 SentenceTransformer | ST 加载 `Qwen3VLModel` 时权重全随机初始化（checkpoint 键带 `model.` 前缀），实测 embedding 无意义 | 需要自写 last-token pooling/图像预处理，加载代码更复杂 |
| 视觉索引范围 | 默认全量图片 | 本语料 caption 覆盖仅 28.6%，selective 仅跳过 4/502，节省 <1%，但 recall 无损失 | 编码/存储成本高 |
| 视觉索引切换 | generation + pointer 单写切换 | 失败时旧索引仍可用，禁止新旧双份可见，崩溃可恢复 | 无跨存储事务，仅文档级一致性 |
| 文本/视觉/rerank 设备 | CPU + GPU VLM | 12GB 显存无法同时容纳 VLM + 文本/视觉模型 | 查询编码慢（CPU 视觉 P50 983ms） |
| 证据充分性 | 逐条 Evidence 词面绑定 | 修复“跨候选拼凑充分”的误判，可解释 | 不能证明语义支撑；q03/q08 类仍需安全门禁兜底 |
| 补检索次数 | 最多 1 次，且只在明确缺失时 | 控制延迟与无效循环 | 复杂多跳问题可能仍不足 |
| 自动入库重建 | 已完成文档禁止原地重建（409） | 避免部分失败破坏旧索引 | 需删除后重传，运维不够便利 |
| 报告插入 | 服务端 HMAC 台账 + chat_message_id | 客户端不能自称“已核验” | 台账文件本机存储，多机部署需共享密钥/存储 |

## 10. 限制与未验证项（必须如实说明）

- M7/M9 评测均为**技术测试集**；M7 的 4 份文档是真实技术报告，不是国别经济研究报告；M9 gold 为 15 道技术问题。
- 未做人工逐题语义核验，不能把 14/15 或 100% 指标写成“人工语义准确率”。
- 视觉数值不能保证绝对正确；系统只能防止未核验数字进入可信结论/自动报告，不能消除 VLM 视觉误读。
- 历史语料图片资产只有约 23.2% 能解析到原始图片，历史文档的视觉索引/图片预览覆盖有限。
- 最终演示中观察到两个 UI/格式缺陷：
  1. **已修复（补丁 PR #70）**：确定性表格回答“相对变化 = 10.6977 亿美元”单位错误。根因是渲染器在 `units` 只有一个值时把表单位附加到所有数值列；现改为按 `columns[].unit` 逐列渲染（相对变化 `%`、百分数指标绝对变化“百分点”），并补充真实 `TableAnalysisResult` 回归测试：4300→4760 输出“绝对变化 = 460 亿美元；相对变化 = 10.6977%”，无“10.6977 亿美元”，确定性结果仍标记 `source_verified`。
  2. 该确定性回答下方出现空的“本次分析的图片”区域（只有表格提示、无图片），未修复，仅记录。
- 报告插入失败时前端只显示通用“插入报告失败，请稍后重试”，未透出服务端 409 detail（API 侧 detail 正确），未修复。
