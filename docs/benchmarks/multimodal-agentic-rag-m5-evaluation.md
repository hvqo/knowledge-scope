# Multimodal Agentic RAG M5 评测报告

## 1. 评测范围

- 知识库：隔离的 M5 评测知识库（4 份 PDF，30 道题）。
- 文档：
  - `m3-vision-acceptance.pdf`：图表、图片型表格、流程图（3 页）。
  - `m2-pdf-a.pdf`：2023/2024 年三国 GDP 表格（1 页）。
  - `m2-pdf-b.pdf`：2023/2024 年三国预测 GDP 表格（1 页）。
  - `m5-math.pdf`：含公式与版式表格的数学教材页面（20 页）。
- 问题类别：RAG 问答（7）、图片理解（4）、图片型表格提取（1）、表格取数（3）、增长率（3）、汇总（3）、跨表/跨文档对比（3）、多轮追问（3，作为一组连续会话）、不存在证据与相似干扰（3）。
- 每道题记录：问题、期望工具、期望关键数字/短语、实际工具、工具参数、最终回答、引用有效性、时延、判定与失败原因。

## 2. 指标定义

| 指标 | 定义 | 目标 |
| --- | --- | --- |
| Tool Selection Accuracy | 实际工具集合与期望工具一致（允许类别中声明的降级选项） | ≥ 85% |
| Parameter Validity | 工具参数未导致服务端参数校验失败 | ≥ 90% |
| Numerical Accuracy | 回答包含期望关键数字；`compare`/`growth`/`aggregate` 执行时以表格工具结果核对 | 100%（确定性计算题） |
| Citation Validity | 回答中出现的 `[Cn]` 均在 citations 事件中，且对应 Evidence 可被服务器读取 | 100% |
| Retrieval Hit@K | 期望文档出现在引用来源中 | 监控 |
| E2E Success | 上述检查与拒答/降级行为均符合预期 | ≥ 80% |
| Latency | 端到端时延（均值、中位数、P95） | 监控 |

## 3. 运行方式

```bash
# 1) 启动 API（读取主工作区 .env，使用 M5 源码；本地 Ollama VLM）
cd /home/amber/PycharmProjects/KnowledgeScope
PYTHONPATH=<M5 worktree>/src \
NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1 \
KNOWLEDGE_SCOPE_VLM_ENABLED=true \
KNOWLEDGE_SCOPE_VLM_BASE_URL=http://127.0.0.1:11434/v1 \
KNOWLEDGE_SCOPE_VLM_MODEL=qwen3-vl:8b \
KNOWLEDGE_SCOPE_VLM_REASONING_EFFORT=none \
KNOWLEDGE_SCOPE_VLM_MAX_OUTPUT_TOKENS=4096 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/uvicorn knowledge_scope.api.app:app --host 127.0.0.1 --port 8000

# 2) 运行评测（真实 DeepSeek Flash 规划/回答 + 本地 qwen3-vl:8b 图片理解）
cd <M5 worktree>
NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1 \
.venv/bin/python scripts/m5_agent_eval.py --output /tmp/m5-eval-results.json
```

评测数据：`scripts/m5_eval_dataset.json`；运行器：`scripts/m5_agent_eval.py`。

## 4. 结果

所有结果均来自真实运行：规划与最终回答使用 DeepSeek Flash，图片理解使用本地
Ollama `qwen3-vl:8b`，表格计算由确定性 `TableAnalysisService` 执行，不使用 Mock 结论。

### 4.1 最终全量运行（30 题）

| 指标 | 结果 | 目标 | 是否达标 |
| --- | --- | --- | --- |
| Tool Selection Accuracy | 96.7%（29/30） | ≥ 85% | 是 |
| Parameter Validity | 100.0%（30/30） | ≥ 90% | 是 |
| Numerical Accuracy | 86.7%（确定性计算题的数值均由工具结果核对） | 100%（确定性计算题） | 否 |
| Citation Validity | 100.0% | 100% | 是 |
| Retrieval Hit@K | 96.7% | 监控 | — |
| End-to-End Success | 83.3%（25/30） | ≥ 80% | 是 |
| Latency（均值 / 中位数 / P95） | 13599.4 ms / 9460.3 ms / 38584.5 ms | 监控 | — |

说明：Numerical Accuracy 的 86.7% 中包含视觉类失败（图表最小值误读、VLM 输出截断导致
“无法判断”），确定性表格工具本身的数值结果均正确。

### 4.2 修复前后对照

| 运行 | Tool Selection | Parameter | Numerical | Citation | Hit@K | E2E |
| --- | --- | --- | --- | --- | --- | --- |
| 首轮（修复前） | 90.0% | 100.0% | 80.0% | 100.0% | 96.7% | 70.0% |
| 最终轮（修复后） | 96.7% | 100.0% | 86.7% | 100.0% | 96.7% | 83.3% |

首轮发现的规划器问题（把 `aggregate` 参数标成 `select`、简单问题过度调用工具、表格取数
被“优先直答”规则误判）已通过操作名归一化、参数键清洗与提示词约束修复；修复后针对性回归
（表格取数与汇总 6 题）为 100%（工具选择、参数、数值、引用、命中、E2E 均 100%），
增长率 3 题在精度提示修复后回归为 100%。

### 4.3 分项观察

- 表格类任务（取数、增长率、汇总、跨表/跨文档对比）：确定性工具结果稳定，`compare` 与
  `growth` 的数值、公式、单位、双来源引用在真实浏览器与评测中均正确。
- 图片理解：流程图描述与柱状图趋势/峰值可稳定回答，但“最低值/精确小数字”读取不稳定。
- 多轮追问：复用既有 `HistoryCompressor` + `QueryRewrite`，在真实浏览器中完成
  “2023 越南 GDP → 2024 → 增长百分之几”的上下文继承与确定性计算。
- 引用：服务端标记过滤后，回答中出现的 `[Cn]` 均存在于 citations 事件且 Evidence 可读取；
  未被引用或未参与工具调用的候选来源不会展示。

## 5. 失败案例分析

| 用例 | 现象 | 归类 | 处理 |
| --- | --- | --- | --- |
| rag-07 | 第二页表格标题在检索候选中缺失，回答未命中期望短语 | 检索缺失（多文档同构表格干扰） | 未修复；记录为检索侧限制 |
| image-01 | VLM 输出被截断，工具返回受控失败，回答明确“无法判断” | 视觉推理稳定性 / 输出预算 | 受控降级正确；未伪造结论 |
| image-03 | VLM 将“最低季度”误读为 Q1≈120，真实图未标注该数值 | 视觉模型幻觉 | 真实 VLM 限制，已知并记录；回答含人工核对提示 |
| growth-01 | 回答把工具结果 10.6977% 四舍五入为 10.70%，触发数字不一致 | 表达精度 | 已修复：回答提示词要求保留工具原始精度；回归 100% |
| agg-03 | 规划器参数校验失败后降级，答案数值正确但未走确定性工具 | 规划偶发失败 | 已增加操作名/参数清洗与重试；全量回归中仍偶发 1 次，记录为剩余风险 |

失败归因明确区分：检索缺失、视觉模型幻觉/截断、表达精度、规划失败，未将 Hit@K 与问答
正确率混为一谈。评测集规模较小（30 题），上述数字用于回归参考，不代表开放域泛化能力。

## 6. 已知限制

- 历史文档缺少解析产物时，图片证据只能给出“原始素材不可用”的受控降级，无法读取图片像素。
- 本地 `qwen3-vl:8b` 在低值/最小值类细粒度数字读取上存在不稳定，必要时需人工核对或接入更强视觉模型。
- 单表复杂合并单元格、多层表头仍可能返回 `not_computable`，属于受控失败而非幻觉。
- 评测集规模较小（30 题、4 份文档），指标用于回归参考，不代表开放域泛化能力。
- 智能体降级回答会显式提示未执行确定性计算；但底层常规 RAG 回答仍可能自行归纳，评测中未将其计为工具成功。
- 规划器对个别表格汇总题的输出仍可能先以错误参数名出现，服务端已做操作名/参数清洗与一次重试，全量回归中仍偶发 1 例降级（agg-03）。
- 浏览器演示中曾出现视觉模型输出截断的受控降级（提示“视觉模型输出被截断”）；将 VLM 输出预算调至 8192 后同一图表问题成功，说明该失败为本地推理稳定性问题而非链路崩溃。
