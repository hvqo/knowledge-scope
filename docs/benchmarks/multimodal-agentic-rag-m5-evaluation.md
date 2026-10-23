# Multimodal Agentic RAG M5 评测报告

## 1. 评测范围

- 知识库：隔离的 M5 评测知识库（4 份 PDF，30 道题）。
- 文档：
  - `m3-vision-acceptance.pdf`：图表、图片型表格、流程图（3 页，由 `scripts/m5_make_eval_pdfs.py` 使用仓库内 PNG fixture 生成）。
  - `m2-pdf-a.pdf`、`m2-pdf-b.pdf`：2023/2024 年三国 GDP 表格（各 1 页，由同一脚本生成）。
  - `m5-math.pdf`：含公式与版式表格的数学教材页面（20 页），来自冻结的 A1.5 benchmark 语料，未随仓库分发。
- 问题类别（严格数据集 v2.0，共 30 题）：RAG 问答（7）、图片理解（4）、图片型表格提取（1）、
  表格取数（3）、增长率（3）、汇总（3）、跨表/跨文档对比（3）、多轮追问（3，同一会话连续三问）、
  不存在证据与相似干扰（3）。
- 数据集与运行器：`scripts/m5_eval_dataset.json`、`scripts/m5_agent_eval.py`。数据集中不写死
  KB/Document UUID，运行时通过 `--fixtures` 注入，见第 0 节与第 3 节。

### 0. 复现说明（重要）

评测资源已不再写死在数据集中：

1. `scripts/m5_make_eval_pdfs.py` 生成可公开复现的三份 PDF（图表/图片表格/流程图、两张 GDP 表）。
2. `scripts/m5_prepare_eval_fixtures.py prepare` 创建临时知识库并上传 4 份 PDF（数学教材 PDF 需
   通过 `--math-pdf` 显式指定）。
3. 停掉 API 后，`index` 阶段用本地 MinerU 3.4.5 + 本地 Embedding/Reranker 完成解析、分块、
   Qdrant 索引与 Evidence 构建；随后重启 API。
4. `scripts/m5_agent_eval.py --fixtures ...` 执行严格评测；`cleanup` 阶段删除本次创建的
   KB/文档与 `data/parsing`、`data/chunking`、`data/evidence` 产物。

测试文档分为三类，报告中不混称：

- **生成文档**：`m3-vision-acceptance.pdf`、`m2-pdf-a/b.pdf`（脚本生成，可完全复现）。
- **真实语料文档**：`m5-math.pdf`（A1.5 冻结语料中的教材页，来源不可从仓库内重新获取；缺少它时
  4 道数学题会标记为不可复现/未验证，而不是伪造结果）。
- **真实模型调用**：DeepSeek Flash（规划 + 最终回答）与本地 Ollama `qwen3-vl:8b`（图片理解）；
  无 Mock 结论。Mock 只用于单元测试。

## 2. 指标定义

### 2.1 严格口径（v2，当前有效）

| 指标 | 定义（分子 / 分母） | 目标 |
| --- | --- | --- |
| Tool Selection Accuracy | 期望工具全部被实际调用、未出现禁止工具的题数 / 全部 30 题 | ≥ 85% |
| Tool Execution Success Rate | 期望工具实际执行成功（或产生该题声明的受控错误）的题数 / **确实调用了必需工具**的题数 | 监控 |
| Parameter Validity (strict) | `agent_tool_trace` 中 operation、目标文档、vision task 与预期一致且可观测的题数 / **确实调用了必需工具**的题数；未调用工具已在 Tool Selection 扣分，不在此重复计入；不可观测字段计失败，不默认通过 | ≥ 90% |
| Exact Calculation Chain Success | 确定性表格题中工具执行成功、参数正确、全部期望数字在容差内、双来源引用完整的题数 / 确定性计算题数（14 题，排除受控错误题） | 监控（完整链路） |
| Executed Tool Numeric Accuracy | 工具实际执行成功的确定性题中，回答全部期望数字在容差内的题数 / 该类题数 | 100% |
| Final Answer Numeric Accuracy | 确定性计算题中最终回答数字全部匹配（Decimal 容差）的题数 / 含数字断言的确定性计算题数 | 100% |
| Visual Number Accuracy | 图片型表格提取题中数字匹配且工具完成的题数 / 图片数值提取题数（标记 `needs_review`，不计入精确计算） | 监控 |
| Visual Controlled Refusal Rate | 无法读数的视觉题中给出受控拒答且未编造数字的题数 / 该类题数 | 100% |
| Cited Document Coverage | 最终引用中包含全部必需文档的题数 / 有必需文档的题数（替代旧“Retrieval Hit@K”） | 监控 |
| Citation Structure Validity | `[Cn]` 标记、Evidence/Chunk 结构与 KB 归属合法的题数 / 需要引用的题数（不含语义支持）；其中“有引用回答”的结构合法性为 26/26 = 100%，rag-07 因未召回到引用单独计入覆盖失败 | 100%（有引用回答） |
| Evidence Accessibility Rate | 可被服务器重新读取的 Evidence 引用数 / Evidence 引用总数 | 100% |
| Required Text Accuracy | 回答包含全部期望定性表述的题数 / 有文本断言的题数 | 监控 |
| E2E Success | 上述检查与该题的拒答/降级/`needs_review` 预期全部通过的题数 / 30 题 | ≥ 80% |
| Latency | 端到端时延（均值 / 中位数 / P95） | 监控 |

**明确标记为 UNVERIFIED 的指标：**

- **True Retrieval Hit@K**：现有公开 SSE 契约只给出最终 citations，不含检索阶段 Top-K 候选，
  因此不能称为 Retrieval Hit@K；本报告改名为 **Cited Document Coverage**。真正的 Hit@K 标记
  UNVERIFIED，本轮不做检索架构改造。
- **Citation Semantic Support**：结构合法与 Evidence 可读取不代表引用在语义上支持每一句结论；
  该指标标记 UNVERIFIED，不宣称“引用真实性 100%”。

### 2.2 旧宽松口径（历史，仅供对照）

旧运行器使用 `any()` 子串命中、任意文档命中、以“未出现 invalid_parameters”判定参数正确，且把
引用文档覆盖率命名为 Retrieval Hit@K；视觉 `needs_review` 与精确计算未分开。该口径下的数字只
作为开发过程记录，**不得与严格口径直接比较**，也不能解读为模型能力提升。

## 3. 运行方式

```bash
# 0) 生成可复现的测试 PDF（图表/图片表格/流程图 + 两张 GDP 表）
cd <M5 worktree>
uv run python scripts/m5_make_eval_pdfs.py --output-dir /tmp/m5-eval-pdfs

# 1) API 运行中：创建隔离知识库并上传 4 份 PDF
uv run python scripts/m5_prepare_eval_fixtures.py prepare \
  --pdf-dir /tmp/m5-eval-pdfs \
  --math-pdf <A1.5 benchmark>/auto/..._origin.pdf \
  --output /tmp/m5-eval-fixtures.json

# 2) 停掉 API（释放 GPU），执行 MinerU/Embedding/Evidence 索引
uv run python scripts/m5_prepare_eval_fixtures.py index --fixtures /tmp/m5-eval-fixtures.json

# 3) 启动 API（读取主工作区 .env，使用 M5 源码；本地 Ollama VLM）
cd /home/amber/PycharmProjects/KnowledgeScope
PYTHONPATH=<M5 worktree>/src \
NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1 \
KNOWLEDGE_SCOPE_VLM_ENABLED=true \
KNOWLEDGE_SCOPE_VLM_BASE_URL=http://127.0.0.1:11434/v1 \
KNOWLEDGE_SCOPE_VLM_MODEL=qwen3-vl:8b \
KNOWLEDGE_SCOPE_VLM_REASONING_EFFORT=none \
KNOWLEDGE_SCOPE_VLM_MAX_OUTPUT_TOKENS=8192 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
.venv/bin/uvicorn knowledge_scope.api.app:app --host 127.0.0.1 --port 8000

# 4) 严格评测（真实 DeepSeek Flash + 本地 qwen3-vl:8b）
cd <M5 worktree>
NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1 \
uv run python scripts/m5_agent_eval.py \
  --fixtures /tmp/m5-eval-fixtures.json \
  --output /tmp/m5-eval-strict.json

# 5) 清理本次创建的 KB/文档/解析产物
uv run python scripts/m5_prepare_eval_fixtures.py cleanup --fixtures /tmp/m5-eval-fixtures.json
```

评测单元测试（评分逻辑本身，不调用模型/数据库）：

```bash
uv run pytest tests/test_m5_eval_scoring.py -q
```

## 4. 结果

所有结果均来自真实运行：规划与最终回答使用 DeepSeek Flash，图片理解使用本地
Ollama `qwen3-vl:8b`，表格计算由确定性 `TableAnalysisService` 执行。

### 4.1 严格口径运行结果（v2，当前有效）

- 运行时间：2026-10-23（本机），单次完整运行 30 题。
- 实际模型：DeepSeek Flash 负责规划与最终回答；本地 Ollama `qwen3-vl:8b` 负责图片理解；
  表格数值由确定性 `TableAnalysisService` 计算，未使用 Mock。
- 数据集/运行器：`scripts/m5_eval_dataset.json`（schema 2.0）、`scripts/m5_agent_eval.py`；
  评测夹具为隔离 KB（4 份 PDF），KB/Document UUID 运行时注入，运行后已清理。
- 评分方式：数值按 Decimal + 容差判定；工具选择、工具执行、参数有效性、引用结构、
  Evidence 可达性、引用文档覆盖分别独立计分；`needs_review` 不作为精确计算成功。

| 指标 | 结果（分子/分母） | 定义与说明 | 验收目标 |
| --- | --- | --- | --- |
| Tool Selection Accuracy | 29/30（96.7%） | 是否选择了要求工具，且未调用禁止/多余工具 | ≥ 85% |
| Tool Execution Success Rate | 16/18（88.9%） | 仅在“确实调用了工具”的题目中统计工具执行成功 | 监控 |
| Parameter Validity（严格） | 18/18（100.0%） | 操作类型、目标文档、Evidence、参数可观测且匹配；未调用工具的题目不在此重复扣分 | ≥ 90% |
| Exact Calculation Chain Success | 12/14（85.7%） | 确定性计算题的完整链路：工具执行 + 参数 + 数值 + 文档 + 引用结构 | 监控 |
| Executed Tool Numeric Accuracy | 12/12（100.0%） | 工具实际执行成功的确定性题中，回答数值是否正确 | 100% |
| Final Answer Numeric Accuracy | 13/13（100.0%） | 带数值期望题目的最终回答数值正确率 | 监控 |
| Visual Numeric Extraction Accuracy | 1/1（100.0%） | 图片型表格数值提取正确（结果为 needs_review，不计为精确计算） | 监控 |
| Visual Controlled Refusal Rate | 0/1（0.0%） | 视觉无法读数时是否受控拒答且不编造数字 | 100%（本轮未达标） |
| Cited Document Coverage | 26/27（96.3%） | 需要多文档的问题中，要求文档是否都出现在 cites | 监控 |
| Citation Structure Validity | 26/27（96.3%） | 有引用的回答结构合法；有引用回答内部为 26/26 = 100% | 100%（有引用回答） |
| Evidence Accessibility | 57/57（100.0%） | citations 中的 Evidence/Chunk 能否在服务器端再次读取 | 100% |
| Required Text Accuracy | 16/17（94.1%） | 关键结论短语/拒答表述命中 | 监控 |
| E2E Success | 26/30（86.7%） | 以上关键检查全部通过 | ≥ 80% |
| Latency（均值 / 中位数 / P95） | 12618.9 / 8320.4 / 60044.6 ms | API 端到端 | 监控 |
| True Retrieval Hit@K | UNVERIFIED | 现 API 不暴露真实 Top-K 候选，仅能统计引用文档覆盖 | — |
| Citation Semantic Support | UNVERIFIED | 本报告只验证引用结构与 Evidence 可达性，不宣称语义支撑已自动验证 | — |

**验收门槛对照（本报告采用的定义）**

- Tool Selection Accuracy ≥ 85%：29/30（96.7%），通过。
- Parameter Validity ≥ 90%：18/18（100.0%），通过。
- 已执行确定性工具的数值正确率 = 100%：12/12（100.0%），通过。
- Citation Validity（有引用回答结构合法 + Evidence 可达） = 100%：
  有引用回答 26/26 合法，Evidence 访问 57/57（100.0%），通过。
- E2E Success ≥ 80%：26/30（86.7%），通过。
- 未达标项单独报告：Visual Controlled Refusal 0/1；Citation 题目级覆盖 26/27（rag-07 无引用）；
  Exact Calculation Chain 12/14（compare-03 工具执行失败、agg-03 规划器降级）。

### 4.1.1 与旧宽松口径的关系

严格口径与旧口径的问题、评分规则不同，**不得把两组数字直接比较或解读为模型能力提升**。
旧宽松口径的历史数字保留在 4.2 节，仅用于开发过程回溯。严格运行的三次波动（Tool Selection
93.3%–96.7%、E2E 80%–86.7%）说明规划器仍存在随机性；本报告以最后一次完整运行为主结果。

### 4.2 旧宽松口径运行结果（历史，不可与严格结果直接比较）

| 指标（旧口径） | 首轮（修复前） | 最终轮（修复后） | 旧目标 |
| --- | --- | --- | --- |
| Tool Selection Accuracy | 90.0% | 96.7% | ≥ 85% |
| Parameter Validity | 100.0% | 100.0% | ≥ 90% |
| Numerical Accuracy | 80.0% | 86.7% | — |
| Citation Validity | 100.0% | 100.0% | 100% |
| “Retrieval Hit@K”（实为 Cited Document Coverage） | 96.7% | 96.7% | 监控 |
| E2E Success | 70.0% | 83.3% | ≥ 80% |
| Latency 均值 / 中位数 / P95 | — | 13599.4 / 9460.3 / 38584.5 ms | 监控 |

首轮发现的规划器问题（把 `aggregate` 参数标成 `select`、简单问题过度调用工具、表格取数
被“优先直答”规则误判）已通过操作名归一化、参数键清洗与提示词约束修复；修复后针对性回归
（表格取数与汇总 6 题）为 100%，增长率 3 题在精度提示修复后回归为 100%。

### 4.3 分项观察

- 表格类任务（取数、增长率、汇总、跨表/跨文档对比）：确定性工具结果稳定，`compare` 与
  `growth` 的数值、公式、单位、双来源引用在真实浏览器与评测中均正确。
- 图片理解：流程图描述与柱状图趋势/峰值可稳定回答，但“最低值/精确小数字”读取不稳定。
- 多轮追问：复用既有 `HistoryCompressor` + `QueryRewrite`，在真实浏览器中完成
  “2023 越南 GDP → 2024 → 增长百分之几”的上下文继承与确定性计算。
- 引用：服务端标记过滤后，回答中出现的 `[Cn]` 均存在于 citations 事件且 Evidence 可读取；
  未被引用或未参与工具调用的候选来源不会展示。

## 5. 失败案例分析

本轮 30 题中 26 题通过 E2E，4 题失败。失败按根因归类如下：

| 题目 | 类别 | 失败检查 | 根因与影响 | 是否产品级阻塞 |
| --- | --- | --- | --- | --- |
| rag-07 | RAG 检索 | document_coverage / citation_structure / required_text / e2e | Unified Retrieval 未召回题干指向的第二页表格标题，回答缺少引用。属于检索召回问题，非引用伪造。 | 否，记录为检索召回限制 |
| image-03 | 图片理解 | tool_execution / unexpected_number / numbers_mismatch / e2e | `qwen3-vl:8b` 对柱状图“最低季度具体值”给出约 120 的编造数值，未受控拒答。属本地 VLM 能力限制。 | 否，已在 UI/answer 保留人工核对提示；建议更强 VLM |
| agg-03 | 汇总计算 | tool_selection / missing_required_tool / tool_execution / parameter_unobservable / e2e | 规划器两次尝试后返回 final 而未调用 `analyze_table`，降级为常规回答；未执行确定性计算。属规划器稳定性残留风险。 | 否，但属于已知可靠性边界 |
| compare-03 | 跨文档对比 | tool_execution / e2e | `compare` 工具要求两表表头同名；英文扫描表（Country/GDP）与中文表（国家/2024年GDP）表头无法对齐，工具受控失败（“未找到列：Country”），回答如实说明无法完成该计算。 | 否，记录为跨语言表头支持限制 |

补充说明：

- 上述失败均未出现编造 Evidence ID、跨 KB 访问、或未执行工具却给出确定性数值的情况；
  `agg-03` 降级回答带有未执行工具提示，未计入精确计算。
- 严格运行的工具执行失败率来自 image-03（VLM 工具超时/失败）与 compare-03（表头不匹配），
  均属可控失败路径，不是静默错误。
- 引用侧：所有实际输出的 citations 均通过服务端 Evidence 读取校验（57/57），且回答中的
  `[Cn]` 标记均属于 citations 集合；rag-07 因无召回而未产生引用。

## 6. 已知限制

- 历史文档缺少解析产物时，图片证据只能给出“原始素材不可用”的受控降级，无法读取图片像素。
- 本地 `qwen3-vl:8b` 在低值/最小值类细粒度数字读取上存在不稳定，必要时需人工核对或接入更强视觉模型。
- 单表复杂合并单元格、多层表头仍可能返回 `not_computable`，属于受控失败而非幻觉。
- 评测集规模较小（30 题、4 份文档），指标用于回归参考，不代表开放域泛化能力。
- 智能体降级回答会显式提示未执行确定性计算；但底层常规 RAG 回答仍可能自行归纳，评测中不将其计为工具成功。
- 规划器对个别表格汇总题的输出仍可能先以错误参数名出现，服务端已做操作名/参数清洗与一次重试；仍有偶发降级风险。
- 浏览器演示中曾出现视觉模型输出截断的受控降级（提示“视觉模型输出被截断”）；将 VLM 输出预算调至 8192 后同一图表问题成功，说明该失败为本地推理稳定性问题而非链路崩溃。
- `compare` 工具按表头名称对齐；当同一指标在英文扫描表与中文表中表头不同名时，工具会受控失败，
  当前版本不支持跨语言表头映射。
- 严格口径中 True Retrieval Hit@K 与 Citation Semantic Support 仍为 UNVERIFIED；评测报告不宣称这两项已达标。
- 严格评测存在运行间波动（规划器随机性）；报告以最后一次完整运行为主结果，并保留三次运行的区间说明。
- 评测集数学 PDF 来自 A1.5 冻结语料，未随仓库分发；缺少该文件时 4 道数学题不可复现，报告中会显式标记。

