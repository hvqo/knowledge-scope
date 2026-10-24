# 视觉向量检索（M7）

## 1. 目标与边界

在现有 `CanonicalDocument → Evidence → 文本/multimodal representation → Unified Retrieval` 链路之上，增加**真实图片像素级视觉向量检索**分支：

```
PDF 图片 → MinerU 解析资产(asset_ref) → 视觉编码(Qwen3-VL-Embedding-2B)
        → 独立 Qdrant 集合(模型指纹隔离) → 文本查询召回图片
        → Evidence/页码定位 → Agent 原图分析(VLM) → 前端展示原始图片
```

视觉向量**不复用**文本 `Qwen3-Embedding-0.6B` 空间，也不把 caption 文本向量当作视觉向量。文本检索、稀疏检索、图谱检索、multimodal representation 检索与 Agent 行为保持兼容；未启用或未建视觉索引的知识库行为不变。

## 2. 模块

| 模块 | 职责 |
| --- | --- |
| `src/knowledge_scope/retrieval/visual_embedding.py` | 懒加载 Qwen3-VL-Embedding-2B；图文同一编码器；模型契约/指纹；编码统计 |
| `src/knowledge_scope/retrieval/visual_policy.py` | 可解释索引策略 `text_only` / `text_and_visual` |
| `src/knowledge_scope/retrieval/visual_index.py` | payload、集合命名、幂等 point id、Qdrant 读写、检索结果 |
| `src/knowledge_scope/retrieval/visual_build.py` | 扫描 Evidence → 解析原始资产 → 编码 → 建索引；all/selective |
| `src/knowledge_scope/retrieval/unified.py` | visual 分支并入 Unified Retrieval，配额融合与分支溯源 |
| `src/knowledge_scope/cli.py` | `visual-index check/build/search/audit` |
| `src/knowledge_scope/api/app.py` | 启用时懒加载 encoder/store 注入 Unified Retrieval |

### 2.1 模型加载注意

官方 HuggingFace checkpoint 是 `Qwen3VLForConditionalGeneration`（权重键以 `model.` 开头）。`sentence-transformers` 封装会加载基座 `Qwen3VLModel` 导致**所有权重随机初始化且不报错**。因此实现直接：

- `Qwen3VLForConditionalGeneration.from_pretrained(..., dtype=bfloat16).model`；
- 自组合 tokenizer + image processor（绕开需要 torchvision 的 video processor）；
- chat template + 最后一个非 pad token 池化 + L2 归一化；
- 查询与图片使用同一编码器、同一向量空间。

冒烟实测：2048 维，加载 ≈13s，2 图编码 ≈1.05s，查询 ≈95ms，显存峰值 ≈4.46GB（RTX 3060 12GB）。

### 2.2 集合与幂等

- 集合名：`<base>_<模型slug>_<指纹10>`，指纹 = 模型契约（model_id/path/dtype/维度/prompt/归一化）的 SHA-256；不同模型不会混写。
- point id：`uuid5(命名空间, 集合指纹|KB|Document|Evidence|asset_index|asset_sha256)`；重复构建幂等。
- 重建单个文档：`replace_document` 先删除该文档旧点，再写入；素材字节变化后旧向量不会被复用。
- 检索按 `knowledge_base_id + collection_fingerprint` 过滤；结果再校验 payload 契约。

## 3. 索引策略（选择性）

`visual-index build --mode selective` 仅对策略判定为 `text_and_visual` 的图片建立视觉索引；`--mode all` 全量。策略不依赖模型，输入 caption/context/图片像素/同页图片数/同页表格/资产大小，输出 mode + 中文 reason + signals。

规则（保守优先，宁可多建视觉索引）：
1. 无 caption：上下文足够（≥100 字、同页单图、无表格、像素≤600k、无视觉关键词）→ `text_only`；否则 `text_and_visual`。
2. 通用编号 caption（如“图 3”）→ `text_and_visual`。
3. caption <15 字 → `text_and_visual`。
4. context <80 字 → `text_and_visual`。
5. 同页有表格 / 多图 / 未知或 >600k 像素 → `text_and_visual`。
6. caption 或 context 含图表/流程/趋势等视觉细节词 → `text_and_visual`。
7. 其余 → `text_only`。

## 4. A/B/C 真实检索实验

### 4.1 数据与问题（无 caption 泄漏）

- 数据：A1.5 冻结语料 retained 的 14 份真实教材 PDF（地理/生物/历史/数学/物理），隔离目录 `/tmp/m7-data`，重建 canonical + Evidence（含同页文本 context），共 **502 张开源图片**，KB `00000000-0000-4000-8000-000000000007`。
- 问题：人工查看 28 张跨 14 类图片后撰写 **28 条视觉特征查询**（不使用 caption 原文），每条标注 gold Evidence。其中 15 条 gold 图片无 caption（纯视觉可答），13 条有 caption。
- 三类检索：
  - A：caption/context 文本向量（Qwen3-Embedding-0.6B，1024 维）
  - B：选择性视觉索引（policy `text_and_visual` 子集，视觉向量 2048 维）
  - C：全量视觉索引（502 张）

### 4.2 结果（Recall@5 / Recall@10 / MRR@10）

| 方案 | Recall@5 | Recall@10 | MRR@10 |
| --- | --- | --- | --- |
| A 文本表示 | 25.0% (7/28) | 35.7% (10/28) | 0.197 |
| B 选择性视觉 | 92.9% (26/28) | 92.9% (26/28) | 0.866 |
| C 全量视觉 | 92.9% (26/28) | 96.4% (27/28) | 0.871 |

- 有 caption 的 13 条：A 命中 3/13，B/C 13/13。
- 无 caption 的 15 条：A 命中 4/15，B/C 13/15。
- 选择性策略在本语料仅把 4/502 张判为 `text_only`，因此 B 与 C 成本几乎相同；B 相比 C 仅少 1 条 @10 命中（q15 的 gold 图片被策略排除，即唯一 policy miss）。

### 4.3 成本

| 指标 | selective | full |
| --- | --- | --- |
| 编码图片数 | 498 / 502 | 502 / 502 |
| 图片编码总耗时（实测，batch=4） | ≈83.4s（按比例估算） | **84.0s** |
| 向量原始字节（2048×float32） | 4.08MB | 4.11MB |
| 文本查询编码 | — | ≈52ms（VLM 编码器） |
| Qdrant 查询延迟 | — | ≈29ms |

结论：**本语料不存在“文本索引已足够”的大规模图片子集**，选择性索引无法带来有意义的成本节省（<1%）；若需要完整视觉召回，应全量建视觉索引。策略的价值在 caption/context 质量更高的语料上才会体现，本仓库保留 all/selective 两种模式并记录 policy miss。

## 5. Unified Retrieval 集成

- visual 分支位于现有 dense/sparse/graph/multimodal 之后；`UnifiedCandidateSource` 与 `source_distribution` 增加 `visual`，候选仍归一到 Evidence。
- 融合：在既有 reranker 排序后保留 `visual_fusion_quota`（默认 2）个视觉来源候选，避免纯视觉候选被文本 reranker 全部淘汰；不跨模型比较原始相似度，只在 reranker 顺序内保配额。
- 每个候选记录 `branches` 溯源（含 `visual` 排名/分数），同一 Evidence 的文本与视觉命中合并去重。
- 未启用/未建索引的 KB：visual 分支为 `empty`，不影响原行为（`tests/test_unified_retrieval.py` 全绿）。

## 6. 已知限制

- 视觉编码器对细粒度类别（如“柱状图” vs “折线图”）仍会混淆；视觉向量提升的是“找到相关图片”，不等于精确图表类型分类。
- 文本查询与图片的跨模态对齐受查询措辞影响；实验中的 28 条查询是人工视觉描述，真实用户表达可能更口语化。
- selective 策略依赖 caption/context 质量；历史语料 caption 覆盖率 28.6%，导致选择性几乎等价于全量。
- 视觉向量空间与文本 embedding 空间不可比，因此不做跨模型分数融合；仅做配额与 Evidence 级去重。
- 建索引仍是离线 CLI 流程，未做增量自动触发；素材缺失会显式计入 `images_missing`，不生成假向量。

## 7. 原始图片资产可用性审计

M7 开始前对历史语料做了只读审计（见 `scripts/m7_prepare_eval_data.py` 与 teammate 报告）：

| 检查项 | 结果 |
| --- | --- |
| `data/parsing/` 解析树 | 不存在（历史解析产物未保留） |
| canonical image blocks / asset_refs | 6152 个，其中仅 1429（23.2%）能在 A1.5 `raw/` 找到原始文件 |
| Evidence image（6153 条）资产可解析率 | 同为 23.2%（192 份文档的 raw 未保留） |
| 原始 PDF | 257 份全部存在于 `/home/amber/Projects/file`，可按需重解析 |
| A1.5 `raw/` 图片 | 1705 张 jpg（276 张 orphan），971 张可用（短边 ≥200px），326 张可用且有 caption |
| caption 覆盖 | 28.6%；抽样 15 张中 3 张 caption 与 asset 错配 |
| 扫描版表格 | 不存在；409 个 table 均为 HTML 结构表 |

处理策略：不重解析全部 255 份文档，也不对历史缺失素材造假；从 retained 文档中挑选 14 份有原始图片的教材 PDF 重建**隔离实验数据目录** `/tmp/m7-data`（502 张图片）用于 A/B/C 实验与索引构建。视觉索引建库时素材缺失会显式计入 `images_missing`，不会生成空向量；生产环境若需要完整视觉召回，应先保证 `data/parsing/<document>/mineru/` 存在并可解析。

## 8. 端到端验收（真实链路）

验收环境：PR #65 合并后的主分支 + 本分支；隔离 KB `1792e9e7-aa58-4053-bc89-9c392bcdb076`（3 份 PDF），数据目录 `/tmp/m7-e2e-data`；真实 DeepSeek Flash + Ollama `qwen3-vl:8b`。链路：

1. PDF（含柱状图）→ MinerU 解析 → canonical/evidence；
2. `chunk-document` + `qdrant index-document` + `multimodal-index build`；
3. `visual-index build --mode all` → 2 张图片写入独立视觉集合（VRAM 峰值 4263MB，编码 3.7s）；
4. API 以 `KNOWLEDGE_SCOPE_VISUAL_EMBEDDING_ENABLED=true` 启动，Unified Retrieval 注入 visual 分支；
5. Agent 提问“分析图中柱状图…”：`retrieval_branch_statuses` 为 `dense/multimodal/visual success`，候选 C1 图片 Evidence 的 `branch_provenance` 同时含 `multimodal` 与 `visual`；
6. Agent 调用 `analyze_image`（真实 qwen3-vl:8b）分析该候选原图，回答趋势与峰值季度并标注“视觉估计需核对”；
7. 前端回答下方直接展示“本次分析图片”卡片：真实原图缩略图、`视觉分析结果需核对` 徽标、查看原图、打开原文 PDF（截图 `/tmp/m7-visual-rag-e2e.png`）。

验收中修复的集成问题：

- `RAGCitationBranchName` 未包含 `visual` → 引用构造 ValidationError；
- 前端 `RAGCitationBranch` 类型与 SSE 解析未接受 `visual` → citations 事件解析失败导致回答被标记为错误；
- `QdrantVisualStore.search` 用 UUID 与字符串比较导致 0 命中；集合缺失时不应自动建集合（已改为返回空结果）。

资源注意事项：RTX 3060 12GB 无法同时驻留 VLM（Ollama ≈6GB）+ 文本 embedding + reranker + 视觉编码器（≈4.4GB），因此端到端验收用 CPU 做查询编码（dtype 必须与建库时契约一致，否则集合指纹不同）。生产建议 ≥16GB 显存或按阶段串行加载；视觉编码是离线建库成本的大头（502 图 ≈84s）。

## 9. 复现命令

```bash
# 1) 构建隔离实验数据（从 A1.5 retained 文档）
uv run --group embedding-benchmark python scripts/m7_prepare_eval_data.py --output-dir /tmp/m7-data

# 2) 全量视觉索引（GPU）
KNOWLEDGE_SCOPE_DATA_DIR=/tmp/m7-data \
KNOWLEDGE_SCOPE_VISUAL_EMBEDDING_ENABLED=true \
KNOWLEDGE_SCOPE_VISUAL_MODEL_PATH=/home/amber/.cache/knowledgescope/qwen3-vl-embedding-2b \
KNOWLEDGE_SCOPE_VISUAL_DEVICE=cuda KNOWLEDGE_SCOPE_VISUAL_DTYPE=bfloat16 \
NO_PROXY=localhost,127.0.0.1 uv run knowledgescope visual-index build \
  --knowledge-base-id 00000000-0000-4000-8000-000000000007 --mode all

# 3) A/B/C 检索实验
uv run --group embedding-benchmark python scripts/m7_retrieval_experiment.py --output /tmp/m7-ab-results.json

# 4) 文本查询召回图片（独立证明）
uv run knowledgescope visual-index search "柱状图 四个季度 销售额 增长趋势 峰值" \
  --knowledge-base-id 00000000-0000-4000-8000-000000000007 --limit 5
```
