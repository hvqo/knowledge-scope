# PDF 自动入库（M8）

本文说明知识库上传 PDF 后的自动入库流程：从 HTTP 上传到解析、分块、四类索引构建与就绪验证。目标是把原本需要手工执行的多条 CLI（`parse-document`、`chunk-document`、`qdrant index-document`、`sparse-index build`、`multimodal-index build`、`visual-index build`）收敛成一个可观测、可重试的后台任务。

## 1. 设计边界

- **不新增数据库迁移**：任务状态、阶段细节和中间产物以原子 JSON 文件保存在 `data_dir/ingestion/` 下，不修改 `Document.status` 枚举（仍只有 `uploaded` / `registered`）。
- **单机、单 worker、单重型任务**：解析、Embedding、视觉编码串行执行，避免 MinerU / Embedding / VLM 同时占用 GPU 造成 OOM。
- **复用现有模块**：解析、分块、Qdrant 文本索引、BM25、多模态 Evidence 与视觉索引都调用现有服务，不重复实现检索算法。
- **失败可见、可重试、可恢复**：阶段状态、错误码、尝试次数写入任务文件；进程退出后重启会将未完成任务标记为中断并自动重排。

## 2. 入库流程

```
上传 PDF (HTTP)
  → documents 表入账 + 原始文件落盘
  → POST /documents 返回 201（立即返回 Document ID）
  → IngestionWorker.enqueue(document_id)
  → [parse]            MinerU 解析 → CanonicalDocument + 原始图片
  → [chunk]            结构化 Chunk
  → [text_index]       Qwen3-Embedding → 预生成 dense points（bundle）
  → [sparse_index]     BM25 records → 预生成（bundle）
  → [multimodal_index] Evidence + caption/context 文本向量 → 预生成（bundle）
  → [visual_index]     原始图片视觉向量 → 预生成（bundle，无图/未启用则跳过）
  → [publish]          按固定顺序发布：Evidence → dense → representation → sparse → visual
  → [verify]           校验 Chunk 数、dense 点数、representation 点数、sparse 命中、visual 点数
  → [graph]            明确 skipped（本流程不自动建图）
  → completed / failed（记录失败阶段、错误码、消息，可 retry）
```

`graph` 阶段固定记录为 `skipped`，不会把未建图的知识库标记为图谱就绪。

## 3. 阶段与状态

阶段顺序固定为：

| 阶段 | 说明 |
| --- | --- |
| `parse` | MinerU 解析 PDF，生成 CanonicalDocument 与图片素材 |
| `chunk` | 生成结构化 Chunk（含页码、区块、asset_refs） |
| `text_index` | 生成 Qdrant dense 向量（bundle 暂存，publish 时写入） |
| `sparse_index` | 生成 BM25 records（bundle 暂存） |
| `multimodal_index` | 生成 Evidence 与文本表示向量（bundle 暂存） |
| `visual_index` | 原始图片视觉向量（bundle 暂存；无图或未启用时跳过） |
| `publish` | 按顺序写入四类索引，任一步失败执行补偿回滚 |
| `verify` | 校验已发布索引与文档内容一致 |
| `graph` | 明确标记 skipped |

任务状态：`queued`、`running`、`completed`、`failed`。每个阶段另有 `pending`、`running`、`completed`、`failed`、`skipped`、`interrupted`。

任务文件位于 `data_dir/ingestion/tasks/<document_id>.json`，bundle 位于 `data_dir/ingestion/<document_id>/bundle.json`。写入使用同目录临时文件 + `fsync` + `os.replace`，并拒绝 symlink 与越界路径。

## 4. 发布一致性与失败处理

- `publish` 之前所有重结果都写入 bundle，不触碰正式索引；因此解析、分块、Embedding 失败不会污染现有检索结果。
- `publish` 顺序为：Evidence 文件 → dense → representation → sparse → visual。
- 视觉向量在 `visual_index` 阶段完成真实编码并写入 bundle；`publish` 只写入预生成的向量（`replace_document_vectors`），不重复加载视觉模型、不重新编码。
- 任一步发布失败时执行补偿：删除本次已写入的 dense / representation / sparse / visual 点，并删除刚写入的 Evidence 文件，然后抛出 `publish_failed`。
- 补偿是尽力而为的回滚，不宣称跨存储（文件系统 + Qdrant）的原子事务；若补偿本身失败，任务进入 `failed` 并保留错误信息，需要人工重试。已知边界记录在文档末尾。

## 5. 恢复与重试

- 服务启动时，worker 扫描任务文件：`queued` / `running` 任务标记为 interrupted，`interruptions += 1`，若尝试次数未达上限则重新入队；达到 `ingestion_max_attempts` 则置为 `failed`。
- 同一文档不会并发处理：任务仍在 `queued` / `running` 时重复 enqueue 直接返回现有任务。
- 已完成阶段根据 bundle 数据判定可复用，重试时从失败阶段继续，不重复执行已成功阶段。
- `POST /knowledge-bases/{kb}/documents/{doc}/ingestion/retry` 会重置 attempts 并重新入队；只允许对失败/未完成任务使用。
- 已完成文档在 M8.1 中禁止重新处理：`retry` / `force=True` 返回 `rebuild_not_supported`（API 409），因为重新解析会使旧 Chunk/Evidence 失效，而当前没有安全回滚旧索引的能力；如需重建，先删除文档再重新上传。已存在文档的旧检索结果、Evidence 与原始素材不受该拒绝逻辑影响。
- 未完成（未通过 verify）的索引不会被视为就绪：只有任务 `completed` 且四类索引标志为真（视觉索引在无图或未启用时可为空）时，前端才显示“已就绪”。

## 6. API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/v1/knowledge-bases/{kb}/documents/{doc}/ingestion` | 单个文档任务状态（任务 + 摘要），无任务时返回 null |
| POST | `/api/v1/knowledge-bases/{kb}/documents/{doc}/ingestion/retry` | 重试失败/中断任务，返回 202；已完成文档返回 409 `rebuild_not_supported` |
| GET | `/api/v1/knowledge-bases/{kb}/ingestion` | 该知识库全部任务状态（摘要列表） |

前端知识库详情页对文档列表轮询单文档状态：处理中每 1.5 秒刷新，完成后停止；失败时展示失败阶段摘要并提供“重试处理”入口。

## 7. 配置

| 配置键 | 默认值 | 说明 |
| --- | --- | --- |
| `KNOWLEDGE_SCOPE_INGESTION_ENABLED` | `true` | 是否启用自动入库 worker |
| `KNOWLEDGE_SCOPE_INGESTION_MAX_ATTEMPTS` | `3` | 单任务最大自动/手动尝试次数 |
| `KNOWLEDGE_SCOPE_INGESTION_VISUAL_MODE` | `all` | 视觉索引模式（`all` / `selective`） |

自动入库依赖既有配置：`DATA_DIR`、Qdrant、Embedding、BM25、MinerU、视觉编码模型（`VISUAL_EMBEDDING_ENABLED`）与 VLM（用于后续问答，不参与入库）。

## 8. 已知边界

1. **跨存储发布非原子**：Evidence 文件与 Qdrant 四个集合不属于同一事务，补偿失败时可能出现部分索引残留；任务会标记 failed，需要重试或人工清理。
2. **任务文件为单机本地状态**：多实例部署时每个实例有独立任务目录，不支持共享队列；当前定位为单机部署。
3. **进程内队列不持久**：服务重启时通过任务文件恢复，队列内存丢失但任务不丢；恢复语义是“重新排队”而非断点续传。
4. **视觉索引依赖独立模型**：视觉编码模型不可用或未启用时，`visual_index` 阶段标记 skipped，文档仍可完成文本/多模态索引；但前端会显示视觉索引未就绪。
5. **图谱不自动构建**：`graph` 阶段始终 skipped，图谱检索需要另行构建。
6. **MinerU 仍为外部运行时**：需要 `KNOWLEDGE_SCOPE_MINERU_COMMAND` 指向可用的 MinerU 可执行文件。
7. **已完成文档不支持重建**：删除文档前不能安全地重新解析并回滚旧索引；M8.1 选择 fail-closed（409），避免“重新入库失败清空旧索引”。
8. **阶段耗时边界**：视觉编码在 CPU 上包含模型加载与编码（实测 2 张图约 75s/首次；后续复用进程内模型更快），`publish` 仍受 BM25 全量代提交与 Qdrant 写入影响（实测约 66s）；本轮不再做增量 BM25 或编码缓存优化。
9. **严格校验**：BM25 以 `SparseIndexStore` 活动代记录数为准，不再用自然语言查询命中作为成功依据；视觉索引必须匹配当前活动代的实际 `(evidence_id, asset_index)` 集合，仅剩 pointer 时验证失败。

## 9. M8.1 真实验收记录（2026-10-26）

环境：`data_dir=/tmp/m8r-data`，CPU 文本/视觉编码 + GPU Ollama `qwen3-vl:8b`，真实 DeepSeek Flash 问答；隔离 KB `f42808db-7b35-4b24-aa95-79267d940bb4`。

| 场景 | 结果 |
| --- | --- |
| 新 PDF 首次完整入库（含图片/统计图） | completed，chunks 1，images 2，text/bm25/multimodal/visual 全就绪，visual_status=ready |
| 无图 PDF 完整入库 | completed，images 0，text/bm25/multimodal true，visual_status=skipped_no_images |
| 已完成文档重新处理 | `POST retry` 409 `rebuild_not_supported`；重复上传同一 PDF 409；retrieval/search 命中 chunk 与 Evidence 素材保持不变 |
| BM25 索引缺失 | 删除活动代后 `verify` 返回 `verify_bm25`；恢复 bundle 记录后再次验证通过 |
| 视觉索引仅 pointer、缺图片点 | 删除数据点后 `verify` 返回 `verify_visual_index`；从真实素材重建 2 张图后验证通过 |
| 解析失败后手动重试 | 首次 `parse_failed`（parsing 目录不可写）；恢复权限后 `POST retry` 202 → completed，四类索引就绪 |
| Agent 图文问答 + 原图预览 | agent+unified 查询返回 analyze_image 工具轨迹、visual 分支 success、visual_sources 1；Evidence detail 200、asset 200（image/jpeg，33452 bytes） |
