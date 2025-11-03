# A2.7 RAG QA

KnowledgeScope 当前提供一个最小的 RAG QA 编排：

默认路径为 `query → Qwen/Qwen3-Embedding-0.6B → Qdrant dense Top-10 → BGE reranker → bounded context → LLM Gateway`。
请求显式指定 `retrieval_mode=unified` 时，改为复用 A4.4 的统一候选池和最终重排，
再进入同一套 context、citation 和 LLM Gateway 流程。

## 当前实现

- `RAGService` 默认复用现有 dense retrieval、`RerankingService` 和 async `LLMGateway`；`unified` 模式复用现有 `UnifiedRetrievalService` 的四条分支和最终候选，不在 RAG 层重复检索或重排。不重新运行 MinerU，也不引入 LangChain。
- 默认检索 10 个 dense candidates，再用显式的 `bge-reranker-v2-m3` 重排；本地 embedding、Qdrant I/O 和 reranker 在 `asyncio.to_thread` 中执行，不占用 FastAPI event loop。每个进程内的模型 adapter 通过小粒度线程锁串行化同一模型的本地推理；RAG service 也会串行化一次 context selection。当前不承诺 GPU 并发吞吐。
- Dense 默认路径只把有文本的 chunk 放入回答上下文；Unified 路径还可以放入有界的 Evidence representation 文本。上下文按对应检索结果顺序选择，并受 `KNOWLEDGE_SCOPE_RAG_CONTEXT_BUDGET_CHARS` 字符预算限制；完整文本放不进剩余预算的候选会被跳过，因此不会为未发送的内容生成 citation。source block 重叠本身不等于重复，因为 A1.6 可能把同一 source block 拆成多个不重叠 chunk；只有同一 lineage 下的精确规范化重复文本会被抑制。没有可用文本时直接返回受控的“当前检索到的资料不足以回答该问题”，不会调用 LLM。
- 字符预算只约束选中 chunk 文本的字符数，不包含 system/user prompt、marker 和协议包装，也不使用 tokenizer；它是近似的工程上限，不是精确的 LLM token context budget。当前没有 source block 的字符/token offset，因此无法可靠地自动消除所有近似重复内容。
- 每个选中的 context item 都生成确定性的请求内 marker（`C1`、`C2`……）。marker 和 document、page、chunk 或 Evidence、source block、section 元数据由应用生成并在最终 SSE citation event 中返回；模型输出中的未知、重复或格式错误 marker 不会被解析为 citation metadata，应用只信任自己的 citation event。
- `unified` 模式下，chunk 候选仍引用真实 `chunk_id`；image、table、formula 等 Evidence 候选引用真实 `evidence_id` 和 `representation_ids`，不伪造 `chunk_id`。representation 文本只是送入模型的可检索上下文载体，权威 citation lineage 仍来自当前 Evidence/Representation 状态；Unified branch 的 rank、score、Graph seed/path 会在 citation provenance 中保留。
- `/chat` 对话工作区使用本节 SSE 接口，并复用本轮问答作为追问上下文；对话与消息的持久化见 [对话工作区](chat-workspace.md)。
- `retrieval_mode=auto`（前端默认「智能」）时，路由会在检索分支之间选择：短、单一子句、无对比/汇总/多跳线索的问题走 `dense`（实测 0.09 s）；礼貌语包裹的单一事实问句（含「多少/何时/哪里/是否」等、去掉语气词后不超过 40 字）同样走 `dense`；其余（含对比、原因、流程、汇总、多文档线索）走 `unified`（实测 4.5 s）。显式选择 `dense`/`unified` 时不做自动降级，`complete.route` 始终回填真正使用的路径，便于前端与评测核对。
- 走 `dense` 之后还有一道**证据校验**：把问题拆成按语序排列的内容词（去掉停用词与疑问词），若组装出的上下文**没有提到问题开头的主体词**（例如「曝气池的溶解氧…」中上下文完全没出现「曝气」），或整体内容词覆盖率低于 `rag_auto_escalate_min_coverage`（默认 0.5，即一半），说明快速路径拿到的材料与问题主体不符，会立即追加一次 `unified` 检索并采信它的上下文，同时在 `complete.retrieval_escalated=true` 标注；前端会在来源面板提示「快速检索得到的证据较弱，已自动扩展为多路检索」。显式选择「快速」不会触发升级；把阈值设为 0 可只保留主体词校验。礼貌语包裹的长问句（如「请问一下这份资料里提到的养鱼池增氧的具体措施是什么呢」）只要主体词命中就保持在快速路径，不会为措辞买单。
- 追问（短问题或含指代词）会先用一次短的非流式调用改写成可独立检索的问题，再用改写结果检索；`complete.rewritten_query` 回填改写结果，回答仍然针对用户原问题。改写后的问句同时用于**决定路由**，因此省略主语的追问不会被误判成简单查表（`complete.rewrite_latency_ms` 回填改写耗时）。改写失败、超长、超时（默认 2.5 s）或与原问题等价时一律回退到原问题（fail-open），可用 `KNOWLEDGE_SCOPE_RAG_FOLLOWUP_REWRITE_ENABLED=false` 关闭。
- 只缓存**检索上下文**而不缓存回答：键由知识库、文档、请求的检索模式与规范化问题哈希而来，命中时 `complete.retrieval_cached=true`，检索延迟从秒级降到毫秒级；缓存值同时记录真正使用的路径，`auto` 命中后仍能正确回填 `route` 与 `retrieval_escalated`。空结果（`insufficient_evidence`）同样缓存，但 TTL 更短（默认 60 s，命中为 300 s），因为「资料里还没有」是最贵且最容易被重复问的一类问题；追问问句的改写结果也进同一缓存。默认后端 `memory`（进程内，最多 256 条），`redis` 可跨 worker 共享，`off` 关闭；缓存异常一律降级为「不缓存」，不会让问题失败。
- 请求进入检索之前先做一次确定性的**查询路由**：问候、致谢、告别以及「你是谁 / 你能做什么 / 怎么用」这类元问题由 `rag/routing.py` 规则识别后直接返回产品级回答，不检索、不调用 LLM，`complete.route` 为 `direct`、`retrieval_latency_ms` 为 0；其余问题一律按请求的 `retrieval_mode` 检索，`complete.route` 回填实际使用的 `dense`/`unified`，便于前端和评测区分两类回答。路由是纯规则匹配（整句锚定 + 长度上限 30 字符 + 内容词兜底），不做模型分类，因此没有额外延迟，也不会随模型漂移。citation 中的 `snippet` 仅用于来源卡片展示：`source` 表示文本 chunk 的原始片段，`representation` 表示可检索表示，后者不替代权威 Evidence。
- prompt 版本为 `rag-qa-v5`，要求回答只依据给定资料、证据不足时说明、不得编造，并只能引用上下文中的应用 marker。v5 增加公式规范：数学内容一律用 LaTeX（行内 `$a^x=N$`、独立成行 `$$x=\log_a N$$`），公式里不使用 Markdown 标记，金额与编号等普通文本不加美元符号；v4 重点规定**表达方式**：第一句就给结论、不复述问题、不写「资料中提到」这类旁白；用「你」称呼提问者；内容多时分条写，每条给具体对象/数值/条件/因果，不用空话；用 Markdown 组织排版（`##` 小标题、`-`/`1.` 列表、`**加粗**`）；资料只能回答一部分时**先把能回答的答完**，末尾用一句话说明缺口，资料完全无法回答时只用一句话说明并停止罗列。引用规则延续 v3：只能使用本轮列出的 `[Cn]`，标记紧贴所支撑句子的句末标点之前，不单独成行、不堆段尾、同一位置最多两个。v2 仍然有效：问题与资料无关（询问身份/能力/用法）时一句话说明并指出资料中没有相关内容，这类回答不加引用标记；更早轮次的引用标记不得复用。

## SSE API

```http
POST /api/v1/rag/query
Content-Type: application/json

{"query":"你的问题","knowledge_base_id":null,"document_id":null,"retrieval_mode":"dense"}
```

要选择 A4.4 统一路径，必须显式指定知识库：

```json
{"query":"你的问题","knowledge_base_id":"<knowledge-base-uuid>","retrieval_mode":"unified"}
```

追问可以把最近几轮问答一并发送，模型据此理解指代；历史只用于理解当前问题，其中的引用标记会由服务端剥离：

```json
{"query":"它有哪些限制？","knowledge_base_id":"<knowledge-base-uuid>","retrieval_mode":"unified","history":[{"role":"user","content":"上一轮问题"},{"role":"assistant","content":"上一轮回答"}]}
```

`history` 上限为 200 条、单条 4,000 字符、合计 60,000 字符；超出最近窗口（默认 8 轮）的部分会被压缩成摘要作为背景，`retrieval_mode` 取 `auto` / `dense` / `unified`，请求还可携带 `memories`（由服务端注入的长期记忆）。请求还可以通过 `retrieval_mode` 选择检索范围：`unified` 覆盖向量、关键词、图谱与多模态分支，`dense` 只做向量检索、明显更快；无论选哪种，元问题都由路由直接回答。历史只参与生成，不会绕过检索：如果本轮没有检索到可用文本，仍然直接返回「当前检索到的资料不足以回答该问题」，不会因为历史存在而调用 LLM。

未知 `retrieval_mode` 会被拒绝；`unified` 缺少 `knowledge_base_id` 也会被拒绝。
未指定该字段的既有客户端继续使用 `dense`。

响应为 `text/event-stream`，事件顺序如下：

1. `answer_delta`：增量回答文本；
2. `citations`：应用生成的 citation metadata；
3. `complete`：`completed`、`insufficient_evidence` 或 `error` 状态，以及 provider、model、token 和端到端延迟（可用时）。

成功请求的顺序是 `answer_delta* → citations → complete`，且只发送一次 `complete`。检索、模型、provider 或 usage 持久化失败会发送受控的 `error`，再发送 `status=error` 的终止 `complete`，不会伪装成成功；客户端取消会传播取消信号并关闭下游 async generator。`complete` 中的 `retrieval_latency_ms`、`llm_latency_ms` 和 `latency_ms` 分别表示 context selection、LLM gateway stream 消费和端到端 wall-clock 范围（后者包含前两阶段及调度开销）；LLM token 字段按 provider 可用性填写。LLM usage 仍由 A2.6 gateway 尝试写入 PostgreSQL，provider 调用、检索和 usage 写入不是原子事务。

`query` 最多接受 `4,000` 个字符；SSE event payload 使用严格的 Pydantic 模型校验。当前 endpoint 没有认证或限流，适用于受信任的本地/内网开发环境，不应直接暴露到公网。客户端取消可以阻止后续 LLM 调用并关闭 provider stream，但已经提交给 `asyncio.to_thread` 的同步 embedding、Qdrant 或 reranker 工作无法被 Python 线程强制中断，可能运行到当前调用结束。

## 配置

上下文和生成预算通过 `Settings` / `.env` 配置：

```dotenv
KNOWLEDGE_SCOPE_RAG_CANDIDATE_LIMIT=10
KNOWLEDGE_SCOPE_RAG_RERANK_LIMIT=5
KNOWLEDGE_SCOPE_RAG_CONTEXT_BUDGET_CHARS=6000
KNOWLEDGE_SCOPE_RAG_MAX_TOKENS=512
```

默认 provider 仍需要按 [A2.6 LLM gateway 说明](llm-gateway.md) 配置 `KNOWLEDGE_SCOPE_LLM_API_KEY`。正常测试使用 fake retrieval/reranker/gateway，不需要 GPU、网络或 API key；前端工作区不包含答案质量 benchmark 或 GraphRAG 功能。

真实运行还需要安装已有的本地模型依赖：

```bash
uv sync --group embedding-benchmark --group reranker-benchmark
```
