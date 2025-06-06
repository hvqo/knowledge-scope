# 本地产品演示流程

这份流程用于在本地检查 KnowledgeScope 的主要产品路径，不是评测协议，也不产生冻结基准结果。
演示应使用不含敏感信息的本地 PDF 和隔离的业务 PostgreSQL 数据源；不要把业务 fixture 导入
KnowledgeScope 的应用数据库。

## 准备环境

```bash
uv sync
cp .env.example .env
docker compose up -d postgres qdrant neo4j
uv run alembic upgrade head
```

`.env` 中的 `KNOWLEDGE_SCOPE_DATABASE_URL` 只连接 KnowledgeScope 应用库。
ChatBI 演示库必须是同一 PostgreSQL 服务中的独立数据库，不能把 fixture 加载到
`knowledgescope` 应用库。

## 初始化 ChatBI 演示库

仓库没有自动加载业务 fixture 的应用启动钩子，也没有数据源注册 CLI。现有可重复的
fixture 是 [`tests/fixtures/chatbi_demo.sql`](../../tests/fixtures/chatbi_demo.sql)。下面的
命令使用已经启动的 Compose PostgreSQL 服务创建独立的 `chatbi_demo` 数据库，并加载该
fixture；如果这个数据库已经存在，只需跳过第一条命令：

```bash
docker compose exec -T postgres createdb -U knowledgescope chatbi_demo
docker compose exec -T postgres psql -U knowledgescope -d chatbi_demo < tests/fixtures/chatbi_demo.sql
```

上面的用户与当前 `compose.yaml` 的本地开发默认值一致；如果本地覆盖了 Compose 用户，
将 `-U knowledgescope` 替换为对应的本地用户。密码不要写入文档或提交到仓库。fixture
会创建 `chatbi_demo` schema、`customers`/`sales` 表和
`region_sales` 视图，重复加载会重建该 schema。

## 启动应用并注册数据源

在启动后端的同一个终端中设置外部连接引用指向的环境变量，并将 ChatBI allow-list
切换到 fixture 的 schema。URL 只保存在本地环境中；使用你本地 PostgreSQL 的实际用户、
密码和端口值：

```bash
export CHATBI_DEMO_DATABASE_URL="postgresql://LOCAL_USER:LOCAL_PASSWORD@127.0.0.1:${KNOWLEDGE_SCOPE_POSTGRES_PORT:-5433}/chatbi_demo"
export KNOWLEDGE_SCOPE_CHATBI_ALLOWED_SCHEMAS='["chatbi_demo"]'
uv run uvicorn knowledge_scope.api.app:app --reload
```

应用启动后，使用现有 REST API 注册数据源。`connection_ref` 是 opaque reference，
不是把数据库 URL 发给 API：

```bash
curl -sS -X POST http://127.0.0.1:8000/api/v1/chatbi/data-sources \
  -H 'Content-Type: application/json' \
  -d '{
    "display_name": "本地 ChatBI 演示库",
    "dialect": "postgresql",
    "enabled": true,
    "connection_ref": "env:CHATBI_DEMO_DATABASE_URL",
    "default_schema": "chatbi_demo"
  }'
```

保存响应中的 `id`（下文记为 `<datasource_id>`）。API 不会在响应中返回
`connection_ref`。可用以下现有 CLI 完成一次只读 schema discovery：

```bash
uv run knowledgescope chatbi schema <datasource_id>
```

也可以直接检查 REST 结果：

```bash
curl -sS http://127.0.0.1:8000/api/v1/chatbi/data-sources/<datasource_id>/schema
```

这两个路径都会重新读取已注册数据源的外部 schema；没有单独的 schema 持久化步骤。

另开终端启动前端：

```bash
cd frontend
npm install
npm run dev
```

使用浏览器打开 `http://localhost:5173`，进入 `/analysis`，确认刚注册的数据源出现在
数据源列表中且为启用状态。若未配置 LLM，仍可以检查知识库、报告编辑和错误状态，跳过
需要生成内容的步骤。

## 逐步检查

1. 创建知识库并上传一个本地 PDF，确认列表显示上传或登记状态。
2. 进入 AI 问答，选择该知识库，发送一个与 PDF 内容相关的问题，确认回答逐段出现，完成后能打开来源卡片。
3. 进入数据分析，选择已启用的 `chatbi_demo` 数据源，提交一个简单的统计问题（例如
   “按地区统计销售额”），确认表格、行数和必要的截断提示来自后端。
4. 创建报告并关联知识库/数据源，打开报告工作区，修改章节正文后切换章节，确认草稿先保存。
5. 在来源面板分别插入一条 RAG 证据和一条 ChatBI 结果，确认来源列表、正文追加和引用信息一致。
6. 生成大纲或章节草稿，检查预览内容；只有点击接受并保存后，内容才进入报告。
7. 分别导出 DOCX 与 PDF，打开文件确认章节、正文、来源清单和 ChatBI 表头/行内容可读。
8. 在请求进行中切换报告或离开页面，确认旧的 RAG、ChatBI、AI 和导出响应不会写入新报告。
9. 对不存在的报告、停用的数据源和超限导出操作执行重试，确认页面显示中文的受控错误。

## 收尾

删除仅为演示创建的报告和本地测试数据；停止基础设施时使用：

```bash
docker compose down
```

不要使用 `docker compose down -v`，除非明确确认要删除本机其他开发数据。
