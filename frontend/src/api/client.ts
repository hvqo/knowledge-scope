import type {
  HealthResponse,
  KnowledgeBase,
  KnowledgeBaseCreateRequest,
  KnowledgeBaseListResponse,
  KnowledgeBaseUpdateRequest,
  MetaResponse,
  Document,
  DocumentListResponse,
  ChatBIAnalysisResult,
  ChatBIDataSourceListResponse,
  ChatBIResultScalar,
  Report,
  ReportCreateRequest,
  ReportListResponse,
  ReportSection,
  ReportSectionCreateRequest,
  ReportSectionUpdateRequest,
  ReportSourceCreateRequest,
  ReportSourceInsertRequest,
  ReportSourceReference,
  ReportUpdateRequest,
  RAGCitation,
  RAGCompleteData,
  RAGStreamEvent,
} from "./types";

const DEFAULT_API_BASE_URL = "/api";
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || DEFAULT_API_BASE_URL).replace(
  /\/$/,
  "",
);

export class ApiError extends Error {
  readonly status: number;
  readonly detail?: string;
  readonly category?: string;

  constructor(status: number, statusText: string, detail?: string, category?: string) {
    super(detail || `API request failed: ${status} ${statusText}`);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
    this.category = category;
  }
}

export interface UserFacingErrorOptions {
  conflictMessage?: string;
}

export function getUserFacingError(
  error: unknown,
  fallback: string,
  options: UserFacingErrorOptions = {},
): string {
  if (!(error instanceof ApiError)) {
    return fallback;
  }

  if (error.status === 404) {
    return "请求的内容不存在或已被删除。";
  }
  if (error.status === 409) {
    if (error.category === "datasource_disabled") {
      return "当前数据源不可用或已停用。";
    }
    return options.conflictMessage ?? fallback;
  }
  if (error.status === 413) {
    return "文件过大，请选择较小的 PDF。";
  }
  if (error.status === 415) {
    return "仅支持 PDF 文件。";
  }
  if (error.status === 422) {
    return "提交内容未通过校验，请检查后重试。";
  }
  if (error.status >= 500) {
    return "服务暂时不可用，请稍后重试。";
  }
  return fallback;
}

interface ParsedErrorDetail {
  detail?: string;
  category?: string;
}

function safeErrorString(value: unknown): string | undefined {
  return typeof value === "string" && value.length <= 2000 ? value : undefined;
}

function safeErrorCategory(value: unknown): string | undefined {
  return typeof value === "string" && /^[a-z][a-z0-9_]{0,63}$/.test(value)
    ? value
    : undefined;
}

async function readErrorDetail(response: Response): Promise<ParsedErrorDetail> {
  try {
    const payload: unknown = await response.json();
    if (typeof payload === "object" && payload !== null && "detail" in payload) {
      const detail = payload.detail;
      if (typeof detail === "string") {
        return { detail: safeErrorString(detail) };
      }
      if (typeof detail === "object" && detail !== null && !Array.isArray(detail)) {
        const detailRecord = detail as Record<string, unknown>;
        return {
          detail: safeErrorString(detailRecord.message),
          category: safeErrorCategory(detailRecord.category),
        };
      }
      if (Array.isArray(detail)) {
        const messages = detail.flatMap((item) => {
          if (typeof item === "object" && item !== null && "msg" in item) {
            const message = safeErrorString(item.msg);
            return message === undefined ? [] : [message];
          }
          return [];
        });
        return messages.length > 0 ? { detail: messages.join("；") } : {};
      }
    }
  } catch {
    return {};
  }
  return {};
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const headers = new Headers(options.headers);
  headers.set("Accept", "application/json");
  if (
    options.body !== undefined &&
    options.body !== null &&
    !(options.body instanceof FormData) &&
    !headers.has("Content-Type")
  ) {
    headers.set("Content-Type", "application/json");
  }

  const response = await fetch(`${API_BASE_URL}${path}`, { ...options, headers });

  if (!response.ok) {
    const errorDetail = await readErrorDetail(response);
    throw new ApiError(
      response.status,
      response.statusText,
      errorDetail.detail,
      errorDetail.category,
    );
  }

  return (await response.json()) as T;
}

async function requestNoContent(path: string, options: RequestInit = {}): Promise<void> {
  const headers = new Headers(options.headers);
  headers.set("Accept", "application/json");
  const response = await fetch(`${API_BASE_URL}${path}`, { ...options, headers });

  if (!response.ok) {
    const errorDetail = await readErrorDetail(response);
    throw new ApiError(
      response.status,
      response.statusText,
      errorDetail.detail,
      errorDetail.category,
    );
  }
}

export function fetchHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/v1/health");
}

export function fetchMeta(): Promise<MetaResponse> {
  return request<MetaResponse>("/v1/meta");
}

export interface KnowledgeBaseListParams {
  limit?: number;
  offset?: number;
}

export function fetchKnowledgeBases({
  limit = 20,
  offset = 0,
}: KnowledgeBaseListParams = {}): Promise<KnowledgeBaseListResponse> {
  const searchParams = new URLSearchParams({
    limit: String(limit),
    offset: String(offset),
  });
  return request<KnowledgeBaseListResponse>(`/v1/knowledge-bases?${searchParams.toString()}`);
}

export function fetchKnowledgeBase(id: string): Promise<KnowledgeBase> {
  return request<KnowledgeBase>(`/v1/knowledge-bases/${encodeURIComponent(id)}`);
}

export function createKnowledgeBase(
  payload: KnowledgeBaseCreateRequest,
): Promise<KnowledgeBase> {
  return request<KnowledgeBase>("/v1/knowledge-bases", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function updateKnowledgeBase(
  id: string,
  payload: KnowledgeBaseUpdateRequest,
): Promise<KnowledgeBase> {
  return request<KnowledgeBase>(`/v1/knowledge-bases/${encodeURIComponent(id)}`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
}

export function deleteKnowledgeBase(id: string): Promise<void> {
  return requestNoContent(`/v1/knowledge-bases/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
}

export interface DocumentListParams {
  limit?: number;
  offset?: number;
}

export function fetchDocuments(
  knowledgeBaseId: string,
  { limit = 10, offset = 0 }: DocumentListParams = {},
): Promise<DocumentListResponse> {
  const searchParams = new URLSearchParams({
    limit: String(limit),
    offset: String(offset),
  });
  return request<DocumentListResponse>(
    `/v1/knowledge-bases/${encodeURIComponent(knowledgeBaseId)}/documents?${searchParams.toString()}`,
  );
}

export function uploadDocument(knowledgeBaseId: string, file: File): Promise<Document> {
  const formData = new FormData();
  formData.append("file", file);
  return request<Document>(
    `/v1/knowledge-bases/${encodeURIComponent(knowledgeBaseId)}/documents`,
    {
      method: "POST",
      body: formData,
    },
  );
}

export function deleteDocument(knowledgeBaseId: string, documentId: string): Promise<void> {
  return requestNoContent(
    `/v1/knowledge-bases/${encodeURIComponent(knowledgeBaseId)}/documents/${encodeURIComponent(documentId)}`,
    { method: "DELETE" },
  );
}

export interface ChatBIDataSourceListParams {
  limit?: number;
  offset?: number;
}

export function fetchChatBIDataSources({
  limit = 100,
  offset = 0,
}: ChatBIDataSourceListParams = {}): Promise<ChatBIDataSourceListResponse> {
  const searchParams = new URLSearchParams({
    limit: String(limit),
    offset: String(offset),
  });
  return request<ChatBIDataSourceListResponse>(
    `/v1/chatbi/data-sources?${searchParams.toString()}`,
  );
}

export interface RAGQueryRequest {
  query: string;
  knowledge_base_id: string;
  retrieval_mode?: "dense" | "unified";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function requiredString(value: unknown, field: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new Error(`RAG stream field ${field} is invalid`);
  }
  return value;
}

function nullableString(value: unknown, field: string): string | null {
  if (value === null || value === undefined) {
    return null;
  }
  return requiredString(value, field);
}

function requiredFiniteNumber(value: unknown, field: string): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`RAG stream field ${field} is invalid`);
  }
  return value;
}

function nullableFiniteNumber(value: unknown, field: string): number | null {
  if (value === null || value === undefined) {
    return null;
  }
  return requiredFiniteNumber(value, field);
}

function nullableBoolean(value: unknown, field: string): boolean | null {
  if (value === null || value === undefined) {
    return null;
  }
  if (typeof value !== "boolean") {
    throw new Error(`RAG stream field ${field} is invalid`);
  }
  return value;
}

function stringArray(value: unknown, field: string): string[] {
  if (!Array.isArray(value) || value.some((item) => typeof item !== "string")) {
    throw new Error(`RAG stream field ${field} is invalid`);
  }
  return value;
}

function parseChatBIScalar(value: unknown, depth = 0): ChatBIResultScalar {
  if (depth > 32) {
    throw new Error("ChatBI result value is too deeply nested");
  }
  if (value === null || typeof value === "string" || typeof value === "boolean") {
    return value;
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) {
      throw new Error("ChatBI result contains a non-finite number");
    }
    return value;
  }
  if (Array.isArray(value)) {
    return value.map((item) => parseChatBIScalar(item, depth + 1));
  }
  if (isRecord(value)) {
    return Object.fromEntries(
      Object.entries(value).map(([key, item]) => [key, parseChatBIScalar(item, depth + 1)]),
    );
  }
  throw new Error("ChatBI result contains an unsupported value");
}

function parseChatBIResult(value: unknown): ChatBIAnalysisResult {
  if (!isRecord(value)) {
    throw new Error("ChatBI response is invalid");
  }
  const status = requiredString(value.execution_status, "execution_status");
  const allowedStatuses = new Set([
    "succeeded",
    "rejected",
    "failed",
    "cancelled",
    "clarify",
    "refuse",
    "eligibility_unavailable",
  ]);
  if (!allowedStatuses.has(status)) {
    throw new Error("ChatBI execution status is invalid");
  }
  const columnsValue = value.columns;
  if (!Array.isArray(columnsValue)) {
    throw new Error("ChatBI columns are invalid");
  }
  const columns = columnsValue.map((item, index) => {
    if (!isRecord(item)) {
      throw new Error("ChatBI column is invalid");
    }
    const ordinal = requiredFiniteNumber(item.ordinal, `columns[${index}].ordinal`);
    if (!Number.isInteger(ordinal) || ordinal < 0) {
      throw new Error("ChatBI column ordinal is invalid");
    }
    return {
      name: requiredString(item.name, `columns[${index}].name`),
      data_type: requiredString(item.data_type, `columns[${index}].data_type`),
      nullable: typeof item.nullable === "boolean" ? item.nullable : (() => {
        throw new Error("ChatBI column nullable flag is invalid");
      })(),
      ordinal,
    };
  });
  if (!Array.isArray(value.rows)) {
    throw new Error("ChatBI rows are invalid");
  }
  const rows = value.rows.map((row) => {
    if (!Array.isArray(row)) {
      throw new Error("ChatBI result row is invalid");
    }
    return row.map((cell) => parseChatBIScalar(cell));
  });
  if (rows.some((row) => row.length !== columns.length)) {
    throw new Error("ChatBI rows do not match the column contract");
  }
  const eligibilityValue = value.eligibility;
  let eligibility: ChatBIAnalysisResult["eligibility"] = null;
  if (eligibilityValue !== null && eligibilityValue !== undefined) {
    if (!isRecord(eligibilityValue)) {
      throw new Error("ChatBI eligibility is invalid");
    }
    const decision = requiredString(eligibilityValue.decision, "eligibility.decision");
    if (decision !== "eligible" && decision !== "clarify" && decision !== "refuse" && decision !== "unavailable") {
      throw new Error("ChatBI eligibility decision is invalid");
    }
    eligibility = {
      decision,
      reason_code: requiredString(eligibilityValue.reason_code, "eligibility.reason_code"),
      user_message: nullableString(eligibilityValue.user_message, "eligibility.user_message"),
      clarification_question: nullableString(
        eligibilityValue.clarification_question,
        "eligibility.clarification_question",
      ),
    };
  }
  const rowCount = requiredFiniteNumber(value.row_count, "row_count");
  const elapsedMs = requiredFiniteNumber(value.elapsed_ms, "elapsed_ms");
  if (!Number.isInteger(rowCount) || rowCount < 0 || elapsedMs < 0) {
    throw new Error("ChatBI result metrics are invalid");
  }
  const truncated = value.truncated;
  if (typeof truncated !== "boolean") {
    throw new Error("ChatBI truncation flag is invalid");
  }
  const truncationReason = nullableString(value.truncation_reason, "truncation_reason");
  if (truncationReason !== null && truncationReason !== "row_limit" && truncationReason !== "payload_bytes") {
    throw new Error("ChatBI truncation reason is invalid");
  }
  if (truncated !== (truncationReason !== null)) {
    throw new Error("ChatBI truncation contract is invalid");
  }
  return {
    query_id: requiredString(value.query_id, "query_id"),
    datasource_id: requiredString(value.datasource_id, "datasource_id"),
    execution_status: status as ChatBIAnalysisResult["execution_status"],
    eligibility,
    answer: nullableString(value.answer, "answer"),
    columns,
    rows,
    row_count: rowCount,
    truncated,
    truncation_reason: truncationReason,
    redacted_sql: nullableString(value.redacted_sql, "redacted_sql"),
    warnings: stringArray(value.warnings, "warnings"),
    error_message: nullableString(value.error_message, "error_message"),
    elapsed_ms: elapsedMs,
  };
}

export function askChatBI(
  datasourceId: string,
  question: string,
  signal?: AbortSignal,
): Promise<ChatBIAnalysisResult> {
  return request<unknown>(
    `/v1/chatbi/data-sources/${encodeURIComponent(datasourceId)}/ask`,
    {
      method: "POST",
      body: JSON.stringify({ question }),
      signal,
    },
  ).then(parseChatBIResult);
}

export interface ReportListParams {
  limit?: number;
  offset?: number;
}

export function fetchReports({
  limit = 20,
  offset = 0,
}: ReportListParams = {}): Promise<ReportListResponse> {
  const searchParams = new URLSearchParams({
    limit: String(limit),
    offset: String(offset),
  });
  return request<ReportListResponse>(`/v1/reports?${searchParams.toString()}`);
}

export function createReport(payload: ReportCreateRequest): Promise<Report> {
  return request<Report>("/v1/reports", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function fetchReport(reportId: string): Promise<Report> {
  return request<Report>(`/v1/reports/${encodeURIComponent(reportId)}`);
}

export function updateReport(reportId: string, payload: ReportUpdateRequest): Promise<Report> {
  return request<Report>(`/v1/reports/${encodeURIComponent(reportId)}`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
}

export function deleteReport(reportId: string): Promise<void> {
  return requestNoContent(`/v1/reports/${encodeURIComponent(reportId)}`, { method: "DELETE" });
}

export function createReportSection(
  reportId: string,
  payload: ReportSectionCreateRequest = {},
): Promise<ReportSection> {
  return request<ReportSection>(`/v1/reports/${encodeURIComponent(reportId)}/sections`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function updateReportSection(
  reportId: string,
  sectionId: string,
  payload: ReportSectionUpdateRequest,
): Promise<ReportSection> {
  return request<ReportSection>(
    `/v1/reports/${encodeURIComponent(reportId)}/sections/${encodeURIComponent(sectionId)}`,
    { method: "PATCH", body: JSON.stringify(payload) },
  );
}

export function deleteReportSection(reportId: string, sectionId: string): Promise<void> {
  return requestNoContent(
    `/v1/reports/${encodeURIComponent(reportId)}/sections/${encodeURIComponent(sectionId)}`,
    { method: "DELETE" },
  );
}

export function createReportSource(
  reportId: string,
  sectionId: string,
  payload: ReportSourceCreateRequest,
): Promise<ReportSourceReference> {
  return request<ReportSourceReference>(
    `/v1/reports/${encodeURIComponent(reportId)}/sections/${encodeURIComponent(sectionId)}/sources`,
    { method: "POST", body: JSON.stringify(payload) },
  );
}

export function insertReportSource(
  reportId: string,
  sectionId: string,
  payload: ReportSourceInsertRequest,
): Promise<ReportSourceReference> {
  return request<ReportSourceReference>(
    `/v1/reports/${encodeURIComponent(reportId)}/sections/${encodeURIComponent(sectionId)}/sources/insert`,
    { method: "POST", body: JSON.stringify(payload) },
  );
}

export function deleteReportSource(reportId: string, sourceId: string): Promise<void> {
  return requestNoContent(
    `/v1/reports/${encodeURIComponent(reportId)}/sources/${encodeURIComponent(sourceId)}`,
    { method: "DELETE" },
  );
}

function parseCitationBranch(value: unknown): RAGCitation["branch_provenance"][number] {
  if (!isRecord(value)) {
    throw new Error("RAG citation branch is invalid");
  }
  const branch = requiredString(value.branch, "branch");
  if (branch !== "dense" && branch !== "sparse" && branch !== "graph" && branch !== "multimodal") {
    throw new Error("RAG citation branch is invalid");
  }
  const graphSeed = nullableString(value.graph_seed_entity_id, "graph_seed_entity_id");
  const graphReason = nullableString(value.graph_retrieval_reason, "graph_retrieval_reason");
  if (branch === "graph" && (graphSeed === null || graphReason === null)) {
    throw new Error("RAG graph citation branch is incomplete");
  }
  if (branch !== "graph" && (graphSeed !== null || graphReason !== null)) {
    throw new Error("RAG non-graph citation branch contains graph metadata");
  }
  return {
    branch,
    rank: requiredFiniteNumber(value.rank, "rank"),
    score: requiredFiniteNumber(value.score, "score"),
    graph_seed_entity_id: graphSeed,
    graph_retrieval_reason: graphReason,
  };
}

function parseCitation(value: unknown): RAGCitation {
  if (!isRecord(value)) {
    throw new Error("RAG citation is invalid");
  }
  const candidateKind = requiredString(value.candidate_kind, "candidate_kind");
  if (candidateKind !== "chunk" && candidateKind !== "evidence") {
    throw new Error("RAG citation kind is invalid");
  }
  const modality = nullableString(value.modality, "modality");
  if (
    modality !== null &&
    modality !== "text" &&
    modality !== "image" &&
    modality !== "table" &&
    modality !== "formula"
  ) {
    throw new Error("RAG citation modality is invalid");
  }
  const citation: RAGCitation = {
    marker: requiredString(value.marker, "marker"),
    document_id: requiredString(value.document_id, "document_id"),
    knowledge_base_id: nullableString(value.knowledge_base_id, "knowledge_base_id"),
    candidate_kind: candidateKind,
    chunk_id: nullableString(value.chunk_id, "chunk_id"),
    evidence_id: nullableString(value.evidence_id, "evidence_id"),
    modality,
    representation_ids: stringArray(value.representation_ids, "representation_ids"),
    asset_refs: stringArray(value.asset_refs, "asset_refs"),
    page_start: requiredFiniteNumber(value.page_start, "page_start"),
    page_end: requiredFiniteNumber(value.page_end, "page_end"),
    source_block_ids: stringArray(value.source_block_ids, "source_block_ids"),
    section_path: stringArray(value.section_path, "section_path"),
    section_title: nullableString(value.section_title, "section_title"),
    snippet: nullableString(value.snippet, "snippet"),
    snippet_kind: (() => {
      const valueKind = nullableString(value.snippet_kind, "snippet_kind");
      if (valueKind !== null && valueKind !== "source" && valueKind !== "representation") {
        throw new Error("RAG citation snippet kind is invalid");
      }
      return valueKind;
    })(),
    final_rank: nullableFiniteNumber(value.final_rank, "final_rank"),
    final_reranker_score: nullableFiniteNumber(value.final_reranker_score, "final_reranker_score"),
    branch_provenance: Array.isArray(value.branch_provenance)
      ? value.branch_provenance.map(parseCitationBranch)
      : (() => {
          throw new Error("RAG branch provenance is invalid");
        })(),
  };
  if (citation.candidate_kind === "chunk") {
    if (citation.chunk_id === null || citation.evidence_id !== null || citation.modality !== null) {
      throw new Error("RAG chunk citation contract is invalid");
    }
  } else if (
    citation.chunk_id !== null ||
    citation.evidence_id === null ||
    citation.modality === null ||
    citation.representation_ids.length === 0
  ) {
    throw new Error("RAG Evidence citation contract is invalid");
  }
  if ((citation.snippet === null) !== (citation.snippet_kind === null)) {
    throw new Error("RAG citation snippet contract is invalid");
  }
  if (
    citation.candidate_kind === "chunk" &&
    citation.snippet_kind !== null &&
    citation.snippet_kind !== "source"
  ) {
    throw new Error("RAG chunk snippet contract is invalid");
  }
  if (
    citation.candidate_kind === "evidence" &&
    citation.snippet_kind !== null &&
    citation.snippet_kind !== "representation"
  ) {
    throw new Error("RAG Evidence snippet contract is invalid");
  }
  return citation;
}

function parseRagStreamEvent(eventName: string, rawData: string): RAGStreamEvent {
  let parsed: unknown;
  try {
    parsed = JSON.parse(rawData) as unknown;
  } catch {
    throw new Error("RAG stream returned invalid JSON");
  }
  if (!isRecord(parsed)) {
    throw new Error("RAG stream payload is not an object");
  }
  if (eventName === "answer_delta") {
    return { event: "answer_delta", data: { text: requiredString(parsed.text, "text") } };
  }
  if (eventName === "citations") {
    const items = Array.isArray(parsed.items) ? parsed.items.map(parseCitation) : null;
    if (items === null) {
      throw new Error("RAG citations payload is invalid");
    }
    return {
      event: "citations",
      data: { prompt_version: requiredString(parsed.prompt_version, "prompt_version"), items },
    };
  }
  if (eventName === "error") {
    return {
      event: "error",
      data: {
        category: requiredString(parsed.category, "category"),
        message: requiredString(parsed.message, "message"),
      },
    };
  }
  if (eventName === "complete") {
    const status = requiredString(parsed.status, "status");
    if (status !== "completed" && status !== "insufficient_evidence" && status !== "error") {
      throw new Error("RAG completion status is invalid");
    }
    const retrievalMode = requiredString(parsed.retrieval_mode, "retrieval_mode");
    if (retrievalMode !== "dense" && retrievalMode !== "unified") {
      throw new Error("RAG retrieval mode is invalid");
    }
    const complete: RAGCompleteData = {
      status,
      prompt_version: requiredString(parsed.prompt_version, "prompt_version"),
      provider: nullableString(parsed.provider, "provider"),
      model: nullableString(parsed.model, "model"),
      retrieval_mode: retrievalMode,
      retrieval_degraded: nullableBoolean(parsed.retrieval_degraded, "retrieval_degraded"),
      retrieval_branch_statuses: isRecord(parsed.retrieval_branch_statuses)
        ? Object.fromEntries(
            Object.entries(parsed.retrieval_branch_statuses).filter(
              ([, value]) => typeof value === "string",
            ),
          ) as RAGCompleteData["retrieval_branch_statuses"]
        : null,
      input_tokens: nullableFiniteNumber(parsed.input_tokens, "input_tokens"),
      output_tokens: nullableFiniteNumber(parsed.output_tokens, "output_tokens"),
      finish_reason: nullableString(parsed.finish_reason, "finish_reason"),
      retrieval_latency_ms: nullableFiniteNumber(parsed.retrieval_latency_ms, "retrieval_latency_ms"),
      llm_latency_ms: nullableFiniteNumber(parsed.llm_latency_ms, "llm_latency_ms"),
      latency_ms: requiredFiniteNumber(parsed.latency_ms, "latency_ms"),
    };
    return { event: "complete", data: complete };
  }
  throw new Error("RAG stream event is unsupported");
}

function parseSseBlock(block: string): { event: string; data: string } | null {
  const data: string[] = [];
  let event = "message";
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      data.push(line.slice(5).trimStart());
    }
  }
  return data.length > 0 ? { event, data: data.join("\n") } : null;
}

export async function* streamRagQuery(
  payload: RAGQueryRequest,
  signal?: AbortSignal,
): AsyncGenerator<RAGStreamEvent> {
  const response = await fetch(`${API_BASE_URL}/v1/rag/query`, {
    method: "POST",
    headers: { Accept: "text/event-stream", "Content-Type": "application/json" },
    body: JSON.stringify({ ...payload, retrieval_mode: payload.retrieval_mode ?? "dense" }),
    signal,
  });
  if (!response.ok) {
    const errorDetail = await readErrorDetail(response);
    throw new ApiError(
      response.status,
      response.statusText,
      errorDetail.detail,
      errorDetail.category,
    );
  }
  if (response.body === null) {
    throw new Error("RAG stream has no response body");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
      let separatorIndex = buffer.indexOf("\n\n");
      while (separatorIndex >= 0) {
        const block = buffer.slice(0, separatorIndex);
        buffer = buffer.slice(separatorIndex + 2);
        const parsed = parseSseBlock(block);
        if (parsed !== null) {
          yield parseRagStreamEvent(parsed.event, parsed.data);
        }
        separatorIndex = buffer.indexOf("\n\n");
      }
      if (done) {
        if (buffer.trim()) {
          const parsed = parseSseBlock(buffer);
          if (parsed !== null) {
            yield parseRagStreamEvent(parsed.event, parsed.data);
          }
        }
        break;
      }
    }
  } finally {
    reader.releaseLock();
  }
}
