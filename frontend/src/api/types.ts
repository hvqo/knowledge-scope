export interface HealthResponse {
  status: "ok";
  project_name: string;
  version: string;
  python_version: string;
  config_status: "ok";
  environment: string;
  log_level: string;
  data_dir: string;
}

export interface MetaResponse {
  project_name: string;
  version: string;
  phase: "A1.4";
  status: "foundation";
  config_status: "ok";
}

export interface KnowledgeBase {
  id: string;
  name: string;
  description: string | null;
  created_at: string;
  updated_at: string;
}

export interface KnowledgeBaseListResponse {
  items: KnowledgeBase[];
  total: number;
  limit: number;
  offset: number;
}

export interface KnowledgeBaseCreateRequest {
  name: string;
  description?: string | null;
}

export interface KnowledgeBaseUpdateRequest {
  name?: string;
  description?: string | null;
}

export interface Document {
  id: string;
  knowledge_base_id: string;
  original_filename: string;
  media_type: "application/pdf";
  size_bytes: number;
  sha256: string;
  status: "uploaded" | "registered";
  created_at: string;
  updated_at: string;
}

export interface DocumentListResponse {
  items: Document[];
  total: number;
  limit: number;
  offset: number;
}

export type ChatBIDataSourceDialect = "postgresql";

export interface ChatBIDataSource {
  id: string;
  display_name: string;
  dialect: ChatBIDataSourceDialect;
  enabled: boolean;
  default_database: string | null;
  default_schema: string | null;
  connection_configured: boolean;
  created_at: string;
  updated_at: string;
}

export interface ChatBIDataSourceListResponse {
  items: ChatBIDataSource[];
  total: number;
  limit: number;
  offset: number;
}

export type ChatBIExecutionStatus =
  | "succeeded"
  | "rejected"
  | "failed"
  | "cancelled"
  | "clarify"
  | "refuse"
  | "eligibility_unavailable";
export type ChatBITruncationReason = "row_limit" | "payload_bytes";
export type ChatBIResultScalar =
  | string
  | number
  | boolean
  | null
  | ChatBIResultScalar[]
  | { [key: string]: ChatBIResultScalar };

export interface ChatBIColumnMetadata {
  name: string;
  data_type: string;
  nullable: boolean;
  ordinal: number;
}

export type ChatBIEligibilityDecision = "eligible" | "clarify" | "refuse" | "unavailable";

export interface ChatBIEligibilityPublic {
  decision: ChatBIEligibilityDecision;
  reason_code: string;
  user_message: string | null;
  clarification_question: string | null;
}

export interface ChatBIAnalysisResult {
  query_id: string;
  datasource_id: string;
  execution_status: ChatBIExecutionStatus;
  eligibility: ChatBIEligibilityPublic | null;
  answer: string | null;
  columns: ChatBIColumnMetadata[];
  rows: ChatBIResultScalar[][];
  row_count: number;
  truncated: boolean;
  truncation_reason: ChatBITruncationReason | null;
  redacted_sql: string | null;
  warnings: string[];
  error_message: string | null;
  elapsed_ms: number;
}

export interface ChatBIQuestionRequest {
  question: string;
}

export interface ReportSummary {
  id: string;
  title: string;
  knowledge_base_id: string | null;
  datasource_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface ReportChartPoint {
  label: string;
  value: number;
}

export interface ReportChartSpec {
  kind: "bar" | "line" | "pie";
  category_column: string;
  value_column: string;
  points: ReportChartPoint[];
}

export interface ReportSourceReference {
  id: string;
  section_id: string;
  kind: "rag_evidence" | "chatbi_result";
  title: string;
  knowledge_base_id: string | null;
  datasource_id: string | null;
  document_id: string | null;
  document_title: string | null;
  evidence_id: string | null;
  chunk_id: string | null;
  page_start: number | null;
  page_end: number | null;
  evidence_type: string | null;
  snippet: string | null;
  section_path: string[];
  source_block_ids: string[];
  question: string | null;
  answer: string | null;
  columns: ChatBIColumnMetadata[];
  rows: ChatBIResultScalar[][];
  row_count: number | null;
  truncated: boolean | null;
  truncation_reason: string | null;
  redacted_sql: string | null;
  chart_spec: ReportChartSpec | null;
  created_at: string;
}

export interface ReportSection {
  id: string;
  report_id: string;
  title: string;
  content: string;
  position: number;
  sources: ReportSourceReference[];
  created_at: string;
  updated_at: string;
}

export interface Report extends ReportSummary {
  sections: ReportSection[];
}

export interface ReportListResponse {
  items: ReportSummary[];
  total: number;
  limit: number;
  offset: number;
}

export interface ReportCreateRequest {
  title: string;
  knowledge_base_id?: string | null;
  datasource_id?: string | null;
}

export interface ReportUpdateRequest {
  title?: string;
  knowledge_base_id?: string | null;
  datasource_id?: string | null;
}

export interface ReportSectionCreateRequest {
  title?: string;
  content?: string;
  position?: number;
}

export interface ReportSectionUpdateRequest {
  title?: string;
  content?: string;
  position?: number;
}

export interface ReportRagSourceCreateRequest {
  kind: "rag_evidence";
  title: string;
  knowledge_base_id: string;
  document_id: string;
  evidence_id?: string | null;
  chunk_id?: string | null;
  document_title: string;
  page_start: number;
  page_end: number;
  evidence_type: RAGCitationModality;
  snippet?: string | null;
  section_path?: string[];
  source_block_ids?: string[];
}

export interface ReportChatBISourceCreateRequest {
  kind: "chatbi_result";
  title: string;
  datasource_id: string;
  datasource_name: string;
  question: string;
  answer?: string | null;
  columns: ChatBIColumnMetadata[];
  /** The API validates the bounded JSON snapshot again on write. */
  rows: unknown[][];
  row_count: number;
  truncated: boolean;
  truncation_reason: ChatBITruncationReason | null;
  redacted_sql?: string | null;
  chart_spec?: ReportChartSpec | null;
}

export type ReportSourceCreateRequest =
  | ReportRagSourceCreateRequest
  | ReportChatBISourceCreateRequest;

export interface ReportSourceInsertRequest {
  source: ReportSourceCreateRequest;
  content_append: string;
}

export interface ReportAIContextRequest {
  source_ids?: string[];
  instruction?: string | null;
}

export interface ReportAIOutlineRequest extends ReportAIContextRequest {
  topic?: string | null;
}

export interface ReportAIEditRequest extends ReportAIContextRequest {
  operation: "rewrite" | "expand" | "summarize";
}

export interface ReportAIOutlineItem {
  title: string;
  summary: string;
}

export interface ReportAIOutlineResponse {
  items: ReportAIOutlineItem[];
}

export interface ReportAICitation {
  marker: string;
  title: string;
  page_start: number | null;
  page_end: number | null;
}

export interface ReportAIDraftResponse {
  operation: "generate" | "rewrite" | "expand" | "summarize";
  content: string;
  citations: ReportAICitation[];
}

export type RAGRetrievalMode = "dense" | "unified";
export type RAGCitationModality = "text" | "image" | "table" | "formula";
export type RAGCitationSnippetKind = "source" | "representation";
export type RAGCompletionStatus = "completed" | "insufficient_evidence" | "error";

export interface RAGCitationBranch {
  branch: "dense" | "sparse" | "graph" | "multimodal";
  rank: number;
  score: number;
  graph_seed_entity_id: string | null;
  graph_retrieval_reason: string | null;
}

export interface RAGCitation {
  marker: string;
  document_id: string;
  knowledge_base_id: string | null;
  candidate_kind: "chunk" | "evidence";
  chunk_id: string | null;
  evidence_id: string | null;
  modality: RAGCitationModality | null;
  representation_ids: string[];
  asset_refs: string[];
  page_start: number;
  page_end: number;
  source_block_ids: string[];
  section_path: string[];
  section_title: string | null;
  snippet: string | null;
  snippet_kind: RAGCitationSnippetKind | null;
  final_rank: number | null;
  final_reranker_score: number | null;
  branch_provenance: RAGCitationBranch[];
}

export interface RAGCompleteData {
  status: RAGCompletionStatus;
  prompt_version: string;
  provider: string | null;
  model: string | null;
  retrieval_mode: RAGRetrievalMode;
  retrieval_degraded: boolean | null;
  retrieval_branch_statuses: Record<string, "success" | "empty" | "failed" | "timed_out" | "cancelled"> | null;
  input_tokens: number | null;
  output_tokens: number | null;
  finish_reason: string | null;
  retrieval_latency_ms: number | null;
  llm_latency_ms: number | null;
  latency_ms: number;
}

export interface RAGCitationsData {
  prompt_version: string;
  items: RAGCitation[];
}

export type RAGStreamEvent =
  | { event: "answer_delta"; data: { text: string } }
  | { event: "citations"; data: RAGCitationsData }
  | { event: "complete"; data: RAGCompleteData }
  | { event: "error"; data: { category: string; message: string } };
