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

export interface DocumentChunk {
  chunk_id: string;
  ordinal: number;
  page_start: number;
  page_end: number;
  section_path: string[];
  content_types: string[];
  asset_refs: string[];
  text: string;
}

export interface DocumentChunkListResponse {
  document_id: string;
  page_count: number | null;
  items: DocumentChunk[];
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

export interface ReportAgentAnswerInsertRequest {
  chat_message_id: string;
}

export interface ReportAgentAnswerInsertResponse {
  section: ReportSection;
  sources: ReportSourceReference[];
  warnings: string[];
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
export type RAGRequestedRetrievalMode = "auto" | RAGRetrievalMode;
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

export type EvidenceRepresentationType =
  | "text"
  | "caption"
  | "context"
  | "markdown"
  | "html"
  | "latex"
  | "asset_ref"
  | "visual_embedding_ref";

export interface EvidenceRepresentationRecord {
  representation_id: string;
  representation_type: EvidenceRepresentationType;
  content: string | null;
  searchable: boolean;
}

export interface EvidenceAssetRecord {
  index: number;
  media_type: "image/jpeg" | "image/png" | "image/webp" | null;
  size_bytes: number | null;
  available: boolean;
}

export interface EvidenceDetail {
  document_id: string;
  knowledge_base_id: string;
  evidence_id: string;
  modality: RAGCitationModality;
  page_start: number;
  page_end: number;
  source_block_ids: string[];
  section_path: string[];
  source_fingerprint: string;
  representations: EvidenceRepresentationRecord[];
  assets: EvidenceAssetRecord[];
}

export type VisionTaskType = "describe_image" | "analyze_chart" | "extract_table";
export type VisionStatus = "ok" | "needs_review" | "not_configured" | "failed";

export interface VisionSource {
  knowledge_base_id: string;
  document_id: string;
  evidence_id: string;
  page_start: number;
  page_end: number;
  source_block_ids: string[];
}

export interface ChartAnalysis {
  chart_type: string | null;
  axes: string[];
  series: string[];
  trends: string[];
  peaks: string[];
  anomalies: string[];
  limitations: string[];
}

export interface ExtractedVisionTable {
  title: string | null;
  unit: string | null;
  columns: string[];
  rows: (string | null)[][];
  merged_cell_info: string[];
  warnings: string[];
  extraction_method: "vlm";
}

export interface VisionAnalysisResult {
  status: VisionStatus;
  task_type: VisionTaskType;
  answer: string | null;
  chart_analysis: ChartAnalysis | null;
  extracted_table: ExtractedVisionTable | null;
  extraction_method: "vlm" | null;
  provider: string | null;
  model: string | null;
  latency_ms: number | null;
  warnings: string[];
  sources: VisionSource[];
  error_code: string | null;
  message: string | null;
}

export interface VisionAnalyzeRequest {
  knowledge_base_id: string;
  document_id: string;
  evidence_id: string;
  asset_index: number;
  task_type: VisionTaskType;
  question?: string | null;
}

export type RAGAnswerMode = "standard" | "agent";
export type RAGToolName = "retrieval" | "analyze_image" | "analyze_table";
export type RAGToolStatusValue = "running" | "completed" | "failed" | "skipped";
export type RAGFactStatus = "source_verified" | "visual_estimate" | "unverified" | "not_available";
export type RAGSafetyGate =
  | "none"
  | "visual_numeric_withheld"
  | "visual_unverified"
  | "vision_unavailable";

export interface RAGVisualSource {
  knowledge_base_id: string;
  document_id: string;
  evidence_id: string;
  page_start: number;
  page_end: number;
  source_block_ids: string[];
  asset_index: number;
  label: string | null;
  fact_status: RAGFactStatus;
  execution_status?: "completed" | "failed";
  vision_task?: string | null;
}

export interface RAGToolStatusData {
  tool: RAGToolName;
  status: RAGToolStatusValue;
  summary: string;
  step: number;
}

export interface RAGCompleteData {
  status: RAGCompletionStatus;
  prompt_version: string;
  provider: string | null;
  model: string | null;
  retrieval_mode: RAGRetrievalMode;
  answer_mode?: RAGAnswerMode;
  tool_calls?: number;
  fact_status?: RAGFactStatus | null;
  safety_gate?: RAGSafetyGate;
  safety_notice?: string | null;
  visual_sources?: RAGVisualSource[];
  answer_generated_by?: "agent" | "rag" | "safety_gate" | null;
  trust_id?: string | null;
  rewritten_query: string | null;
  retrieval_cached: boolean | null;
  retrieval_escalated: boolean | null;
  rewrite_latency_ms: number | null;
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
  | { event: "tool_status"; data: RAGToolStatusData }
  | { event: "complete"; data: RAGCompleteData }
  | { event: "error"; data: { category: string; message: string } };

export type ChatMessageRole = "user" | "assistant";
export type ChatMessageStatus = "complete" | "error";

export interface ChatConversation {
  id: string;
  title: string;
  knowledge_base_id: string | null;
  project_id: string | null;
  message_count: number;
  created_at: string;
  updated_at: string;
}

export interface ChatProject {
  id: string;
  title: string;
  description: string | null;
  conversation_count: number;
  created_at: string;
  updated_at: string;
}

export interface ChatMemory {
  id: string;
  knowledge_base_id: string;
  content: string;
  source_conversation_id: string | null;
  created_at: string;
}

export interface ChatConversationListResponse {
  items: ChatConversation[];
  total: number;
  limit: number;
  offset: number;
}

export interface ChatMessageRecord {
  id: string;
  role: ChatMessageRole;
  status: ChatMessageStatus;
  content: string;
  citations: RAGCitation[];
  position: number;
  created_at: string;
}

export interface ChatConversationDetail {
  conversation: ChatConversation;
  messages: ChatMessageRecord[];
}

/** Persisted statuses plus the client-side state of a streaming answer. */
export type ChatMessageViewStatus = ChatMessageStatus | "streaming";

export interface ChatMessageView extends Omit<ChatMessageRecord, "status"> {
  status: ChatMessageViewStatus;
  safetyNotice?: string | null;
  safetyGate?: RAGSafetyGate;
  factStatus?: RAGFactStatus | null;
  visualSources?: RAGVisualSource[];
}

export type GraphNeighborDirection = "forward" | "reverse" | "canonical_bridge";

export interface GraphNode {
  entity_id: string;
  canonical_name: string;
  entity_type: string;
  aliases: string[];
  document_id: string;
  canonical_entity_ids: string[];
  evidence_count: number;
  degree: number;
}

export type GraphEdgeKind = "relation" | "canonical_bridge";

export interface GraphEdge {
  relation_id: string;
  source_entity_id: string;
  target_entity_id: string;
  relation_type: string;
  kind: GraphEdgeKind;
  document_id: string | null;
}

export interface GraphEntityTypeOption {
  entity_type: string;
  entity_count: number;
}

export interface GraphOverview {
  knowledge_base_id: string;
  entity_count: number;
  relation_count: number;
  entity_types: GraphEntityTypeOption[];
  truncated: boolean;
  nodes: GraphNode[];
  edges: GraphEdge[];
}

export interface GraphEvidenceItem {
  chunk_id: string;
  document_id: string;
  page_start: number;
  page_end: number;
  section_path: string[];
}

export interface GraphCanonicalEntity {
  canonical_entity_id: string;
  canonical_name: string;
  entity_type: string;
}

export interface GraphNeighborItem {
  entity_id: string;
  canonical_name: string;
  entity_type: string;
  document_id: string;
  direction: GraphNeighborDirection;
  relation_id: string | null;
  relation_type: string | null;
  page_start: number | null;
  page_end: number | null;
  shared_canonical_name: string | null;
}

export interface GraphEntityDetail {
  entity: GraphNode;
  canonical_entities: GraphCanonicalEntity[];
  evidence: GraphEvidenceItem[];
  neighbors: GraphNeighborItem[];
}
