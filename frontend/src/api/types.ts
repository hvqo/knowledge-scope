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
  status: "uploaded";
  created_at: string;
  updated_at: string;
}

export interface DocumentListResponse {
  items: Document[];
  total: number;
  limit: number;
  offset: number;
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
