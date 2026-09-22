import { afterEach, describe, expect, it, vi } from "vitest";

import {
  ApiError,
  askChatBI,
  getUserFacingError,
  streamRagQuery,
  uploadDocument,
} from "./client";

function sseResponse(frames: string[]): Response {
  const encoder = new TextEncoder();
  let index = 0;
  const stream = new ReadableStream<Uint8Array>({
    pull(controller) {
      if (index >= frames.length) {
        controller.close();
        return;
      }
      controller.enqueue(encoder.encode(frames[index]));
      index += 1;
    },
  });
  return new Response(stream, {
    status: 200,
    headers: { "Content-Type": "text/event-stream" },
  });
}

const citation = {
  marker: "C1",
  document_id: "11111111-1111-4111-8111-111111111111",
  knowledge_base_id: "22222222-2222-4222-8222-222222222222",
  candidate_kind: "chunk",
  chunk_id: "chunk-1",
  evidence_id: null,
  modality: null,
  representation_ids: [],
  asset_refs: [],
  page_start: 1,
  page_end: 1,
  source_block_ids: ["block-1"],
  section_path: ["章节"],
  section_title: "章节",
  snippet: "原始片段",
  snippet_kind: "source",
  final_rank: null,
  final_reranker_score: null,
  branch_provenance: [],
};

describe("streamRagQuery", () => {
  afterEach(() => vi.restoreAllMocks());

  it("parses split SSE frames and preserves citation lineage", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        sseResponse([
          'event: answer_delta\ndata: {"text":"回答"}\n\n',
          `event: citations\ndata: ${JSON.stringify({ prompt_version: "rag-qa-v1", items: [citation] })}\n\n`,
          'event: complete\ndata: {"status":"completed","prompt_version":"rag-qa-v1","retrieval_mode":"dense","latency_ms":12}\n\n',
        ]),
      );

    const events = [];
    for await (const event of streamRagQuery({
      query: "问题",
      knowledge_base_id: "22222222-2222-4222-8222-222222222222",
    })) {
      events.push(event);
    }

    expect(fetchMock).toHaveBeenCalledOnce();
    expect(events.map((event) => event.event)).toEqual([
      "answer_delta",
      "citations",
      "complete",
    ]);
    const citationsEvent = events[1];
    expect(citationsEvent.event).toBe("citations");
    if (citationsEvent.event === "citations") {
      expect(citationsEvent.data.items[0].snippet).toBe("原始片段");
      expect(citationsEvent.data.items[0].chunk_id).toBe("chunk-1");
    }
  });

  it("forwards cancellation and rejects malformed stream payloads", async () => {
    const controller = new AbortController();
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(sseResponse(['event: answer_delta\ndata: {"text":}\n\n']));

    await expect(
      (async () => {
        for await (const event of streamRagQuery(
          { query: "问题", knowledge_base_id: "22222222-2222-4222-8222-222222222222" },
          controller.signal,
        )) {
          // The malformed payload is expected to stop this iterator.
          void event;
        }
      })(),
    ).rejects.toThrow("invalid JSON");
    expect(fetchMock.mock.calls[0][1]?.signal).toBe(controller.signal);
  });
});

describe("askChatBI", () => {
  afterEach(() => vi.restoreAllMocks());

  it("maps a disabled datasource 409 without treating it as duplicate content", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(JSON.stringify({
        detail: {
          category: "datasource_disabled",
          message: "当前数据源已停用",
        },
      }), { status: 409, statusText: "Conflict", headers: { "Content-Type": "application/json" } }),
    );

    let caught: unknown;
    try {
      await askChatBI("datasource-1", "统计数量");
    } catch (error) {
      caught = error;
    }

    expect(caught).toBeInstanceOf(ApiError);
    expect(caught).toMatchObject({
      status: 409,
      category: "datasource_disabled",
      detail: "当前数据源已停用",
    });
    expect(getUserFacingError(caught, "分析暂时无法完成，请稍后重试。")).toBe(
      "当前数据源不可用或已停用。",
    );
    expect(getUserFacingError(caught, "分析暂时无法完成，请稍后重试。")).not.toContain(
      "已有相同内容",
    );
  });

  it("keeps duplicate-content wording opt-in for the document upload flow", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(JSON.stringify({ detail: "同一知识库中已存在相同文件" }), {
        status: 409,
        statusText: "Conflict",
        headers: { "Content-Type": "application/json" },
      }),
    );

    let caught: unknown;
    try {
      await uploadDocument(
        "knowledge-base-1",
        new File(["pdf"], "report.pdf", { type: "application/pdf" }),
      );
    } catch (error) {
      caught = error;
    }

    expect(caught).toBeInstanceOf(ApiError);
    expect(getUserFacingError(caught, "文档上传失败，请稍后重试。")).toBe(
      "文档上传失败，请稍后重试。",
    );
    expect(
      getUserFacingError(caught, "文档上传失败，请稍后重试。", {
        conflictMessage: "已有相同内容，请检查后重试。",
      }),
    ).toBe("已有相同内容，请检查后重试。");
  });

  it("keeps other status mappings and does not expose raw backend errors", () => {
    const error = new ApiError(
      500,
      "Internal Server Error",
      "Traceback: password=super-secret",
    );

    expect(getUserFacingError(error, "请求失败，请稍后重试。")).toBe(
      "服务暂时不可用，请稍后重试。",
    );
    expect(getUserFacingError(new Error("internal stack"), "请求失败，请稍后重试。")).toBe(
      "请求失败，请稍后重试。",
    );
  });

  it("sends only the question and parses bounded tabular output", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(JSON.stringify({
        query_id: "query-1",
        datasource_id: "datasource-1",
        execution_status: "succeeded",
        eligibility: {
          decision: "eligible",
          reason_code: "eligible_analytical",
          user_message: null,
          clarification_question: null,
        },
        answer: "完成",
        columns: [{ name: "count", data_type: "bigint", nullable: false, ordinal: 0 }],
        rows: [[3]],
        row_count: 1,
        truncated: false,
        truncation_reason: null,
        redacted_sql: "SELECT COUNT(*) FROM public.sales",
        warnings: [],
        error_message: null,
        elapsed_ms: 4,
      }), { status: 200, headers: { "Content-Type": "application/json" } }),
    );

    const result = await askChatBI("datasource-1", "统计数量");

    expect(result.rows).toEqual([[3]]);
    expect(result.columns[0].name).toBe("count");
    expect(fetchMock.mock.calls[0][0]).toContain("/chatbi/data-sources/datasource-1/ask");
    expect(JSON.parse(String(fetchMock.mock.calls[0][1]?.body))).toEqual({ question: "统计数量" });
  });

  it("rejects a response whose rows do not match its columns", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValueOnce(
      new Response(JSON.stringify({
        query_id: "query-1",
        datasource_id: "datasource-1",
        execution_status: "succeeded",
        eligibility: null,
        answer: "完成",
        columns: [{ name: "count", data_type: "bigint", nullable: false, ordinal: 0 }],
        rows: [[1, 2]],
        row_count: 1,
        truncated: false,
        truncation_reason: null,
        redacted_sql: null,
        warnings: [],
        error_message: null,
        elapsed_ms: 1,
      }), { status: 200, headers: { "Content-Type": "application/json" } }),
    );

    await expect(askChatBI("datasource-1", "问题")).rejects.toThrow(
      "rows do not match the column contract",
    );
  });
});
