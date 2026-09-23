import { afterEach, describe, expect, it, vi } from "vitest";
import { VueQueryPlugin, QueryClient } from "@tanstack/vue-query";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia } from "pinia";

import ChatView from "./ChatView.vue";

function streamResponse(frames: string[]): Response {
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

function jsonResponse(payload: unknown): Response {
  return new Response(JSON.stringify(payload), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

describe("ChatView", () => {
  afterEach(() => vi.restoreAllMocks());

  it("uses the selected knowledge base and completes a citation-backed stream", async () => {
    const citation = {
      marker: "C1",
      document_id: "doc-1",
      knowledge_base_id: "kb-1",
      candidate_kind: "chunk",
      chunk_id: "chunk-1",
      evidence_id: null,
      modality: null,
      representation_ids: [],
      asset_refs: [],
      page_start: 1,
      page_end: 1,
      source_block_ids: ["block-1"],
      section_path: ["产品说明"],
      section_title: "产品说明",
      snippet: "这是来源片段",
      snippet_kind: "source",
      final_rank: null,
      final_reranker_score: null,
      branch_provenance: [],
    };
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.includes("/knowledge-bases?")) {
        return jsonResponse({ items: [
          {
            id: "kb-1",
            name: "产品资料",
            description: null,
            created_at: "2026-01-01T00:00:00Z",
            updated_at: "2026-01-01T00:00:00Z",
          },
        ], total: 1, limit: 100, offset: 0 });
      }
      if (url.includes("/documents?")) {
        return jsonResponse({ items: [
          {
            id: "doc-1",
            knowledge_base_id: "kb-1",
            original_filename: "产品资料.pdf",
            media_type: "application/pdf",
            size_bytes: 10,
            sha256: "a".repeat(64),
            status: "uploaded",
            created_at: "2026-01-01T00:00:00Z",
            updated_at: "2026-01-01T00:00:00Z",
          },
        ], total: 1, limit: 100, offset: 0 });
      }
      expect(init?.method).toBe("POST");
      return streamResponse([
        'event: answer_delta\ndata: {"text":"回答 "}\n\n',
        'event: answer_delta\ndata: {"text":"[C1]"}\n\n',
        `event: citations\ndata: ${JSON.stringify({ prompt_version: "rag-qa-v1", items: [citation] })}\n\n`,
        'event: complete\ndata: {"status":"completed","prompt_version":"rag-qa-v1","retrieval_mode":"unified","latency_ms":4}\n\n',
      ]);
    });

    const wrapper = mount(ChatView, {
      global: {
        plugins: [
          createPinia(),
          [VueQueryPlugin, { queryClient: new QueryClient({ defaultOptions: { queries: { retry: false } } }) }],
        ],
      },
    });
    await vi.waitFor(() => expect(wrapper.find("select").element.value).toBe("kb-1"));

    const textarea = wrapper.get("textarea");
    await textarea.setValue("产品有哪些说明？");
    await textarea.trigger("keydown", { key: "Enter", shiftKey: false });
    await flushPromises();
    await vi.waitFor(() => expect(wrapper.text()).toContain("回答 [C1]"));

    expect(wrapper.text()).toContain("产品资料.pdf");
    expect(wrapper.text()).toContain("这是来源片段");
    expect(wrapper.text()).toContain("1 个来源块");
    const postCall = fetchMock.mock.calls.find(([input, init]) =>
      String(input).includes("/rag/query") && init?.method === "POST",
    );
    expect(postCall).toBeDefined();
    expect(JSON.parse(String(postCall?.[1]?.body))).toMatchObject({
      query: "产品有哪些说明？",
      knowledge_base_id: "kb-1",
      retrieval_mode: "unified",
    });
  });
});
