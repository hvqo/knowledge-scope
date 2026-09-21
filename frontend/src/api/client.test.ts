import { afterEach, describe, expect, it, vi } from "vitest";

import { streamRagQuery } from "./client";

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
