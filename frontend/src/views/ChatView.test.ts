import { VueQueryPlugin, QueryClient } from "@tanstack/vue-query";
import { flushPromises, mount } from "@vue/test-utils";
import { createPinia } from "pinia";
import { afterEach, describe, expect, it, vi } from "vitest";

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

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

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

interface StoredMessage {
  id: string;
  role: string;
  status: string;
  content: string;
  citations: unknown[];
  position: number;
  created_at: string;
}

function createChatBackend() {
  const conversations: Record<string, unknown>[] = [];
  const messages = new Map<string, StoredMessage[]>();
  const calls: { url: string; method: string; body: unknown }[] = [];

  function handle(url: string, init?: RequestInit): Response {
    const method = (init?.method ?? "GET").toUpperCase();
    const body = typeof init?.body === "string" ? JSON.parse(init.body) : undefined;
    calls.push({ url, method, body });

    if (url.includes("/v1/knowledge-bases?") && method === "GET") {
      return jsonResponse({
        items: [
          {
            id: "kb-1",
            name: "产品资料",
            description: null,
            created_at: "2026-01-01T00:00:00Z",
            updated_at: "2026-01-01T00:00:00Z",
          },
        ],
        total: 1,
        limit: 100,
        offset: 0,
      });
    }
    if (url.includes("/documents?") && method === "GET") {
      return jsonResponse({
        items: [
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
        ],
        total: 1,
        limit: 100,
        offset: 0,
      });
    }
    if (url.includes("/v1/chat/conversations?") && method === "GET") {
      return jsonResponse({
        items: conversations,
        total: conversations.length,
        limit: 100,
        offset: 0,
      });
    }
    if (url.endsWith("/v1/chat/conversations") && method === "POST") {
      const conversation = {
        id: "conversation-1",
        title: "新对话",
        knowledge_base_id: body?.knowledge_base_id ?? null,
        message_count: 0,
        created_at: "2026-01-01T00:00:00Z",
        updated_at: "2026-01-01T00:00:00Z",
      };
      conversations.unshift(conversation);
      messages.set("conversation-1", []);
      return jsonResponse(conversation, 201);
    }
    if (url.endsWith("/messages") && method === "POST") {
      const conversationId = url.split("/").at(-2) ?? "";
      const stored = messages.get(conversationId) ?? [];
      const message: StoredMessage = {
        id: `message-${stored.length + 1}`,
        role: String(body?.role),
        status: String(body?.status ?? "complete"),
        content: String(body?.content ?? ""),
        citations: body?.citations ?? [],
        position: stored.length,
        created_at: "2026-01-01T00:00:00Z",
      };
      stored.push(message);
      messages.set(conversationId, stored);
      const conversation = conversations.find((item) => item.id === conversationId);
      if (conversation !== undefined) {
        conversation.message_count = stored.length;
        if (message.role === "user" && conversation.title === "新对话") {
          conversation.title = message.content;
        }
      }
      return jsonResponse(message, 201);
    }
    if (url.includes("/v1/chat/conversations/conversation-1") && method === "GET") {
      const stored = messages.get("conversation-1") ?? [];
      return jsonResponse({
        conversation: { ...conversations[0], message_count: stored.length },
        messages: stored,
      });
    }
    throw new Error(`unexpected request: ${method} ${url}`);
  }

  return { handle, calls, messages, conversations };
}

function mountView({ reduceMotion = true }: { reduceMotion?: boolean } = {}) {
  // jsdom has no matchMedia; the stub opts every test into reduced motion so
  // answers render instantly and tests do not depend on playback timing.  Pass
  // reduceMotion: false to exercise the paced playback.
  vi.stubGlobal(
    "matchMedia",
    vi
      .fn()
      .mockReturnValue({ matches: reduceMotion, addEventListener: vi.fn(), removeEventListener: vi.fn() }),
  );
  return mount(ChatView, {
    global: {
      plugins: [
        createPinia(),
        [
          VueQueryPlugin,
          { queryClient: new QueryClient({ defaultOptions: { queries: { retry: false } } }) },
        ],
      ],
    },
  });
}

describe("ChatView", () => {
  afterEach(() => vi.restoreAllMocks());

  it("streams a grounded answer and stores both messages", async () => {
    const backend = createChatBackend();
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        if (url.includes("/rag/query")) {
          backend.calls.push({ url, method: "POST", body: JSON.parse(String(init?.body)) });
          return streamResponse([
            'event: answer_delta\ndata: {"text":"回答 "}\n\n',
            'event: answer_delta\ndata: {"text":"[C1]"}\n\n',
            `event: citations\ndata: ${JSON.stringify({ prompt_version: "rag-qa-v1", items: [citation] })}\n\n`,
            'event: complete\ndata: {"status":"completed","prompt_version":"rag-qa-v1","retrieval_mode":"unified","latency_ms":4}\n\n',
          ]);
        }
        return backend.handle(url, init);
      });

    const wrapper = mountView();
    await flushPromises();
    expect(wrapper.text()).toContain("今天想了解什么？");

    const textarea = wrapper.get("textarea");
    await textarea.setValue("产品有哪些说明？");
    await textarea.trigger("keydown", { key: "Enter", shiftKey: false });
    await vi.waitFor(() => expect(wrapper.text()).toContain("回答 [C1]"));
    await flushPromises();

    const ragCall = backend.calls.find((call) => call.url.includes("/rag/query"));
    expect(ragCall?.body).toMatchObject({
      query: "产品有哪些说明？",
      knowledge_base_id: "kb-1",
      // The default scope is automatic: the backend picks the path.
      retrieval_mode: "auto",
    });

    const assistantCall = backend.calls.find(
      (call) => call.url.endsWith("/messages") && (call.body as { role: string }).role === "assistant",
    );
    expect(assistantCall?.body).toMatchObject({
      role: "assistant",
      content: "回答 [C1]",
      status: "complete",
    });
    expect((assistantCall?.body as { citations: unknown[] }).citations).toHaveLength(1);

    expect(backend.calls.filter((call) => call.url.endsWith("/v1/chat/conversations"))).toHaveLength(1);
    expect(wrapper.text()).toContain("产品有哪些说明？");
    expect(fetchMock).toHaveBeenCalled();
  });

  it("plays cached answers back at a pace instead of one burst", async () => {
    // Motion allowed: the pacer reveals the answer over several ticks.
    vi.stubGlobal("matchMedia", vi.fn().mockReturnValue({ matches: false }));
    const backend = createChatBackend();
    const sentence = "这是缓存命中后回放的一句回答。";
    vi.spyOn(globalThis, "fetch").mockImplementation(
      async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        if (url.includes("/rag/query")) {
          backend.calls.push({ url, method: "POST", body: JSON.parse(String(init?.body)) });
          return streamResponse([
            `event: answer_delta\ndata: {"text":"${sentence.repeat(4)}"}\n\n`,
            'event: complete\ndata: {"status":"completed","prompt_version":"rag-qa-v2","retrieval_mode":"dense","latency_ms":4}\n\n',
          ]);
        }
        return backend.handle(url, init);
      },
    );

    const wrapper = mountView({ reduceMotion: false });
    await flushPromises();
    const textarea = wrapper.get("textarea");
    await textarea.setValue("产品有哪些说明？");
    await textarea.trigger("keydown", { key: "Enter", shiftKey: false });

    const full = sentence.repeat(4);
    let sawPartialPlayback = false;
    const deadline = Date.now() + 2_000;
    while (Date.now() < deadline) {
      const text = wrapper.text();
      if (text.includes(full)) {
        break;
      }
      // Mid-playback the first sentence is visible but the rest is not.
      sawPartialPlayback ||= text.includes(sentence) && !text.includes(sentence.repeat(2));
      await new Promise((resolve) => window.setTimeout(resolve, 2));
    }
    await vi.waitFor(() => expect(wrapper.text()).toContain(full));

    expect(sawPartialPlayback).toBe(true);
  });

  it("replays the recent turns so a follow-up keeps its context", async () => {
    const backend = createChatBackend();
    vi.spyOn(globalThis, "fetch").mockImplementation(
      async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        if (url.includes("/rag/query")) {
          backend.calls.push({ url, method: "POST", body: JSON.parse(String(init?.body)) });
          return streamResponse([
            'event: answer_delta\ndata: {"text":"回答"}\n\n',
            'event: complete\ndata: {"status":"completed","prompt_version":"rag-qa-v2","retrieval_mode":"unified","latency_ms":4}\n\n',
          ]);
        }
        return backend.handle(url, init);
      },
    );

    const wrapper = mountView();
    await flushPromises();

    for (const question of ["第一个问题", "再展开讲讲"]) {
      const textarea = wrapper.get("textarea");
      await textarea.setValue(question);
      await textarea.trigger("keydown", { key: "Enter", shiftKey: false });
      // Reduced motion renders answers instantly; wait until the turn is fully
      // stored before sending the next one.
      await vi.waitFor(() => expect(wrapper.text()).toContain("回答"));
      await flushPromises();
      await flushPromises();
    }

    const ragCalls = backend.calls.filter((call) => call.url.includes("/rag/query"));
    expect(ragCalls).toHaveLength(2);
    expect((ragCalls[0]?.body as { history: unknown[] }).history).toEqual([]);
    expect((ragCalls[1]?.body as { history: { role: string; content: string }[] }).history).toEqual([
      { role: "user", content: "第一个问题" },
      { role: "assistant", content: "回答" },
    ]);
  });

  it("shows the rewritten follow-up query in the sources drawer", async () => {
    const backend = createChatBackend();
    vi.spyOn(globalThis, "fetch").mockImplementation(
      async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        if (url.includes("/rag/query")) {
          backend.calls.push({ url, method: "POST", body: JSON.parse(String(init?.body)) });
          return streamResponse([
            'event: answer_delta\ndata: {"text":"回答 [C1]"}\n\n',
            `event: citations\ndata: ${JSON.stringify({ prompt_version: "rag-qa-v2", items: [citation] })}\n\n`,
            'event: complete\ndata: {"status":"completed","prompt_version":"rag-qa-v2","retrieval_mode":"unified","rewritten_query":"二次沉淀池有哪些限制","retrieval_cached":true,"retrieval_escalated":true,"latency_ms":4}\n\n',
          ]);
        }
        return backend.handle(url, init);
      },
    );

    const wrapper = mountView();
    await flushPromises();
    const textarea = wrapper.get("textarea");
    await textarea.setValue("它有哪些限制？");
    await textarea.trigger("keydown", { key: "Enter", shiftKey: false });
    await vi.waitFor(() => expect(wrapper.text()).toContain("回答 [C1]"));
    await flushPromises();

    await wrapper
      .findAll(".message-action")
      .find((action) => action.text().includes("来源"))
      ?.trigger("click");
    await flushPromises();

    expect(wrapper.text()).toContain("追问已按「二次沉淀池有哪些限制」检索");
    expect(wrapper.text()).toContain("已自动扩展为多路检索");
  });

  it("sends the retrieval scope the user selected", async () => {
    const backend = createChatBackend();
    vi.spyOn(globalThis, "fetch").mockImplementation(
      async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        if (url.includes("/rag/query")) {
          backend.calls.push({ url, method: "POST", body: JSON.parse(String(init?.body)) });
          return streamResponse([
            'event: answer_delta\ndata: {"text":"回答"}\n\n',
            'event: complete\ndata: {"status":"completed","prompt_version":"rag-qa-v2","retrieval_mode":"dense","latency_ms":4}\n\n',
          ]);
        }
        return backend.handle(url, init);
      },
    );

    const wrapper = mountView();
    await flushPromises();

    await wrapper.findAll(".composer__mode")[2].trigger("click");
    const textarea = wrapper.get("textarea");
    await textarea.setValue("这份资料讲了什么？");
    await textarea.trigger("keydown", { key: "Enter", shiftKey: false });
    await vi.waitFor(() => expect(wrapper.text()).toContain("回答"));
    await flushPromises();

    const ragCall = backend.calls.find((call) => call.url.includes("/rag/query"));
    expect((ragCall?.body as { retrieval_mode: string }).retrieval_mode).toBe("dense");
  });

  it("keeps a failed answer visible and stores it as an error", async () => {
    const backend = createChatBackend();
    vi.spyOn(globalThis, "fetch").mockImplementation(
      async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        if (url.includes("/rag/query")) {
          backend.calls.push({ url, method: "POST", body: JSON.parse(String(init?.body)) });
          return streamResponse([
            'event: error\ndata: {"category":"retrieval","message":"failed"}\n\n',
          ]);
        }
        return backend.handle(url, init);
      },
    );

    const wrapper = mountView();
    await flushPromises();
    const textarea = wrapper.get("textarea");
    await textarea.setValue("读取失败的问题");
    await textarea.trigger("keydown", { key: "Enter", shiftKey: false });
    await vi.waitFor(() => expect(wrapper.text()).toContain("暂时无法读取当前知识库"));
    await flushPromises();

    const assistantCall = backend.calls.find(
      (call) => call.url.endsWith("/messages") && (call.body as { role: string }).role === "assistant",
    );
    expect(assistantCall?.body).toMatchObject({ status: "error" });
  });

  it("asks for a knowledge base before accepting a question", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/v1/knowledge-bases?")) {
        return jsonResponse({ items: [], total: 0, limit: 100, offset: 0 });
      }
      if (url.includes("/v1/chat/conversations?")) {
        return jsonResponse({ items: [], total: 0, limit: 100, offset: 0 });
      }
      throw new Error(`unexpected request: ${url}`);
    });

    const wrapper = mountView();
    await flushPromises();

    // Without a knowledge base the composer and the prompts stay disabled.
    expect(wrapper.get(".prompt-card").attributes("disabled")).toBeDefined();
    expect(wrapper.get(".composer__button").attributes("disabled")).toBeDefined();
  });
});
