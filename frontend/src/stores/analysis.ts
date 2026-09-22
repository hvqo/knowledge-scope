import { computed, ref, type Ref } from "vue";
import { defineStore } from "pinia";

import type { ChatBIAnalysisResult } from "../api/types";

export type AnalysisSessionStatus = "idle" | "loading" | "complete" | "error";

export interface AnalysisSession {
  id: string;
  datasourceId: string | null;
  question: string;
  status: AnalysisSessionStatus;
  result: ChatBIAnalysisResult | null;
  errorMessage: string | null;
  createdAt: number;
  updatedAt: number;
}

function createId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `analysis-${crypto.randomUUID()}`;
  }
  return `analysis-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function createSession(datasourceId: string | null = null): AnalysisSession {
  const now = Date.now();
  return {
    id: createId(),
    datasourceId,
    question: "",
    status: "idle",
    result: null,
    errorMessage: null,
    createdAt: now,
    updatedAt: now,
  };
}

export const useAnalysisStore = defineStore("analysis", () => {
  const sessions = ref([]) as Ref<AnalysisSession[]>;
  const activeSessionId = ref<string | null>(null);
  const selectedDatasourceId = ref<string | null>(null);

  const activeSession = computed(
    () => sessions.value.find((session) => session.id === activeSessionId.value) ?? null,
  );

  function newAnalysis(datasourceId = selectedDatasourceId.value): string {
    const session = createSession(datasourceId);
    sessions.value.unshift(session);
    activeSessionId.value = session.id;
    return session.id;
  }

  function ensureActiveSession(): AnalysisSession {
    if (activeSession.value !== null) {
      return activeSession.value;
    }
    const id = newAnalysis();
    const session = sessions.value.find((item) => item.id === id);
    if (session === undefined) {
      throw new Error("analysis session could not be created");
    }
    return session;
  }

  function selectSession(id: string): void {
    const session = sessions.value.find((item) => item.id === id);
    if (session === undefined) {
      return;
    }
    activeSessionId.value = id;
    if (session.datasourceId !== null) {
      selectedDatasourceId.value = session.datasourceId;
    }
  }

  function setDatasource(id: string | null): void {
    selectedDatasourceId.value = id;
    const session = activeSession.value;
    if (session !== null && session.status !== "loading" && !session.question) {
      session.datasourceId = id;
      session.updatedAt = Date.now();
    }
  }

  function updateSession(id: string, update: (session: AnalysisSession) => void): void {
    const session = sessions.value.find((item) => item.id === id);
    if (session === undefined) {
      return;
    }
    update(session);
    session.updatedAt = Date.now();
  }

  return {
    sessions,
    activeSessionId,
    activeSession,
    selectedDatasourceId,
    newAnalysis,
    ensureActiveSession,
    selectSession,
    setDatasource,
    updateSession,
  };
});
