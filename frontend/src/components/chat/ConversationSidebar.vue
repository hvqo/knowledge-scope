<script setup lang="ts">
import type { KnowledgeBase } from "../../api/types";
import type { ChatConversation } from "../../stores/chat";

defineProps<{
  conversations: ChatConversation[];
  activeConversationId: string | null;
  knowledgeBases: KnowledgeBase[];
  selectedKnowledgeBaseId: string | null;
  knowledgeBasesLoading: boolean;
  knowledgeBasesError: boolean;
  knowledgeBasesErrorMessage: string;
}>();

const emit = defineEmits<{
  newConversation: [];
  selectConversation: [id: string];
  selectKnowledgeBase: [id: string | null];
  retryKnowledgeBases: [];
}>();

function handleKnowledgeBaseChange(event: Event): void {
  const value = (event.currentTarget as HTMLSelectElement).value;
  emit("selectKnowledgeBase", value || null);
}
</script>

<template>
  <aside class="conversation-sidebar">
    <div class="conversation-sidebar__header">
      <div>
        <span class="eyebrow">工作区</span>
        <h2>对话</h2>
      </div>
      <button
        class="icon-action"
        type="button"
        aria-label="新建对话"
        title="新建对话"
        @click="emit('newConversation')"
      >
        <span aria-hidden="true">＋</span>
      </button>
    </div>

    <button
      class="new-conversation-button"
      type="button"
      @click="emit('newConversation')"
    >
      <span aria-hidden="true">✦</span>
      <span>新建对话</span>
    </button>

    <div class="knowledge-base-picker">
      <label for="chat-knowledge-base">当前知识库</label>
      <div
        v-if="knowledgeBasesLoading"
        class="picker-state"
        role="status"
      >
        正在加载知识库…
      </div>
      <div
        v-else-if="knowledgeBasesError"
        class="picker-state picker-state--error"
        role="alert"
      >
        <span>{{ knowledgeBasesErrorMessage }}</span>
        <button
          type="button"
          class="text-action"
          @click="emit('retryKnowledgeBases')"
        >
          重试
        </button>
      </div>
      <div
        v-else-if="knowledgeBases.length === 0"
        class="picker-state"
      >
        暂无知识库，请先建立一个。
      </div>
      <select
        v-else
        id="chat-knowledge-base"
        class="knowledge-base-select"
        :value="selectedKnowledgeBaseId ?? ''"
        aria-label="选择当前知识库"
        @change="handleKnowledgeBaseChange"
      >
        <option value="">
          请选择知识库
        </option>
        <option
          v-for="knowledgeBase in knowledgeBases"
          :key="knowledgeBase.id"
          :value="knowledgeBase.id"
        >
          {{ knowledgeBase.name }}
        </option>
      </select>
    </div>

    <div
      class="conversation-list"
      aria-label="历史对话"
    >
      <div
        v-if="conversations.length === 0"
        class="conversation-empty"
      >
        <span
          class="conversation-empty__mark"
          aria-hidden="true"
        >◌</span>
        <p>还没有对话</p>
        <span>从一个问题开始探索你的资料。</span>
      </div>
      <template v-else>
        <button
          v-for="conversation in conversations"
          :key="conversation.id"
          class="conversation-item"
          :class="{ 'is-active': conversation.id === activeConversationId }"
          type="button"
          @click="emit('selectConversation', conversation.id)"
        >
          <span
            class="conversation-item__dot"
            aria-hidden="true"
          />
          <span class="conversation-item__body">
            <strong>{{ conversation.title }}</strong>
            <small>{{ conversation.messages.length }} 条消息</small>
          </span>
        </button>
      </template>
    </div>
  </aside>
</template>

<style scoped>
.conversation-sidebar {
  display: flex;
  min-width: 232px;
  width: 232px;
  flex-direction: column;
  gap: 18px;
  padding: 22px 16px 18px;
  background: var(--ks-surface-subtle);
  border-right: 1px solid var(--ks-border);
}

.conversation-sidebar__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.eyebrow {
  color: var(--ks-accent);
  font-size: 11px;
  font-weight: 720;
  letter-spacing: 0.08em;
}

h2 {
  margin: 4px 0 0;
  color: var(--ks-ink);
  font-size: 20px;
  letter-spacing: -0.04em;
}

.icon-action,
.new-conversation-button,
.conversation-item,
.text-action {
  border: 0;
  cursor: pointer;
}

.icon-action {
  display: grid;
  width: 32px;
  height: 32px;
  place-items: center;
  color: var(--ks-accent-strong);
  font-size: 20px;
  background: var(--ks-surface);
  border: 1px solid var(--ks-border);
  border-radius: 9px;
}

.icon-action:hover {
  background: var(--ks-accent-soft);
}

.new-conversation-button {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 38px;
  gap: 8px;
  color: var(--ks-surface);
  font-size: 13px;
  font-weight: 680;
  background: var(--ks-accent);
  border-radius: var(--ks-radius-sm);
  box-shadow: 0 3px 8px rgb(55 116 102 / 14%);
}

.new-conversation-button:hover {
  background: var(--ks-accent-strong);
}

.knowledge-base-picker {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.knowledge-base-picker label {
  color: var(--ks-muted);
  font-size: 11px;
  font-weight: 680;
}

.knowledge-base-select {
  width: 100%;
  min-height: 38px;
  padding: 0 10px;
  color: var(--ks-ink);
  font-size: 13px;
  background: var(--ks-surface);
  border: 1px solid var(--ks-border);
  border-radius: var(--ks-radius-sm);
}

.picker-state {
  display: flex;
  min-height: 38px;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 8px 10px;
  color: var(--ks-muted);
  font-size: 12px;
  line-height: 1.5;
  background: var(--ks-surface);
  border: 1px dashed var(--ks-border-strong);
  border-radius: var(--ks-radius-sm);
}

.picker-state--error {
  align-items: flex-start;
  color: var(--ks-danger);
}

.text-action {
  flex: 0 0 auto;
  padding: 0;
  color: var(--ks-accent-strong);
  font-size: 12px;
  font-weight: 680;
  background: transparent;
}

.conversation-list {
  display: flex;
  min-height: 0;
  flex: 1;
  flex-direction: column;
  gap: 4px;
  overflow-y: auto;
}

.conversation-empty {
  display: flex;
  align-items: center;
  padding: 24px 8px;
  color: var(--ks-muted);
  font-size: 12px;
  line-height: 1.6;
  text-align: center;
  flex-direction: column;
}

.conversation-empty p {
  margin: 8px 0 2px;
  color: var(--ks-text);
  font-weight: 650;
}

.conversation-empty__mark {
  display: grid;
  width: 38px;
  height: 38px;
  place-items: center;
  color: var(--ks-accent);
  font-size: 24px;
  background: var(--ks-accent-soft);
  border-radius: 50%;
}

.conversation-item {
  display: flex;
  min-height: 54px;
  align-items: center;
  gap: 10px;
  padding: 9px 10px;
  color: var(--ks-muted);
  text-align: left;
  background: transparent;
  border-radius: var(--ks-radius-sm);
}

.conversation-item:hover,
.conversation-item.is-active {
  color: var(--ks-ink);
  background: var(--ks-surface);
}

.conversation-item.is-active {
  box-shadow: 0 0 0 1px var(--ks-border) inset;
}

.conversation-item__dot {
  width: 7px;
  height: 7px;
  flex: 0 0 7px;
  background: var(--ks-border-strong);
  border-radius: 50%;
}

.is-active .conversation-item__dot {
  background: var(--ks-accent);
}

.conversation-item__body {
  display: flex;
  min-width: 0;
  flex-direction: column;
  gap: 4px;
}

.conversation-item strong {
  overflow: hidden;
  color: currentColor;
  font-size: 13px;
  font-weight: 630;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.conversation-item small {
  color: var(--ks-faint);
  font-size: 10px;
}

@media (max-width: 1040px) {
  .conversation-sidebar {
    min-width: 206px;
    width: 206px;
  }
}
</style>
