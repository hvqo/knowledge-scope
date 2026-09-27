<script setup lang="ts">
import { ElMessageBox } from "element-plus";
import { computed, ref } from "vue";

import type { ChatConversation, ChatProject, KnowledgeBase } from "../../api/types";

const props = defineProps<{
  conversations: ChatConversation[];
  projects: ChatProject[];
  activeConversationId: string | null;
  conversationsLoading: boolean;
  conversationsError: boolean;
  conversationsErrorMessage: string;
  knowledgeBases: KnowledgeBase[];
  selectedKnowledgeBaseId: string | null;
  knowledgeBasesLoading: boolean;
  knowledgeBasesError: boolean;
  knowledgeBasesErrorMessage: string;
  memoryCount: number;
  isBusy: boolean;
  isOpen: boolean;
}>();

const emit = defineEmits<{
  newConversation: [];
  selectConversation: [id: string];
  selectKnowledgeBase: [id: string | null];
  renameConversation: [id: string, title: string];
  deleteConversation: [id: string];
  moveConversation: [id: string, projectId: string | null];
  createProject: [title: string];
  deleteProject: [id: string];
  renameProject: [id: string, title: string];
  openMemories: [];
  retryConversations: [];
  retryKnowledgeBases: [];
  close: [];
}>();

const renamingId = ref<string | null>(null);
const renameDraft = ref("");
const renamingProjectId = ref<string | null>(null);
const projectRenameDraft = ref("");
const moveMenuId = ref<string | null>(null);

function focusRenameInput(element: Element | { $el?: Element } | null): void {
  const target = element instanceof HTMLInputElement ? element : null;
  if (target === null) {
    return;
  }
  target.focus();
  target.select();
}

const knowledgeBaseNames = computed(() =>
  Object.fromEntries(props.knowledgeBases.map((item) => [item.id, item.name])),
);

const projectsOpen = ref(true);
const collapsedProjectIds = ref<string[]>([]);

const recentConversations = computed(() =>
  props.conversations.filter((conversation) => conversation.project_id === null),
);

function projectConversations(project: ChatProject): ChatConversation[] {
  return props.conversations.filter(
    (conversation) => conversation.project_id === project.id,
  );
}

function isProjectExpanded(project: ChatProject): boolean {
  return !collapsedProjectIds.value.includes(project.id);
}

function toggleProject(project: ChatProject): void {
  collapsedProjectIds.value = isProjectExpanded(project)
    ? [...collapsedProjectIds.value, project.id]
    : collapsedProjectIds.value.filter((id) => id !== project.id);
}

function collapseAllProjects(): void {
  collapsedProjectIds.value = props.projects.map((project) => project.id);
}

function expandAllProjects(): void {
  collapsedProjectIds.value = [];
}

async function createProject(): Promise<void> {
  try {
    const { value } = await ElMessageBox.prompt(
      "给项目起个名字，例如「中考化学复习」。",
      "新建项目",
      {
        confirmButtonText: "创建",
        cancelButtonText: "取消",
        inputPlaceholder: "项目名称",
        inputPattern: /\S/,
        inputErrorMessage: "项目名称不能为空",
      },
    );
    emit("createProject", value.trim());
  } catch {
    return;
  }
}

function formatRelativeTime(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "";
  }
  const diffMinutes = Math.round((Date.now() - date.getTime()) / 60_000);
  if (diffMinutes < 1) {
    return "刚刚";
  }
  if (diffMinutes < 60) {
    return `${diffMinutes} 分钟前`;
  }
  if (diffMinutes < 24 * 60) {
    return `${Math.round(diffMinutes / 60)} 小时前`;
  }
  return new Intl.DateTimeFormat("zh-CN", { month: "numeric", day: "numeric" }).format(date);
}

function knowledgeBaseLabel(conversation: ChatConversation): string {
  if (conversation.knowledge_base_id === null) {
    return "未绑定知识库";
  }
  return knowledgeBaseNames.value[conversation.knowledge_base_id] ?? "知识库已删除";
}

function handleKnowledgeBaseChange(event: Event): void {
  const value = (event.currentTarget as HTMLSelectElement).value;
  emit("selectKnowledgeBase", value || null);
}

function startRename(conversation: ChatConversation): void {
  renamingId.value = conversation.id;
  renameDraft.value = conversation.title;
}

function cancelRename(): void {
  renamingId.value = null;
  renameDraft.value = "";
}

function commitRename(conversation: ChatConversation): void {
  const title = renameDraft.value.trim();
  renamingId.value = null;
  if (title === "" || title === conversation.title) {
    return;
  }
  emit("renameConversation", conversation.id, title);
}

function toggleMoveMenu(conversation: ChatConversation): void {
  moveMenuId.value = moveMenuId.value === conversation.id ? null : conversation.id;
}

function moveTo(conversation: ChatConversation, projectId: string | null): void {
  moveMenuId.value = null;
  if (conversation.project_id === projectId) {
    return;
  }
  emit("moveConversation", conversation.id, projectId);
}

async function confirmDeleteProject(project: ChatProject): Promise<void> {
  try {
    await ElMessageBox.confirm(
      `删除项目不会删除里面的对话，它们会回到「全部对话」。`,
      `删除项目「${project.title}」？`,
      {
        confirmButtonText: "删除项目",
        cancelButtonText: "取消",
        type: "warning",
        confirmButtonClass: "el-button--danger",
      },
    );
  } catch {
    return;
  }
  emit("deleteProject", project.id);
}

function startProjectRename(project: ChatProject): void {
  renamingProjectId.value = project.id;
  projectRenameDraft.value = project.title;
}

function commitProjectRename(project: ChatProject): void {
  const title = projectRenameDraft.value.trim();
  renamingProjectId.value = null;
  if (title === "" || title === project.title) {
    return;
  }
  emit("renameProject", project.id, title);
}

async function confirmDelete(conversation: ChatConversation): Promise<void> {
  try {
    await ElMessageBox.confirm(
      `删除后「${conversation.title}」的对话记录无法恢复。`,
      "删除这段对话？",
      {
        confirmButtonText: "删除",
        cancelButtonText: "取消",
        type: "warning",
        confirmButtonClass: "el-button--danger",
      },
    );
  } catch {
    return;
  }
  emit("deleteConversation", conversation.id);
}
</script>

<template>
  <aside
    class="chat-sidebar"
    :class="{ 'is-open': isOpen }"
    aria-label="对话列表"
  >
    <div class="chat-sidebar__head">
      <button
        class="new-conversation"
        type="button"
        :disabled="isBusy"
        @click="emit('newConversation')"
      >
        <span aria-hidden="true">＋</span>
        <span>新对话</span>
      </button>
      <button
        class="chat-sidebar__close"
        type="button"
        aria-label="收起对话列表"
        @click="emit('close')"
      >
        ✕
      </button>
    </div>

    <div class="knowledge-base-field">
      <label for="chat-knowledge-base">提问依据</label>
      <div
        v-if="knowledgeBasesLoading"
        class="field-state"
        role="status"
      >
        正在加载知识库…
      </div>
      <div
        v-else-if="knowledgeBasesError"
        class="field-state field-state--error"
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
        class="field-state"
      >
        暂无知识库，请先建立一个。
      </div>
      <select
        v-else
        id="chat-knowledge-base"
        class="knowledge-base-select"
        :value="selectedKnowledgeBaseId ?? ''"
        aria-label="选择提问依据的知识库"
        :disabled="isBusy"
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

    <button
      class="memory-entry"
      type="button"
      title="查看与编辑跨对话的长期记忆"
      @click="emit('openMemories')"
    >
      <span
        class="memory-entry__icon"
        aria-hidden="true"
      >◉</span>
      <span>长期记忆</span>
      <span
        v-if="memoryCount > 0"
        class="memory-entry__count"
      >{{ memoryCount }}</span>
    </button>

    <div class="projects-block">
      <div class="projects-head">
        <button
          class="projects-head__toggle"
          type="button"
          :aria-expanded="projectsOpen"
          @click="projectsOpen = !projectsOpen"
        >
          <span>项目</span>
          <span
            class="projects-head__caret"
            :class="{ 'is-open': projectsOpen }"
            aria-hidden="true"
          >▸</span>
        </button>
        <div class="projects-head__actions">
          <button
            type="button"
            class="projects-action"
            aria-label="新建项目"
            title="新建项目"
            @click="createProject()"
          >
            ＋
          </button>
          <button
            type="button"
            class="projects-action"
            aria-label="收起全部项目"
            title="收起全部项目"
            @click="collapseAllProjects()"
          >
            ···
          </button>
          <button
            type="button"
            class="projects-action"
            aria-label="展开全部项目"
            title="展开全部项目"
            @click="expandAllProjects()"
          >
            ↻
          </button>
        </div>
      </div>

      <div
        v-if="projectsOpen"
        class="projects-list"
      >
        <div
          v-if="projects.length === 0"
          class="projects-empty"
        >
          还没有项目，用 ＋ 建一个，把相关对话归到一起。
        </div>
        <div
          v-for="project in projects"
          :key="project.id"
          class="project-row"
        >
          <div class="project-row__line">
            <button
              type="button"
              class="project-row__main"
              :aria-expanded="isProjectExpanded(project)"
              @click="toggleProject(project)"
            >
              <span
                class="project-row__caret"
                :class="{ 'is-open': isProjectExpanded(project) }"
                aria-hidden="true"
              >▸</span>
              <span class="project-row__name">{{ project.title }}</span>
              <span class="project-row__count">{{ projectConversations(project).length }}</span>
            </button>
            <div class="project-row__actions">
              <button
                type="button"
                class="conversation-action"
                aria-label="重命名项目"
                title="重命名项目"
                @click="startProjectRename(project)"
              >
                ✎
              </button>
              <button
                type="button"
                class="conversation-action conversation-action--danger"
                aria-label="删除项目"
                title="删除项目"
                @click="confirmDeleteProject(project)"
              >
                ⌫
              </button>
            </div>
          </div>

          <input
            v-if="renamingProjectId === project.id"
            :ref="focusRenameInput"
            v-model="projectRenameDraft"
            class="conversation-rename"
            type="text"
            maxlength="200"
            aria-label="重命名项目"
            @keydown.enter.prevent="commitProjectRename(project)"
            @keydown.esc.prevent="renamingProjectId = null"
            @blur="commitProjectRename(project)"
          >

          <div
            v-if="isProjectExpanded(project)"
            class="project-row__conversations"
          >
            <div
              v-if="projectConversations(project).length === 0"
              class="project-row__empty"
            >
              还没有对话
            </div>
            <div
              v-for="conversation in projectConversations(project)"
              :key="conversation.id"
              class="conversation-item is-nested"
              :class="{ 'is-active': conversation.id === activeConversationId }"
              role="listitem"
            >
              <input
                v-if="renamingId === conversation.id"
                :ref="focusRenameInput"
                v-model="renameDraft"
                class="conversation-rename"
                type="text"
                maxlength="200"
                aria-label="重命名对话"
                @keydown.enter.prevent="commitRename(conversation)"
                @keydown.esc.prevent="cancelRename"
                @blur="commitRename(conversation)"
              >
              <button
                v-else
                type="button"
                class="conversation-main"
                :disabled="isBusy"
                @click="emit('selectConversation', conversation.id)"
              >
                <span class="conversation-title">{{ conversation.title }}</span>
                <span class="conversation-meta">
                  {{ knowledgeBaseLabel(conversation) }} ·
                  {{ formatRelativeTime(conversation.updated_at) }}
                </span>
              </button>
              <div class="conversation-actions">
                <button
                  type="button"
                  class="conversation-action"
                  aria-label="移动对话"
                  title="移动到项目"
                  @click="toggleMoveMenu(conversation)"
                >
                  ⇢
                </button>
                <button
                  type="button"
                  class="conversation-action conversation-action--danger"
                  :aria-label="`删除 ${conversation.title}`"
                  title="删除"
                  @click="confirmDelete(conversation)"
                >
                  ⌫
                </button>
              </div>
              <div
                v-if="moveMenuId === conversation.id"
                class="move-menu"
              >
                <button
                  v-for="other in projects"
                  :key="other.id"
                  type="button"
                  class="move-menu__option"
                  @click="moveTo(conversation, other.id)"
                >
                  {{ other.title }}
                  <span
                    v-if="conversation.project_id === other.id"
                    class="move-menu__current"
                  >当前</span>
                </button>
                <button
                  type="button"
                  class="move-menu__option"
                  @click="moveTo(conversation, null)"
                >
                  移出项目
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="conversation-block">
      <div class="conversation-block__title">
        <span>最近</span>
        <span
          v-if="recentConversations.length > 0"
          class="conversation-block__count"
        >{{ recentConversations.length }}</span>
      </div>

      <div
        v-if="conversationsLoading"
        class="conversation-state"
        role="status"
      >
        <span class="skeleton-line" />
        <span class="skeleton-line" />
        <span class="skeleton-line" />
      </div>

      <div
        v-else-if="conversationsError"
        class="conversation-state conversation-state--error"
        role="alert"
      >
        <span>{{ conversationsErrorMessage }}</span>
        <button
          type="button"
          class="text-action"
          @click="emit('retryConversations')"
        >
          重试
        </button>
      </div>

      <div
        v-else-if="recentConversations.length === 0"
        class="conversation-state"
      >
        还没有最近的对话，提一个问题就会自动保存。
      </div>

      <div
        v-else
        class="conversation-list"
        role="list"
      >
        <div
          v-for="conversation in recentConversations"
          :key="conversation.id"
          class="conversation-item"
          :class="{ 'is-active': conversation.id === activeConversationId }"
          role="listitem"
        >
          <input
            v-if="renamingId === conversation.id"
            :ref="focusRenameInput"
            v-model="renameDraft"
            class="conversation-rename"
            type="text"
            maxlength="200"
            aria-label="重命名对话"
            @keydown.enter.prevent="commitRename(conversation)"
            @keydown.esc.prevent="cancelRename"
            @blur="commitRename(conversation)"
          >
          <button
            v-else
            type="button"
            class="conversation-main"
            :disabled="isBusy"
            @click="emit('selectConversation', conversation.id)"
          >
            <span class="conversation-title">{{ conversation.title }}</span>
            <span class="conversation-meta">
              {{ knowledgeBaseLabel(conversation) }} ·
              {{ formatRelativeTime(conversation.updated_at) }}
            </span>
          </button>

          <div class="conversation-actions">
            <button
              type="button"
              class="conversation-action"
              aria-label="移动对话"
              title="移动到项目"
              @click="toggleMoveMenu(conversation)"
            >
              ⇢
            </button>
            <button
              type="button"
              class="conversation-action"
              aria-label="重命名对话"
              title="重命名"
              @click="startRename(conversation)"
            >
              ✎
            </button>
            <button
              type="button"
              class="conversation-action conversation-action--danger"
              aria-label="删除对话"
              title="删除"
              @click="confirmDelete(conversation)"
            >
              ⌫
            </button>
          </div>

          <div
            v-if="moveMenuId === conversation.id"
            class="move-menu"
          >
            <button
              v-for="project in projects"
              :key="project.id"
              type="button"
              class="move-menu__option"
              @click="moveTo(conversation, project.id)"
            >
              {{ project.title }}
              <span
                v-if="conversation.project_id === project.id"
                class="move-menu__current"
              >当前</span>
            </button>
            <button
              type="button"
              class="move-menu__option"
              @click="moveTo(conversation, null)"
            >
              移出项目
            </button>
          </div>
        </div>
      </div>
    </div>
  </aside>
</template>

<style scoped>
/* 侧边栏样式：从旧版 ConversationSidebar 的设计语言重建。 */
.chat-sidebar {
  display: flex;
  min-width: 232px;
  width: 232px;
  flex-direction: column;
  gap: 14px;
  padding: 20px 14px 16px;
  overflow-y: auto;
  background: var(--ks-surface-subtle);
  border-right: 1px solid var(--ks-border);
}

.chat-sidebar__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.new-conversation {
  display: flex;
  flex: 1;
  align-items: center;
  justify-content: center;
  min-height: 38px;
  gap: 8px;
  color: var(--ks-surface);
  font-size: 13px;
  font-weight: 680;
  background: var(--ks-accent);
  border: 0;
  border-radius: var(--ks-radius-sm);
  box-shadow: 0 3px 8px rgb(55 116 102 / 14%);
}

.new-conversation:hover:not(:disabled) {
  background: var(--ks-accent-strong);
}

.new-conversation:disabled {
  opacity: 0.6;
}

.chat-sidebar__close {
  display: none;
  width: 32px;
  height: 32px;
  flex: 0 0 32px;
  color: var(--ks-muted);
  font-size: 14px;
  background: var(--ks-surface);
  border: 1px solid var(--ks-border);
  border-radius: 9px;
}

.knowledge-base-field,
.conversation-block {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.knowledge-base-field label,
.conversation-block__title {
  color: var(--ks-muted);
  font-size: 11px;
  font-weight: 680;
}

.conversation-block__title {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.conversation-block__count {
  color: var(--ks-faint);
  font-weight: 620;
}

.field-state,
.conversation-state {
  display: flex;
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

.field-state--error,
.conversation-state--error {
  color: var(--ks-danger);
}

.text-action {
  flex: 0 0 auto;
  padding: 0;
  color: var(--ks-accent-strong);
  font-size: 12px;
  font-weight: 680;
  background: transparent;
  border: 0;
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

.skeleton-line {
  height: 12px;
  flex: 1;
  background: var(--ks-border);
  border-radius: 6px;
}

.memory-entry {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 9px 10px;
  color: var(--ks-muted);
  font-size: 12px;
  font-weight: 620;
  text-align: left;
  background: var(--ks-surface);
  border: 1px solid var(--ks-border);
  border-radius: var(--ks-radius-sm);
}

.memory-entry:hover {
  color: var(--ks-accent-strong);
  background: var(--ks-accent-soft);
}

.memory-entry__icon {
  color: var(--ks-accent);
}

.memory-entry__count {
  margin-left: auto;
  color: var(--ks-faint);
}

.conversation-groups {
  display: flex;
  min-height: 0;
  flex: 1;
  flex-direction: column;
  gap: 12px;
  overflow-y: auto;
}

.conversation-group {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.conversation-group__head {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 0 4px;
}

.conversation-group__name {
  overflow: hidden;
  color: var(--ks-ink);
  font-size: 12px;
  font-weight: 680;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.conversation-group__count {
  color: var(--ks-faint);
  font-size: 11px;
}

.conversation-group__head .conversation-action {
  display: none;
}

.conversation-group__head:hover .conversation-action {
  display: grid;
}

.conversation-group__empty {
  padding: 4px 6px 2px;
  color: var(--ks-faint);
  font-size: 11px;
  line-height: 1.5;
}

.conversation-item {
  position: relative;
  display: flex;
  min-height: 50px;
  align-items: center;
  gap: 8px;
  padding: 8px 8px 8px 10px;
  background: transparent;
  border-radius: var(--ks-radius-sm);
}

.conversation-item:hover,
.conversation-item.is-active {
  background: var(--ks-surface);
}

.conversation-item.is-active {
  box-shadow: 0 0 0 1px var(--ks-border) inset;
}

.conversation-main {
  display: flex;
  min-width: 0;
  flex: 1;
  flex-direction: column;
  gap: 4px;
  padding: 0;
  text-align: left;
  background: transparent;
  border: 0;
}

.conversation-title {
  overflow: hidden;
  color: var(--ks-text);
  font-size: 13px;
  font-weight: 630;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.conversation-meta {
  color: var(--ks-faint);
  font-size: 10px;
}

.conversation-actions {
  display: flex;
  flex: 0 0 auto;
  gap: 2px;
  opacity: 0;
  transition: opacity var(--ks-duration-fast) var(--ks-ease-out);
}

.conversation-item:hover .conversation-actions,
.conversation-item.is-active .conversation-actions {
  opacity: 1;
}

.conversation-action {
  display: grid;
  width: 24px;
  height: 24px;
  place-items: center;
  color: var(--ks-muted);
  font-size: 12px;
  background: transparent;
  border: 0;
  border-radius: 6px;
}

.conversation-action:hover {
  color: var(--ks-accent-strong);
  background: var(--ks-accent-soft);
}

.conversation-action--danger:hover {
  color: var(--ks-danger);
  background: #fbeaea;
}

.conversation-rename {
  width: 100%;
  min-height: 30px;
  padding: 0 8px;
  color: var(--ks-ink);
  font-size: 13px;
  background: var(--ks-surface);
  border: 1px solid var(--ks-accent);
  border-radius: var(--ks-radius-sm);
}

.move-menu {
  position: absolute;
  top: calc(100% - 4px);
  right: 6px;
  z-index: 20;
  display: flex;
  min-width: 180px;
  flex-direction: column;
  padding: 4px;
  background: var(--ks-surface);
  border: 1px solid var(--ks-border);
  border-radius: var(--ks-radius-sm);
  box-shadow: var(--ks-shadow-md);
}

.move-menu__option {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 7px 9px;
  color: var(--ks-text);
  font-size: 12px;
  text-align: left;
  background: transparent;
  border: 0;
  border-radius: 6px;
}

.move-menu__option:hover {
  background: var(--ks-accent-soft);
}

.move-menu__option--create {
  color: var(--ks-accent-strong);
  font-weight: 650;
}

.move-menu__current {
  color: var(--ks-faint);
  font-size: 10px;
}

@media (max-width: 1040px) {
  .chat-sidebar {
    min-width: 206px;
    width: 206px;
  }
}

@media (max-width: 900px) {
  .chat-sidebar {
    position: fixed;
    inset: 0 auto 0 0;
    z-index: 40;
    width: 260px;
    background: var(--ks-surface-subtle);
    box-shadow: var(--ks-shadow-md);
    transform: translateX(-105%);
    transition: transform var(--ks-duration-fast) var(--ks-ease-out);
  }

  .chat-sidebar.is-open {
    transform: none;
  }

  .chat-sidebar__close {
    display: grid;
    place-items: center;
  }
}
</style>

<style scoped>
/* 项目区：模仿 ChatGPT 的「项目 / 最近」两段结构。 */
.projects-block {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.projects-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 6px;
}

.projects-head__toggle {
  display: flex;
  flex: 1;
  align-items: center;
  gap: 6px;
  padding: 4px 6px;
  color: var(--ks-ink);
  font-size: 12px;
  font-weight: 680;
  text-align: left;
  background: transparent;
  border: 0;
  border-radius: 6px;
}

.projects-head__toggle:hover {
  background: var(--ks-accent-soft);
}

.projects-head__caret {
  display: inline-block;
  color: var(--ks-faint);
  font-size: 10px;
  transition: transform var(--ks-duration-fast) var(--ks-ease-out);
}

.projects-head__caret.is-open {
  transform: rotate(90deg);
}

.projects-head__actions {
  display: flex;
  gap: 2px;
  opacity: 0;
  transition: opacity var(--ks-duration-fast) var(--ks-ease-out);
}

.projects-head:hover .projects-head__actions,
.projects-head__actions:focus-within {
  opacity: 1;
}

.projects-action {
  display: grid;
  width: 24px;
  height: 24px;
  place-items: center;
  color: var(--ks-muted);
  font-size: 13px;
  background: transparent;
  border: 0;
  border-radius: 6px;
}

.projects-action:hover {
  color: var(--ks-accent-strong);
  background: var(--ks-accent-soft);
}

.projects-list,
.project-row {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.projects-empty {
  padding: 6px 8px;
  color: var(--ks-faint);
  font-size: 11px;
  line-height: 1.6;
}

.project-row__line {
  display: flex;
  align-items: center;
  gap: 4px;
}

.project-row__main {
  display: flex;
  min-width: 0;
  flex: 1;
  align-items: center;
  gap: 6px;
  padding: 6px 6px;
  color: var(--ks-text);
  font-size: 12px;
  font-weight: 650;
  text-align: left;
  background: transparent;
  border: 0;
  border-radius: 6px;
}

.project-row__main:hover {
  background: var(--ks-accent-soft);
}

.project-row__caret {
  display: inline-block;
  color: var(--ks-faint);
  font-size: 10px;
  transition: transform var(--ks-duration-fast) var(--ks-ease-out);
}

.project-row__caret.is-open {
  transform: rotate(90deg);
}

.project-row__name {
  overflow: hidden;
  flex: 1;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.project-row__count {
  color: var(--ks-faint);
  font-size: 10px;
}

.project-row__actions {
  display: flex;
  gap: 2px;
  opacity: 0;
  transition: opacity var(--ks-duration-fast) var(--ks-ease-out);
}

.project-row__line:hover .project-row__actions {
  opacity: 1;
}

.project-row__conversations {
  display: flex;
  flex-direction: column;
  gap: 2px;
  margin-left: 14px;
}

.project-row__empty {
  padding: 2px 8px 4px;
  color: var(--ks-faint);
  font-size: 11px;
}

.conversation-item.is-nested {
  min-height: 44px;
}
</style>
