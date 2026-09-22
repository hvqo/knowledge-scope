import { createRouter, createWebHistory } from "vue-router";

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: "/",
      redirect: { name: "knowledge-bases" },
    },
    {
      path: "/knowledge-bases",
      name: "knowledge-bases",
      component: () => import("../views/KnowledgeBaseListView.vue"),
      meta: {
        title: "知识库",
      },
    },
    {
      path: "/chat",
      name: "chat",
      component: () => import("../views/ChatView.vue"),
      meta: {
        title: "AI 问答",
      },
    },
    {
      path: "/analysis",
      name: "analysis",
      component: () => import("../views/AnalysisView.vue"),
      meta: {
        title: "数据分析",
      },
    },
    {
      path: "/reports",
      name: "reports",
      component: () => import("../views/ReportListView.vue"),
      meta: {
        title: "报告创作",
      },
    },
    {
      path: "/reports/:id",
      name: "report-workspace",
      component: () => import("../views/ReportWorkspaceView.vue"),
      meta: {
        title: "报告工作区",
      },
    },
    {
      path: "/knowledge-bases/:id",
      name: "knowledge-base-detail",
      component: () => import("../views/KnowledgeBaseDetailView.vue"),
      meta: {
        title: "知识库",
      },
    },
    {
      path: "/:pathMatch(.*)*",
      name: "not-found",
      component: () => import("../views/NotFoundView.vue"),
      meta: {
        title: "页面不存在",
      },
    },
  ],
});

export default router;
