<script setup lang="ts">
// Kept next to the component so the chat route pulls the stylesheet, not the
// whole app.
import "katex/dist/katex.min.css";
import katex from "katex";
import { onMounted, ref, watch } from "vue";

const props = withDefaults(
  defineProps<{
    latex: string;
    display?: boolean;
  }>(),
  { display: false },
);

const host = ref<HTMLElement | null>(null);
const failed = ref(false);

/**
 * Render LaTeX into the element with KaTeX's own DOM API — no innerHTML.
 *
 * KaTeX escapes its input and `trust` stays off, so a formula can only produce
 * KaTeX markup.  An unparseable formula shows its source instead of an error or
 * a blank spot, so nothing the model wrote is ever lost.
 */
function renderMath(): void {
  const element = host.value;
  if (element === null) {
    return;
  }
  try {
    katex.render(props.latex, element, {
      displayMode: props.display,
      throwOnError: true,
      trust: false,
      strict: false,
    });
    failed.value = false;
  } catch {
    element.textContent = props.display ? `$$${props.latex}$$` : `$${props.latex}$`;
    failed.value = true;
  }
}

onMounted(renderMath);
watch(() => [props.latex, props.display], renderMath);
</script>

<template>
  <span
    ref="host"
    class="answer-math"
    :class="{ 'is-display': display, 'is-raw': failed }"
    :title="latex"
  />
</template>

<style scoped>
.answer-math.is-display {
  display: block;
  margin: 12px 0;
  overflow-x: auto;
  text-align: center;
}

.answer-math.is-raw {
  color: var(--ks-muted);
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.94em;
}

.answer-math :deep(.katex) {
  font-size: 1.02em;
}

.answer-math.is-display :deep(.katex-display) {
  margin: 0;
}
</style>
