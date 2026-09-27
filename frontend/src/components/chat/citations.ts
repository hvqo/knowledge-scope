import type { RAGCitation } from "../../api/types";

const CITATION_MARKER_PATTERN = /\[C[1-9][0-9]*\]/g;

/**
 * Normalize how citation markers sit inside the answer text.
 *
 * Models occasionally emit a marker on its own line, which renders as a
 * floating chip between paragraphs, so markers are glued back to the sentence
 * they support.  Everything else — spacing and punctuation — is left exactly
 * as the model wrote it.
 */
export function normalizeCitationLayout(content: string): string {
  return content.replace(/([^\S\n]*\n+)+\s*(\[C[1-9][0-9]*\])/g, "$2");
}

/**
 * Keep only the citations the answer actually referenced.
 *
 * The server returns every candidate it retrieved, but a grounded answer often
 * cites a subset, so "来源" should list what the text points at.
 */
export function referencedCitations(
  content: string,
  citations: RAGCitation[],
): RAGCitation[] {
  const markers = new Set(
    [...content.matchAll(CITATION_MARKER_PATTERN)].map((match) => match[0].slice(1, -1)),
  );
  return citations.filter((citation) => markers.has(citation.marker));
}
