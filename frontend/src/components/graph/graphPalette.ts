import { hashUnit } from "./graphLayout";

const PALETTE: readonly string[] = [
  "#377466",
  "#3f6f9c",
  "#a66f32",
  "#6f5c9c",
  "#3f8063",
  "#b45e56",
  "#5c7a86",
  "#8a6d3b",
];

/** Stable colour per entity type so one type keeps its colour between renders. */
export function colorForType(entityType: string): string {
  const index = Math.min(PALETTE.length - 1, Math.floor(hashUnit(entityType) * PALETTE.length));
  return PALETTE[index];
}
