"""A/B/C retrieval experiment for visual image vectors on an isolated data dir.

A: caption/context text embeddings (Qwen3-Embedding-0.6B), B: selective visual
index (C hits filtered to policy-selected images), C: full visual image index.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from knowledge_scope.retrieval.embedding import QwenEmbeddingModel
from knowledge_scope.retrieval.visual_embedding import VisualEmbeddingEncoder
from knowledge_scope.retrieval.visual_index import QdrantVisualStore
from knowledge_scope.shared.config import Settings

KB_ID = "00000000-0000-4000-8000-000000000007"
VECTOR_BYTES = 2048 * 4


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _cosine_rank(
    query_vector: list[float],
    doc_vectors: list[list[float]],
    doc_ids: list[str | None],
    top_k: int,
) -> list[str]:
    scores: list[tuple[float, int]] = []
    for index, vector in enumerate(doc_vectors):
        dot = sum(a * b for a, b in zip(query_vector, vector, strict=True))
        norm = math.sqrt(sum(a * a for a in query_vector)) * math.sqrt(sum(b * b for b in vector))
        scores.append((dot / norm if norm else 0.0, index))
    scores.sort(reverse=True)
    ranked: list[str] = []
    for _score, index in scores[:top_k]:
        if doc_ids[index]:
            ranked.append(doc_ids[index])
    return ranked


def _mrr(ranked: list[str], gold: set[str]) -> float:
    for position, item in enumerate(ranked, start=1):
        if item in gold:
            return 1.0 / position
    return 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/tmp/m7-data")
    parser.add_argument("--catalog", default="/tmp/m7-catalog.json")
    parser.add_argument("--gold", default="/tmp/m7-gold.json")
    parser.add_argument("--output", default="/tmp/m7-experiment.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    settings = Settings(
        _env_file=None,
        data_dir=str(data_dir),
        visual_embedding_enabled=True,
        visual_model_path="/home/amber/.cache/knowledgescope/qwen3-vl-embedding-2b",
    )
    catalog = _load_json(Path(args.catalog))
    gold_queries = _load_json(Path(args.gold))["queries"]

    # --- A: text representation index from caption/context ------------------
    text_docs: list[str] = []
    text_doc_ids: list[str | None] = []
    for item in catalog:
        caption = (item.get("caption") or "").strip()
        context = (item.get("context") or "").strip()
        text = caption or context
        text_docs.append(text)
        text_doc_ids.append(item["evidence_id"] if text else None)

    text_model = QwenEmbeddingModel(settings)
    started = time.perf_counter()
    text_vectors = text_model.encode_documents(text_docs)
    text_doc_latency_ms = (time.perf_counter() - started) * 1000.0

    # --- C: full visual index ----------------------------------------------
    visual_encoder = VisualEmbeddingEncoder(settings)
    store = QdrantVisualStore(settings, visual_encoder)
    selected = {item["evidence_id"] for item in catalog if item["policy_mode"] == "text_and_visual"}

    results: dict[str, dict[str, list[str]]] = {"A": {}, "B": {}, "C": {}}
    latencies: dict[str, list[float]] = {"A": [], "B": [], "C": []}
    visual_query_ms: list[float] = []
    for gold in gold_queries:
        query = gold["query"]
        gold_id = gold["evidence_id"]
        gold_set = {gold_id}

        started = time.perf_counter()
        query_text_vector = text_model.encode_query(query)
        a_ranked = _cosine_rank(query_text_vector, text_vectors, text_doc_ids, 10)
        latencies["A"].append((time.perf_counter() - started) * 1000.0)

        started = time.perf_counter()
        visual_query_vector = visual_encoder.encode_query(query)
        visual_query_ms.append((time.perf_counter() - started) * 1000.0)
        search_started = time.perf_counter()
        hits = store.search(query, KB_ID, top_k=10)
        latencies["C"].append((time.perf_counter() - search_started) * 1000.0)
        c_ranked = [hit.payload.evidence_id for hit in hits.items]
        b_ranked = [evidence_id for evidence_id in c_ranked if evidence_id in selected]

        results["A"][gold["query_id"]] = a_ranked
        results["B"][gold["query_id"]] = b_ranked
        results["C"][gold["query_id"]] = c_ranked
        del visual_query_vector

    summary: dict[str, Any] = {}
    for variant in ("A", "B", "C"):
        recalls5 = recalls10 = 0
        reciprocal = 0.0
        for gold in gold_queries:
            ranked = results[variant][gold["query_id"]]
            gold_set = {gold["evidence_id"]}
            if gold["evidence_id"] in ranked[:5]:
                recalls5 += 1
            if gold["evidence_id"] in ranked[:10]:
                recalls10 += 1
            reciprocal += _mrr(ranked, gold_set)
        count = len(gold_queries)
        summary[variant] = {
            "recall_at_5": round(recalls5 / count, 4),
            "recall_at_10": round(recalls10 / count, 4),
            "mrr_at_10": round(reciprocal / count, 4),
            "hits_at_10": recalls10,
            "total": count,
            "avg_latency_ms": round(sum(latencies[variant]) / count, 2),
        }

    selected_count = len(selected)
    total_images = len(catalog)
    summary["cost"] = {
        "text_document_encode_ms": round(text_doc_latency_ms, 2),
        "visual_query_encode_avg_ms": round(sum(visual_query_ms) / len(visual_query_ms), 2),
        "visual_image_encode_total_ms_from_build": 84021.254,
        "visual_image_encode_avg_ms": round(84021.254 / total_images, 2),
        "images_total": total_images,
        "images_selected": selected_count,
        "images_text_only": total_images - selected_count,
        "selective_vector_bytes": selected_count * VECTOR_BYTES,
        "full_vector_bytes": total_images * VECTOR_BYTES,
        "selective_image_encode_ms_estimate": round(selected_count * 84021.254 / total_images, 2),
        "full_image_encode_ms_estimate": 84021.254,
    }
    policy_misses = [
        gold["query_id"] for gold in gold_queries if gold["evidence_id"] not in selected
    ]
    summary["policy_misses"] = policy_misses
    summary["policy_miss_count"] = len(policy_misses)

    payload = {
        "variant_definitions": {
            "A": "caption/context text embeddings (Qwen3-Embedding-0.6B)",
            "B": "selective visual index (policy text_and_visual subset of C)",
            "C": "full visual image index (Qwen3-VL-Embedding-2B)",
        },
        "summary": summary,
        "per_query": {
            gold["query_id"]: {
                "query": gold["query"],
                "gold_evidence_id": gold["evidence_id"],
                "category": gold["category"],
                "has_caption": gold["has_caption"],
                "policy_mode": gold["policy_mode"],
                "A": results["A"][gold["query_id"]],
                "B": results["B"][gold["query_id"]],
                "C": results["C"][gold["query_id"]],
            }
            for gold in gold_queries
        },
    }
    Path(args.output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
