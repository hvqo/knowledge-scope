"""Measure visual Top-K -> Unified Top-8 -> Agent Top-6 candidate retention."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

from qdrant_client import QdrantClient

from knowledge_scope.agent.tools import AgentRetrievalTool
from knowledge_scope.graph.retrieval import GraphRetrievalResult
from knowledge_scope.retrieval.embedding import QwenEmbeddingModel
from knowledge_scope.retrieval.qdrant import QdrantVectorStore
from knowledge_scope.retrieval.representation_index import (
    MultimodalRepresentationRetrievalService,
    QdrantRepresentationStore,
)
from knowledge_scope.retrieval.reranking import create_local_reranker
from knowledge_scope.retrieval.service import DenseRetrievalService
from knowledge_scope.retrieval.sparse import SparseSearchResponse
from knowledge_scope.retrieval.unified import UnifiedRetrievalConfig, UnifiedRetrievalService
from knowledge_scope.retrieval.visual_embedding import VisualEmbeddingEncoder
from knowledge_scope.retrieval.visual_index import QdrantVisualStore
from knowledge_scope.shared.config import Settings


class _EmptySparse:
    def search(self, query, *, knowledge_base_id, top_k, document_id=None):
        return SparseSearchResponse(
            query=query,
            knowledge_base_id=knowledge_base_id,
            document_id=document_id,
            top_k=top_k,
            index_fingerprint="0" * 64,
            items=[],
        )


class _EmptyGraph:
    def search(self, query, knowledge_base_id, *, document_id=None):
        return GraphRetrievalResult(query=query, knowledge_base_id=knowledge_base_id, items=[])


def _gold_set(entry: dict) -> set[str]:
    return set(entry["gold_evidence_ids"])


def _position(items: list[str], gold: set[str]) -> int | None:
    for index, evidence_id in enumerate(items):
        if evidence_id in gold:
            return index + 1
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixtures = json.loads(args.fixtures.read_text())
    kb = UUID(fixtures["knowledge_base_id"])
    documents = {item["id"]: item for item in fixtures["documents"]}
    payload = json.loads(args.gold.read_text())
    diagnostics = payload.get("diagnostics")
    gold_items = diagnostics["queries"] if diagnostics else payload["queries"]

    settings = Settings(
        _env_file=None,
        data_dir=str(args.data_dir),
        visual_embedding_enabled=True,
        visual_model_path=str(args.model_path),
        visual_device="cuda",
        visual_dtype="bfloat16",
        visual_batch_size=1,
        visual_candidate_limit=8,
        visual_fusion_quota=2,
        visual_quota_ambiguous=1,
    )
    client = QdrantClient(url=settings.qdrant_url, trust_env=False)
    store = QdrantVectorStore(settings, client=client)
    embedder = QwenEmbeddingModel(settings)
    dense = DenseRetrievalService(store, embedder)
    representation_store = QdrantRepresentationStore(settings, client=client)
    multimodal = MultimodalRepresentationRetrievalService(representation_store, embedder)
    reranker = create_local_reranker(settings, model_key="bge-reranker-v2-m3")
    visual_encoder = VisualEmbeddingEncoder(settings)
    visual_store = QdrantVisualStore(settings, visual_encoder, client=client)
    unified = UnifiedRetrievalService(
        dense,
        _EmptySparse(),
        _EmptyGraph(),
        multimodal,
        reranker,
        chunk_lookup=store,
        config=UnifiedRetrievalConfig(
            candidate_pool_limit=20,
            result_limit=8,
            visual_candidate_limit=8,
            visual_fusion_quota=2,
            visual_quota_ambiguous=1,
        ),
        visual_retrieval=visual_store,
    )

    async def loader(knowledge_base_id: UUID, document_id: UUID):
        item = documents.get(str(document_id))
        if item is None:
            return None
        return SimpleNamespace(
            id=UUID(item["id"]),
            knowledge_base_id=knowledge_base_id,
            original_filename=item["name"],
        )

    tool = AgentRetrievalTool(settings, unified, document_loader=loader, result_limit=6)
    results = []
    for item in gold_items:
        query = item["query"]
        gold = _gold_set(item)
        started = time.perf_counter()
        visual = visual_store.search(query, kb, top_k=8)
        visual_ids = [hit.payload.evidence_id for hit in visual.items]
        visual_rank = _position(visual_ids, gold)
        unified_result = asyncio.run(unified.search(query, kb))
        unified_ids = [
            candidate.evidence_id for candidate in unified_result.items if candidate.evidence_id
        ]
        unified_rank = _position(unified_ids, gold)
        retrieval = asyncio.run(tool.retrieve(query, kb))
        agent_ids = [
            candidate.evidence_id for candidate in retrieval.candidates if candidate.evidence_id
        ]
        agent_rank = _position(agent_ids, gold)
        results.append(
            {
                "id": item["id"],
                "query": query,
                "gold_evidence_ids": sorted(gold),
                "visual_top8_rank": visual_rank,
                "unified_top8_rank": unified_rank,
                "agent_top6_rank": agent_rank,
                "visual_intent": unified_result.visual_intent,
                "visual_quota_applied": unified_result.visual_quota_applied,
                "branch_statuses": {
                    branch.branch: branch.status for branch in unified_result.branches
                },
                "branch_errors": {
                    branch.branch: branch.error
                    for branch in unified_result.branches
                    if branch.error
                },
                "candidates": {
                    "visual_top8": len(visual_ids),
                    "unified_top8": len(unified_ids),
                    "agent_top6": len(agent_ids),
                },
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            }
        )
        print(
            f"{item['id']}: visual={visual_rank} unified={unified_rank} agent={agent_rank} "
            f"intent={unified_result.visual_intent} quota={unified_result.visual_quota_applied}"
        )
    args.output.write_text(
        json.dumps({"fixtures": fixtures, "results": results}, ensure_ascii=False, indent=2)
    )
    visual_hits = sum(1 for item in results if item["visual_top8_rank"])
    unified_hits = sum(1 for item in results if item["unified_top8_rank"])
    agent_hits = sum(1 for item in results if item["agent_top6_rank"])
    print(
        json.dumps(
            {
                "queries": len(results),
                "visual_top8_hits": visual_hits,
                "unified_top8_hits": unified_hits,
                "agent_top6_hits": agent_hits,
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
