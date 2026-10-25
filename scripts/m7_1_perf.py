"""Measure visual encoder/query/Qdrant/reranker latency on the local host."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from qdrant_client import QdrantClient, models

from knowledge_scope.retrieval.reranking import create_local_reranker
from knowledge_scope.retrieval.visual_embedding import VisualEmbeddingEncoder
from knowledge_scope.shared.config import Settings

QUERIES = [
    "柱状图展示不同模型的准确率对比",
    "折线图显示训练损失随轮次下降",
    "系统架构图中的数据流转",
    "流程图展示算法处理步骤",
    "表格截图包含多列数值指标",
    "散点图展示预测值与真实值关系",
]


def _percentile(values: list[float], ratio: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round(ratio * (len(ordered) - 1))))
    return round(ordered[index], 2)


def _stats(values: list[float]) -> dict:
    return {
        "count": len(values),
        "p50_ms": _percentile(values, 0.5),
        "p95_ms": _percentile(values, 0.95),
        "avg_ms": round(statistics.fmean(values), 2) if values else 0.0,
    }


def _encoder(model_path: Path, device: str) -> VisualEmbeddingEncoder:
    settings = Settings(
        _env_file=None,
        data_dir="/tmp/m7-perf-data",
        visual_embedding_enabled=True,
        visual_model_path=str(model_path),
        visual_device=device,
        visual_dtype="bfloat16",
        visual_batch_size=1,
    )
    encoder = VisualEmbeddingEncoder(settings)
    started = time.perf_counter()
    encoder.encode_query("预热查询")
    load_ms = round((time.perf_counter() - started) * 1000, 1)
    return encoder, load_ms  # type: ignore[return-value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixtures = json.loads(args.fixtures.read_text())
    kb = fixtures["knowledge_base_id"]
    result: dict = {}

    gpu_encoder, gpu_load_ms = _encoder(args.model_path, "cuda")
    torch.cuda.reset_peak_memory_stats()
    query_vectors = []
    gpu_encode = []
    for query in QUERIES:
        started = time.perf_counter()
        vector, _encode_meta = gpu_encoder.encode_query(query)
        query_vectors.append(vector)
        gpu_encode.append((time.perf_counter() - started) * 1000)
    gpu_peak = gpu_encoder.vram_peak_mb()
    result["gpu"] = {
        "model_load_ms": gpu_load_ms,
        "query_encode": _stats(gpu_encode),
        "peak_vram_mb": round(gpu_peak, 1) if gpu_peak else None,
        "embedding_dimension": gpu_encoder.embedding_dimension(),
    }

    client = QdrantClient(url="http://127.0.0.1:6333", trust_env=False)
    collection_name = None
    for item in client.get_collections().collections:
        if item.name.startswith("knowledgescope_visual_v1_"):
            collection_name = item.name
    qdrant_latency = []
    if collection_name and query_vectors:
        for _repeat in range(5):
            for vector in query_vectors:
                started = time.perf_counter()
                client.query_points(
                    collection_name=collection_name,
                    query=vector,
                    query_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="knowledge_base_id", match=models.MatchValue(value=kb)
                            ),
                            models.FieldCondition(
                                key="active", match=models.MatchValue(value=True)
                            ),
                        ]
                    ),
                    limit=8,
                    with_payload=False,
                )
                qdrant_latency.append((time.perf_counter() - started) * 1000)
    result["qdrant_visual_query"] = {
        "collection": collection_name,
        **_stats(qdrant_latency),
    }

    gpu_encoder.close()
    torch.cuda.empty_cache()
    cpu_encoder, cpu_load_ms = _encoder(args.model_path, "cpu")
    cpu_encode = []
    for query in QUERIES:
        started = time.perf_counter()
        cpu_encoder.encode_query(query)
        cpu_encode.append((time.perf_counter() - started) * 1000)
    cpu_encoder.close()
    result["cpu"] = {"model_load_ms": cpu_load_ms, "query_encode": _stats(cpu_encode)}

    settings = Settings(_env_file=None, data_dir="/tmp/m7-perf-data")
    reranker = create_local_reranker(settings, model_key="bge-reranker-v2-m3")
    passages = [
        "季风气候统计图展示了不同地区降水量变化",
        "柱状图对比不同模型的准确率",
        "折线图显示训练损失下降",
        "系统架构图包含数据采集与处理模块",
        "流程图描述算法执行步骤",
        "表格列出多组实验数值",
    ]
    reranker.score_pairs("柱状图展示模型准确率对比", passages)
    rerank_latency = []
    for _ in range(5):
        started = time.perf_counter()
        reranker.score_pairs("柱状图展示模型准确率对比", passages)
        rerank_latency.append((time.perf_counter() - started) * 1000)
    result["reranker"] = _stats(rerank_latency)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
