# ruff: noqa: RUF001
from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from knowledge_scope.evidence.lifecycle import write_evidence_artifact
from knowledge_scope.evidence.models import (
    MultimodalEvidenceDocument,
    representation_id_for,
)
from knowledge_scope.parsing.models import ImageBlock
from knowledge_scope.retrieval.unified import UnifiedRetrievalConfig, UnifiedRetrievalService
from knowledge_scope.retrieval.visual_build import VisualIndexBuilder
from knowledge_scope.retrieval.visual_index import (
    VISUAL_INDEX_SCHEMA_VERSION,
    QdrantVisualStore,
    VisualRepresentationRef,
    VisualSearchResult,
    VisualVectorPayload,
    visual_collection_name,
    visual_encoder_fingerprint,
    visual_point_id,
)
from knowledge_scope.retrieval.visual_policy import decide_image_index
from knowledge_scope.shared.config import Settings

_PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
    b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


@dataclass
class _FakeContract:
    schema_version: str = "1.0"
    model_id: str = "fake/visual-embedding"
    model_path: str = ""
    dtype: str = "float32"
    normalize: bool = True
    dimension: int = 4
    query_prompt: str = "query prompt"

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "model_path": self.model_path,
            "dtype": self.dtype,
            "normalize": self.normalize,
            "dimension": self.dimension,
            "query_prompt": self.query_prompt,
        }


class _FakeEncoder:
    """Deterministic tiny encoder; visual relevance = first byte / 255."""

    vram_peak_mb = None

    def __init__(self) -> None:
        self.contract = _FakeContract()

    def encode_images(self, image_bytes: list[bytes]):
        return [self._vector(payload) for payload in image_bytes], None

    def encode_texts(self, texts: list[str]):
        return [self._vector_text(text) for text in texts], None

    def encode_query(self, query: str):
        vector = self._vector_text(query)
        return vector, None

    @staticmethod
    def _vector(payload: bytes) -> list[float]:
        value = payload[0] / 255.0 if payload else 0.0
        return [value, 1.0 - value, 0.0, 0.0]

    @staticmethod
    def _vector_text(text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [digest[0] / 255.0, 0.5, 0.0, 0.0]


def _settings(data_dir: Path) -> Settings:
    return Settings(
        _env_file=None,
        environment="test",
        data_dir=str(data_dir),
        visual_embedding_enabled=True,
        visual_model_path="fake-model",
        visual_collection_name="knowledgescope_visual_v1",
    )


def _payload(
    *,
    knowledge_base_id: UUID,
    document_id: UUID,
    evidence_id: str,
    page_number: int = 2,
    asset_index: int = 0,
    asset_ref: str = "doc/auto/images/figure.png",
    asset_bytes: bytes = b"\x89PNG\r\n\x1a\nA",
    caption: str | None = "设备结构示意图",
) -> tuple[VisualVectorPayload, bytes]:
    plain = b"x" * 20
    representation_id = representation_id_for(
        evidence_id,
        "image",
        "caption",
        content=caption or "",
        reference=None,
    )
    payload = VisualVectorPayload(
        schema_version=VISUAL_INDEX_SCHEMA_VERSION,
        collection_fingerprint="0" * 64,
        visual_branch="image",
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
        evidence_id=evidence_id,
        page_number=page_number,
        source_block_ids=["p2-b1"],
        asset_index=asset_index,
        asset_ref=asset_ref,
        asset_sha256=hashlib.sha256(asset_bytes).hexdigest(),
        asset_mime="image/png",
        image_width=800,
        image_height=600,
        caption=caption,
        context="页面说明文字",
        representation_ids=[representation_id],
        representations=[],
        policy_mode="text_and_visual",
        policy_reason="测试",
        policy_version="visual-index-policy-v1",
        model_id="fake/visual-embedding",
        model_fingerprint="0" * 64,
        embedding_dim=4,
        indexed_at=datetime.now(UTC),
    )
    return payload, plain


def test_visual_collection_name_is_model_fingerprinted() -> None:
    settings = _settings(Path("/tmp/m7-unused"))
    contract = _FakeContract()
    name = visual_collection_name(settings, contract)
    assert name != settings.visual_collection_name
    assert len(name) <= 63
    assert visual_encoder_fingerprint(contract)[:10] in name
    other = _FakeContract(model_id="other/model")
    assert visual_collection_name(settings, other) != name


def test_visual_point_id_is_idempotent_and_tracks_asset_bytes() -> None:
    kb, doc = UUID(int=1), UUID(int=2)
    first, _ = _payload(
        knowledge_base_id=kb, document_id=doc, evidence_id="evidence_v1_" + "a" * 64
    )
    second, _ = _payload(
        knowledge_base_id=kb,
        document_id=doc,
        evidence_id="evidence_v1_" + "a" * 64,
        asset_bytes=b"\x89PNG\r\n\x1a\nB",
    )
    assert visual_point_id(first) == visual_point_id(first)
    assert visual_point_id(first) != visual_point_id(second)


def test_policy_is_conservative_for_complex_images() -> None:
    assert (
        decide_image_index(
            caption=None, context="上下文", image_width=100, image_height=100, page_image_count=1
        ).mode
        == "text_and_visual"
    )
    assert (
        decide_image_index(
            caption="图 1", context="上下文", image_width=100, image_height=100, page_image_count=1
        ).mode
        == "text_and_visual"
    )
    assert (
        decide_image_index(
            caption="森林中的树木和地面植被覆盖情况记录",
            context=(
                "本章介绍森林生态系统的组成与功能。阅读材料描述了不同区域的植被覆盖情况，"
                "并说明了观测记录的方法与注意事项，提供了详细的文字说明与数据支撑。"
                "读者可以根据正文中的描述理解该对象的主体内容，无需依赖视觉细节。"
            ),
            image_width=400,
            image_height=300,
            page_image_count=1,
        ).mode
        == "text_only"
    )
    decision = decide_image_index(
        caption="季风气候统计图",
        context="上下文",
        image_width=100,
        image_height=100,
        page_image_count=1,
    )
    assert decision.mode == "text_and_visual"
    assert "视觉" in decision.reason or "标题" in decision.reason


def test_visual_store_roundtrip_and_scope_filtering(tmp_path: Path) -> None:
    from qdrant_client import QdrantClient

    settings = _settings(tmp_path)
    encoder = _FakeEncoder()
    store = QdrantVisualStore(settings, encoder, client=QdrantClient(location=":memory:"))
    kb_a, kb_b, doc = UUID(int=11), UUID(int=12), UUID(int=13)
    entry_a, bytes_a = _payload(
        knowledge_base_id=kb_a,
        document_id=doc,
        evidence_id="evidence_v1_" + "1" * 64,
        asset_bytes=b"\x00",
    )
    entry_b, bytes_b = _payload(
        knowledge_base_id=kb_b,
        document_id=doc,
        evidence_id="evidence_v1_" + "2" * 64,
        asset_bytes=b"\xff",
    )
    store.ensure_collection()
    entry_a = entry_a.model_copy(
        update={
            "collection_fingerprint": store.collection_fingerprint,
            "model_fingerprint": store.collection_fingerprint,
        }
    )
    entry_b = entry_b.model_copy(
        update={
            "collection_fingerprint": store.collection_fingerprint,
            "model_fingerprint": store.collection_fingerprint,
        }
    )
    assert store.replace_document(doc, [(entry_a, bytes_a), (entry_b, bytes_b)]) == 2

    result = store.search("quarterly chart", kb_a, top_k=5)
    assert len(result.items) == 1
    assert result.items[0].payload.evidence_id == entry_a.evidence_id
    assert result.items[0].payload.knowledge_base_id == kb_a
    assert store.readiness()["points_count"] == 2

    filter_result = store.search("quarterly chart", kb_b, top_k=5, document_id=doc)
    assert len(filter_result.items) == 1
    assert filter_result.items[0].payload.knowledge_base_id == kb_b


def test_store_deletes_stale_points_when_document_rebuilt(tmp_path: Path) -> None:
    from qdrant_client import QdrantClient

    settings = _settings(tmp_path)
    encoder = _FakeEncoder()
    store = QdrantVisualStore(settings, encoder, client=QdrantClient(location=":memory:"))
    doc = UUID(int=21)
    entry, payload_bytes = _payload(
        knowledge_base_id=UUID(int=22), document_id=doc, evidence_id="evidence_v1_" + "3" * 64
    )
    store.ensure_collection()
    entry = entry.model_copy(
        update={
            "collection_fingerprint": store.collection_fingerprint,
            "model_fingerprint": store.collection_fingerprint,
        }
    )
    store.replace_document(doc, [(entry, payload_bytes)])
    assert store.readiness()["points_count"] == 1
    store.replace_document(doc, [])
    assert store.readiness()["points_count"] == 0


def _evidence_document(
    *,
    canonical,
    knowledge_base_id: UUID,
    context_by_block_id: dict[str, str] | None = None,
) -> MultimodalEvidenceDocument:
    from knowledge_scope.evidence.service import build_evidence_document

    return build_evidence_document(
        canonical,
        knowledge_base_id,
        context_by_block_id=context_by_block_id or {},
    )


def _write_document(
    data_dir: Path,
    *,
    document_id: UUID,
    knowledge_base_id: UUID,
    caption: str | None,
    context: str,
) -> None:
    from knowledge_scope.evidence.service import build_evidence_document
    from knowledge_scope.parsing.models import BoundingBox, CanonicalDocument, Page, TextBlock

    image_path = (
        data_dir / "parsing" / str(document_id) / "mineru" / "auto" / "images" / "figure.png"
    )
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(_PNG_1X1)
    canonical = CanonicalDocument(
        schema_version="1.0",
        document_id=document_id,
        pages=[
            Page(
                page_number=1,
                blocks=[
                    TextBlock(block_id="p1-b1", reading_order=0, text=context),
                    ImageBlock(
                        block_id="p1-b2",
                        reading_order=1,
                        bbox=BoundingBox(x0=0.1, y0=0.2, x1=0.9, y1=0.8),
                        asset_ref="auto/images/figure.png",
                        caption=caption,
                    ),
                ],
            )
        ],
    )
    canonical_path = data_dir / "parsing" / str(document_id) / "canonical.json"
    canonical_path.write_text(canonical.model_dump_json(indent=2) + "\n")
    artifact = build_evidence_document(
        canonical,
        knowledge_base_id,
        context_by_block_id={"p1-b2": context},
    )
    write_evidence_artifact(data_dir, artifact)


def test_builder_indexes_visual_assets_and_skips_text_only_in_selective_mode(
    tmp_path: Path,
) -> None:
    from qdrant_client import QdrantClient

    settings = _settings(tmp_path)
    encoder = _FakeEncoder()
    store = QdrantVisualStore(settings, encoder, client=QdrantClient(location=":memory:"))
    kb = UUID(int=31)
    visual_doc = UUID(int=32)
    text_only_doc = UUID(int=33)
    _write_document(
        tmp_path,
        document_id=visual_doc,
        knowledge_base_id=kb,
        caption="季风气候统计图",
        context="该图展示了2023年四个季度的销售变化趋势与峰值情况，用于辅助分析。",
    )
    _write_document(
        tmp_path,
        document_id=text_only_doc,
        knowledge_base_id=kb,
        caption="森林中的树木和地面植被覆盖情况记录",
        context=(
            "本章介绍森林生态系统的组成与功能。阅读材料描述了不同区域的植被覆盖情况，"
            "并说明了观测记录的方法与注意事项，帮助读者在不查看原始画面的情况下"
            "理解该对象的主体内容、相关背景知识以及本节的学习目标。"
        ),
    )

    stats = VisualIndexBuilder(settings, store).build(kb, mode="selective")
    assert stats.documents_scanned == 2
    assert stats.images_total == 2
    assert stats.images_skipped_by_policy == 1
    assert stats.images_indexed == 1
    assert stats.document_ids == [str(visual_doc)]
    assert store.readiness()["points_count"] == 1

    all_stats = VisualIndexBuilder(settings, store).build(kb, mode="all")
    assert all_stats.images_indexed == 2
    assert store.readiness()["points_count"] == 2


def test_builder_reports_missing_assets_without_fake_vectors(tmp_path: Path) -> None:
    from qdrant_client import QdrantClient

    settings = _settings(tmp_path)
    encoder = _FakeEncoder()
    store = QdrantVisualStore(settings, encoder, client=QdrantClient(location=":memory:"))
    kb = UUID(int=41)
    doc = UUID(int=42)
    _write_document(
        tmp_path,
        document_id=doc,
        knowledge_base_id=kb,
        caption="季风气候统计图",
        context="该图展示了2023年四个季度的销售变化趋势与峰值情况，用于辅助分析。",
    )
    (tmp_path / "parsing" / str(doc) / "mineru" / "auto" / "images" / "figure.png").unlink()
    stats = VisualIndexBuilder(settings, store).build(kb, mode="all")
    assert stats.images_missing == 1
    assert stats.images_indexed == 0
    assert store.readiness()["points_count"] == 0


def test_unified_visual_branch_and_fusion_quota() -> None:
    from knowledge_scope.retrieval.representation_index import RepresentationRetrievalResult
    from knowledge_scope.retrieval.service import RetrievalResult
    from knowledge_scope.retrieval.sparse import SparseSearchResponse

    kb, doc = UUID(int=51), UUID(int=52)
    evidence_id = "evidence_v1_" + "5" * 64
    representation_id = representation_id_for(
        evidence_id,
        "image",
        "caption",
        content="季风气候统计图",
        reference=None,
    )
    hit_payload = VisualVectorPayload(
        schema_version=VISUAL_INDEX_SCHEMA_VERSION,
        collection_fingerprint="0" * 64,
        visual_branch="image",
        knowledge_base_id=kb,
        document_id=doc,
        evidence_id=evidence_id,
        page_number=2,
        source_block_ids=["p2-b1"],
        asset_index=0,
        asset_ref="doc/auto/images/figure.png",
        asset_sha256="a" * 64,
        asset_mime="image/png",
        image_width=800,
        image_height=600,
        caption="季风气候统计图",
        context="图表",
        representation_ids=[representation_id],
        representations=[
            VisualRepresentationRef(
                representation_id=representation_id,
                representation_type="caption",
                content="季风气候统计图",
                reference=None,
            )
        ],
        policy_mode="text_and_visual",
        policy_reason="测试",
        policy_version="visual-index-policy-v1",
        model_id="fake/visual-embedding",
        model_fingerprint="0" * 64,
        embedding_dim=4,
        indexed_at=datetime.now(UTC),
    )
    from knowledge_scope.retrieval.visual_index import VisualSearchHit

    class _Visual:
        def search(self, query, knowledge_base_id, *, top_k, document_id=None):
            return VisualSearchResult(
                query=query,
                knowledge_base_id=knowledge_base_id,
                collection_name="visual",
                collection_fingerprint="0" * 64,
                model_id="fake/visual-embedding",
                embedding_dim=4,
                items=[VisualSearchHit(rank=1, score=0.99, point_id="p1", payload=hit_payload)],
                latency_ms=1.0,
            )

    class _Dense:
        def search(self, query, *, limit, knowledge_base_id, document_id=None):
            return RetrievalResult(query, limit, "dense", "chunks", [])

    class _Sparse:
        def search(self, query, *, knowledge_base_id, top_k, document_id=None):
            return SparseSearchResponse(
                query=query,
                knowledge_base_id=knowledge_base_id,
                document_id=document_id,
                top_k=top_k,
                index_fingerprint="b" * 64,
                items=[],
            )

    class _Graph:
        def search(self, query, knowledge_base_id, *, document_id=None):
            from knowledge_scope.graph.retrieval import GraphRetrievalResult

            return GraphRetrievalResult(query=query, knowledge_base_id=knowledge_base_id, items=[])

    class _Multimodal:
        def search(self, query, *, knowledge_base_id, top_k, document_id=None):
            return RepresentationRetrievalResult(
                query=query,
                knowledge_base_id=knowledge_base_id,
                items=[],
                raw_hits=[],
            )

    class _Reranker:
        model_id = "BAAI/bge-reranker-v2-m3"

        def score_pairs(self, _query, passages):
            return [0.0 for _ in passages]

    class _Lookup:
        def get_chunk_payload(self, *, knowledge_base_id, document_id, chunk_id):
            return None

    service = UnifiedRetrievalService(
        _Dense(),
        _Sparse(),
        _Graph(),
        _Multimodal(),
        _Reranker(),
        chunk_lookup=_Lookup(),
        visual_retrieval=_Visual(),
        config=UnifiedRetrievalConfig(result_limit=2, visual_fusion_quota=1),
    )
    result = asyncio.run(service.search("最低季度销售额", kb))
    assert [branch.branch for branch in result.branches].count("visual") == 1
    assert result.source_distribution["visual"] == 1
    assert result.items[0].evidence_id == evidence_id
    assert result.visual_intent == "ambiguous"
    assert result.visual_quota_applied == 1


def test_visual_search_without_index_returns_empty_without_creating_collection(
    tmp_path: Path,
) -> None:
    from qdrant_client import QdrantClient

    settings = _settings(tmp_path)
    client = QdrantClient(location=":memory:")
    store = QdrantVisualStore(settings, _FakeEncoder(), client=client)
    result = store.search("quarterly chart", UUID(int=31), top_k=5)
    assert result.items == []
    assert result.collection_missing is True
    assert client.get_collections().collections == []


def _store(tmp_path: Path) -> QdrantVisualStore:
    from qdrant_client import QdrantClient

    store = QdrantVisualStore(
        _settings(tmp_path), _FakeEncoder(), client=QdrantClient(location=":memory:")
    )
    store.ensure_collection()
    return store


def _fit(store: QdrantVisualStore, entry: VisualVectorPayload) -> VisualVectorPayload:
    return entry.model_copy(
        update={
            "collection_fingerprint": store.collection_fingerprint,
            "model_fingerprint": store.collection_fingerprint,
        }
    )


def _two_generations(
    store: QdrantVisualStore,
) -> tuple[UUID, UUID, VisualVectorPayload, bytes, VisualVectorPayload, bytes]:
    kb, doc = UUID(int=41), UUID(int=42)
    old, old_bytes = _payload(
        knowledge_base_id=kb,
        document_id=doc,
        evidence_id="evidence_v1_" + "4" * 64,
        asset_bytes=b"\x01",
    )
    new, new_bytes = _payload(
        knowledge_base_id=kb,
        document_id=doc,
        evidence_id="evidence_v1_" + "5" * 64,
        asset_bytes=b"\x02",
    )
    return kb, doc, _fit(store, old), old_bytes, _fit(store, new), new_bytes


def test_replace_document_encode_failure_keeps_previous_generation(
    tmp_path: Path, monkeypatch
) -> None:
    import pytest

    from knowledge_scope.retrieval.visual_index import VisualIndexError

    store = _store(tmp_path)
    kb, doc, old, old_bytes, new, new_bytes = _two_generations(store)
    store.replace_document(doc, [(old, old_bytes)])

    def fail_encode(_images):
        raise RuntimeError("encoder crashed")

    monkeypatch.setattr(store.encoder, "encode_images", fail_encode)
    with pytest.raises(VisualIndexError, match="旧索引保持不变"):
        store.replace_document(doc, [(new, new_bytes)])

    result = store.search("quarterly chart", kb, top_k=5)
    assert len(result.items) == 1
    assert result.items[0].payload.evidence_id == old.evidence_id


def test_replace_document_upsert_failure_keeps_previous_generation(
    tmp_path: Path, monkeypatch
) -> None:
    import pytest

    from knowledge_scope.retrieval.visual_index import VisualIndexError

    store = _store(tmp_path)
    kb, doc, old, old_bytes, new, new_bytes = _two_generations(store)
    store.replace_document(doc, [(old, old_bytes)])

    client = store._get_client()
    original_upsert = client.upsert

    def fail_data_upsert(*args, **kwargs):
        for point in kwargs.get("points", ()):
            if (getattr(point, "payload", None) or {}).get("kind") != "generation_pointer":
                raise RuntimeError("qdrant write failed")
        return original_upsert(*args, **kwargs)

    monkeypatch.setattr(client, "upsert", fail_data_upsert)
    with pytest.raises(VisualIndexError, match="旧索引保持不变"):
        store.replace_document(doc, [(new, new_bytes)])
    monkeypatch.undo()

    result = store.search("quarterly chart", kb, top_k=5)
    assert len(result.items) == 1
    assert result.items[0].payload.evidence_id == old.evidence_id


def test_partial_switch_crash_hides_pending_generation_until_recovery(tmp_path: Path) -> None:
    from qdrant_client import models

    store = _store(tmp_path)
    kb, doc, old, old_bytes, new, new_bytes = _two_generations(store)
    store.replace_document(doc, [(old, old_bytes)])

    pending = new.model_copy(update={"generation": "a" * 32, "active": False})
    vector, _stats = store.encoder.encode_images([new_bytes])
    store._get_client().upsert(
        collection_name=store.collection_name,
        points=[
            models.PointStruct(
                id=str(visual_point_id(pending)),
                vector=vector[0],
                payload=pending.model_dump(mode="json"),
            )
        ],
        wait=True,
    )

    result = store.search("quarterly chart", kb, top_k=5)
    assert len(result.items) == 1
    assert result.items[0].payload.evidence_id == old.evidence_id

    assert store.recover_document(doc) is not None
    result = store.search("quarterly chart", kb, top_k=5)
    assert len(result.items) == 1
    assert result.items[0].payload.evidence_id == old.evidence_id
    remaining = [
        record
        for record in store._scroll_records(document_id=doc)
        if isinstance(record.payload, dict) and record.payload.get("kind") != "generation_pointer"
    ]
    assert len(remaining) == 1


def test_switch_failure_rolls_back_to_previous_generation(tmp_path: Path, monkeypatch) -> None:
    import pytest

    from knowledge_scope.retrieval.visual_index import VisualIndexError

    store = _store(tmp_path)
    kb, doc, old, old_bytes, new, new_bytes = _two_generations(store)
    store.replace_document(doc, [(old, old_bytes)])

    def fail_cleanup(document_id, *, keep):
        raise RuntimeError("switch failed")

    monkeypatch.setattr(store, "_delete_other_generations", fail_cleanup)
    with pytest.raises(VisualIndexError, match="已回滚到旧索引"):
        store.replace_document(doc, [(new, new_bytes)])
    monkeypatch.undo()

    result = store.search("quarterly chart", kb, top_k=5)
    assert len(result.items) == 1
    assert result.items[0].payload.evidence_id == old.evidence_id


def test_backend_unavailable_is_not_reported_as_missing_collection(
    tmp_path: Path, monkeypatch
) -> None:
    import pytest

    from knowledge_scope.retrieval.visual_index import VisualIndexError

    store = _store(tmp_path)

    def fail_collection_info():
        raise ConnectionError("qdrant down")

    monkeypatch.setattr(store, "_collection_info", fail_collection_info)
    with pytest.raises(VisualIndexError, match="视觉检索后端不可用"):
        store.search("quarterly chart", UUID(int=41), top_k=5)


def test_unified_visual_quota_respects_query_intent() -> None:
    from knowledge_scope.graph.retrieval import GraphRetrievalResult
    from knowledge_scope.retrieval.service import (
        RetrievalResult,
        RetrievedChunk,
    )
    from knowledge_scope.retrieval.sparse import SparseSearchResponse
    from knowledge_scope.retrieval.unified import UnifiedRetrievalConfig, UnifiedRetrievalService

    kb = UUID(int=51)
    doc = UUID(int=52)

    def _chunk(index: int) -> RetrievedChunk:
        from knowledge_scope.retrieval.qdrant import (
            QDRANT_COLLECTION_SCHEMA_VERSION,
            ChunkVectorPayload,
        )

        payload = ChunkVectorPayload(
            collection_schema_version=QDRANT_COLLECTION_SCHEMA_VERSION,
            chunk_id=f"chunk-{index}",
            document_id=doc,
            knowledge_base_id=kb,
            page_start=1,
            page_end=1,
            source_block_ids=[f"p1-b{index + 1}"],
            section_path=["第一章"],
            content_types=["text"],
            asset_refs=[],
            text=f"chunk-{index} 表格之外的正文说明",
            chunking_config_fingerprint="a" * 64,
            embedding_model="Qwen/Qwen3-Embedding-0.6B",
            embedding_model_revision="revision",
            embedding_config_fingerprint="b" * 64,
        )
        return RetrievedChunk(point_id=UUID(int=100 + index), score=0.5, payload=payload)

    def _visual_hit(index: int):
        payload, _bytes = _payload(
            knowledge_base_id=kb,
            document_id=doc,
            evidence_id="evidence_v1_" + ("f" if index == 0 else "e") * 64,
            asset_ref=f"doc/auto/images/figure-{index}.png",
        )
        from knowledge_scope.retrieval.visual_index import (
            VisualRepresentationRef,
            VisualSearchHit,
        )

        payload = payload.model_copy(
            update={
                "representations": [
                    VisualRepresentationRef(
                        representation_id=payload.representation_ids[0],
                        representation_type="caption",
                        content=payload.caption or "",
                        reference=None,
                    )
                ]
            }
        )
        return VisualSearchHit(rank=index + 1, score=0.9, point_id=f"v{index}", payload=payload)

    class _Dense:
        def search(self, query, *, limit, knowledge_base_id, document_id=None):
            return RetrievalResult(
                query, limit, "dense", "chunks", [_chunk(0), _chunk(1), _chunk(2)]
            )

    class _Sparse:
        def search(self, query, *, knowledge_base_id, top_k, document_id=None):
            return SparseSearchResponse(
                query=query,
                knowledge_base_id=knowledge_base_id,
                document_id=document_id,
                top_k=top_k,
                index_fingerprint="b" * 64,
                items=[],
            )

    class _Graph:
        def search(self, query, knowledge_base_id, *, document_id=None):
            return GraphRetrievalResult(query=query, knowledge_base_id=knowledge_base_id, items=[])

    class _Multimodal:
        def search(self, query, *, knowledge_base_id, top_k, document_id=None):
            from knowledge_scope.retrieval.representation_index import (
                RepresentationRetrievalResult,
            )

            return RepresentationRetrievalResult(
                query=query, knowledge_base_id=knowledge_base_id, items=[], raw_hits=[]
            )

    class _Reranker:
        model_id = "BAAI/bge-reranker-v2-m3"

        def score_pairs(self, _query, passages):
            result = []
            for passage in passages:
                score = 0.01
                for key, value in {"chunk-0": 0.9, "chunk-1": 0.5, "chunk-2": 0.4}.items():
                    if key in passage:
                        score = value
                        break
                result.append(score)
            return result

    class _Lookup:
        def get_chunk_payload(self, *, knowledge_base_id, document_id, chunk_id):
            return None

    class _Visual:
        def search(self, query, knowledge_base_id, *, top_k, document_id=None):
            from knowledge_scope.retrieval.visual_index import VisualSearchResult

            return VisualSearchResult(
                query=query,
                knowledge_base_id=knowledge_base_id,
                collection_name="visual",
                collection_fingerprint="0" * 64,
                model_id="fake/visual-embedding",
                embedding_dim=4,
                items=[_visual_hit(0), _visual_hit(1)],
                latency_ms=1.0,
            )

    def _run(query: str):
        service = UnifiedRetrievalService(
            _Dense(),
            _Sparse(),
            _Graph(),
            _Multimodal(),
            _Reranker(),
            chunk_lookup=_Lookup(),
            visual_retrieval=_Visual(),
            config=UnifiedRetrievalConfig(
                result_limit=3,
                visual_fusion_quota=2,
                visual_quota_ambiguous=1,
            ),
        )
        return asyncio.run(service.search(query, kb))

    plain = _run("光合作用的过程是什么")
    assert plain.visual_intent == "plain"
    assert plain.visual_quota_applied == 0
    assert plain.source_distribution["visual"] == 0

    explicit = _run("分析图中的柱状图，指出最低季度的具体销售额趋势")
    assert explicit.visual_intent == "explicit"
    assert explicit.visual_quota_applied == 2
    assert explicit.source_distribution["visual"] == 2
    assert explicit.items[0].chunk_id == "chunk-0"

    ambiguous = _run("2024 年越南 GDP 的具体数值是多少")
    assert ambiguous.visual_intent == "ambiguous"
    assert ambiguous.visual_quota_applied == 1
    assert ambiguous.source_distribution["visual"] == 1


def test_crash_before_pointer_switch_keeps_old_generation_visible(tmp_path: Path) -> None:
    """A staged-but-uncommitted generation must not become visible without a pointer."""
    from qdrant_client import models

    store = _store(tmp_path)
    kb, doc, old, old_bytes, new, new_bytes = _two_generations(store)
    store.replace_document(doc, [(old, old_bytes)])
    pointer_before = store._read_pointer(doc)
    assert pointer_before is not None

    pending_generation = "b" * 32
    pending = new.model_copy(update={"generation": pending_generation, "active": False})
    vector, _stats = store.encoder.encode_images([new_bytes])
    store._get_client().upsert(
        collection_name=store.collection_name,
        points=[
            models.PointStruct(
                id=str(visual_point_id(pending)),
                vector=vector[0],
                payload=pending.model_dump(mode="json"),
            )
        ],
        wait=True,
    )

    result = store.search("quarterly chart", kb, top_k=5)
    assert [item.payload.evidence_id for item in result.items] == [old.evidence_id]
    assert store._read_pointer(doc)["active_generation"] == (pointer_before["active_generation"])

    recovered = store.recover_document(doc)
    assert recovered == pointer_before["active_generation"]
    remaining = [
        record
        for record in store._scroll_records(document_id=doc)
        if isinstance(record.payload, dict) and record.payload.get("kind") != "generation_pointer"
    ]
    assert len(remaining) == 1
    assert remaining[0].payload["evidence_id"] == old.evidence_id


def test_crash_after_pointer_switch_exposes_new_generation_before_active_flag(
    tmp_path: Path,
) -> None:
    """The pointer is authoritative: new generation is visible even if active flags lag."""
    from qdrant_client import models

    store = _store(tmp_path)
    kb, doc, old, old_bytes, new, new_bytes = _two_generations(store)
    store.replace_document(doc, [(old, old_bytes)])

    generation = "c" * 32
    pending = new.model_copy(update={"generation": generation, "active": False})
    vector, _stats = store.encoder.encode_images([new_bytes])
    store._get_client().upsert(
        collection_name=store.collection_name,
        points=[
            models.PointStruct(
                id=str(visual_point_id(pending)),
                vector=vector[0],
                payload=pending.model_dump(mode="json"),
            )
        ],
        wait=True,
    )
    store._write_pointer(doc, kb, generation, point_count=1)

    result = store.search("quarterly chart", kb, top_k=5)
    assert [item.payload.evidence_id for item in result.items] == [new.evidence_id]

    recovered = store.recover_document(doc)
    assert recovered == generation
    flags = [
        record.payload.get("active")
        for record in store._scroll_records(document_id=doc)
        if isinstance(record.payload, dict) and record.payload.get("kind") != "generation_pointer"
    ]
    assert flags == [True]
    result = store.search("quarterly chart", kb, top_k=5)
    assert [item.payload.evidence_id for item in result.items] == [new.evidence_id]


def test_first_build_pending_residue_is_invisible_and_recovered(tmp_path: Path) -> None:
    """Crash during the very first build: pending points must never be served."""
    from qdrant_client import models

    store = _store(tmp_path)
    kb, doc, _old, _old_bytes, new, new_bytes = _two_generations(store)
    pending = new.model_copy(update={"generation": "d" * 32, "active": False})
    vector, _stats = store.encoder.encode_images([new_bytes])
    store._get_client().upsert(
        collection_name=store.collection_name,
        points=[
            models.PointStruct(
                id=str(visual_point_id(pending)),
                vector=vector[0],
                payload=pending.model_dump(mode="json"),
            )
        ],
        wait=True,
    )

    result = store.search("quarterly chart", kb, top_k=5)
    assert result.items == []
    assert store.recover_document(doc) is None
    remaining = store._scroll_records(document_id=doc)
    assert remaining == []


def test_concurrent_query_during_switch_sees_old_then_new_never_empty(
    tmp_path: Path, monkeypatch
) -> None:
    """Queries racing with the switch always see one complete generation."""
    store = _store(tmp_path)
    kb, doc, old, old_bytes, new, new_bytes = _two_generations(store)
    store.replace_document(doc, [(old, old_bytes)])

    observed: list[tuple[str, str]] = []
    original_encode = store.encoder.encode_images
    original_set_active = store._set_generation_active

    def encode_with_probe(images):
        vectors = original_encode(images)
        result = store.search("quarterly chart", kb, top_k=5)
        observed.append(("before_pointer", result.items[0].payload.evidence_id))
        return vectors

    def set_active_with_probe(*args, **kwargs):
        # The pointer already points at the new generation here, but its
        # ``active`` flag has not been flipped yet: this is the historical
        # visibility-gap window.
        result = store.search("quarterly chart", kb, top_k=5)
        observed.append(("pointer_switched_active_pending", result.items[0].payload.evidence_id))
        original_set_active(*args, **kwargs)

    monkeypatch.setattr(store.encoder, "encode_images", encode_with_probe)
    monkeypatch.setattr(store, "_set_generation_active", set_active_with_probe)
    store.replace_document(doc, [(new, new_bytes)])
    monkeypatch.undo()

    # Encoding happens before the pointer moves: the probe there must see the
    # old generation; the final flag flip switches queries to the new one.
    # Recovery may normalize the old generation first, so intermediate probes
    # may legitimately see either complete generation — never an empty window.
    assert observed[0] == ("before_pointer", old.evidence_id)
    assert observed[-1] == ("pointer_switched_active_pending", new.evidence_id)
    assert {evidence for _label, evidence in observed} <= {old.evidence_id, new.evidence_id}
    result = store.search("quarterly chart", kb, top_k=5)
    assert [item.payload.evidence_id for item in result.items] == [new.evidence_id]


def test_search_ignores_dormant_generations_when_window_is_oversampled(
    tmp_path: Path,
) -> None:
    """Dormant older generations must not crowd a valid new generation out of the window."""
    store = _store(tmp_path)
    kb, doc, old, old_bytes, new, new_bytes = _two_generations(store)
    dormant_entries = []
    for index in range(8):
        entry, asset = _payload(
            knowledge_base_id=kb,
            document_id=doc,
            evidence_id="evidence_v1_" + f"{index + 6:064x}",
            asset_bytes=bytes([index + 1]),
        )
        dormant_entries.append((_fit(store, entry), asset))
    store.replace_document(doc, [*dormant_entries, (old, old_bytes)])

    # Simulate a switch where the new generation has a single point but the
    # dormant generation has many points with equally high scores.
    generation = "e" * 32
    pending = new.model_copy(update={"generation": generation, "active": False})
    vector, _stats = store.encoder.encode_images([new_bytes])
    from qdrant_client import models

    store._get_client().upsert(
        collection_name=store.collection_name,
        points=[
            models.PointStruct(
                id=str(visual_point_id(pending)),
                vector=vector[0],
                payload=pending.model_dump(mode="json"),
            )
        ],
        wait=True,
    )
    store._write_pointer(doc, kb, generation, point_count=1)

    result = store.search("quarterly chart", kb, top_k=2)
    assert [item.payload.evidence_id for item in result.items] == [new.evidence_id]
