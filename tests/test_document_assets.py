"""Evidence-centric original asset access tests for the multimodal preview chain."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest
from httpx import AsyncClient

from knowledge_scope.documents.assets import (
    MAX_ASSET_BYTES,
    AssetNotFoundError,
    AssetTooLargeError,
    AssetUnsupportedError,
    _resolve_asset_path,
    get_evidence,
    read_evidence_asset_bytes,
    resolve_evidence_asset,
)
from knowledge_scope.evidence.lifecycle import (
    evidence_artifact_path,
    write_evidence_artifact,
)
from knowledge_scope.evidence.models import MultimodalEvidenceDocument
from knowledge_scope.evidence.service import build_evidence_document
from knowledge_scope.parsing.models import (
    CanonicalBlock,
    CanonicalDocument,
    FormulaBlock,
    ImageBlock,
    Page,
    TableBlock,
    TextBlock,
)
from knowledge_scope.parsing.service import PARSING_DIRECTORY_NAME
from knowledge_scope.shared.config import Settings

VALID_PDF = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<<>>\n%%EOF\n"
SECOND_PDF = VALID_PDF.replace(b"1 0 obj", b"2 0 obj")
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 24
GIF_BYTES = b"GIF89a" + b"\x00" * 24
DOCUMENT_NAME = "sample-textbook"
IMAGE_ASSET_REF = f"{DOCUMENT_NAME}/auto/images/asset-0.png"
TABLE_ASSET_REF = f"{DOCUMENT_NAME}/auto/images/asset-1.jpg"
IMAGE_CAPTION = "图 5-14 光合作用过程的示意图"
TABLE_CAPTION = "表 5-1 光反应与暗反应比较"
TABLE_MARKDOWN = "| 阶段 | 场所 |\n| --- | --- |\n| 光反应 | 类囊体薄膜 |"
FORMULA_LATEX = r"6CO_2+6H_2O \rightarrow C_6H_{12}O_6+6O_2"


def build_canonical(
    document_id: UUID,
    *,
    text: str = "光合作用的过程包括光反应和暗反应。",
    image_asset_ref: str = IMAGE_ASSET_REF,
    table_asset_ref: str = TABLE_ASSET_REF,
) -> CanonicalDocument:
    """Build one canonical page containing every multimodal block type."""
    blocks: list[CanonicalBlock] = [
        TextBlock(block_id="p1-b1", reading_order=0, text=text),
        ImageBlock(
            block_id="p1-b2",
            reading_order=1,
            asset_ref=image_asset_ref,
            caption=IMAGE_CAPTION,
        ),
        TableBlock(
            block_id="p1-b3",
            reading_order=2,
            markdown=TABLE_MARKDOWN,
            caption=TABLE_CAPTION,
            asset_ref=table_asset_ref,
        ),
        FormulaBlock(block_id="p1-b4", reading_order=3, latex=FORMULA_LATEX),
    ]
    return CanonicalDocument(document_id=document_id, pages=[Page(page_number=1, blocks=blocks)])


def write_parsing_artifact(
    test_data_dir: Path,
    document_id: UUID,
    *,
    canonical: CanonicalDocument | None = None,
    assets: dict[str, bytes] | None = None,
) -> CanonicalDocument:
    """Persist a canonical artifact and its MinerU image files exactly like parsing does."""
    canonical = canonical or build_canonical(document_id)
    document_root = test_data_dir / PARSING_DIRECTORY_NAME / str(document_id)
    document_root.mkdir(parents=True, exist_ok=True)
    (document_root / "canonical.json").write_text(
        canonical.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    asset_payloads = (
        assets
        if assets is not None
        else {
            IMAGE_ASSET_REF: PNG_BYTES,
            TABLE_ASSET_REF: JPEG_BYTES,
        }
    )
    mineru_root = document_root / "mineru"
    for asset_ref, payload in asset_payloads.items():
        asset_path = mineru_root / asset_ref
        asset_path.parent.mkdir(parents=True, exist_ok=True)
        asset_path.write_bytes(payload)
    return canonical


def write_evidence(
    test_data_dir: Path, canonical: CanonicalDocument, knowledge_base_id: str
) -> None:
    """Persist a validated evidence artifact for the current canonical document."""
    artifact = build_evidence_document(canonical, UUID(knowledge_base_id))
    write_evidence_artifact(test_data_dir, artifact)


def evidence_id_for_modality(
    test_data_dir: Path,
    document_id: UUID,
    modality: str = "image",
) -> str:
    """Read the persisted evidence artifact and return one modality's evidence id."""
    artifact = MultimodalEvidenceDocument.model_validate_json(
        evidence_artifact_path(test_data_dir, document_id).read_bytes()
    )
    return next(item.evidence_id for item in artifact.evidence if item.modality == modality)


async def create_knowledge_base(client: AsyncClient, name: str = "素材预览知识库") -> str:
    response = await client.post(
        "/api/v1/knowledge-bases",
        json={"name": name, "description": "M1 evidence asset tests"},
    )
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


async def upload_document(
    client: AsyncClient,
    knowledge_base_id: str,
    *,
    filename: str = "evidence-source.pdf",
    content: bytes = VALID_PDF,
) -> dict[str, object]:
    response = await client.post(
        f"/api/v1/knowledge-bases/{knowledge_base_id}/documents",
        files={"file": (filename, content, "application/pdf")},
    )
    assert response.status_code == 201, response.text
    return dict(response.json())


async def prepare_evidence_document(
    client: AsyncClient,
    test_data_dir: Path,
    *,
    knowledge_base_id: str | None = None,
    filename: str = "evidence-source.pdf",
    content: bytes = VALID_PDF,
) -> tuple[str, dict[str, object]]:
    kb_id = knowledge_base_id or await create_knowledge_base(client)
    document = await upload_document(client, kb_id, filename=filename, content=content)
    document_id = UUID(str(document["id"]))
    canonical = write_parsing_artifact(test_data_dir, document_id)
    write_evidence(test_data_dir, canonical, kb_id)
    return kb_id, document


def evidence_detail_url(kb_id: str, document_id: str, evidence_id: str) -> str:
    return f"/api/v1/knowledge-bases/{kb_id}/documents/{document_id}/evidence/{evidence_id}"


def evidence_asset_url(kb_id: str, document_id: str, evidence_id: str, index: int) -> str:
    return f"{evidence_detail_url(kb_id, document_id, evidence_id)}/assets/{index}"


@pytest.mark.anyio
async def test_image_evidence_detail_exposes_content_without_internal_paths(
    client: AsyncClient,
    test_data_dir: Path,
) -> None:
    kb_id, document = await prepare_evidence_document(client, test_data_dir)
    document_id = str(document["id"])
    evidence_id = evidence_id_for_modality(test_data_dir, UUID(document_id), "image")

    response = await client.get(evidence_detail_url(kb_id, document_id, evidence_id))

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["evidence_id"] == evidence_id
    assert payload["document_id"] == document_id
    assert payload["knowledge_base_id"] == kb_id
    assert payload["modality"] == "image"
    assert payload["page_start"] == 1
    assert payload["page_end"] == 1
    assert payload["source_block_ids"] == ["p1-b2"]
    assert payload["assets"] == [
        {
            "index": 0,
            "media_type": "image/png",
            "size_bytes": len(PNG_BYTES),
            "available": True,
        }
    ]
    representation_types = {item["representation_type"] for item in payload["representations"]}
    assert representation_types == {"caption", "asset_ref"}
    caption = next(
        item for item in payload["representations"] if item["representation_type"] == "caption"
    )
    assert caption["content"] == IMAGE_CAPTION
    assert caption["searchable"] is True
    asset_reference = next(
        item for item in payload["representations"] if item["representation_type"] == "asset_ref"
    )
    assert asset_reference["content"] is None
    assert asset_reference["searchable"] is False
    for internal_fragment in ("auto/images", "mineru", str(test_data_dir)):
        assert internal_fragment not in response.text


@pytest.mark.anyio
async def test_image_asset_endpoint_streams_validated_bytes(
    client: AsyncClient,
    test_data_dir: Path,
) -> None:
    kb_id, document = await prepare_evidence_document(client, test_data_dir)
    document_id = str(document["id"])
    evidence_id = evidence_id_for_modality(test_data_dir, UUID(document_id), "image")

    response = await client.get(evidence_asset_url(kb_id, document_id, evidence_id, 0))

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "private, max-age=0, must-revalidate"
    assert response.content == PNG_BYTES


@pytest.mark.anyio
async def test_table_evidence_detail_returns_structured_markdown_and_asset(
    client: AsyncClient,
    test_data_dir: Path,
) -> None:
    kb_id, document = await prepare_evidence_document(client, test_data_dir)
    document_id = str(document["id"])
    evidence_id = evidence_id_for_modality(test_data_dir, UUID(document_id), "table")

    response = await client.get(evidence_detail_url(kb_id, document_id, evidence_id))

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["modality"] == "table"
    representations = {item["representation_type"]: item for item in payload["representations"]}
    assert representations["markdown"]["content"] == TABLE_MARKDOWN
    assert representations["caption"]["content"] == TABLE_CAPTION
    assert representations["asset_ref"]["searchable"] is False
    assert payload["assets"][0]["available"] is True
    assert payload["assets"][0]["media_type"] == "image/jpeg"


@pytest.mark.anyio
async def test_formula_and_text_evidence_detail_has_no_original_assets(
    client: AsyncClient,
    test_data_dir: Path,
) -> None:
    kb_id, document = await prepare_evidence_document(client, test_data_dir)
    document_id = str(document["id"])

    formula_id = evidence_id_for_modality(test_data_dir, UUID(document_id), "formula")
    formula_response = await client.get(evidence_detail_url(kb_id, document_id, formula_id))
    assert formula_response.status_code == 200, formula_response.text
    formula_payload = formula_response.json()
    assert formula_payload["modality"] == "formula"
    assert formula_payload["assets"] == []
    latex = next(
        item
        for item in formula_payload["representations"]
        if item["representation_type"] == "latex"
    )
    assert latex["content"] == FORMULA_LATEX

    text_id = evidence_id_for_modality(test_data_dir, UUID(document_id), "text")
    text_response = await client.get(evidence_detail_url(kb_id, document_id, text_id))
    assert text_response.status_code == 200, text_response.text
    text_payload = text_response.json()
    assert text_payload["modality"] == "text"
    assert text_payload["assets"] == []
    assert text_payload["representations"][0]["content"] == "光合作用的过程包括光反应和暗反应。"


@pytest.mark.anyio
async def test_evidence_and_asset_are_scoped_to_the_requested_document(
    client: AsyncClient,
    test_data_dir: Path,
) -> None:
    kb_id, first_document = await prepare_evidence_document(client, test_data_dir)
    _, second_document = await prepare_evidence_document(
        client,
        test_data_dir,
        knowledge_base_id=kb_id,
        filename="second-source.pdf",
        content=SECOND_PDF,
    )
    first_id = str(first_document["id"])
    second_id = str(second_document["id"])
    second_evidence_id = evidence_id_for_modality(test_data_dir, UUID(second_id), "image")

    # The evidence belongs to the second document even though both share a KB.
    response = await client.get(evidence_detail_url(kb_id, first_id, second_evidence_id))
    assert response.status_code == 404
    assert response.json()["detail"] == "证据不存在"

    response = await client.get(evidence_asset_url(kb_id, first_id, second_evidence_id, 0))
    assert response.status_code == 404
    assert response.json()["detail"] == "证据不存在"


@pytest.mark.anyio
async def test_asset_access_is_scoped_to_the_knowledge_base(
    client: AsyncClient,
    test_data_dir: Path,
) -> None:
    _first_kb, document = await prepare_evidence_document(client, test_data_dir)
    document_id = str(document["id"])
    evidence_id = evidence_id_for_modality(test_data_dir, UUID(document_id), "image")
    second_kb = await create_knowledge_base(client, name="另一个知识库")

    response = await client.get(evidence_detail_url(second_kb, document_id, evidence_id))
    assert response.status_code == 404
    assert response.json()["detail"] == "文档不存在"

    response = await client.get(evidence_asset_url(second_kb, document_id, evidence_id, 0))
    assert response.status_code == 404
    assert response.json()["detail"] == "文档不存在"


@pytest.mark.anyio
async def test_missing_parsing_artifact_returns_controlled_not_found(
    client: AsyncClient,
    test_data_dir: Path,
) -> None:
    kb_id = await create_knowledge_base(client)
    document = await upload_document(client, kb_id)
    document_id = str(document["id"])
    evidence_id = "evidence_v1_" + "a" * 64

    response = await client.get(evidence_detail_url(kb_id, document_id, evidence_id))
    assert response.status_code == 404
    assert response.json()["detail"] == "文档尚未完成解析"
    assert "mineru" not in response.text

    response = await client.get(evidence_asset_url(kb_id, document_id, evidence_id, 0))
    assert response.status_code == 404


@pytest.mark.anyio
async def test_unknown_or_malformed_evidence_id_is_rejected(
    client: AsyncClient,
    test_data_dir: Path,
) -> None:
    kb_id, document = await prepare_evidence_document(client, test_data_dir)
    document_id = str(document["id"])

    for evidence_id in ("not-an-evidence-id", "evidence_v1_" + "0" * 64):
        response = await client.get(evidence_detail_url(kb_id, document_id, evidence_id))
        assert response.status_code == 404
        assert response.json()["detail"] == "证据不存在"


@pytest.mark.anyio
async def test_historical_missing_asset_reports_unavailable_but_detail_still_renders(
    client: AsyncClient,
    test_data_dir: Path,
) -> None:
    kb_id, document = await prepare_evidence_document(client, test_data_dir)
    document_id = str(document["id"])
    evidence_id = evidence_id_for_modality(test_data_dir, UUID(document_id), "image")
    (test_data_dir / PARSING_DIRECTORY_NAME / document_id / "mineru" / IMAGE_ASSET_REF).unlink()

    detail_response = await client.get(evidence_detail_url(kb_id, document_id, evidence_id))
    assert detail_response.status_code == 200, detail_response.text
    detail_payload = detail_response.json()
    assert detail_payload["assets"] == [
        {"index": 0, "media_type": None, "size_bytes": None, "available": False}
    ]

    asset_response = await client.get(evidence_asset_url(kb_id, document_id, evidence_id, 0))
    assert asset_response.status_code == 404
    assert asset_response.json()["detail"] == "素材文件不存在"
    assert str(test_data_dir) not in asset_response.text


@pytest.mark.anyio
async def test_stale_evidence_is_rejected_instead_of_serving_old_asset(
    client: AsyncClient,
    test_data_dir: Path,
) -> None:
    kb_id, document = await prepare_evidence_document(client, test_data_dir)
    document_id = UUID(str(document["id"]))
    evidence_id = evidence_id_for_modality(test_data_dir, document_id, "image")

    stale_canonical = build_canonical(document_id, text="重新解析后的正文内容已经变化。")
    write_parsing_artifact(test_data_dir, document_id, canonical=stale_canonical)

    response = await client.get(evidence_detail_url(kb_id, str(document_id), evidence_id))
    assert response.status_code == 404
    assert response.json()["detail"] == "证据产物不存在、已过期或不可用"

    response = await client.get(evidence_asset_url(kb_id, str(document_id), evidence_id, 0))
    assert response.status_code == 404


@pytest.mark.anyio
async def test_asset_index_out_of_range_is_rejected(
    client: AsyncClient,
    test_data_dir: Path,
) -> None:
    kb_id, document = await prepare_evidence_document(client, test_data_dir)
    document_id = str(document["id"])
    evidence_id = evidence_id_for_modality(test_data_dir, UUID(document_id), "image")

    response = await client.get(evidence_asset_url(kb_id, document_id, evidence_id, 1))
    assert response.status_code == 404
    assert response.json()["detail"] == "证据素材不存在"


@pytest.mark.anyio
async def test_symlinked_asset_is_not_served(
    client: AsyncClient,
    test_data_dir: Path,
    tmp_path: Path,
) -> None:
    kb_id, document = await prepare_evidence_document(client, test_data_dir)
    document_id = str(document["id"])
    evidence_id = evidence_id_for_modality(test_data_dir, UUID(document_id), "image")
    asset_path = test_data_dir / PARSING_DIRECTORY_NAME / document_id / "mineru" / IMAGE_ASSET_REF
    outside = tmp_path / "outside.png"
    outside.write_bytes(PNG_BYTES)
    asset_path.unlink()
    asset_path.symlink_to(outside)

    response = await client.get(evidence_asset_url(kb_id, document_id, evidence_id, 0))
    assert response.status_code == 404
    assert response.json()["detail"] == "素材文件不可用"


@pytest.mark.anyio
async def test_unsupported_asset_format_is_rejected(
    client: AsyncClient,
    test_data_dir: Path,
) -> None:
    kb_id, document = await prepare_evidence_document(client, test_data_dir)
    document_id = UUID(str(document["id"]))
    gif_ref = f"{DOCUMENT_NAME}/auto/images/asset-0.gif"
    canonical = build_canonical(document_id, image_asset_ref=gif_ref)
    write_parsing_artifact(
        test_data_dir,
        document_id,
        canonical=canonical,
        assets={gif_ref: GIF_BYTES, TABLE_ASSET_REF: JPEG_BYTES},
    )
    write_evidence(test_data_dir, canonical, kb_id)
    evidence_id = evidence_id_for_modality(test_data_dir, document_id, "image")

    detail_response = await client.get(evidence_detail_url(kb_id, str(document_id), evidence_id))
    assert detail_response.status_code == 200, detail_response.text
    assert detail_response.json()["assets"][0]["available"] is False

    asset_response = await client.get(evidence_asset_url(kb_id, str(document_id), evidence_id, 0))
    assert asset_response.status_code == 415
    assert asset_response.json()["detail"] == "素材格式不受支持"


@pytest.mark.anyio
async def test_oversized_asset_is_rejected(
    client: AsyncClient,
    test_data_dir: Path,
) -> None:
    kb_id, document = await prepare_evidence_document(client, test_data_dir)
    document_id = str(document["id"])
    evidence_id = evidence_id_for_modality(test_data_dir, UUID(document_id), "image")
    asset_path = test_data_dir / PARSING_DIRECTORY_NAME / document_id / "mineru" / IMAGE_ASSET_REF
    with asset_path.open("wb") as oversized:
        oversized.write(b"\x89PNG\r\n\x1a\n")
        oversized.seek(MAX_ASSET_BYTES)
        oversized.write(b"\x00")

    response = await client.get(evidence_asset_url(kb_id, document_id, evidence_id, 0))
    assert response.status_code == 413
    assert response.json()["detail"] == "素材文件超过预览大小限制"


def test_asset_path_resolver_rejects_traversal_and_absolute_references(
    test_data_dir: Path,
) -> None:
    document_id = UUID("4c1f0b9e-8f4a-4b26-9d4e-3a2f6b7c8d90")
    write_parsing_artifact(test_data_dir, document_id)
    settings = Settings(_env_file=None, data_dir=test_data_dir)

    for unsafe_ref in (
        "../outside.png",
        "/etc/passwd",
        "a//b.png",
        "a/./b.png",
        "a/../../b.png",
    ):
        with pytest.raises(AssetNotFoundError):
            _resolve_asset_path(settings, document_id, unsafe_ref)


def test_resolved_asset_can_be_read_for_future_vlm_consumers(test_data_dir: Path) -> None:
    document_id = UUID("c7a8b9d0-1e2f-4a3b-8c4d-5e6f70819203")
    write_parsing_artifact(test_data_dir, document_id)
    settings = Settings(_env_file=None, data_dir=test_data_dir)
    document = SimpleNamespace(id=document_id, knowledge_base_id=UUID(int=1))
    write_evidence(test_data_dir, build_canonical(document_id), str(UUID(int=1)))
    evidence_id = evidence_id_for_modality(test_data_dir, document_id, "image")

    asset = resolve_evidence_asset(settings, document, evidence_id, 0)  # type: ignore[arg-type]

    assert asset.media_type == "image/png"
    assert asset.size_bytes == len(PNG_BYTES)
    assert read_evidence_asset_bytes(asset) == PNG_BYTES

    asset.path.unlink()
    with pytest.raises(AssetNotFoundError):
        read_evidence_asset_bytes(asset)


def test_asset_resolver_helpers_reject_unsupported_and_oversized_files(
    test_data_dir: Path,
) -> None:
    document_id = UUID("d8b9c0e1-2f3a-4b5c-9d6e-7f8091a2b304")
    asset_path = write_parsing_artifact(test_data_dir, document_id)
    settings = Settings(_env_file=None, data_dir=test_data_dir)
    document = SimpleNamespace(id=document_id, knowledge_base_id=UUID(int=1))
    write_evidence(test_data_dir, asset_path, str(UUID(int=1)))
    image_path = (
        test_data_dir / PARSING_DIRECTORY_NAME / str(document_id) / "mineru" / IMAGE_ASSET_REF
    )

    image_path.write_bytes(GIF_BYTES)
    with pytest.raises(AssetUnsupportedError):
        _resolve_asset_path(settings, document_id, IMAGE_ASSET_REF)

    image_path.write_bytes(PNG_BYTES[:-4] + b"\x00\x00\x00\x00")
    image_path.unlink()
    with image_path.open("wb") as oversized:
        oversized.write(b"\x89PNG\r\n\x1a\n")
        oversized.seek(MAX_ASSET_BYTES)
        oversized.write(b"\x00")
    with pytest.raises(AssetTooLargeError):
        _resolve_asset_path(settings, document_id, IMAGE_ASSET_REF)

    with pytest.raises(AssetNotFoundError):
        get_evidence(settings, document, "evidence_v1_" + "f" * 64)  # type: ignore[arg-type]


def test_evidence_detail_does_not_leak_asset_reference_paths_via_schema() -> None:
    from knowledge_scope.api.schemas import EvidenceDetailResponse

    fields = set(EvidenceDetailResponse.model_fields)
    assert "asset_refs" not in fields
    assert "asset_path" not in fields
    assert "assets" in fields
