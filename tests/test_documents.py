from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException, Request
from httpx import AsyncClient, Response
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from knowledge_scope.api.documents import (
    _delete_document_vectors,
    delete_document,
    get_document_file,
    list_document_chunks,
)
from knowledge_scope.chunking.models import Chunk, ChunkedDocument
from knowledge_scope.chunking.service import CHUNKING_DIRECTORY_NAME, CHUNKS_ARTIFACT_FILENAME
from knowledge_scope.documents.models import (
    DOCUMENT_MEDIA_TYPE_PDF,
    DOCUMENT_STATUS_REGISTERED,
    DOCUMENT_STORAGE_KIND_EXTERNAL_REFERENCE,
    Document,
)
from knowledge_scope.documents.storage import (
    StorageError,
    resolve_external_source_path,
    storage_key_for_document,
)
from knowledge_scope.evidence.lifecycle import EVIDENCE_DIRECTORY_NAME
from knowledge_scope.knowledge_bases.models import KnowledgeBase
from knowledge_scope.retrieval.qdrant import ChunkVectorPayload

VALID_PDF = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n<<>>\n%%EOF\n"


def write_chunk_artifact(
    test_data_dir: Path,
    document_id: UUID,
    chunks: list[Chunk],
    *,
    page_count: int,
) -> Path:
    """Persist one chunk artifact exactly where the chunking stage writes it."""
    artifact = ChunkedDocument(
        document_id=document_id,
        page_count=page_count,
        config_fingerprint="a" * 64,
        chunks=chunks,
    )
    path = test_data_dir / CHUNKING_DIRECTORY_NAME / str(document_id) / CHUNKS_ARTIFACT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(artifact.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


async def create_knowledge_base(client: AsyncClient, name: str = "测试知识库") -> dict[str, object]:
    response = await client.post("/api/v1/knowledge-bases", json={"name": name})
    assert response.status_code == 201
    return response.json()


async def upload_pdf(
    client: AsyncClient,
    knowledge_base_id: str,
    *,
    filename: str = "manual.pdf",
    content: bytes = VALID_PDF,
    content_type: str = "application/pdf",
) -> Response:
    response = await client.post(
        f"/api/v1/knowledge-bases/{knowledge_base_id}/documents",
        files={"file": (filename, content, content_type)},
    )
    return response


@pytest.mark.anyio
async def test_upload_persists_pdf_metadata_and_uses_fixed_storage_path(
    client: AsyncClient,
    test_data_dir: Path,
) -> None:
    knowledge_base = await create_knowledge_base(client)
    response = await upload_pdf(
        client,
        str(knowledge_base["id"]),
        filename="../原始文件.PDF",
        content_type="text/plain",
    )

    assert response.status_code == 201
    document = response.json()
    document_id = UUID(document["id"])
    knowledge_base_id = UUID(knowledge_base["id"])
    assert set(document) == {
        "id",
        "knowledge_base_id",
        "original_filename",
        "media_type",
        "size_bytes",
        "sha256",
        "status",
        "created_at",
        "updated_at",
    }
    expected_storage_key = storage_key_for_document(knowledge_base_id, document_id)
    expected_path = test_data_dir / expected_storage_key

    assert document["knowledge_base_id"] == str(knowledge_base_id)
    assert document["original_filename"] == "原始文件.PDF"
    assert document["media_type"] == "application/pdf"
    assert document["size_bytes"] == len(VALID_PDF)
    assert document["sha256"] == hashlib.sha256(VALID_PDF).hexdigest()
    assert document["status"] == "uploaded"
    assert expected_path.read_bytes() == VALID_PDF
    assert expected_path.name == "original.pdf"
    assert "原始文件" not in str(expected_path)

    detail_response = await client.get(
        f"/api/v1/knowledge-bases/{knowledge_base_id}/documents/{document_id}"
    )
    list_response = await client.get(f"/api/v1/knowledge-bases/{knowledge_base_id}/documents")
    assert detail_response.status_code == 200
    assert detail_response.json() == document
    assert list_response.status_code == 200
    assert list_response.json() == {
        "items": [document],
        "total": 1,
        "limit": 20,
        "offset": 0,
    }


@pytest.mark.anyio
async def test_duplicate_pdf_is_rejected_only_within_the_same_knowledge_base(
    client: AsyncClient,
    test_data_dir: Path,
) -> None:
    knowledge_base = await create_knowledge_base(client)
    knowledge_base_id = str(knowledge_base["id"])

    first_response = await upload_pdf(client, knowledge_base_id, filename="first.pdf")
    files_after_first_upload = sorted(
        path.relative_to(test_data_dir) for path in test_data_dir.rglob("*") if path.is_file()
    )
    duplicate_response = await upload_pdf(client, knowledge_base_id, filename="renamed.pdf")

    assert first_response.status_code == 201
    assert duplicate_response.status_code == 409
    assert "相同文件" in duplicate_response.json()["detail"]
    assert files_after_first_upload == sorted(
        path.relative_to(test_data_dir) for path in test_data_dir.rglob("*") if path.is_file()
    )

    list_response = await client.get(f"/api/v1/knowledge-bases/{knowledge_base_id}/documents")
    assert list_response.json()["total"] == 1


@pytest.mark.anyio
async def test_same_pdf_can_be_uploaded_to_different_knowledge_bases(
    client: AsyncClient,
) -> None:
    first_knowledge_base = await create_knowledge_base(client, "第一个知识库")
    second_knowledge_base = await create_knowledge_base(client, "第二个知识库")

    first_response = await upload_pdf(client, str(first_knowledge_base["id"]))
    second_response = await upload_pdf(client, str(second_knowledge_base["id"]))

    assert first_response.status_code == 201
    assert second_response.status_code == 201
    assert first_response.json()["sha256"] == second_response.json()["sha256"]
    first_storage_key = storage_key_for_document(
        UUID(first_knowledge_base["id"]), UUID(first_response.json()["id"])
    )
    second_storage_key = storage_key_for_document(
        UUID(second_knowledge_base["id"]), UUID(second_response.json()["id"])
    )
    assert first_storage_key != second_storage_key


@pytest.mark.anyio
async def test_document_list_is_newest_first_and_supports_bounded_pagination(
    client: AsyncClient,
) -> None:
    knowledge_base = await create_knowledge_base(client)
    knowledge_base_id = str(knowledge_base["id"])
    created_documents = [
        (
            await upload_pdf(
                client,
                knowledge_base_id,
                filename=f"manual-{index}.pdf",
                content=VALID_PDF + str(index).encode(),
            )
        ).json()
        for index in range(3)
    ]

    all_response = await client.get(
        f"/api/v1/knowledge-bases/{knowledge_base_id}/documents?limit=100&offset=0"
    )
    page_response = await client.get(
        f"/api/v1/knowledge-bases/{knowledge_base_id}/documents?limit=2&offset=1"
    )

    assert all_response.status_code == 200
    assert page_response.status_code == 200
    all_items = all_response.json()["items"]
    expected_items = sorted(
        created_documents,
        key=lambda item: (item["created_at"], item["id"]),
        reverse=True,
    )
    assert all_items == expected_items
    assert page_response.json() == {
        "items": expected_items[1:3],
        "total": 3,
        "limit": 2,
        "offset": 1,
    }
    assert (
        await client.get(f"/api/v1/knowledge-bases/{knowledge_base_id}/documents?limit=101")
    ).status_code == 422


@pytest.mark.anyio
async def test_invalid_uploads_return_controlled_errors_without_orphan_files(
    client: AsyncClient,
    test_data_dir: Path,
) -> None:
    knowledge_base = await create_knowledge_base(client)
    knowledge_base_id = str(knowledge_base["id"])
    invalid_uploads = (
        ("manual.txt", VALID_PDF, 415),
        ("fake.pdf", b"not a PDF", 415),
        ("empty.pdf", b"", 400),
    )

    for filename, content, expected_status in invalid_uploads:
        response = await upload_pdf(
            client,
            knowledge_base_id,
            filename=filename,
            content=content,
        )
        assert response.status_code == expected_status

    missing_filename_response = await client.post(
        f"/api/v1/knowledge-bases/{knowledge_base_id}/documents",
        files={"file": (None, VALID_PDF, "application/pdf")},
    )
    assert missing_filename_response.status_code == 422

    assert not [path for path in test_data_dir.rglob("*") if path.is_file()]


@pytest.mark.anyio
async def test_upload_size_limit_is_configurable_and_staged_file_is_removed(
    limited_client: AsyncClient,
    test_data_dir: Path,
) -> None:
    knowledge_base = await create_knowledge_base(limited_client)
    response = await upload_pdf(
        limited_client,
        str(knowledge_base["id"]),
        content=VALID_PDF,
    )

    assert response.status_code == 413
    assert not [path for path in test_data_dir.rglob("*") if path.is_file()]


@pytest.mark.anyio
async def test_missing_knowledge_base_does_not_create_storage(
    client: AsyncClient,
    test_data_dir: Path,
) -> None:
    response = await upload_pdf(client, str(uuid4()))

    assert response.status_code == 404
    assert not test_data_dir.exists()


@pytest.mark.anyio
async def test_document_is_owned_by_its_knowledge_base(
    client: AsyncClient,
    test_data_dir: Path,
) -> None:
    first_knowledge_base = await create_knowledge_base(client, "文档所属知识库")
    second_knowledge_base = await create_knowledge_base(client, "另一个知识库")
    document_response = await upload_pdf(client, str(first_knowledge_base["id"]))
    document = document_response.json()
    first_knowledge_base_id = UUID(first_knowledge_base["id"])
    document_id = UUID(document["id"])
    stored_path = test_data_dir / storage_key_for_document(first_knowledge_base_id, document_id)

    get_response = await client.get(
        f"/api/v1/knowledge-bases/{second_knowledge_base['id']}/documents/{document_id}"
    )
    delete_response = await client.delete(
        f"/api/v1/knowledge-bases/{second_knowledge_base['id']}/documents/{document_id}"
    )

    assert get_response.status_code == 404
    assert delete_response.status_code == 404
    assert stored_path.exists()


@pytest.mark.anyio
async def test_delete_removes_document_metadata_and_file_and_allows_empty_kb_delete(
    client: AsyncClient,
    test_data_dir: Path,
) -> None:
    knowledge_base = await create_knowledge_base(client)
    knowledge_base_id = str(knowledge_base["id"])
    document = (await upload_pdf(client, knowledge_base_id)).json()
    stored_path = test_data_dir / storage_key_for_document(
        UUID(knowledge_base_id), UUID(document["id"])
    )

    blocked_delete = await client.delete(f"/api/v1/knowledge-bases/{knowledge_base_id}")
    assert blocked_delete.status_code == 409
    assert stored_path.exists()

    delete_response = await client.delete(
        f"/api/v1/knowledge-bases/{knowledge_base_id}/documents/{document['id']}"
    )
    knowledge_base_delete_response = await client.delete(
        f"/api/v1/knowledge-bases/{knowledge_base_id}"
    )

    assert delete_response.status_code == 204
    assert knowledge_base_delete_response.status_code == 204
    assert not stored_path.exists()
    assert not stored_path.parent.exists()
    assert not (test_data_dir / "parsing" / document["id"]).exists()
    assert not (test_data_dir / "chunking" / document["id"]).exists()
    assert (
        await client.get(f"/api/v1/knowledge-bases/{knowledge_base_id}/documents/{document['id']}")
    ).status_code == 404


@pytest.mark.anyio
async def test_delete_removes_parsing_and_chunking_artifacts_with_document(
    client: AsyncClient,
    test_data_dir: Path,
) -> None:
    knowledge_base = await create_knowledge_base(client)
    knowledge_base_id = UUID(str(knowledge_base["id"]))
    document = (await upload_pdf(client, str(knowledge_base_id))).json()
    document_id = UUID(document["id"])
    stored_path = test_data_dir / storage_key_for_document(knowledge_base_id, document_id)
    parsing_dir = test_data_dir / "parsing" / str(document_id)
    chunking_dir = test_data_dir / "chunking" / str(document_id)
    evidence_dir = test_data_dir / EVIDENCE_DIRECTORY_NAME / str(document_id)
    (parsing_dir / "mineru" / "images").mkdir(parents=True)
    (chunking_dir / "chunks.json").parent.mkdir(parents=True)
    (parsing_dir / "canonical.json").write_text("{}", encoding="utf-8")
    (parsing_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (parsing_dir / "mineru" / "images" / "image.png").write_bytes(b"asset")
    (chunking_dir / "chunks.json").write_text("chunks", encoding="utf-8")
    (evidence_dir / "evidence.json").parent.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "evidence.json").write_text("evidence", encoding="utf-8")

    response = await client.delete(
        f"/api/v1/knowledge-bases/{knowledge_base_id}/documents/{document_id}"
    )

    assert response.status_code == 204
    assert not stored_path.exists()
    assert not parsing_dir.exists()
    assert not chunking_dir.exists()
    assert not evidence_dir.exists()
    assert not list((test_data_dir / "documents").glob(".delete-*"))
    assert (
        await client.get(f"/api/v1/knowledge-bases/{knowledge_base_id}/documents/{document_id}")
    ).status_code == 404


@pytest.mark.anyio
async def test_qdrant_cleanup_runs_outside_the_event_loop_thread() -> None:
    event_loop_thread = threading.get_ident()
    cleanup_threads: list[int] = []

    class ThreadRecordingVectorStore:
        def delete_document(self, _document_id: UUID) -> int:
            cleanup_threads.append(threading.get_ident())
            return 0

    request = cast(
        Request,
        SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(vector_store=ThreadRecordingVectorStore()))
        ),
    )

    await _delete_document_vectors(request, uuid4())

    assert cleanup_threads
    assert cleanup_threads[0] != event_loop_thread


@pytest.mark.anyio
async def test_delete_external_reference_cleans_evidence_without_documents_root(
    postgres_test_engine: AsyncEngine,
    tmp_path: Path,
) -> None:
    knowledge_base_id = uuid4()
    document_id = uuid4()
    data_dir = tmp_path / "data"
    evidence_dir = data_dir / EVIDENCE_DIRECTORY_NAME / str(document_id)
    evidence_dir.mkdir(parents=True)
    (evidence_dir / "evidence.json").write_text("evidence", encoding="utf-8")

    factory = async_sessionmaker(postgres_test_engine, expire_on_commit=False)
    async with factory() as session:
        session.add(KnowledgeBase(id=knowledge_base_id, name="external evidence"))
        await session.flush()
        session.add(
            Document(
                id=document_id,
                knowledge_base_id=knowledge_base_id,
                original_filename="external.pdf",
                storage_key=None,
                storage_kind=DOCUMENT_STORAGE_KIND_EXTERNAL_REFERENCE,
                source_ref="benchmark/test/external.pdf",
                media_type=DOCUMENT_MEDIA_TYPE_PDF,
                size_bytes=1,
                sha256="c" * 64,
                status=DOCUMENT_STATUS_REGISTERED,
            )
        )
        await session.commit()

    class EmptyGraphStore:
        def delete_document(self, *_: object, **__: object) -> None:
            return None

    class EmptyVectorStore:
        def delete_document(self, *_: object, **__: object) -> None:
            return None

    request = cast(
        Request,
        SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    settings=SimpleNamespace(data_dir=data_dir),
                    graph_store=EmptyGraphStore(),
                    vector_store=EmptyVectorStore(),
                )
            )
        ),
    )
    async with factory() as session:
        response = await delete_document(knowledge_base_id, document_id, request, session)

    assert response.status_code == 204
    assert not evidence_dir.exists()
    assert not (data_dir / "documents").exists()

    async with factory() as session:
        knowledge_base = await session.get(KnowledgeBase, knowledge_base_id)
        assert knowledge_base is not None
        await session.delete(knowledge_base)
        await session.commit()


@pytest.mark.anyio
async def test_delete_restores_source_and_parsing_artifacts_when_database_delete_fails(
    client: AsyncClient,
    test_data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    knowledge_base = await create_knowledge_base(client)
    knowledge_base_id = UUID(str(knowledge_base["id"]))
    document = (await upload_pdf(client, str(knowledge_base_id))).json()
    document_id = UUID(document["id"])
    stored_path = test_data_dir / storage_key_for_document(knowledge_base_id, document_id)
    parsing_dir = test_data_dir / "parsing" / str(document_id)
    chunking_dir = test_data_dir / "chunking" / str(document_id)
    evidence_dir = test_data_dir / EVIDENCE_DIRECTORY_NAME / str(document_id)
    (parsing_dir / "mineru").mkdir(parents=True)
    chunking_dir.mkdir(parents=True)
    evidence_dir.mkdir(parents=True)
    (parsing_dir / "canonical.json").write_text("canonical", encoding="utf-8")
    (parsing_dir / "mineru" / "raw.json").write_text("raw", encoding="utf-8")
    (chunking_dir / "chunks.json").write_text("chunks", encoding="utf-8")
    (evidence_dir / "evidence.json").write_text("evidence", encoding="utf-8")

    async def fail_commit(_session: object) -> None:
        raise SQLAlchemyError("simulated database failure")

    monkeypatch.setattr("sqlalchemy.ext.asyncio.AsyncSession.commit", fail_commit)

    response = await client.delete(
        f"/api/v1/knowledge-bases/{knowledge_base_id}/documents/{document_id}"
    )

    assert response.status_code == 500
    assert "文档删除失败" in response.json()["detail"]
    assert stored_path.read_bytes() == VALID_PDF
    assert (parsing_dir / "canonical.json").read_text(encoding="utf-8") == "canonical"
    assert (parsing_dir / "mineru" / "raw.json").read_text(encoding="utf-8") == "raw"
    assert (chunking_dir / "chunks.json").read_text(encoding="utf-8") == "chunks"
    assert (evidence_dir / "evidence.json").read_text(encoding="utf-8") == "evidence"
    assert not list((test_data_dir / "documents").glob(".delete-*"))
    assert (
        await client.get(f"/api/v1/knowledge-bases/{knowledge_base_id}/documents/{document_id}")
    ).status_code == 200


@pytest.mark.anyio
async def test_document_file_endpoint_streams_inline_pdf_and_supports_ranges(
    client: AsyncClient,
) -> None:
    knowledge_base = await create_knowledge_base(client)
    knowledge_base_id = str(knowledge_base["id"])
    document = (await upload_pdf(client, knowledge_base_id, filename="../原始文件.PDF")).json()
    file_url = f"/api/v1/knowledge-bases/{knowledge_base_id}/documents/{document['id']}/file"

    response = await client.get(file_url)

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    disposition = response.headers["content-disposition"]
    assert disposition.startswith("inline")
    assert "filename*=utf-8''" in disposition
    assert response.content == VALID_PDF

    range_response = await client.get(file_url, headers={"Range": "bytes=0-4"})

    assert range_response.status_code == 206
    assert range_response.content == VALID_PDF[:5]


@pytest.mark.anyio
async def test_document_preview_endpoints_are_scoped_to_their_knowledge_base(
    client: AsyncClient,
) -> None:
    first_knowledge_base = await create_knowledge_base(client, "第一个知识库")
    second_knowledge_base = await create_knowledge_base(client, "第二个知识库")
    document = (await upload_pdf(client, str(first_knowledge_base["id"]))).json()
    prefix = f"/api/v1/knowledge-bases/{second_knowledge_base['id']}/documents/{document['id']}"

    assert (await client.get(f"{prefix}/file")).status_code == 404
    assert (await client.get(f"{prefix}/chunks")).status_code == 404


@pytest.mark.anyio
async def test_document_chunks_endpoint_returns_persisted_preview_chunks(
    client: AsyncClient,
    test_data_dir: Path,
) -> None:
    knowledge_base = await create_knowledge_base(client)
    knowledge_base_id = UUID(str(knowledge_base["id"]))
    document = (await upload_pdf(client, str(knowledge_base_id))).json()
    document_id = UUID(document["id"])
    chunks_url = f"/api/v1/knowledge-bases/{knowledge_base_id}/documents/{document_id}/chunks"

    empty_response = await client.get(chunks_url)

    assert empty_response.status_code == 200
    assert empty_response.json() == {
        "document_id": str(document_id),
        "page_count": None,
        "items": [],
    }

    write_chunk_artifact(
        test_data_dir,
        document_id,
        [
            Chunk(
                chunk_id="chunk-1",
                document_id=document_id,
                ordinal=0,
                text="第一章 质量控制要求。",
                page_start=1,
                page_end=2,
                source_block_ids=["block-1"],
                section_path=["质量控制"],
                content_types=["text"],
            ),
            Chunk(
                chunk_id="chunk-2",
                document_id=document_id,
                ordinal=1,
                text="",
                page_start=3,
                page_end=3,
                source_block_ids=["block-2"],
                section_path=["质量控制", "检验"],
                content_types=["table"],
                asset_refs=["asset-1"],
            ),
        ],
        page_count=3,
    )

    response = await client.get(chunks_url)

    assert response.status_code == 200
    assert response.json() == {
        "document_id": str(document_id),
        "page_count": 3,
        "items": [
            {
                "chunk_id": "chunk-1",
                "ordinal": 0,
                "page_start": 1,
                "page_end": 2,
                "section_path": ["质量控制"],
                "content_types": ["text"],
                "asset_refs": [],
                "text": "第一章 质量控制要求。",
            },
            {
                "chunk_id": "chunk-2",
                "ordinal": 1,
                "page_start": 3,
                "page_end": 3,
                "section_path": ["质量控制", "检验"],
                "content_types": ["table"],
                "asset_refs": ["asset-1"],
                "text": "",
            },
        ],
    }


@pytest.mark.anyio
async def test_document_chunks_endpoint_reports_unusable_artifacts(
    client: AsyncClient,
    test_data_dir: Path,
) -> None:
    knowledge_base = await create_knowledge_base(client)
    knowledge_base_id = UUID(str(knowledge_base["id"]))
    document = (await upload_pdf(client, str(knowledge_base_id))).json()
    document_id = UUID(document["id"])
    chunks_url = f"/api/v1/knowledge-bases/{knowledge_base_id}/documents/{document_id}/chunks"
    artifact_path = (
        test_data_dir / CHUNKING_DIRECTORY_NAME / str(document_id) / CHUNKS_ARTIFACT_FILENAME
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    artifact_path.write_text("not a chunk artifact", encoding="utf-8")
    corrupt_response = await client.get(chunks_url)

    assert corrupt_response.status_code == 500
    assert corrupt_response.json()["detail"] == "文档切片数据不可用"

    mismatched_artifact = ChunkedDocument(
        document_id=uuid4(),
        page_count=1,
        config_fingerprint="b" * 64,
        chunks=[],
    )
    artifact_path.write_text(mismatched_artifact.model_dump_json(indent=2) + "\n", encoding="utf-8")
    mismatched_response = await client.get(chunks_url)

    assert mismatched_response.status_code == 500
    assert mismatched_response.json()["detail"] == "文档切片数据不可用"

    artifact_path.unlink()
    artifact_path.mkdir()
    unreadable_response = await client.get(chunks_url)

    assert unreadable_response.status_code == 500
    assert unreadable_response.json()["detail"] == "文档切片暂时无法读取"


# The full-width parentheses are part of the real corpus filename.
EXTERNAL_SOURCE_REF = "benchmark/a1-5/part_9/普通高中教科书·数学（A版）必修_第一册_256-270.pdf"  # noqa: RUF001


async def insert_external_document(
    engine: AsyncEngine,
    *,
    knowledge_base_id: UUID,
    document_id: UUID,
    source_ref: str = EXTERNAL_SOURCE_REF,
) -> None:
    """Register one external-reference document exactly like corpus registration does."""
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        session.add(KnowledgeBase(id=knowledge_base_id, name="外部语料知识库"))
        await session.flush()
        session.add(
            Document(
                id=document_id,
                knowledge_base_id=knowledge_base_id,
                original_filename="片段.pdf",
                storage_key=None,
                storage_kind=DOCUMENT_STORAGE_KIND_EXTERNAL_REFERENCE,
                source_ref=source_ref,
                media_type=DOCUMENT_MEDIA_TYPE_PDF,
                size_bytes=len(VALID_PDF),
                sha256="e" * 64,
                status=DOCUMENT_STATUS_REGISTERED,
            )
        )
        await session.commit()


async def remove_external_document(
    engine: AsyncEngine,
    *,
    knowledge_base_id: UUID,
    document_id: UUID,
) -> None:
    """Remove test-only external rows so the shared test database stays clean."""
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as session:
        document = await session.get(Document, document_id)
        if document is not None:
            await session.delete(document)
            await session.flush()
        knowledge_base = await session.get(KnowledgeBase, knowledge_base_id)
        if knowledge_base is not None:
            await session.delete(knowledge_base)
        await session.commit()


def preview_request(
    data_dir: Path,
    *,
    corpus_source_dir: Path | None,
    vector_store: object | None = None,
) -> Request:
    return cast(
        Request,
        SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    settings=SimpleNamespace(
                        data_dir=data_dir,
                        corpus_source_dir=corpus_source_dir,
                    ),
                    vector_store=vector_store,
                )
            )
        ),
    )


def test_external_source_path_stays_inside_the_corpus_root(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()

    resolved = resolve_external_source_path(corpus_root, EXTERNAL_SOURCE_REF)

    assert resolved == (corpus_root / "part_9" / Path(EXTERNAL_SOURCE_REF).name).resolve()
    with pytest.raises(StorageError):
        resolve_external_source_path(corpus_root, "uploads/片段.pdf")
    with pytest.raises(StorageError):
        resolve_external_source_path(corpus_root, "benchmark/a1-5/../../escape.pdf")


@pytest.mark.anyio
async def test_external_corpus_document_file_streams_from_the_configured_root(
    postgres_test_engine: AsyncEngine,
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    corpus_root = tmp_path / "corpus"
    stored_pdf = corpus_root / "part_9" / Path(EXTERNAL_SOURCE_REF).name
    stored_pdf.parent.mkdir(parents=True)
    stored_pdf.write_bytes(VALID_PDF)
    knowledge_base_id = uuid4()
    document_id = uuid4()
    await insert_external_document(
        postgres_test_engine,
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
    )

    factory = async_sessionmaker(postgres_test_engine, expire_on_commit=False)
    try:
        async with factory() as session:
            response = await get_document_file(
                knowledge_base_id,
                document_id,
                preview_request(data_dir, corpus_source_dir=corpus_root),
                session,
            )

        assert Path(response.path).resolve() == stored_pdf.resolve()
        assert response.media_type == DOCUMENT_MEDIA_TYPE_PDF
        assert response.headers["content-disposition"].startswith("inline")
    finally:
        await remove_external_document(
            postgres_test_engine,
            knowledge_base_id=knowledge_base_id,
            document_id=document_id,
        )


@pytest.mark.anyio
async def test_external_corpus_document_file_requires_a_configured_root(
    postgres_test_engine: AsyncEngine,
    tmp_path: Path,
) -> None:
    knowledge_base_id = uuid4()
    document_id = uuid4()
    await insert_external_document(
        postgres_test_engine,
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
    )

    factory = async_sessionmaker(postgres_test_engine, expire_on_commit=False)
    try:
        async with factory() as session:
            with pytest.raises(HTTPException) as unconfigured:
                await get_document_file(
                    knowledge_base_id,
                    document_id,
                    preview_request(tmp_path / "data", corpus_source_dir=None),
                    session,
                )

            assert unconfigured.value.status_code == 404
            assert unconfigured.value.detail == "该文档来自外部语料, 未配置原始文件目录"

            with pytest.raises(HTTPException) as missing_file:
                await get_document_file(
                    knowledge_base_id,
                    document_id,
                    preview_request(tmp_path / "data", corpus_source_dir=tmp_path / "empty"),
                    session,
                )

        assert missing_file.value.status_code == 404
        assert missing_file.value.detail == "文档文件不存在"
    finally:
        await remove_external_document(
            postgres_test_engine,
            knowledge_base_id=knowledge_base_id,
            document_id=document_id,
        )


@pytest.mark.anyio
async def test_document_chunks_fall_back_to_indexed_payloads(
    postgres_test_engine: AsyncEngine,
    tmp_path: Path,
) -> None:
    knowledge_base_id = uuid4()
    document_id = uuid4()
    await insert_external_document(
        postgres_test_engine,
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
    )
    payloads = (
        ChunkVectorPayload(
            collection_schema_version="1.0",
            chunk_id="chunk-indexed-1",
            document_id=document_id,
            knowledge_base_id=knowledge_base_id,
            page_start=2,
            page_end=3,
            source_block_ids=["p2-b1"],
            section_path=["索引章节"],
            content_types=["text"],
            asset_refs=[],
            text="来自向量索引的切片。",
            chunking_config_fingerprint="c" * 64,
            embedding_model="test-embedding",
            embedding_model_revision=None,
            embedding_config_fingerprint="d" * 64,
        ),
    )

    class IndexedChunkStore:
        def list_document_chunk_payloads(self, **_: object) -> tuple[ChunkVectorPayload, ...]:
            return payloads

    factory = async_sessionmaker(postgres_test_engine, expire_on_commit=False)
    try:
        async with factory() as session:
            response = await list_document_chunks(
                knowledge_base_id,
                document_id,
                preview_request(
                    tmp_path / "data",
                    corpus_source_dir=None,
                    vector_store=IndexedChunkStore(),
                ),
                session,
            )

        assert response.document_id == document_id
        assert response.page_count is None
        assert [item.chunk_id for item in response.items] == ["chunk-indexed-1"]
        assert response.items[0].ordinal == 0
        assert response.items[0].page_start == 2
        assert response.items[0].page_end == 3
        assert response.items[0].section_path == ["索引章节"]
        assert response.items[0].text == "来自向量索引的切片。"
    finally:
        await remove_external_document(
            postgres_test_engine,
            knowledge_base_id=knowledge_base_id,
            document_id=document_id,
        )
