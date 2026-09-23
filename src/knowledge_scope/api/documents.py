"""PDF document upload and metadata endpoints."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from knowledge_scope.chunking.models import ChunkedDocument
from knowledge_scope.chunking.service import CHUNKING_DIRECTORY_NAME, chunk_artifact_path
from knowledge_scope.documents.models import (
    DOCUMENT_MEDIA_TYPE_PDF,
    DOCUMENT_STATUS_UPLOADED,
    DOCUMENT_STORAGE_KIND_EXTERNAL_REFERENCE,
    DOCUMENT_STORAGE_KIND_MANAGED,
    Document,
)
from knowledge_scope.documents.storage import (
    StagedUpload,
    StorageError,
    TrashedResource,
    UploadValidationError,
    filesystem_path_for_storage_key,
    move_to_trash,
    permanently_remove_trash,
    promote_staged_upload,
    remove_file,
    remove_staged_upload,
    resolve_external_source_path,
    restore_from_trash,
    stage_pdf,
    storage_key_for_document,
)
from knowledge_scope.evidence.lifecycle import (
    EvidenceArtifactError,
    move_evidence_artifact_to_trash,
)
from knowledge_scope.graph.neo4j import GraphStoreError
from knowledge_scope.parsing.service import PARSING_DIRECTORY_NAME
from knowledge_scope.retrieval.qdrant import ChunkVectorPayload, VectorStoreError
from knowledge_scope.retrieval.representation_index import (
    RepresentationIndexError,
    RepresentationVisibilitySnapshot,
)
from knowledge_scope.shared.config import Settings
from knowledge_scope.shared.database import get_session

from .knowledge_bases import _get_knowledge_base
from .schemas import (
    DocumentChunkListResponse,
    DocumentChunkResponse,
    DocumentListResponse,
    DocumentResponse,
)

router = APIRouter(
    prefix="/knowledge-bases/{knowledge_base_id}/documents",
    tags=["documents"],
)


class ChunkArtifactError(RuntimeError):
    """Raised when a persisted chunk artifact cannot be read for preview."""


def _runtime_settings(request: Request) -> Settings:
    return request.app.state.settings


async def _get_document(
    session: AsyncSession,
    knowledge_base_id: UUID,
    document_id: UUID,
) -> Document:
    document = await session.scalar(
        select(Document).where(
            Document.id == document_id,
            Document.knowledge_base_id == knowledge_base_id,
        )
    )
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="文档不存在")
    return document


def _resolve_document_source_path(settings: Settings, document: Document) -> Path:
    """Resolve the source PDF of a managed upload or a registered external corpus."""
    if document.storage_kind == DOCUMENT_STORAGE_KIND_MANAGED:
        if document.storage_key is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="文档文件不可用")
        try:
            return filesystem_path_for_storage_key(settings.data_dir, document.storage_key)
        except StorageError:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="文档文件不可用",
            ) from None
    if document.storage_kind == DOCUMENT_STORAGE_KIND_EXTERNAL_REFERENCE:
        if document.source_ref is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="文档文件不可用")
        if settings.corpus_source_dir is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="该文档来自外部语料, 未配置原始文件目录",
            )
        try:
            return resolve_external_source_path(settings.corpus_source_dir, document.source_ref)
        except StorageError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="外部语料原始文件不可用",
            ) from None
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="文档文件不可用")


async def _list_indexed_chunk_payloads(
    request: Request,
    knowledge_base_id: UUID,
    document_id: UUID,
) -> tuple[ChunkVectorPayload, ...]:
    """Fall back to indexed chunks so registered corpora stay previewable."""
    store = getattr(request.app.state, "vector_store", None)
    if store is None:
        return ()
    try:
        return await asyncio.to_thread(
            store.list_document_chunk_payloads,
            knowledge_base_id=knowledge_base_id,
            document_id=document_id,
        )
    except VectorStoreError:
        return ()


def _read_chunk_artifact(settings: Settings, document_id: UUID) -> ChunkedDocument | None:
    """Read one persisted chunk artifact, or None when chunking has not run yet."""
    path = chunk_artifact_path(settings, document_id)
    try:
        payload = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise ChunkArtifactError("文档切片暂时无法读取") from error
    try:
        chunked = ChunkedDocument.model_validate_json(payload)
    except (ValueError, UnicodeDecodeError) as error:
        raise ChunkArtifactError("文档切片数据不可用") from error
    if chunked.document_id != document_id:
        raise ChunkArtifactError("文档切片数据不可用")
    return chunked


def _is_duplicate_error(error: IntegrityError) -> bool:
    return "uq_documents_knowledge_base_sha256" in str(error)


def _cleanup_upload(staged: StagedUpload | None, final_path: Path | None) -> None:
    if final_path is not None:
        remove_file(final_path)
    if staged is not None:
        remove_staged_upload(staged.directory, staged.path)


def _restore_deleted_resources(resources: tuple[TrashedResource | None, ...]) -> bool:
    """Restore staged resources and report whether any restoration failed."""
    restoration_failed = False
    for resource in reversed(resources):
        if resource is None:
            continue
        try:
            restore_from_trash(resource)
        except (OSError, StorageError):
            restoration_failed = True
    return restoration_failed


async def _delete_document_vectors(request: Request, document_id: UUID) -> None:
    """Run synchronous Qdrant cleanup outside the FastAPI event loop."""
    try:
        await asyncio.to_thread(request.app.state.vector_store.delete_document, document_id)
    except VectorStoreError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="文档已删除, 但向量清理失败",
        ) from None


async def _delete_document_graph(
    request: Request,
    knowledge_base_id: UUID,
    document_id: UUID,
) -> None:
    """Clean graph state without blocking the FastAPI event loop."""
    try:
        await asyncio.to_thread(
            request.app.state.graph_store.delete_document,
            document_id,
            knowledge_base_id=knowledge_base_id,
        )
    except GraphStoreError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="文档已删除, 但知识图谱清理失败",
        ) from None


async def _delete_document_representations(
    request: Request,
    knowledge_base_id: UUID,
    document_id: UUID,
) -> None:
    """Remove A4.2 points outside the event loop when the index is configured."""
    store = getattr(request.app.state, "representation_store", None)
    if store is None:
        return
    try:
        await asyncio.to_thread(
            store.delete_document,
            document_id,
            knowledge_base_id=knowledge_base_id,
        )
    except RepresentationIndexError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="文档已删除, 但多模态表示清理失败",
        ) from None
    except Exception as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="文档已删除, 但多模态表示清理失败",
        ) from error


async def _quarantine_document_representations(
    request: Request,
    knowledge_base_id: UUID,
    document_id: UUID,
) -> RepresentationVisibilitySnapshot | None:
    """Hide representation points before committing authoritative deletion."""
    store = getattr(request.app.state, "representation_store", None)
    if store is None:
        return None
    try:
        return await asyncio.to_thread(
            store.quarantine_document,
            document_id,
            knowledge_base_id=knowledge_base_id,
        )
    except RepresentationIndexError:
        raise
    except Exception as error:
        raise RepresentationIndexError("多模态表示在权威删除前无法隔离") from error


async def _restore_document_representations(
    request: Request,
    snapshot: RepresentationVisibilitySnapshot | None,
) -> bool:
    """Restore a pre-commit representation quarantine and report failures."""
    if snapshot is None:
        return False
    store = getattr(request.app.state, "representation_store", None)
    if store is None:
        return True
    try:
        await asyncio.to_thread(store.restore_document_visibility, snapshot)
    except RepresentationIndexError:
        return True
    except Exception:
        return True
    return False


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    knowledge_base_id: UUID,
    request: Request,
    file: Annotated[UploadFile, File(...)],
    session: AsyncSession = Depends(get_session),
) -> Document:
    await _get_knowledge_base(session, knowledge_base_id)
    settings = _runtime_settings(request)
    staged: StagedUpload | None = None
    final_path: Path | None = None
    metadata_committed = False

    try:
        staged = await stage_pdf(file, settings.data_dir, settings.max_upload_size_bytes)
        existing_id = await session.scalar(
            select(Document.id).where(
                Document.knowledge_base_id == knowledge_base_id,
                Document.sha256 == staged.sha256,
            )
        )
        if existing_id is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="同一知识库中已存在相同文件",
            )

        document_id = uuid4()
        storage_key = storage_key_for_document(knowledge_base_id, document_id)
        final_path = filesystem_path_for_storage_key(settings.data_dir, storage_key)
        promote_staged_upload(staged, final_path)

        document = Document(
            id=document_id,
            knowledge_base_id=knowledge_base_id,
            original_filename=staged.original_filename,
            storage_key=storage_key,
            storage_kind=DOCUMENT_STORAGE_KIND_MANAGED,
            source_ref=None,
            media_type=DOCUMENT_MEDIA_TYPE_PDF,
            size_bytes=staged.size_bytes,
            sha256=staged.sha256,
            status=DOCUMENT_STATUS_UPLOADED,
        )
        session.add(document)
        try:
            await session.commit()
        except IntegrityError as error:
            await session.rollback()
            _cleanup_upload(staged, final_path)
            if _is_duplicate_error(error):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="同一知识库中已存在相同文件",
                ) from None
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="文档元数据保存失败",
            ) from None
        metadata_committed = True
        await session.refresh(document)
        return document
    except UploadValidationError as error:
        await session.rollback()
        _cleanup_upload(staged, final_path)
        raise HTTPException(status_code=error.status_code, detail=str(error)) from None
    except HTTPException:
        await session.rollback()
        if not metadata_committed:
            _cleanup_upload(staged, final_path)
        raise
    except (OSError, StorageError, SQLAlchemyError):
        await session.rollback()
        if not metadata_committed:
            _cleanup_upload(staged, final_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="文档保存失败",
        ) from None
    finally:
        if staged is not None:
            remove_staged_upload(staged.directory, staged.path)


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    knowledge_base_id: UUID,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_session),
) -> DocumentListResponse:
    await _get_knowledge_base(session, knowledge_base_id)

    total = int(
        await session.scalar(
            select(func.count())
            .select_from(Document)
            .where(Document.knowledge_base_id == knowledge_base_id)
        )
        or 0
    )
    result = await session.scalars(
        select(Document)
        .where(Document.knowledge_base_id == knowledge_base_id)
        .order_by(Document.created_at.desc(), Document.id.desc())
        .offset(offset)
        .limit(limit)
    )
    return DocumentListResponse(
        items=list(result.all()),
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    knowledge_base_id: UUID,
    document_id: UUID,
    session: AsyncSession = Depends(get_session),
) -> Document:
    await _get_knowledge_base(session, knowledge_base_id)
    return await _get_document(session, knowledge_base_id, document_id)


@router.get(
    "/{document_id}/file",
    response_class=FileResponse,
    responses={200: {"content": {"application/pdf": {}}}},
)
async def get_document_file(
    knowledge_base_id: UUID,
    document_id: UUID,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> FileResponse:
    """Stream the source PDF so the product UI can preview it in place."""
    await _get_knowledge_base(session, knowledge_base_id)
    document = await _get_document(session, knowledge_base_id, document_id)
    path = _resolve_document_source_path(_runtime_settings(request), document)
    if not path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="文档文件不存在")

    return FileResponse(
        path,
        media_type=document.media_type,
        filename=document.original_filename,
        content_disposition_type="inline",
        headers={"Cache-Control": "private, max-age=0, must-revalidate"},
    )


@router.get("/{document_id}/chunks", response_model=DocumentChunkListResponse)
async def list_document_chunks(
    knowledge_base_id: UUID,
    document_id: UUID,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> DocumentChunkListResponse:
    """Expose chunk page ranges so the UI can navigate the source PDF."""
    await _get_knowledge_base(session, knowledge_base_id)
    document = await _get_document(session, knowledge_base_id, document_id)
    settings = _runtime_settings(request)
    try:
        chunked = await asyncio.to_thread(_read_chunk_artifact, settings, document.id)
    except ChunkArtifactError as error:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(error),
        ) from None

    if chunked is not None:
        return DocumentChunkListResponse(
            document_id=document.id,
            page_count=chunked.page_count,
            items=[DocumentChunkResponse.model_validate(chunk) for chunk in chunked.chunks],
        )

    indexed_payloads = await _list_indexed_chunk_payloads(request, knowledge_base_id, document.id)
    return DocumentChunkListResponse(
        document_id=document.id,
        page_count=None,
        items=[
            DocumentChunkResponse(
                chunk_id=payload.chunk_id,
                ordinal=ordinal,
                page_start=payload.page_start,
                page_end=payload.page_end,
                section_path=list(payload.section_path),
                content_types=list(payload.content_types),
                asset_refs=list(payload.asset_refs),
                text=payload.text,
            )
            for ordinal, payload in enumerate(indexed_payloads)
        ],
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    knowledge_base_id: UUID,
    document_id: UUID,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> Response:
    await _get_knowledge_base(session, knowledge_base_id)
    document = await _get_document(session, knowledge_base_id, document_id)

    settings = _runtime_settings(request)
    trashed_source: TrashedResource | None = None
    trashed_parsing: TrashedResource | None = None
    trashed_chunking: TrashedResource | None = None
    trashed_evidence: TrashedResource | None = None
    representation_visibility: RepresentationVisibilitySnapshot | None = None
    try:
        if document.storage_kind == DOCUMENT_STORAGE_KIND_MANAGED:
            if document.storage_key is None:
                raise StorageError("managed document has no storage key")
            final_path = filesystem_path_for_storage_key(settings.data_dir, document.storage_key)
            trashed_source = move_to_trash(final_path, settings.data_dir)
        elif document.storage_kind != DOCUMENT_STORAGE_KIND_EXTERNAL_REFERENCE:
            raise StorageError("document has an unsupported storage kind")

        parsing_path = Path(settings.data_dir).resolve() / PARSING_DIRECTORY_NAME / str(document.id)
        if parsing_path.is_symlink() or parsing_path.exists():
            if parsing_path.is_symlink() or not parsing_path.is_dir():
                raise StorageError("parsing artifact directory is invalid")
            trashed_parsing = move_to_trash(parsing_path, settings.data_dir)

        chunking_path = (
            Path(settings.data_dir).resolve() / CHUNKING_DIRECTORY_NAME / str(document.id)
        )
        if chunking_path.is_symlink() or chunking_path.exists():
            if chunking_path.is_symlink() or not chunking_path.is_dir():
                raise StorageError("chunking artifact directory is invalid")
            trashed_chunking = move_to_trash(chunking_path, settings.data_dir)

        trashed_evidence = move_evidence_artifact_to_trash(settings.data_dir, document.id)
        representation_visibility = await _quarantine_document_representations(
            request,
            knowledge_base_id,
            document.id,
        )
    except (OSError, StorageError, EvidenceArtifactError, RepresentationIndexError):
        restoration_failed = _restore_deleted_resources(
            (trashed_source, trashed_parsing, trashed_chunking, trashed_evidence)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "文档删除失败, 且文件恢复失败" if restoration_failed else "文档文件不可用, 删除失败"
            ),
        ) from None

    try:
        await session.delete(document)
        await session.commit()
    except SQLAlchemyError:
        await session.rollback()
        resources_restore_failed = _restore_deleted_resources(
            (trashed_source, trashed_parsing, trashed_chunking, trashed_evidence)
        )
        representations_restore_failed = await _restore_document_representations(
            request,
            representation_visibility,
        )
        if resources_restore_failed or representations_restore_failed:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="文档删除失败, 且文件或索引恢复失败",
            ) from None
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="文档删除失败",
        ) from None

    await _delete_document_graph(request, knowledge_base_id, document.id)
    await _delete_document_representations(request, knowledge_base_id, document.id)

    cleanup_failed = False
    for resource in (trashed_evidence, trashed_chunking, trashed_parsing, trashed_source):
        if resource is None:
            continue
        try:
            permanently_remove_trash(resource)
        except (OSError, StorageError):
            cleanup_failed = True
    if cleanup_failed:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="文档已删除, 但文件清理失败",
        ) from None
    await _delete_document_vectors(request, document.id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
