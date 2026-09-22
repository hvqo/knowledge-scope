"""FastAPI application for the current KnowledgeScope product surface."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Final

from fastapi import APIRouter, FastAPI, HTTPException, Request, status
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncEngine

from knowledge_scope import __version__
from knowledge_scope.chatbi.agent import ChatBIAgentLimits, ChatBIAgentService
from knowledge_scope.chatbi.api import router as chatbi_router
from knowledge_scope.chatbi.credentials import EnvironmentCredentialResolver
from knowledge_scope.chatbi.discovery import create_postgres_schema_discovery_service
from knowledge_scope.chatbi.eligibility import ChatBIEligibilityService
from knowledge_scope.chatbi.execution import PostgresExecutionAdapter, SQLExecutionService
from knowledge_scope.chatbi.nl2sql import NL2SQLService
from knowledge_scope.chatbi.registry import DatabaseDataSourceProvider
from knowledge_scope.graph.neo4j import Neo4jGraphStore
from knowledge_scope.graph.retrieval import GraphRetrievalConfig
from knowledge_scope.graph.retrieval_service import GraphRetrievalService
from knowledge_scope.llm.gateway import LLMGateway
from knowledge_scope.llm.providers import create_llm_provider
from knowledge_scope.llm.usage import DatabaseUsageRecorder
from knowledge_scope.rag.service import RAGService
from knowledge_scope.reports.api import router as reports_router
from knowledge_scope.retrieval.embedding import QwenEmbeddingModel
from knowledge_scope.retrieval.qdrant import QdrantVectorStore
from knowledge_scope.retrieval.representation_index import (
    MultimodalRepresentationRetrievalService,
    QdrantRepresentationStore,
    validate_representation_collection_role,
)
from knowledge_scope.retrieval.reranking import RerankingService, create_local_reranker
from knowledge_scope.retrieval.service import DenseRetrievalService
from knowledge_scope.retrieval.sparse import SparseIndexConfig, SparseIndexStore
from knowledge_scope.retrieval.unified import UnifiedRetrievalConfig, UnifiedRetrievalService
from knowledge_scope.shared import build_health_report, get_settings
from knowledge_scope.shared.config import Settings
from knowledge_scope.shared.database import create_database_engine, create_session_factory

from .documents import router as documents_router
from .knowledge_bases import router as knowledge_bases_router
from .rag import router as rag_router
from .retrieval import router as retrieval_router
from .schemas import HealthResponse, MetaResponse, Neo4jHealthResponse, QdrantHealthResponse

API_PREFIX: Final = "/api/v1"
CURRENT_PHASE: Final = "A3.1"
PROJECT_STATUS: Final = "foundation"


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Initialize the default RAG graph and dispose owned clients on shutdown."""
    provider = None
    gateway: LLMGateway | None = None
    try:
        if application.state.representation_store is None:
            application.state.representation_store = QdrantRepresentationStore(
                application.state.settings
            )
        if application.state.rag_service is None or application.state.chatbi_agent_service is None:
            settings: Settings = application.state.settings
            if application.state.sparse_store is None:
                application.state.sparse_store = SparseIndexStore(
                    settings.sparse_index_path,
                    config=SparseIndexConfig(
                        bm25_k1=settings.sparse_bm25_k1,
                        bm25_b=settings.sparse_bm25_b,
                    ),
                )
            sparse_store = application.state.sparse_store
            if sparse_store is None:  # pragma: no cover - initialized above.
                raise RuntimeError("sparse store is not initialized")
            provider = create_llm_provider(settings)
            application.state.llm_provider = provider
            gateway = LLMGateway(
                provider,
                DatabaseUsageRecorder(application.state.db_session_factory),
                settings,
            )
        if application.state.rag_service is None:
            settings = application.state.settings
            if gateway is None:  # pragma: no cover - provider is created above.
                raise RuntimeError("LLM gateway is not initialized")
            retrieval = DenseRetrievalService(
                application.state.vector_store,
                application.state.embedding_model,
            )
            reranker = create_local_reranker(settings, model_key="bge-reranker-v2-m3")
            reranking = RerankingService(reranker)
            graph_retrieval = GraphRetrievalService(
                application.state.graph_store,
                config=GraphRetrievalConfig(
                    max_seed_entities=settings.graph_retrieval_max_seed_entities,
                    max_hops=settings.graph_retrieval_max_hops,
                    max_neighbors=settings.graph_retrieval_max_neighbors,
                    max_relations=settings.graph_retrieval_max_relations,
                    max_evidence=settings.graph_retrieval_max_evidence,
                    max_entity_scan=settings.graph_retrieval_max_entity_scan,
                    lexical_threshold=settings.graph_retrieval_lexical_threshold,
                ),
            )
            representation_store = application.state.representation_store
            if representation_store is None:  # pragma: no cover - initialized above.
                raise RuntimeError("representation store is not initialized")
            unified_retrieval = UnifiedRetrievalService(
                retrieval,
                sparse_store,
                graph_retrieval,
                MultimodalRepresentationRetrievalService(
                    representation_store,
                    application.state.embedding_model,
                ),
                reranker,
                chunk_lookup=application.state.vector_store,
                config=UnifiedRetrievalConfig(
                    dense_candidate_limit=settings.unified_dense_candidate_limit,
                    sparse_candidate_limit=settings.unified_sparse_candidate_limit,
                    graph_candidate_limit=settings.unified_graph_candidate_limit,
                    multimodal_candidate_limit=settings.unified_multimodal_candidate_limit,
                    candidate_pool_limit=settings.unified_candidate_pool_limit,
                    result_limit=settings.unified_result_limit,
                    rerank_text_max_chars=settings.unified_rerank_text_max_chars,
                    failure_mode=settings.unified_failure_mode,
                    reranker_model_id=reranker.model_id,
                ),
            )
            application.state.rag_service = RAGService(
                retrieval,
                reranking,
                gateway,
                settings,
                unified_retrieval=unified_retrieval,
            )
        if application.state.chatbi_agent_service is None:
            settings = application.state.settings
            if gateway is None:  # pragma: no cover - provider is created above.
                raise RuntimeError("LLM gateway is not initialized")
            schema_discovery = create_postgres_schema_discovery_service(
                connection_timeout_seconds=settings.chatbi_schema_connection_timeout_seconds,
                statement_timeout_ms=settings.chatbi_statement_timeout_ms,
            )
            nl2sql = NL2SQLService(
                gateway,
                schema_discovery=schema_discovery,
                data_source_provider=DatabaseDataSourceProvider(
                    application.state.db_session_factory
                ),
                max_tokens=settings.chatbi_nl2sql_max_tokens,
            )
            eligibility = ChatBIEligibilityService(nl2sql, gateway)
            execution = SQLExecutionService(
                nl2sql,
                EnvironmentCredentialResolver(),
                PostgresExecutionAdapter(
                    connection_timeout_seconds=settings.chatbi_schema_connection_timeout_seconds,
                    statement_timeout_ms=settings.chatbi_statement_timeout_ms,
                ),
            )
            application.state.chatbi_agent_service = ChatBIAgentService(
                nl2sql,
                execution,
                gateway,
                limits=ChatBIAgentLimits.from_settings(settings),
                eligibility_service=eligibility,
            )
        yield
    finally:
        if provider is not None:
            await provider.aclose()
        await application.state.db_engine.dispose()
        application.state.vector_store.close()
        if application.state.representation_store is not None:
            application.state.representation_store.close()
        if application.state.sparse_store is not None:
            application.state.sparse_store.close()
        application.state.graph_store.close()


def create_app(
    settings: Settings | None = None,
    *,
    database_engine: AsyncEngine | None = None,
    vector_store: QdrantVectorStore | None = None,
    embedding_model: QwenEmbeddingModel | None = None,
    rag_service: RAGService | None = None,
    graph_store: Neo4jGraphStore | None = None,
    representation_store: QdrantRepresentationStore | None = None,
    sparse_store: SparseIndexStore | None = None,
    chatbi_agent_service: ChatBIAgentService | None = None,
) -> FastAPI:
    """Create the API application with validated runtime settings."""
    runtime_settings = settings if settings is not None else get_settings()
    validate_representation_collection_role(runtime_settings)
    engine = (
        database_engine if database_engine is not None else create_database_engine(runtime_settings)
    )
    application = FastAPI(
        title="KnowledgeScope API",
        version=__version__,
        lifespan=lifespan,
    )
    application.state.db_engine = engine
    application.state.db_session_factory = create_session_factory(engine)
    application.state.settings = runtime_settings
    application.state.vector_store = vector_store or QdrantVectorStore(runtime_settings)
    application.state.embedding_model = embedding_model or QwenEmbeddingModel(runtime_settings)
    application.state.rag_service = rag_service
    application.state.graph_store = graph_store or Neo4jGraphStore(runtime_settings)
    application.state.representation_store = representation_store
    application.state.sparse_store = sparse_store
    application.state.chatbi_agent_service = chatbi_agent_service
    application.state.llm_provider = None
    application.add_middleware(
        CORSMiddleware,
        allow_origins=runtime_settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Accept", "Content-Type"],
    )

    @application.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        """Keep invalid ChatBI request bodies out of validation responses."""
        data_source_prefix = f"{API_PREFIX}/chatbi/data-sources"
        if request.url.path == data_source_prefix or request.url.path.startswith(
            f"{data_source_prefix}/"
        ):
            safe_errors = [
                {key: value for key, value in item.items() if key not in {"input", "ctx"}}
                for item in error.errors()
            ]
            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                content={"detail": safe_errors},
            )
        return await request_validation_exception_handler(request, error)

    router = APIRouter(prefix=API_PREFIX)

    @router.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        report = build_health_report(runtime_settings)
        return HealthResponse.model_validate({**report.as_dict(), "status": "ok"})

    @router.get("/meta", response_model=MetaResponse, tags=["system"])
    def meta() -> MetaResponse:
        report = build_health_report(runtime_settings)
        return MetaResponse(
            project_name=report.project_name,
            version=report.version,
            phase=CURRENT_PHASE,
            status=PROJECT_STATUS,
            config_status=report.config_status,
        )

    @router.get("/health/qdrant", response_model=QdrantHealthResponse, tags=["system"])
    def qdrant_health(request: Request) -> QdrantHealthResponse:
        readiness = request.app.state.vector_store.readiness()
        if readiness.status == "unavailable":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=readiness.error or "Qdrant is unavailable",
            )
        return QdrantHealthResponse.model_validate(readiness.model_dump())

    @router.get("/health/neo4j", response_model=Neo4jHealthResponse, tags=["system"])
    def neo4j_health(request: Request) -> Neo4jHealthResponse:
        readiness = request.app.state.graph_store.readiness()
        if readiness.status == "unavailable":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=readiness.error or "Neo4j is unavailable",
            )
        return Neo4jHealthResponse.model_validate(readiness.model_dump())

    application.include_router(router)
    application.include_router(knowledge_bases_router, prefix=API_PREFIX)
    application.include_router(documents_router, prefix=API_PREFIX)
    application.include_router(retrieval_router, prefix=API_PREFIX)
    application.include_router(rag_router, prefix=API_PREFIX)
    application.include_router(chatbi_router, prefix=API_PREFIX)
    application.include_router(reports_router, prefix=API_PREFIX)
    return application


app = create_app()
