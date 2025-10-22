from uuid import UUID, uuid4

import pytest
from httpx import AsyncClient

from knowledge_scope.api.app import create_app
from knowledge_scope.chatbi import (
    ChatBIResult,
    ChatBIUsageSummary,
    ColumnMetadata,
    DataSource,
    QueryLifecycleState,
    QueryPolicy,
    SchemaColumn,
    SchemaDiscoveryResult,
    SchemaObjectKind,
    SchemaRelation,
    SchemaSnapshot,
    SQLDialect,
    build_semantic_schema_context,
)
from knowledge_scope.shared.config import Settings


async def create_data_source(client: AsyncClient) -> dict[str, object]:
    response = await client.post(
        "/api/v1/chatbi/data-sources",
        json={
            "display_name": "  销售数据库  ",
            "connection_ref": "env:CHATBI_DEMO_DATABASE_URL",
            "default_schema": " public ",
        },
    )
    assert response.status_code == 201
    return response.json()


@pytest.mark.anyio
async def test_datasource_api_never_returns_connection_reference(client: AsyncClient) -> None:
    created = await create_data_source(client)

    assert created["display_name"] == "销售数据库"
    assert created["dialect"] == "postgresql"
    assert created["default_schema"] == "public"
    assert created["connection_configured"] is True
    assert "connection_ref" not in created

    detail = await client.get(f"/api/v1/chatbi/data-sources/{created['id']}")
    listing = await client.get("/api/v1/chatbi/data-sources?limit=10")

    assert detail.status_code == 200
    assert detail.json() == created
    assert listing.status_code == 200
    assert listing.json()["items"] == [created]
    assert listing.json()["total"] == 1


@pytest.mark.anyio
async def test_datasource_api_updates_safe_metadata_and_supports_disable_delete(
    client: AsyncClient,
) -> None:
    created = await create_data_source(client)

    updated_response = await client.patch(
        f"/api/v1/chatbi/data-sources/{created['id']}",
        json={"display_name": "销售数据库 (停用)", "enabled": False},
    )

    assert updated_response.status_code == 200
    updated = updated_response.json()
    assert updated["display_name"] == "销售数据库 (停用)"
    assert updated["enabled"] is False
    assert updated["connection_configured"] is True
    assert "connection_ref" not in updated

    delete_response = await client.delete(f"/api/v1/chatbi/data-sources/{created['id']}")
    missing_response = await client.get(f"/api/v1/chatbi/data-sources/{created['id']}")

    assert delete_response.status_code == 204
    assert delete_response.content == b""
    assert missing_response.status_code == 404


@pytest.mark.anyio
async def test_datasource_api_rejects_raw_urls_and_invalid_patches(client: AsyncClient) -> None:
    raw_url_response = await client.post(
        "/api/v1/chatbi/data-sources",
        json={
            "display_name": "不安全数据源",
            "connection_ref": "postgresql://user:password@db.example.test/business",
        },
    )
    empty_patch_response = await client.patch(
        "/api/v1/chatbi/data-sources/00000000-0000-0000-0000-000000000000",
        json={},
    )

    assert raw_url_response.status_code == 422
    assert "password" not in raw_url_response.text
    assert empty_patch_response.status_code == 422


class _FakeSchemaDiscoveryService:
    async def discover(
        self,
        data_source: DataSource,
        _policy: QueryPolicy,
        *,
        max_chars: int,
    ) -> SchemaDiscoveryResult:
        snapshot = SchemaSnapshot(
            datasource_id=data_source.id,
            dialect=SQLDialect.POSTGRESQL,
            database_name="business",
            schemas=("public",),
            relations=(
                SchemaRelation(
                    schema_name="public",
                    name="sales",
                    kind=SchemaObjectKind.TABLE,
                    columns=(
                        SchemaColumn(
                            name="id",
                            normalized_type="integer",
                            nullable=False,
                            ordinal=1,
                        ),
                    ),
                    primary_key=("id",),
                ),
            ),
        )
        return SchemaDiscoveryResult(
            snapshot=snapshot,
            fingerprint=snapshot.fingerprint,
            context=build_semantic_schema_context(snapshot, max_chars=max_chars),
        )


class _FakeChatBIAgent:
    def __init__(self, datasource_id: UUID) -> None:
        self.datasource_id = datasource_id
        self.questions: list[str] = []

    async def ask(self, datasource_id: UUID, question: str, **_kwargs: object) -> ChatBIResult:
        self.questions.append(question)
        assert datasource_id == self.datasource_id
        return ChatBIResult(
            query_id=uuid4(),
            datasource_id=datasource_id,
            answer="按地区统计完成。",
            execution_status=QueryLifecycleState.SUCCEEDED,
            redacted_sql="SELECT region_code, COUNT(*) FROM public.sales GROUP BY region_code",
            columns=[
                ColumnMetadata(name="region_code", data_type="text", ordinal=0),
                ColumnMetadata(name="count", data_type="bigint", ordinal=1),
            ],
            rows=[["华东", 3], ["华南", 2]],
            row_count=2,
            sql_attempts=1,
            repair_attempts=0,
            usage=ChatBIUsageSummary(
                llm_calls=2,
                provider_attempts=2,
                provider="stub",
                model="stub-model",
                task_type_counts={"chatbi_eligibility": 1, "nl2sql": 1},
            ),
            error_category=None,
            error_message=None,
        )


@pytest.mark.anyio
async def test_analysis_endpoint_projects_bounded_agent_result_without_internal_metadata(
    postgres_test_engine,
) -> None:
    datasource_id = uuid4()
    agent = _FakeChatBIAgent(datasource_id)
    application = create_app(
        Settings(_env_file=None, environment="test"),
        database_engine=postgres_test_engine,
        vector_store=object(),  # type: ignore[arg-type]
        graph_store=object(),  # type: ignore[arg-type]
        chatbi_agent_service=agent,  # type: ignore[arg-type]
    )

    from httpx import ASGITransport

    async with AsyncClient(
        transport=ASGITransport(app=application), base_url="http://testserver"
    ) as http_client:
        response = await http_client.post(
            f"/api/v1/chatbi/data-sources/{datasource_id}/ask",
            json={"question": "统计每个地区的客户数量。"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["datasource_id"] == str(datasource_id)
    assert payload["rows"] == [["华东", 3], ["华南", 2]]
    assert payload["row_count"] == 2
    assert payload["redacted_sql"] is not None
    assert "usage" not in payload
    assert "schema_fingerprint" not in response.text
    assert agent.questions == ["统计每个地区的客户数量。"]


@pytest.mark.anyio
async def test_schema_endpoint_returns_safe_snapshot_and_context(
    client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = await create_data_source(client)
    monkeypatch.setattr(
        "knowledge_scope.chatbi.api._build_schema_discovery_service",
        lambda _settings: _FakeSchemaDiscoveryService(),
    )

    response = await client.get(f"/api/v1/chatbi/data-sources/{created['id']}/schema")

    assert response.status_code == 200
    payload = response.json()
    assert payload["snapshot"]["datasource_id"] == created["id"]
    assert payload["context"]["included_relations"] == ["public.sales"]
    assert "connection_ref" not in response.text
    assert "business" in response.text


@pytest.mark.anyio
async def test_schema_endpoint_rejects_disabled_datasource_without_resolving_credentials(
    client: AsyncClient,
) -> None:
    created = await create_data_source(client)
    updated = await client.patch(
        f"/api/v1/chatbi/data-sources/{created['id']}",
        json={"enabled": False},
    )
    assert updated.status_code == 200

    response = await client.get(f"/api/v1/chatbi/data-sources/{created['id']}/schema")

    assert response.status_code == 409
    assert response.json()["detail"] == {
        "category": "datasource_disabled",
        "message": "the data source is disabled",
    }
