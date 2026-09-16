from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from knowledge_scope.chatbi import (
    ChatBIAgentService,
    DataSource,
    EnvironmentCredentialResolver,
    NL2SQLInput,
    NL2SQLService,
    PostgresExecutionAdapter,
    QueryLifecycleState,
    QueryPolicy,
    SchemaDiscoveryService,
    SQLDialect,
    SQLExecutionService,
)
from knowledge_scope.chatbi.postgres_schema import PostgresSchemaInspector
from knowledge_scope.llm.schemas import LLMRequest, LLMResult

DATASOURCE_ID = UUID("22222222-2222-4222-8222-222222222222")


class _DeterministicAgentGateway:
    """A provider-free gateway stub that still exercises both agent calls."""

    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResult:
        self.requests.append(request)
        if request.task_type == "nl2sql":
            text = json.dumps(
                {
                    "result_contract": {
                        "row_grain": "grouped",
                        "grain_keys": ["chatbi_demo.customers.customer_id"],
                        "output_columns": [
                            {
                                "kind": "source",
                                "source": "chatbi_demo.customers.customer_name",
                            },
                            {
                                "kind": "aggregate",
                                "function": "sum",
                                "source": "chatbi_demo.sales.amount",
                                "alias": "total_amount",
                            },
                        ],
                        "group_by": [
                            "chatbi_demo.customers.customer_id",
                            "chatbi_demo.customers.customer_name",
                        ],
                        "order_by": [
                            {
                                "key": "chatbi_demo.customers.customer_name",
                                "direction": "asc",
                            }
                        ],
                        "limit": None,
                    },
                    "sql": (
                        "SELECT c.customer_name, SUM(s.amount) AS total_amount "
                        "FROM chatbi_demo.sales AS s "
                        "JOIN chatbi_demo.customers AS c ON c.customer_id = s.customer_id "
                        "GROUP BY c.customer_id, c.customer_name ORDER BY c.customer_name"
                    ),
                },
                ensure_ascii=False,
            )
            input_tokens, output_tokens = 10, 5
        elif request.task_type == "chatbi_analysis":
            text = '{"answer":"结果包含 2 位客户。","warning":null}'
            input_tokens, output_tokens = 11, 7
        else:
            raise AssertionError(f"unexpected task type: {request.task_type}")
        return LLMResult(
            text=text,
            provider="fake-agent",
            model="fake-agent-model",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=1,
            provider_attempts=1,
        )


class _RegisteredDataSource:
    def __init__(self, source: DataSource) -> None:
        self.source = source
        self.calls = 0

    async def get(self, datasource_id: UUID) -> DataSource | None:
        assert datasource_id == self.source.id
        self.calls += 1
        return self.source


class _DeterministicContractGateway:
    """Return a fixed sequence of ResultContract + SQL payloads."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResult:
        self.requests.append(request)
        if request.task_type != "nl2sql" or not self.responses:
            raise AssertionError("unexpected or exhausted NL2SQL request")
        return LLMResult(
            text=self.responses.pop(0),
            provider="fake-contract",
            model="fake-contract-model",
            input_tokens=10,
            output_tokens=5,
            latency_ms=1,
            provider_attempts=1,
        )


def _contract_payload(contract: dict[str, object], sql: str) -> str:
    return json.dumps(
        {"result_contract": contract, "sql": sql},
        ensure_ascii=False,
    )


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("KNOWLEDGE_SCOPE_RUN_CHATBI_AGENT_INTEGRATION") != "1",
    reason="set KNOWLEDGE_SCOPE_RUN_CHATBI_AGENT_INTEGRATION=1 to run",
)
@pytest.mark.anyio
async def test_real_postgresql_chatbi_agent_end_to_end(
    postgres_test_engine: Any,
    postgres_test_database: str,
) -> None:
    """Run the agent through real discovery, validation, execution and normalization."""
    fixture_path = Path(__file__).parent / "fixtures" / "chatbi_demo.sql"
    fixture_sql = fixture_path.read_text(encoding="utf-8")
    async with postgres_test_engine.begin() as connection:
        for statement in fixture_sql.split(";"):
            if statement.strip():
                await connection.exec_driver_sql(statement)

    os.environ["CHATBI_AGENT_TEST_DATABASE_URL"] = postgres_test_database
    try:
        timestamp = datetime(2026, 1, 1, tzinfo=UTC)
        source = DataSource(
            id=DATASOURCE_ID,
            display_name="ChatBI 演示库",
            dialect=SQLDialect.POSTGRESQL,
            enabled=True,
            connection_ref="env:CHATBI_AGENT_TEST_DATABASE_URL",
            default_database=None,
            default_schema="chatbi_demo",
            created_at=timestamp,
            updated_at=timestamp,
        )
        registry = _RegisteredDataSource(source)
        discovery = SchemaDiscoveryService(
            EnvironmentCredentialResolver(),
            PostgresSchemaInspector(),
        )
        gateway = _DeterministicAgentGateway()
        generation = NL2SQLService(
            gateway,
            schema_discovery=discovery,
            data_source_provider=registry,
        )
        execution = SQLExecutionService(
            generation,
            EnvironmentCredentialResolver(),
            PostgresExecutionAdapter(),
        )

        result = await ChatBIAgentService(generation, execution, gateway).ask(
            DATASOURCE_ID,
            "按客户统计销售额",
            policy=QueryPolicy(allowed_schemas=("chatbi_demo",), max_rows=10),
            max_chars=20_000,
        )

        assert result.execution_status is QueryLifecycleState.SUCCEEDED
        assert result.datasource_id == DATASOURCE_ID
        assert result.answer == "结果包含 2 位客户。"
        assert result.row_count == 2
        assert result.truncated is False
        assert result.truncation_reason is None
        assert result.error_category is None
        assert result.redacted_sql is not None
        assert "chatbi_demo" in result.redacted_sql
        assert result.usage.llm_calls == 2
        assert result.usage.provider_attempts == 2
        assert result.usage.provider == "fake-agent"
        assert result.usage.model == "fake-agent-model"
        assert result.usage.input_tokens == 21
        assert result.usage.output_tokens == 12
        assert [request.task_type for request in gateway.requests] == [
            "nl2sql",
            "chatbi_analysis",
        ]
        assert registry.calls >= 2
        assert [event.event for event in result.trace] == [
            "schema_prepared",
            "sql_generated",
            "validation_accepted",
            "sql_executed",
            "result_normalized",
            "analysis_requested",
            "analysis_completed",
        ]

        safe_result = result.to_deterministic_json()
        assert "CHATBI_AGENT_TEST_DATABASE_URL" not in safe_result
        assert "knowledgescope:knowledgescope" not in safe_result
        assert "password" not in safe_result.lower()
    finally:
        os.environ.pop("CHATBI_AGENT_TEST_DATABASE_URL", None)


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("KNOWLEDGE_SCOPE_RUN_CHATBI_AGENT_INTEGRATION") != "1",
    reason="set KNOWLEDGE_SCOPE_RUN_CHATBI_AGENT_INTEGRATION=1 to run",
)
@pytest.mark.anyio
async def test_real_postgresql_result_contract_shapes(
    postgres_test_engine: Any,
    postgres_test_database: str,
) -> None:
    """Exercise representative contracts against the isolated demo database."""
    fixture_path = Path(__file__).parent / "fixtures" / "chatbi_demo.sql"
    fixture_sql = fixture_path.read_text(encoding="utf-8")
    async with postgres_test_engine.begin() as connection:
        for statement in fixture_sql.split(";"):
            if statement.strip():
                await connection.exec_driver_sql(statement)

    os.environ["CHATBI_CONTRACT_TEST_DATABASE_URL"] = postgres_test_database
    try:
        timestamp = datetime(2026, 1, 1, tzinfo=UTC)
        source = DataSource(
            id=DATASOURCE_ID,
            display_name="ChatBI 契约演示库",
            dialect=SQLDialect.POSTGRESQL,
            enabled=True,
            connection_ref="env:CHATBI_CONTRACT_TEST_DATABASE_URL",
            default_database=None,
            default_schema="chatbi_demo",
            created_at=timestamp,
            updated_at=timestamp,
        )
        registry = _RegisteredDataSource(source)
        discovery = SchemaDiscoveryService(
            EnvironmentCredentialResolver(),
            PostgresSchemaInspector(),
        )
        policy = QueryPolicy(allowed_schemas=("chatbi_demo",), max_rows=10)
        cases = (
            (
                "标量聚合",
                {
                    "row_grain": "scalar",
                    "output_columns": [
                        {
                            "kind": "aggregate",
                            "function": "sum",
                            "source": "chatbi_demo.sales.amount",
                        }
                    ],
                },
                "SELECT SUM(s.amount) FROM chatbi_demo.sales AS s",
            ),
            (
                "每笔销售明细",
                {
                    "row_grain": "detail",
                    "grain_keys": ["chatbi_demo.sales.sale_id"],
                    "output_columns": [
                        {"kind": "source", "source": "chatbi_demo.sales.sale_id"},
                        {"kind": "source", "source": "chatbi_demo.sales.amount"},
                    ],
                    "order_by": [{"key": "chatbi_demo.sales.sale_id", "direction": "asc"}],
                },
                "SELECT s.sale_id, s.amount FROM chatbi_demo.sales AS s ORDER BY s.sale_id ASC",
            ),
            (
                "按客户分组",
                {
                    "row_grain": "grouped",
                    "grain_keys": ["chatbi_demo.customers.customer_id"],
                    "output_columns": [
                        {
                            "kind": "source",
                            "source": "chatbi_demo.customers.customer_id",
                        },
                        {
                            "kind": "aggregate",
                            "function": "sum",
                            "source": "chatbi_demo.sales.amount",
                        },
                    ],
                    "group_by": ["chatbi_demo.customers.customer_id"],
                    "order_by": [
                        {
                            "key": "chatbi_demo.customers.customer_id",
                            "direction": "asc",
                        }
                    ],
                },
                "SELECT c.customer_id, SUM(s.amount) "
                "FROM chatbi_demo.sales AS s "
                "JOIN chatbi_demo.customers AS c ON c.customer_id = s.customer_id "
                "GROUP BY c.customer_id ORDER BY c.customer_id ASC",
            ),
            (
                "按客户和地区分组",
                {
                    "row_grain": "grouped",
                    "grain_keys": [
                        "chatbi_demo.customers.customer_id",
                        "chatbi_demo.sales.region",
                    ],
                    "output_columns": [
                        {
                            "kind": "source",
                            "source": "chatbi_demo.customers.customer_id",
                        },
                        {"kind": "source", "source": "chatbi_demo.sales.region"},
                        {
                            "kind": "aggregate",
                            "function": "sum",
                            "source": "chatbi_demo.sales.amount",
                        },
                    ],
                    "group_by": [
                        "chatbi_demo.customers.customer_id",
                        "chatbi_demo.sales.region",
                    ],
                    "order_by": [
                        {
                            "key": "chatbi_demo.customers.customer_id",
                            "direction": "asc",
                        },
                        {"key": "chatbi_demo.sales.region", "direction": "asc"},
                    ],
                },
                "SELECT c.customer_id, s.region, SUM(s.amount) "
                "FROM chatbi_demo.sales AS s "
                "JOIN chatbi_demo.customers AS c ON c.customer_id = s.customer_id "
                "GROUP BY c.customer_id, s.region "
                "ORDER BY c.customer_id ASC, s.region ASC",
            ),
            (
                "金额 Top-K",
                {
                    "row_grain": "detail",
                    "grain_keys": ["chatbi_demo.sales.sale_id"],
                    "output_columns": [
                        {"kind": "source", "source": "chatbi_demo.sales.sale_id"},
                        {"kind": "source", "source": "chatbi_demo.sales.amount"},
                    ],
                    "order_by": [
                        {"key": "chatbi_demo.sales.amount", "direction": "desc"},
                        {"key": "chatbi_demo.sales.sale_id", "direction": "asc"},
                    ],
                    "limit": 2,
                },
                "SELECT s.sale_id, s.amount FROM chatbi_demo.sales AS s "
                "ORDER BY s.amount DESC, s.sale_id ASC LIMIT 2",
            ),
            (
                "派生平均值",
                {
                    "row_grain": "scalar",
                    "output_columns": [
                        {
                            "kind": "derived",
                            "expression": "COALESCE(SUM(s.amount), 0)",
                            "source_columns": [
                                "chatbi_demo.sales.amount",
                            ],
                            "alias": "total_amount",
                        }
                    ],
                },
                "SELECT COALESCE(SUM(s.amount), 0) AS total_amount FROM chatbi_demo.sales AS s",
            ),
        )
        gateway = _DeterministicContractGateway(
            [_contract_payload(contract, sql) for _name, contract, sql in cases]
        )
        generation = NL2SQLService(
            gateway,
            schema_discovery=discovery,
            data_source_provider=registry,
        )
        execution = SQLExecutionService(
            generation,
            EnvironmentCredentialResolver(),
            PostgresExecutionAdapter(),
        )

        for name, _contract, _sql in cases:
            generated = await generation.generate_for_registered_data_source(
                DATASOURCE_ID,
                NL2SQLInput(datasource_id=DATASOURCE_ID, question=name),
                policy=policy,
                max_chars=20_000,
            )
            outcome = await execution.execute_for_registered_data_source(
                DATASOURCE_ID,
                generated.candidate,
                policy=policy,
                max_chars=20_000,
            )
            assert outcome.result.state is QueryLifecycleState.SUCCEEDED, name
            assert outcome.result.columns, name

        assert gateway.responses == []
        assert len(gateway.requests) == len(cases)
        assert registry.calls >= len(cases) * 2
    finally:
        os.environ.pop("CHATBI_CONTRACT_TEST_DATABASE_URL", None)
