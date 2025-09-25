from __future__ import annotations

from uuid import UUID

import pytest
from pydantic import ValidationError

from knowledge_scope.chatbi import (
    ChatBIError,
    ChatBIErrorCategory,
    QueryPolicy,
    ResultContract,
    ResultGrain,
    ResultOrderByTerm,
    ResultOutputColumn,
    SchemaColumn,
    SchemaObjectKind,
    SchemaRelation,
    SchemaSnapshot,
    SortDirection,
    SQLCandidate,
    SQLDialect,
    build_semantic_schema_context,
    validate_result_contract,
    validate_sql_result_contract,
)
from knowledge_scope.chatbi.sql_validation import _SQLSafetyValidator

DATASOURCE_ID = UUID("33333333-3333-4333-8333-333333333333")


def _relation(name: str, columns: tuple[str, ...]) -> SchemaRelation:
    return SchemaRelation(
        schema_name="public",
        name=name,
        kind=SchemaObjectKind.TABLE,
        columns=tuple(
            SchemaColumn(
                name=column,
                normalized_type="numeric" if column == "amount" else "text",
                nullable=False,
                ordinal=index,
            )
            for index, column in enumerate(columns, start=1)
        ),
    )


def _snapshot() -> SchemaSnapshot:
    return SchemaSnapshot(
        datasource_id=DATASOURCE_ID,
        dialect=SQLDialect.POSTGRESQL,
        database_name="business",
        schemas=("public",),
        relations=(
            _relation("customers", ("customer_id", "customer_name")),
            _relation("sales", ("sale_id", "customer_id", "region", "amount")),
        ),
    )


def _policy(**changes: object) -> QueryPolicy:
    return QueryPolicy(allowed_schemas=("public",), **changes)


def _validated_contract(contract: ResultContract) -> ResultContract:
    snapshot = _snapshot()
    context = build_semantic_schema_context(snapshot, max_chars=100_000)
    return validate_result_contract(
        contract,
        schema_snapshot=snapshot,
        semantic_context=context,
        policy=_policy(),
    )


def _check_sql(sql: str, contract: ResultContract, *, policy: QueryPolicy | None = None) -> None:
    snapshot = _snapshot()
    context = build_semantic_schema_context(snapshot, max_chars=100_000)
    validate_sql_result_contract(
        sql,
        contract,
        schema_snapshot=snapshot,
        semantic_context=context,
        policy=policy or _policy(),
    )


def _source(source: str, *, alias: str | None = None) -> ResultOutputColumn:
    return ResultOutputColumn(kind="source", source=source, alias=alias)


def test_valid_scalar_detail_and_grouped_contracts() -> None:
    scalar = ResultContract(
        row_grain=ResultGrain.SCALAR,
        output_columns=(ResultOutputColumn(kind="derived", expression="1"),),
    )
    detail = ResultContract(
        row_grain=ResultGrain.DETAIL,
        grain_keys=("public.sales.sale_id",),
        output_columns=(_source("public.sales.amount"),),
    )
    grouped = ResultContract(
        row_grain=ResultGrain.GROUPED,
        grain_keys=("public.customers.customer_id",),
        output_columns=(
            _source("public.customers.customer_name"),
            ResultOutputColumn(
                kind="aggregate",
                function="sum",
                source="public.sales.amount",
                alias="total_amount",
            ),
        ),
        group_by=("public.customers.customer_id", "public.customers.customer_name"),
        order_by=(ResultOrderByTerm(key="public.customers.customer_name", direction="asc"),),
    )

    assert _validated_contract(scalar) == scalar
    assert _validated_contract(detail) == detail
    assert _validated_contract(grouped) == grouped
    _check_sql("SELECT 1", scalar)
    _check_sql("SELECT s.amount FROM public.sales AS s", detail)
    _check_sql(
        "SELECT c.customer_name, SUM(s.amount) AS total_amount "
        "FROM public.sales AS s JOIN public.customers AS c "
        "ON c.customer_id = s.customer_id "
        "GROUP BY c.customer_id, c.customer_name ORDER BY c.customer_name",
        grouped,
    )


@pytest.mark.parametrize(
    "contract",
    (
        ResultContract(
            row_grain="detail",
            output_columns=(_source("public.missing.value"),),
        ),
        ResultContract(
            row_grain="detail",
            grain_keys=("public.missing.id",),
            output_columns=(_source("public.sales.amount"),),
        ),
        ResultContract(
            row_grain="grouped",
            grain_keys=("public.customers.customer_id",),
            output_columns=(_source("public.customers.customer_name"),),
            group_by=("public.customers.customer_id", "public.missing.label"),
        ),
        ResultContract(
            row_grain="detail",
            output_columns=(_source("public.sales.amount"),),
            order_by=(ResultOrderByTerm(key="public.missing.id", direction="asc"),),
        ),
    ),
)
def test_unknown_contract_references_are_rejected(contract: ResultContract) -> None:
    with pytest.raises(ChatBIError) as error:
        _validated_contract(contract)
    assert error.value.category is ChatBIErrorCategory.RESULT_CONTRACT_INVALID


def test_invalid_direction_and_limit_are_strict() -> None:
    with pytest.raises(ValidationError):
        ResultOrderByTerm(key="public.sales.amount", direction="sideways")
    with pytest.raises(ValidationError):
        ResultContract(
            row_grain="scalar",
            output_columns=(ResultOutputColumn(kind="derived", expression="1"),),
            limit=-1,
        )
    too_large = ResultContract(
        row_grain="scalar",
        output_columns=(ResultOutputColumn(kind="derived", expression="1"),),
        limit=11,
    )
    with pytest.raises(ChatBIError) as error:
        validate_result_contract(
            too_large,
            schema_snapshot=_snapshot(),
            semantic_context=build_semantic_schema_context(_snapshot(), max_chars=100_000),
            policy=_policy(max_rows=10),
        )
    assert error.value.category is ChatBIErrorCategory.RESULT_CONTRACT_INVALID


@pytest.mark.parametrize(
    "sql",
    (
        "SELECT s.sale_id FROM public.sales AS s",
        "SELECT s.amount, s.sale_id FROM public.sales AS s",
    ),
)
def test_projection_missing_extra_and_order_mismatch_fail(sql: str) -> None:
    contract = ResultContract(
        row_grain="detail",
        grain_keys=("public.sales.sale_id",),
        output_columns=(_source("public.sales.sale_id"), _source("public.sales.amount")),
    )
    with pytest.raises(ChatBIError) as error:
        _check_sql(sql, contract)
    assert error.value.category is ChatBIErrorCategory.RESULT_CONTRACT_INCONSISTENT


def test_exact_source_projection_passes() -> None:
    contract = ResultContract(
        row_grain="detail",
        grain_keys=("public.sales.sale_id",),
        output_columns=(_source("public.sales.sale_id"), _source("public.sales.amount")),
    )
    _check_sql("SELECT s.sale_id, s.amount FROM public.sales AS s", contract)


def test_alias_only_difference_does_not_fail() -> None:
    contract = ResultContract(
        row_grain="detail",
        grain_keys=("public.sales.sale_id",),
        output_columns=(_source("public.sales.amount", alias="reported_amount"),),
    )
    _check_sql("SELECT s.amount AS another_alias FROM public.sales AS s", contract)


def test_aggregate_and_derived_projection_match_structurally() -> None:
    aggregate = ResultContract(
        row_grain="scalar",
        output_columns=(
            ResultOutputColumn(
                kind="aggregate",
                function="sum",
                source="public.sales.amount",
                alias="total",
            ),
        ),
    )
    derived = ResultContract(
        row_grain="detail",
        grain_keys=("public.sales.sale_id",),
        output_columns=(
            ResultOutputColumn(
                kind="derived",
                expression="s.amount + 1",
                source_columns=("public.sales.amount",),
                alias="adjusted",
            ),
        ),
    )
    _check_sql("SELECT SUM(s.amount) AS total FROM public.sales AS s", aggregate)
    _check_sql("SELECT s.amount + 1 AS adjusted FROM public.sales AS s", derived)


def test_grain_key_need_not_be_projected_and_distinct_is_rejected() -> None:
    contract = ResultContract(
        row_grain="detail",
        grain_keys=("public.sales.sale_id",),
        output_columns=(_source("public.sales.amount"),),
    )
    _check_sql("SELECT s.amount FROM public.sales AS s", contract)
    with pytest.raises(ChatBIError) as error:
        _check_sql("SELECT DISTINCT s.amount FROM public.sales AS s", contract)
    assert error.value.category is ChatBIErrorCategory.RESULT_CONTRACT_INCONSISTENT


def test_grouping_must_retain_stable_identity_not_only_display_name() -> None:
    contract = ResultContract(
        row_grain="grouped",
        grain_keys=("public.customers.customer_id",),
        output_columns=(_source("public.customers.customer_name"),),
        group_by=("public.customers.customer_id", "public.customers.customer_name"),
    )
    _check_sql(
        "SELECT c.customer_name FROM public.customers AS c GROUP BY c.customer_id, c.customer_name",
        contract,
    )
    with pytest.raises(ChatBIError) as error:
        _check_sql(
            "SELECT c.customer_name FROM public.customers AS c GROUP BY c.customer_name",
            contract,
        )
    assert error.value.category is ChatBIErrorCategory.RESULT_CONTRACT_INCONSISTENT


def test_ordering_requires_direction_and_declared_tie_break() -> None:
    contract = ResultContract(
        row_grain="detail",
        grain_keys=("public.sales.sale_id",),
        output_columns=(_source("public.sales.region"), _source("public.sales.amount")),
        order_by=(
            ResultOrderByTerm(key="public.sales.region", direction=SortDirection.ASC),
            ResultOrderByTerm(key="public.sales.sale_id", direction=SortDirection.DESC),
        ),
    )
    _check_sql(
        "SELECT s.region, s.amount FROM public.sales AS s ORDER BY s.region ASC, s.sale_id DESC",
        contract,
    )
    for sql in (
        "SELECT s.region, s.amount FROM public.sales AS s ORDER BY s.region DESC, s.sale_id DESC",
        "SELECT s.region, s.amount FROM public.sales AS s ORDER BY s.region ASC",
    ):
        with pytest.raises(ChatBIError) as error:
            _check_sql(sql, contract)
        assert error.value.category is ChatBIErrorCategory.RESULT_CONTRACT_INCONSISTENT


def test_limit_contract_is_exact_and_none_forbids_unexpected_limit() -> None:
    limited = ResultContract(
        row_grain="scalar",
        output_columns=(ResultOutputColumn(kind="derived", expression="1"),),
        limit=2,
    )
    _check_sql("SELECT 1 LIMIT 2", limited)
    for sql in ("SELECT 1", "SELECT 1 LIMIT 3"):
        with pytest.raises(ChatBIError) as error:
            _check_sql(sql, limited)
        assert error.value.category is ChatBIErrorCategory.RESULT_CONTRACT_INCONSISTENT

    no_limit = limited.model_copy(update={"limit": None})
    _check_sql("SELECT 1", no_limit)
    with pytest.raises(ChatBIError) as error:
        _check_sql("SELECT 1 LIMIT 2", no_limit)
    assert error.value.category is ChatBIErrorCategory.RESULT_CONTRACT_INCONSISTENT


def test_contract_consistency_does_not_replace_a5_3_safety_validation() -> None:
    contract = ResultContract(
        row_grain="scalar",
        output_columns=(
            ResultOutputColumn(
                kind="derived",
                expression="pg_catalog.pg_read_file('secret')",
            ),
        ),
    )
    _check_sql("SELECT pg_catalog.pg_read_file('secret')", contract)
    candidate = SQLCandidate(
        datasource_id=DATASOURCE_ID,
        dialect=SQLDialect.POSTGRESQL,
        question="q",
        sql="SELECT pg_catalog.pg_read_file('secret')",
        context_fingerprint=_snapshot().fingerprint,
        provider="test",
        model="test",
        result_contract=contract,
    )
    snapshot = _snapshot()
    context = build_semantic_schema_context(snapshot, max_chars=100_000)
    with pytest.raises(ChatBIError):
        _SQLSafetyValidator().validate(
            candidate,
            schema_snapshot=snapshot,
            semantic_context=context,
            policy=_policy(),
        )


def test_contract_models_reject_extra_fields_and_non_structured_payload() -> None:
    with pytest.raises(ValidationError):
        ResultContract.model_validate(
            {
                "row_grain": "scalar",
                "output_columns": [{"kind": "derived", "expression": "1"}],
                "unexpected": True,
            }
        )
