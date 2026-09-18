from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

import knowledge_scope.evaluation.semantic_evidence as semantic_module
from knowledge_scope.chatbi.agent import ChatBIResult, ChatBITraceEvent, ChatBIUsageSummary
from knowledge_scope.chatbi.errors import ChatBIErrorCategory
from knowledge_scope.chatbi.nl2sql_models import ValidatedSQL
from knowledge_scope.chatbi.policy import SQLDialect
from knowledge_scope.chatbi.schemas import (
    ColumnMetadata,
    QueryExecutionResult,
    QueryLifecycleState,
)
from knowledge_scope.evaluation.semantic_evidence import (
    MAX_EXACT_RESULT_BYTES,
    MAX_EXACT_ROWS,
    EvidenceStatus,
    SemanticEvidenceError,
    _canonical_json,
    _canonical_rows,
    capture_semantic_evidence,
    evidence_digest_for,
)
from knowledge_scope.llm import LLMProviderInvocation

DATASET = "a" * 64
FIXTURE = "b" * 64
SCHEMA = "c" * 64


def _validated(sql: str, *, datasource_id=None) -> ValidatedSQL:
    datasource_id = datasource_id or uuid4()
    return ValidatedSQL._from_validator(
        validation_version="a5.3-v1",
        datasource_id=datasource_id,
        dialect=SQLDialect.POSTGRESQL,
        provider="fake",
        model="fake-model",
        prompt_version="a5.3-v3",
        original_sql=sql,
        normalized_sql=sql,
        referenced_schemas=("chatbi_demo",),
        referenced_relations=("chatbi_demo.sales",),
        schema_fingerprint=SCHEMA,
        policy_fingerprint="d" * 64,
        limit_bounded=True,
        effective_limit=100,
        limit_source="policy_default",
    )


def _result(datasource_id, rows: list[list[object]], *, columns: tuple[str, ...] = ("value",)):
    metadata = [
        ColumnMetadata(name=name, data_type="text", nullable=True, ordinal=index)
        for index, name in enumerate(columns)
    ]
    return QueryExecutionResult(
        query_id=uuid4(),
        datasource_id=datasource_id,
        state=QueryLifecycleState.SUCCEEDED,
        columns=metadata,
        rows=rows,
        row_count=len(rows),
        max_rows=max(1, len(rows)),
        duration_ms=1,
    )


def _capture(
    sql: str,
    *,
    rows=None,
    fixture: str = FIXTURE,
    columns: tuple[str, ...] = ("value",),
    case_id: str = "simple-01",
    provider_invocations: Iterable[LLMProviderInvocation] | None = None,
):
    datasource_id = uuid4()
    result = _result(
        datasource_id,
        rows if rows is not None else [["ok"]],
        columns=columns,
    )
    chatbi_result = ChatBIResult(
        query_id=result.query_id,
        datasource_id=datasource_id,
        execution_status=QueryLifecycleState.SUCCEEDED,
        row_count=result.row_count,
        sql_attempts=1,
        repair_attempts=0,
        usage=ChatBIUsageSummary(llm_calls=1, provider_attempts=1),
    )
    return capture_semantic_evidence(
        case_id=case_id,
        split="dev",
        datasource_id=datasource_id,
        dataset_fingerprint=DATASET,
        fixture_fingerprint=fixture,
        schema_fingerprint=SCHEMA,
        synthetic_fixture_fingerprint=FIXTURE,
        validated_sql=_validated(sql, datasource_id=datasource_id),
        execution_result=result,
        chatbi_result=chatbi_result,
        provider_invocations=provider_invocations if provider_invocations is not None else [],
        prompt_version="a5.3-v3",
        reasoning_mode="disabled",
    )


def _generation_invocation(index: int, *, stage: str = "generation") -> LLMProviderInvocation:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return LLMProviderInvocation(
        case_id=f"generation-{index}",
        logical_stage=stage,
        attempt_index=index,
        provider="fake",
        model="fake-model",
        started_at=now,
        completed_at=now,
        duration_ms=1,
        outcome="success",
        finish_reason="stop",
        input_tokens=10,
        output_tokens=5,
        output_token_budget=1024,
        llm_result_returned=True,
        response_parse_outcome="parsed",
    )


def test_evidence_digest_is_deterministic_and_semantic_changes_are_visible() -> None:
    evidence = _capture("SELECT amount FROM chatbi_demo.sales")
    assert evidence_digest_for(evidence) == evidence.evidence_digest

    reloaded = type(evidence).model_validate(evidence.model_dump(mode="json"))
    assert reloaded.evidence_digest == evidence.evidence_digest

    changed_evidence = _capture(
        "SELECT amount FROM chatbi_demo.sales",
        case_id="simple-02",
    )
    assert changed_evidence.evidence_digest != evidence.evidence_digest


@pytest.mark.parametrize(
    "replacement",
    [None, "", "0" * 63, "g" * 64, "A" * 64, "9" * 64],
)
def test_serialized_evidence_requires_a_valid_current_digest(replacement: str | None) -> None:
    evidence = _capture("SELECT amount FROM chatbi_demo.sales")
    payload = evidence.model_dump(mode="json")
    if replacement is None:
        payload.pop("evidence_digest")
    else:
        payload["evidence_digest"] = replacement
    with pytest.raises(ValidationError):
        type(evidence).model_validate(payload)

    stale = evidence.model_dump(mode="json")
    stale["case_id"] = "stale-digest"
    with pytest.raises(ValidationError):
        type(evidence).model_validate(stale)


def test_serialized_evidence_rejects_extra_version_and_completeness_values() -> None:
    evidence = _capture("SELECT amount FROM chatbi_demo.sales")

    extra = evidence.model_dump(mode="json")
    extra["unexpected"] = True
    with pytest.raises(ValidationError):
        type(evidence).model_validate(extra)

    version = evidence.model_dump(mode="json")
    version["evidence_version"] = "future-v2"
    with pytest.raises(ValidationError):
        type(evidence).model_validate(version)

    completeness = evidence.model_dump(mode="json")
    assert completeness["sql"] is not None
    completeness["sql"]["completeness"] = "unknown"
    with pytest.raises(ValidationError):
        type(evidence).model_validate(completeness)

    with pytest.raises(SemanticEvidenceError, match="provenance"):
        evidence.validate_provenance(
            dataset_fingerprint=DATASET,
            fixture_fingerprint="e" * 64,
            schema_fingerprint=SCHEMA,
        )


def test_provider_failure_without_parse_outcome_is_captureable() -> None:
    datasource_id = uuid4()
    result = ChatBIResult(
        query_id=uuid4(),
        datasource_id=datasource_id,
        execution_status=QueryLifecycleState.FAILED,
        row_count=0,
        sql_attempts=1,
        repair_attempts=0,
        usage=ChatBIUsageSummary(llm_calls=1, provider_attempts=1),
        error_category=ChatBIErrorCategory.GENERATION_FAILED,
        error_message="safe provider failure",
    )
    now = datetime(2026, 1, 1, tzinfo=UTC)
    evidence = capture_semantic_evidence(
        case_id="provider-failure",
        split="dev",
        datasource_id=datasource_id,
        dataset_fingerprint=DATASET,
        fixture_fingerprint=FIXTURE,
        schema_fingerprint=SCHEMA,
        synthetic_fixture_fingerprint=FIXTURE,
        validated_sql=None,
        execution_result=None,
        chatbi_result=result,
        provider_invocations=[
            LLMProviderInvocation(
                case_id="provider-failure",
                logical_stage="generation",
                attempt_index=1,
                provider="fake",
                model="fake-model",
                started_at=now,
                completed_at=now,
                duration_ms=1,
                outcome="failure",
                error_category="provider_api_error",
                llm_result_returned=False,
                response_parse_outcome=None,
            )
        ],
        prompt_version="a5.3-v3",
        reasoning_mode="disabled",
    )
    assert evidence.generation.initial_parse_status == "no_response"
    assert evidence.generation.attempts[0].lifecycle_status == "no_response"
    assert "execution_not_reached" in evidence.unavailable_reasons


def test_capture_lifecycle_records_unreached_sql_validation_and_execution() -> None:
    datasource_id = uuid4()
    no_candidate = ChatBIResult(
        query_id=uuid4(),
        datasource_id=datasource_id,
        execution_status=QueryLifecycleState.FAILED,
        row_count=0,
        sql_attempts=1,
        repair_attempts=0,
        usage=ChatBIUsageSummary(llm_calls=1, provider_attempts=1),
        error_category=ChatBIErrorCategory.MALFORMED_MODEL_OUTPUT,
        error_message="safe malformed output",
    )
    unavailable = capture_semantic_evidence(
        case_id="no-candidate",
        split="dev",
        datasource_id=datasource_id,
        dataset_fingerprint=DATASET,
        fixture_fingerprint=FIXTURE,
        schema_fingerprint=SCHEMA,
        synthetic_fixture_fingerprint=FIXTURE,
        validated_sql=None,
        execution_result=None,
        chatbi_result=no_candidate,
        provider_invocations=[],
        prompt_version="a5.3-v3",
        reasoning_mode="disabled",
    )
    assert "sql_candidate_unavailable" in unavailable.unavailable_reasons
    assert "validation_not_reached" in unavailable.unavailable_reasons
    assert "execution_not_reached" in unavailable.unavailable_reasons

    candidate_but_not_validated = no_candidate.model_copy(
        update={"redacted_sql": "SELECT amount FROM chatbi_demo.sales"}
    )
    candidate_evidence = capture_semantic_evidence(
        case_id="candidate-not-validated",
        split="dev",
        datasource_id=datasource_id,
        dataset_fingerprint=DATASET,
        fixture_fingerprint=FIXTURE,
        schema_fingerprint=SCHEMA,
        synthetic_fixture_fingerprint=FIXTURE,
        validated_sql=None,
        execution_result=None,
        chatbi_result=candidate_but_not_validated,
        provider_invocations=[],
        prompt_version="a5.3-v3",
        reasoning_mode="disabled",
    )
    assert "sql_candidate_unavailable" not in candidate_evidence.unavailable_reasons
    assert "validation_not_reached" in candidate_evidence.unavailable_reasons

    execution_not_reached = ChatBIResult(
        query_id=uuid4(),
        datasource_id=datasource_id,
        execution_status=QueryLifecycleState.FAILED,
        redacted_sql="SELECT amount FROM chatbi_demo.sales",
        row_count=0,
        sql_attempts=1,
        repair_attempts=0,
        usage=ChatBIUsageSummary(llm_calls=1, provider_attempts=1),
        error_category=ChatBIErrorCategory.EXECUTION_FAILED,
        error_message="safe execution failure",
        trace=[ChatBITraceEvent(event="validation_accepted", step=1)],
    )
    execution_evidence = capture_semantic_evidence(
        case_id="execution-not-reached",
        split="dev",
        datasource_id=datasource_id,
        dataset_fingerprint=DATASET,
        fixture_fingerprint=FIXTURE,
        schema_fingerprint=SCHEMA,
        synthetic_fixture_fingerprint=FIXTURE,
        validated_sql=_validated(
            "SELECT amount FROM chatbi_demo.sales", datasource_id=datasource_id
        ),
        execution_result=None,
        chatbi_result=execution_not_reached,
        provider_invocations=[],
        prompt_version="a5.3-v3",
        reasoning_mode="disabled",
    )
    assert execution_evidence.sql is not None
    assert execution_evidence.execution is not None
    assert "execution_result_unavailable" in execution_evidence.execution.unavailable_reasons
    assert "execution_not_reached" in execution_evidence.unavailable_reasons


def test_malformed_completed_response_is_parse_failed_not_provider_failure() -> None:
    datasource_id = uuid4()
    result = ChatBIResult(
        query_id=uuid4(),
        datasource_id=datasource_id,
        execution_status=QueryLifecycleState.FAILED,
        row_count=0,
        sql_attempts=1,
        repair_attempts=0,
        usage=ChatBIUsageSummary(llm_calls=1, provider_attempts=1),
        error_category=ChatBIErrorCategory.MALFORMED_MODEL_OUTPUT,
        error_message="safe malformed output",
    )
    now = datetime(2026, 1, 1, tzinfo=UTC)
    evidence = capture_semantic_evidence(
        case_id="parse-failure",
        split="dev",
        datasource_id=datasource_id,
        dataset_fingerprint=DATASET,
        fixture_fingerprint=FIXTURE,
        schema_fingerprint=SCHEMA,
        synthetic_fixture_fingerprint=FIXTURE,
        validated_sql=None,
        execution_result=None,
        chatbi_result=result,
        provider_invocations=[
            LLMProviderInvocation(
                case_id="parse-failure",
                logical_stage="generation",
                attempt_index=1,
                provider="fake",
                model="fake-model",
                started_at=now,
                completed_at=now,
                duration_ms=1,
                outcome="success",
                llm_result_returned=True,
                response_parse_outcome="structured_output_parse_error",
            )
        ],
        prompt_version="a5.3-v3",
        reasoning_mode="disabled",
    )
    assert evidence.generation.initial_parse_status == "parse_failed"
    assert "parse_failed" in evidence.unavailable_reasons


@pytest.mark.parametrize("attempt_count", (128, 129))
def test_generation_attempt_metadata_is_bounded_and_deterministic(attempt_count: int) -> None:
    invocations = [_generation_invocation(index + 1) for index in range(attempt_count)]
    first = _capture("SELECT amount FROM chatbi_demo.sales", provider_invocations=invocations)
    second = _capture("SELECT amount FROM chatbi_demo.sales", provider_invocations=invocations)

    assert len(first.generation.attempts) == min(attempt_count, 128)
    assert first.generation.attempts_truncated is (attempt_count > 128)
    assert first.generation.completeness == ("partial" if attempt_count > 128 else "complete")
    assert first.generation.model_dump(mode="json") == second.generation.model_dump(mode="json")
    if attempt_count > 128:
        assert "attempt_metadata_truncated" in first.generation.completeness_reasons
        assert "attempt_metadata_truncated" in first.unavailable_reasons


def test_generation_attempt_metadata_does_not_materialize_a_large_iterable() -> None:
    consumed: list[int] = []

    def invocations() -> Iterable[LLMProviderInvocation]:
        for index in range(1000):
            consumed.append(index)
            yield _generation_invocation(index + 1)

    evidence = _capture("SELECT amount FROM chatbi_demo.sales", provider_invocations=invocations())

    assert len(evidence.generation.attempts) == 128
    assert evidence.generation.attempts_truncated is True
    assert len(consumed) == 129


def test_combined_evidence_overflow_uses_a_bounded_degradation() -> None:
    columns = tuple(f"column_{index}" for index in range(50))
    rows = [["x" * 62_000, *range(49)]]
    now = datetime(2026, 1, 1, tzinfo=UTC)
    invocations = [
        LLMProviderInvocation(
            case_id="overflow-" + "x" * 119,
            logical_stage="generation",
            attempt_index=index + 1,
            provider="p" * 64,
            model="m" * 255,
            started_at=now,
            completed_at=now,
            duration_ms=1,
            outcome="success",
            finish_reason="f" * 64,
            input_tokens=100,
            output_tokens=100,
            output_token_budget=1024,
            llm_result_returned=True,
            response_parse_outcome="parsed",
        )
        for index in range(128)
    ]
    evidence = _capture(
        "SELECT " + ", ".join(f"{index}" for index in range(50)),
        rows=rows,
        columns=columns,
        case_id="overflow",
        provider_invocations=invocations,
    )
    assert len(evidence.model_dump_json().encode("utf-8")) <= 128 * 1024
    assert evidence.evidence_status is EvidenceStatus.TRUNCATED
    assert "evidence_record_bytes_bound" in evidence.unavailable_reasons


def test_static_fallback_is_directly_bounded_and_non_recursive(monkeypatch) -> None:
    evidence = _capture("SELECT amount FROM chatbi_demo.sales")
    calls = 0
    real_record_candidate = semantic_module._record_candidate

    def fail_first_three(**kwargs):
        nonlocal calls
        calls += 1
        if calls <= 4:
            return None
        return real_record_candidate(**kwargs)

    monkeypatch.setattr(semantic_module, "_record_candidate", fail_first_three)
    fallback = semantic_module._bounded_record(
        case_id="static-fallback",
        split="dev",
        datasource_id=evidence.datasource_id,
        dataset_fingerprint=DATASET,
        fixture_fingerprint=FIXTURE,
        schema_fingerprint=SCHEMA,
        evidence_mode="synthetic_exact_v1",
        generation=evidence.generation,
        sql=evidence.sql,
        execution=evidence.execution,
        status=EvidenceStatus.AVAILABLE,
        reasons=("forced_static_fallback",),
    )

    assert calls == 5
    assert fallback.evidence_status is EvidenceStatus.TRUNCATED
    assert fallback.sql is None
    assert fallback.execution is None
    assert fallback.generation.attempts == ()
    assert fallback.datasource_id == evidence.datasource_id
    assert fallback.dataset_fingerprint == DATASET
    assert evidence_digest_for(fallback) == fallback.evidence_digest
    assert len(fallback.model_dump_json().encode("utf-8")) <= 128 * 1024
    assert "evidence_record_bytes_bound" in fallback.unavailable_reasons


def test_projection_predicate_join_group_order_and_bounds_are_canonical() -> None:
    evidence = _capture(
        "SELECT s.amount, SUM(DISTINCT s.amount) AS total "
        "FROM chatbi_demo.sales AS s "
        "JOIN chatbi_demo.customers AS c ON s.customer_id = c.customer_id "
        "WHERE s.amount > 10 AND s.region IS NOT NULL "
        "GROUP BY s.amount ORDER BY s.amount DESC NULLS LAST LIMIT 2 OFFSET 1"
    )
    assert evidence.sql is not None
    assert [item.kind for item in evidence.sql.projections] == ["source_column", "aggregate"]
    assert evidence.sql.projections[0].expression.relation == "chatbi_demo.sales"
    assert evidence.sql.joins[0].join_type == "inner"
    assert evidence.sql.joins[0].keys
    assert evidence.sql.group_by
    assert evidence.sql.distinct is False
    assert evidence.sql.order_by[0].direction == "desc"
    assert evidence.sql.limit.value == 2
    assert evidence.sql.offset.value == 1
    assert evidence.sql.predicates[0].kind == "and"
    assert evidence.execution is not None
    assert evidence.execution.row_order_semantics == "ordered"


def test_limit_all_offset_all_and_fetch_are_explicitly_classified() -> None:
    all_bounds = _capture("SELECT amount FROM chatbi_demo.sales LIMIT ALL OFFSET ALL")
    assert all_bounds.sql is not None
    assert all_bounds.sql.limit.kind == "all"
    assert all_bounds.sql.offset.kind == "all"
    assert all_bounds.sql.fetch.kind == "absent"

    fetch = _capture("SELECT amount FROM chatbi_demo.sales FETCH FIRST 4 ROWS ONLY")
    assert fetch.sql is not None
    assert fetch.sql.limit.kind == "absent"
    assert fetch.sql.fetch.kind == "value"
    assert fetch.sql.fetch.value == 4


def test_set_operations_are_represented_without_raw_sql() -> None:
    evidence = _capture(
        "SELECT customer_id FROM chatbi_demo.customers "
        "UNION ALL SELECT customer_id FROM chatbi_demo.customers"
    )
    assert evidence.sql is not None
    assert evidence.sql.set_operations[0].operation == "union"
    assert evidence.sql.set_operations[0].all is True
    assert "UNION ALL" not in evidence.model_dump_json()


def test_set_operation_by_name_is_explicitly_partial() -> None:
    evidence = _capture(
        "SELECT customer_id FROM chatbi_demo.customers "
        "UNION BY NAME SELECT customer_id FROM chatbi_demo.customers"
    )
    assert evidence.sql is not None
    assert evidence.sql.completeness == "partial"
    assert "unsupported_set_by_name" in evidence.sql.completeness_reasons


def test_set_operation_propagates_partial_child_completeness() -> None:
    evidence = _capture(
        "SELECT customer_id FROM chatbi_demo.customers UNION "
        "SELECT * FROM chatbi_demo.sales TABLESAMPLE BERNOULLI (10)"
    )
    assert evidence.sql is not None
    assert evidence.sql.completeness == "partial"
    assert "child_sql_partial" in evidence.sql.completeness_reasons
    assert "unsupported_statement_clause" in evidence.sql.completeness_reasons


@pytest.mark.parametrize(
    ("sql", "mode", "expression_count"),
    [
        ("SELECT DISTINCT customer_id FROM chatbi_demo.customers", "distinct", 0),
        (
            "SELECT DISTINCT ON (customer_id) customer_id FROM chatbi_demo.customers",
            "distinct_on",
            1,
        ),
        (
            "SELECT DISTINCT ON (customer_id, customer_name) customer_id "
            "FROM chatbi_demo.customers",
            "distinct_on",
            2,
        ),
    ],
)
def test_distinct_modes_are_captured(sql: str, mode: str, expression_count: int) -> None:
    evidence = _capture(sql)
    assert evidence.sql is not None
    assert evidence.sql.distinct_mode == mode
    assert len(evidence.sql.distinct_on) == expression_count


def test_unsupported_distinct_on_expression_is_partial() -> None:
    evidence = _capture("SELECT DISTINCT ON (RANDOM()) customer_id FROM chatbi_demo.customers")
    assert evidence.sql is not None
    assert evidence.sql.completeness == "partial"
    assert "unsupported_distinct_on" in evidence.sql.completeness_reasons


@pytest.mark.parametrize(
    "sql",
    [
        "WITH recent AS (SELECT customer_id FROM chatbi_demo.customers) "
        "SELECT customer_id FROM recent UNION SELECT customer_id FROM recent",
        "WITH recent AS (SELECT customer_id FROM chatbi_demo.customers) "
        "SELECT customer_id FROM recent UNION ALL SELECT customer_id FROM recent",
        "WITH recent AS (SELECT customer_id FROM chatbi_demo.customers) "
        "SELECT customer_id FROM recent UNION "
        "SELECT customer_id FROM chatbi_demo.customers",
    ],
)
def test_root_with_set_operation_is_not_claimed_complete(sql: str) -> None:
    evidence = _capture(sql)
    assert evidence.sql is not None
    assert evidence.sql.completeness == "partial"
    assert "unresolved_root_cte_scope" in evidence.sql.completeness_reasons


def test_recursive_and_nested_cte_scopes_are_not_claimed_complete() -> None:
    recursive = _capture(
        "WITH RECURSIVE numbers AS (SELECT 1 UNION ALL SELECT 2) SELECT * FROM numbers"
    )
    nested = _capture(
        "SELECT * FROM (WITH local_rows AS (SELECT 1 AS value) "
        "SELECT value FROM local_rows) AS nested_rows"
    )
    assert recursive.sql and nested.sql
    assert recursive.sql.completeness == "partial"
    assert "unsupported_recursive_cte" in recursive.sql.completeness_reasons
    assert nested.sql.completeness == "partial"
    assert "unresolved_nested_cte_scope" in nested.sql.completeness_reasons


def test_nested_partial_sql_child_propagates_to_parent() -> None:
    evidence = _capture(
        "SELECT * FROM (SELECT * FROM chatbi_demo.sales TABLESAMPLE BERNOULLI (10)) AS sampled"
    )
    assert evidence.sql is not None
    assert evidence.sql.completeness == "partial"
    assert "nested_child_sql_partial" in evidence.sql.completeness_reasons


@pytest.mark.parametrize(
    ("sql", "reason"),
    [
        (
            "SELECT * FROM (VALUES (1), (2)) AS values_source(value)",
            "unsupported_values_source",
        ),
        (
            "SELECT * FROM chatbi_demo.customers AS c "
            "JOIN LATERAL (SELECT c.customer_id) AS lateral_source(customer_id) ON TRUE",
            "unsupported_lateral_source",
        ),
        (
            "SELECT * FROM (SELECT amount FROM chatbi_demo.sales) AS derived_source",
            "unresolved_derived_source_lineage",
        ),
    ],
)
def test_unmodeled_sources_are_partial_without_raw_source_payload(sql: str, reason: str) -> None:
    evidence = _capture(sql)
    assert evidence.sql is not None
    assert evidence.sql.completeness == "partial"
    assert reason in evidence.sql.completeness_reasons
    serialized = evidence.model_dump_json()
    assert "SELECT" not in serialized
    assert "VALUES" not in serialized
    assert "customer_id) AS lateral_source" not in serialized


@pytest.mark.parametrize("operator", ("UNION ALL", "INTERSECT", "EXCEPT"))
def test_nested_set_operations_are_partial_without_silent_omission(operator: str) -> None:
    sql = (
        "SELECT customer_id FROM chatbi_demo.customers "
        f"UNION (SELECT customer_id FROM chatbi_demo.customers {operator} "
        "SELECT customer_id FROM chatbi_demo.customers)"
    )
    evidence = _capture(sql)
    assert evidence.sql is not None
    assert evidence.sql.completeness == "partial"
    assert "unsupported_nested_set_operation" in evidence.sql.completeness_reasons
    assert "UNION" not in evidence.model_dump_json()


def test_generic_unmodeled_source_dispatch_cannot_remain_complete() -> None:
    evidence = _capture("SELECT * FROM UNNEST(ARRAY[1, 2])")
    assert evidence.sql is not None
    assert evidence.sql.completeness == "partial"
    assert "unsupported_source_type" in evidence.sql.completeness_reasons


@pytest.mark.parametrize(
    ("sql", "operation", "bound_field"),
    [
        (
            "SELECT customer_id FROM chatbi_demo.customers "
            "UNION ALL SELECT customer_id FROM chatbi_demo.customers "
            "ORDER BY customer_id DESC LIMIT 5",
            "union",
            "limit",
        ),
        (
            "SELECT customer_id FROM chatbi_demo.customers "
            "INTERSECT SELECT customer_id FROM chatbi_demo.customers OFFSET 2",
            "intersect",
            "offset",
        ),
        (
            "SELECT customer_id FROM chatbi_demo.customers "
            "EXCEPT SELECT customer_id FROM chatbi_demo.customers FETCH FIRST 3 ROWS ONLY",
            "except",
            "fetch",
        ),
    ],
)
def test_set_operation_preserves_outer_order_and_bounds(
    sql: str, operation: str, bound_field: str
) -> None:
    evidence = _capture(sql)
    assert evidence.sql is not None
    assert evidence.sql.set_operations[-1].operation == operation
    assert getattr(evidence.sql, bound_field).kind == "value"
    if operation == "union":
        assert evidence.sql.order_by[0].direction == "desc"
        assert evidence.execution is not None
        assert evidence.execution.row_order_semantics == "ordered"


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM chatbi_demo.sales TABLESAMPLE BERNOULLI (10)",
        "SELECT amount FROM chatbi_demo.sales QUALIFY ROW_NUMBER() OVER (PARTITION BY amount) = 1",
        "SELECT * FROM chatbi_demo.sales PIVOT (SUM(amount) FOR region IN ('x'))",
    ],
)
def test_unsupported_select_clauses_are_explicitly_partial(sql: str) -> None:
    evidence = _capture(sql)
    assert evidence.sql is not None
    assert evidence.sql.completeness == "partial"
    assert "unsupported_statement_clause" in evidence.sql.completeness_reasons


def test_unsupported_outer_set_modifier_is_partial() -> None:
    evidence = _capture(
        "SELECT customer_id FROM chatbi_demo.customers "
        "UNION SELECT customer_id FROM chatbi_demo.customers ORDER BY RANDOM()"
    )
    assert evidence.sql is not None
    assert evidence.sql.completeness == "partial"
    assert "unsupported_outer_ordering" in evidence.sql.completeness_reasons


def test_literal_identity_is_typed_and_does_not_store_values() -> None:
    first = _capture("SELECT amount FROM chatbi_demo.sales WHERE amount = 12")
    same = _capture("SELECT amount FROM chatbi_demo.sales WHERE amount = 12")
    different = _capture("SELECT amount FROM chatbi_demo.sales WHERE amount = 13")
    assert first.sql and same.sql and different.sql
    first_literal = first.sql.predicates[0].values[1].literal
    same_literal = same.sql.predicates[0].values[1].literal
    different_literal = different.sql.predicates[0].values[1].literal
    assert first_literal and same_literal and different_literal
    assert first_literal.identity == same_literal.identity
    assert first_literal.identity != different_literal.identity
    assert first_literal.value_type == "integer"
    assert first_literal.model_dump(mode="json") == {
        "value_type": "integer",
        "identity": first_literal.identity,
        "hash_mode": "sha256-typed-v1",
    }
    assert "canonical_value" not in first.model_dump_json()

    null_evidence = _capture("SELECT amount FROM chatbi_demo.sales WHERE region IS NULL")
    assert null_evidence.sql and null_evidence.sql.predicates[0].kind == "is_null"


def test_result_fingerprints_preserve_duplicates_nulls_and_exact_rows() -> None:
    evidence = _capture("SELECT region FROM chatbi_demo.sales", rows=[["华东"], [None], ["华东"]])
    assert evidence.execution is not None
    assert evidence.execution.exact_rows == (("华东",), (None,), ("华东",))
    assert evidence.execution.row_order_semantics == "unordered"
    assert evidence.execution.row_multiset_sha256

    changed = _capture("SELECT region FROM chatbi_demo.sales", rows=[["华东"], [None]])
    assert changed.execution
    assert changed.execution.row_multiset_sha256 != evidence.execution.row_multiset_sha256


def test_oversized_exact_rows_are_bounded_without_failing_capture() -> None:
    evidence = _capture("SELECT region FROM chatbi_demo.sales", rows=[["x" * 70_000]])
    assert evidence.execution is not None
    assert evidence.execution.exact_rows == ()
    assert evidence.execution.exact_rows_bytes == 2
    assert evidence.execution.status is EvidenceStatus.TRUNCATED
    assert "exact_result_bytes_bound" in evidence.execution.unavailable_reasons


def test_exact_row_and_column_bounds_are_bounded_without_failing_capture() -> None:
    exact = _capture(
        "SELECT region FROM chatbi_demo.sales",
        rows=[[index] for index in range(MAX_EXACT_ROWS)],
    )
    assert exact.execution is not None
    assert len(exact.execution.exact_rows or ()) == MAX_EXACT_ROWS
    assert exact.execution.status is EvidenceStatus.AVAILABLE

    evidence = _capture(
        "SELECT region FROM chatbi_demo.sales",
        rows=[[index] for index in range(257)],
    )
    assert evidence.execution is not None
    assert len(evidence.execution.exact_rows or ()) == MAX_EXACT_ROWS
    assert evidence.execution.row_count == 257
    assert "exact_result_rows_bound" in evidence.execution.unavailable_reasons

    column_names = tuple(f"column_{index}" for index in range(65))
    wide = _capture(
        "SELECT region FROM chatbi_demo.sales",
        columns=column_names,
        rows=[list(range(65))],
    )
    assert wide.execution is not None
    assert wide.execution.status is EvidenceStatus.TRUNCATED
    assert "result_columns_bound" in wide.execution.unavailable_reasons


def test_large_row_iterable_keeps_hashes_complete_but_exact_rows_bounded() -> None:
    rows = [[index] for index in range(1000)]
    evidence = _capture("SELECT region FROM chatbi_demo.sales", rows=rows)
    assert evidence.execution is not None
    assert evidence.execution.row_count == 1000
    assert len(evidence.execution.exact_rows or ()) == MAX_EXACT_ROWS
    assert evidence.execution.row_multiset_sha256 is not None
    assert evidence.execution.row_order_sha256 is not None


def test_exact_row_byte_bound_is_applied_before_storing_a_cell() -> None:
    empty_row_bytes = len(_canonical_json([[""]]).encode("utf-8"))
    fitting = "x" * (MAX_EXACT_RESULT_BYTES - empty_row_bytes)
    overflowing = fitting + "x"

    accepted = _capture("SELECT region FROM chatbi_demo.sales", rows=[[fitting]])
    rejected = _capture("SELECT region FROM chatbi_demo.sales", rows=[[overflowing]])

    assert accepted.execution is not None
    assert accepted.execution.exact_rows == ((fitting,),)
    assert accepted.execution.exact_rows_bytes == MAX_EXACT_RESULT_BYTES
    assert rejected.execution is not None
    assert rejected.execution.exact_rows == ()
    assert rejected.execution.exact_rows_bytes == 2
    assert "exact_result_bytes_bound" in rejected.execution.unavailable_reasons


def test_streaming_row_fingerprints_consume_each_row_once() -> None:
    consumed: list[int] = []

    def rows() -> Iterable[tuple[int]]:
        for index in range(1000):
            consumed.append(index)
            yield (index,)

    result = _canonical_rows(rows())

    assert len(consumed) == 1000
    assert len(result.exact_rows) == MAX_EXACT_ROWS
    assert result.exact_rows_truncated_reason == "exact_result_rows_bound"


def test_list_bounds_return_truncated_evidence() -> None:
    sql = "SELECT " + ", ".join(str(index) for index in range(129))
    evidence = _capture(sql)
    assert evidence.evidence_status is EvidenceStatus.TRUNCATED
    assert "projection_entries" in evidence.unavailable_reasons


def test_ast_node_bounds_return_truncated_evidence() -> None:
    arguments = ", ".join("s.amount" for _ in range(513))
    evidence = _capture(f"SELECT COALESCE({arguments}) FROM chatbi_demo.sales")
    assert evidence.evidence_status is EvidenceStatus.TRUNCATED
    assert "semantic_ast_nodes" in evidence.unavailable_reasons


def test_expression_depth_bound_returns_truncated_evidence() -> None:
    sql = "s.amount"
    for _ in range(33):
        sql = f"ABS({sql})"
    evidence = _capture(f"SELECT {sql} FROM chatbi_demo.sales")
    assert evidence.evidence_status is EvidenceStatus.TRUNCATED
    assert "expression_depth" in evidence.unavailable_reasons


def test_non_synthetic_or_mismatched_fixture_fails_closed() -> None:
    with pytest.raises(SemanticEvidenceError, match="frozen fixture"):
        _capture("SELECT 1", fixture="e" * 64)
    with pytest.raises(SemanticEvidenceError, match="only synthetic_exact_v1"):
        datasource_id = uuid4()
        capture_semantic_evidence(
            case_id="simple-01",
            split="dev",
            datasource_id=datasource_id,
            dataset_fingerprint=DATASET,
            fixture_fingerprint=FIXTURE,
            schema_fingerprint=SCHEMA,
            synthetic_fixture_fingerprint=FIXTURE,
            validated_sql=None,
            execution_result=None,
            chatbi_result=ChatBIResult(
                query_id=uuid4(),
                datasource_id=datasource_id,
                execution_status=QueryLifecycleState.SUCCEEDED,
                row_count=0,
                sql_attempts=0,
                repair_attempts=0,
                usage=ChatBIUsageSummary(llm_calls=0, provider_attempts=0),
            ),
            provider_invocations=[],
            prompt_version="a5.3-v3",
            reasoning_mode="disabled",
            evidence_mode="sensitive_future",
        )
