from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from uuid import UUID, uuid4

import pytest

import knowledge_scope.evaluation.chatbi_semantic_evaluation as semantic_evaluation
from knowledge_scope.chatbi.schemas import (
    ColumnMetadata,
    QueryExecutionResult,
    QueryLifecycleState,
)
from knowledge_scope.evaluation.chatbi_evaluation import (
    ExpectedQueryResult,
    compare_normalized_result,
)
from knowledge_scope.evaluation.chatbi_evaluation_v2_provider import (
    EXPECTED_FIXTURE_SCHEMA_FINGERPRINT_V2,
)
from knowledge_scope.evaluation.chatbi_semantic_evaluation import (
    DEFAULT_SEMANTIC_POLICY_PATH,
    OutputShapeStatus,
    SemanticCasePolicy,
    SemanticEvaluationError,
    SemanticRelationMetadata,
    SemanticRowGrain,
    SemanticSchemaMetadata,
    build_default_semantic_policy,
    compare_semantic_result,
    evaluate_provider_artifact,
    load_semantic_policy,
    policy_for_case,
    projected_field_identities,
    semantic_schema_metadata_fingerprint,
)

DATASOURCE_ID = UUID("00000000-0000-4000-8000-000000000057")


def _result(columns: list[str], rows: list[list[object]]) -> QueryExecutionResult:
    return QueryExecutionResult(
        query_id=uuid4(),
        datasource_id=DATASOURCE_ID,
        state=QueryLifecycleState.SUCCEEDED,
        columns=[
            ColumnMetadata(name=name, data_type="text", nullable=True, ordinal=index)
            for index, name in enumerate(columns)
        ],
        rows=rows,
        row_count=len(rows),
        max_rows=max(1, len(rows)),
        duration_ms=0,
    )


def _policy(
    required: tuple[str, ...],
    *,
    grain: SemanticRowGrain = SemanticRowGrain.DETAIL,
    optional: tuple[str, ...] = (),
    identity: tuple[str, ...] = (),
    ordered: bool = False,
) -> SemanticCasePolicy:
    return SemanticCasePolicy(
        row_grain=grain,
        required_fields=required,
        optional_fields=optional,
        identity_fields=identity,
        semantic_order_required=ordered,
    )


def _schema_variant(mutator) -> SemanticSchemaMetadata:
    """Build a validated sidecar schema after one explicit test mutation."""

    raw = load_semantic_policy().schema.model_dump(mode="json")
    mutator(raw)
    for relation in raw["relations"]:
        columns = relation["columns"]
        not_null_columns = set(relation["not_null_columns"])
        for column in relation["column_metadata"]:
            column["name"] = columns[column["position"] - 1]
            column["nullable"] = column["name"] not in not_null_columns
    raw["relations"] = tuple(
        SemanticRelationMetadata.model_validate(relation) for relation in raw["relations"]
    )
    candidate = SemanticSchemaMetadata.model_construct(**raw)
    fingerprint = semantic_schema_metadata_fingerprint(candidate)
    raw["schema_metadata_fingerprint"] = fingerprint
    raw["fixture_schema_fingerprint"] = fingerprint
    return SemanticSchemaMetadata.model_validate(raw)


def test_projection_identities_ignore_aliases_but_keep_derived_semantics() -> None:
    assert projected_field_identities(
        "SELECT amount AS total, AVG(amount) AS average_amount FROM sales",
        ["total", "average_amount"],
    ) == ("amount", "aggregate:avg:amount")


def test_projection_identity_handles_distinct_aggregate_argument() -> None:
    assert projected_field_identities(
        "SELECT COUNT(DISTINCT customer_id) AS customer_count FROM sales",
        ["customer_count"],
    ) == ("aggregate:count:distinct:customer_id",)


def test_column_order_is_presentation_not_semantic_equivalence() -> None:
    expected = ExpectedQueryResult(columns=["sale_id", "amount"], rows=[[1, 10]])
    actual = _result(["amount", "sale_id"], [[10, 1]])

    comparison = compare_semantic_result(
        actual,
        expected,
        policy=_policy(("sale_id", "amount")),
        actual_sql="SELECT amount, sale_id FROM sales",
        expected_sql="SELECT sale_id, amount FROM sales",
    )

    assert comparison.equivalent is True
    assert comparison.shape.shape_compliant is False
    assert comparison.shape.column_order_difference is True


def test_safe_optional_extra_field_is_shape_deviation_only() -> None:
    expected = ExpectedQueryResult(columns=["amount"], rows=[[10]])
    actual = _result(["amount", "sold_on"], [[10, "2026-01-01"]])

    comparison = compare_semantic_result(
        actual,
        expected,
        policy=_policy(("amount",), optional=("sold_on",)),
        actual_sql="SELECT amount, sold_on FROM sales",
        expected_sql="SELECT amount FROM sales",
    )

    assert comparison.equivalent is True
    assert comparison.shape.shape_compliant is False
    assert comparison.optional_extra_fields == ("sold_on",)


def test_missing_optional_display_field_stays_semantically_valid() -> None:
    expected = ExpectedQueryResult(columns=["customer_name", "amount"], rows=[["A", 10]])
    actual = _result(["amount"], [[10]])

    comparison = compare_semantic_result(
        actual,
        expected,
        policy=_policy(("amount",), optional=("customer_name",)),
        actual_sql="SELECT amount FROM sales",
        expected_sql="SELECT customer_name, amount FROM sales",
    )

    assert comparison.equivalent is True
    assert comparison.shape.optional_omitted_fields == ("customer_name",)


def test_hidden_detail_identity_does_not_need_to_be_visible() -> None:
    expected = ExpectedQueryResult(
        columns=["sale_id", "customer_name", "amount"],
        rows=[[1, "A", 10], [2, "A", 20]],
    )
    actual = _result(["customer_name", "amount"], [["A", 10], ["A", 20]])

    comparison = compare_semantic_result(
        actual,
        expected,
        policy=_policy(
            ("customer_name", "amount"),
            optional=("sale_id",),
            identity=("sale_id",),
        ),
        actual_sql="SELECT customer_name, amount FROM sales",
        expected_sql="SELECT sale_id, customer_name, amount FROM sales",
    )

    assert comparison.equivalent is True


def test_duplicate_entity_label_requires_stable_identity() -> None:
    expected = ExpectedQueryResult(
        columns=["customer_id", "customer_name"],
        rows=[[1, "A"], [2, "A"]],
    )
    actual = _result(["customer_name"], [["A"], ["A"]])

    comparison = compare_semantic_result(
        actual,
        expected,
        policy=_policy(
            ("customer_id", "customer_name"),
            identity=("customer_id",),
            grain=SemanticRowGrain.ENTITY,
        ),
        actual_sql="SELECT customer_name FROM customers GROUP BY customer_name",
        expected_sql=(
            "SELECT customer_id, customer_name FROM customers GROUP BY customer_id, customer_name"
        ),
    )

    assert comparison.equivalent is False
    assert comparison.required_fields_present is False


def test_distinct_collapse_is_a_detail_grain_failure() -> None:
    expected = ExpectedQueryResult(columns=["customer_name"], rows=[["A"], ["A"]])
    actual = _result(["customer_name"], [["A"]])

    comparison = compare_semantic_result(
        actual,
        expected,
        policy=_policy(("customer_name",)),
        actual_sql="SELECT DISTINCT customer_name FROM sales",
        expected_sql="SELECT customer_name FROM sales",
    )

    assert comparison.equivalent is False
    assert comparison.row_grain_match is False


def test_grouped_policy_rejects_detail_projection() -> None:
    expected = ExpectedQueryResult(columns=["customer_name"], rows=[["A"]])
    actual = _result(["customer_name"], [["A"]])

    comparison = compare_semantic_result(
        actual,
        expected,
        policy=_policy(("customer_name",), grain=SemanticRowGrain.GROUPED),
        actual_sql="SELECT customer_name FROM customers",
        expected_sql="SELECT customer_name FROM customers GROUP BY customer_name",
    )

    assert comparison.equivalent is False
    assert comparison.row_grain_match is False


def test_wrong_join_source_field_fails_closed() -> None:
    expected = ExpectedQueryResult(columns=["region_name"], rows=[["华东"]])
    actual = _result(["region_code"], [["R1"]])

    comparison = compare_semantic_result(
        actual,
        expected,
        policy=_policy(("region_name",)),
        actual_sql="SELECT region_code FROM sales",
        expected_sql="SELECT region_name FROM regions",
    )

    assert comparison.equivalent is False
    assert comparison.failure_reason == "required semantic field is missing"


def test_missing_actual_sql_lineage_is_unassessable() -> None:
    expected = ExpectedQueryResult(columns=["amount"], rows=[[10]])
    actual = _result(["amount"], [[10]])

    comparison = compare_semantic_result(
        actual,
        expected,
        policy=_policy(("amount",)),
        expected_sql="SELECT amount FROM sales",
    )

    assert comparison.assessable is False
    assert comparison.failure_reason == "unassessable: projection lineage is unavailable"


def test_derived_value_can_be_required_without_optional_base_value() -> None:
    expected = ExpectedQueryResult(
        columns=["amount", "delta"], rows=[[10, 9]], numeric_columns=[0, 1]
    )
    actual = _result(["difference"], [[9]])

    comparison = compare_semantic_result(
        actual,
        expected,
        policy=_policy(("expression:amount - 1",), optional=("amount",)),
        actual_sql="SELECT amount - 1 AS difference FROM sales",
        expected_sql="SELECT amount, amount - 1 AS delta FROM sales",
    )

    assert comparison.equivalent is True


def test_numeric_tolerance_and_null_semantics_are_preserved() -> None:
    expected = ExpectedQueryResult(
        columns=["amount", "note"], rows=[["1.0", None]], numeric_columns=[0]
    )
    actual = _result(["amount", "note"], [[Decimal("1.0000005"), None]])

    comparison = compare_semantic_result(
        actual,
        expected,
        policy=_policy(("amount", "note")),
        actual_sql="SELECT amount, note FROM sales",
        expected_sql="SELECT amount, note FROM sales",
    )

    assert comparison.equivalent is True


def test_empty_result_allows_optional_schema_difference() -> None:
    expected = ExpectedQueryResult(columns=["customer_id", "customer_name"], rows=[])
    actual = _result(["customer_id"], [])

    comparison = compare_semantic_result(
        actual,
        expected,
        policy=_policy(("customer_id",), optional=("customer_name",)),
        actual_sql="SELECT customer_id FROM customers WHERE false",
        expected_sql="SELECT customer_id, customer_name FROM customers WHERE false",
    )

    assert comparison.equivalent is True


def test_empty_results_with_different_predicates_do_not_pass() -> None:
    expected = ExpectedQueryResult(columns=["customer_id"], rows=[])
    actual = _result(["customer_id"], [])

    comparison = compare_semantic_result(
        actual,
        expected,
        policy=_policy(("customer_id",)),
        actual_sql="SELECT customer_id FROM customers WHERE FALSE",
        expected_sql="SELECT customer_id FROM customers WHERE customer_id = 999",
    )

    assert comparison.equivalent is False
    assert comparison.failure_reason == "empty-result query semantics differ"


def test_empty_results_allow_a_provable_ast_equivalent_rewrite() -> None:
    expected = ExpectedQueryResult(columns=["customer_id"], rows=[])
    actual = _result(["customer_id"], [])

    comparison = compare_semantic_result(
        actual,
        expected,
        policy=_policy(("customer_id",)),
        actual_sql="SELECT customer_id FROM customers WHERE (customer_id = 999)",
        expected_sql="SELECT customer_id FROM customers WHERE customer_id = 999",
    )

    assert comparison.equivalent is True


def test_extra_group_field_is_not_a_safe_presentation_difference() -> None:
    expected = ExpectedQueryResult(
        columns=["customer_id", "total"],
        rows=[[1, 100], [2, 200]],
        numeric_columns=[0, 1],
    )
    actual = _result(
        ["customer_id", "total", "sold_on"],
        [[1, 100, "2026-01-01"], [2, 200, "2026-01-02"]],
    )

    comparison = compare_semantic_result(
        actual,
        expected,
        policy=_policy(
            ("customer_id", "aggregate:sum:amount"),
            grain=SemanticRowGrain.GROUPED,
        ),
        actual_sql=(
            "SELECT customer_id, SUM(amount) AS total, sold_on "
            "FROM sales GROUP BY customer_id, sold_on"
        ),
        expected_sql=("SELECT customer_id, SUM(amount) AS total FROM sales GROUP BY customer_id"),
    )

    assert comparison.equivalent is False
    assert comparison.row_grain_match is False


def test_unordered_rows_compare_as_a_multiset() -> None:
    expected = ExpectedQueryResult(columns=["customer_id"], rows=[[1], [2]])
    actual = _result(["customer_id"], [[2], [1]])

    comparison = compare_semantic_result(
        actual,
        expected,
        policy=_policy(("customer_id",)),
        actual_sql="SELECT customer_id FROM customers",
        expected_sql="SELECT customer_id FROM customers",
    )

    assert comparison.equivalent is True
    assert comparison.ordering_semantics_match is True


def test_unordered_rows_preserve_duplicate_multiplicity() -> None:
    expected = ExpectedQueryResult(columns=["customer_name"], rows=[["A"], ["A"]])
    actual = _result(["customer_name"], [["A"], ["B"]])

    comparison = compare_semantic_result(
        actual,
        expected,
        policy=_policy(("customer_name",)),
        actual_sql="SELECT customer_name FROM customers",
        expected_sql="SELECT customer_name FROM customers",
    )

    assert comparison.equivalent is False
    assert comparison.row_set_match is False


def test_required_row_order_remains_positional() -> None:
    expected = ExpectedQueryResult(columns=["customer_id"], rows=[[1], [2]])
    actual = _result(["customer_id"], [[2], [1]])

    comparison = compare_semantic_result(
        actual,
        expected,
        policy=_policy(("customer_id",), ordered=True),
        actual_sql="SELECT customer_id FROM customers ORDER BY customer_id DESC",
        expected_sql="SELECT customer_id FROM customers ORDER BY customer_id",
    )

    assert comparison.equivalent is False
    assert comparison.ordering_semantics_match is False


def test_sum_and_average_do_not_pass_on_coincidentally_equal_values() -> None:
    expected = ExpectedQueryResult(columns=["total"], rows=[[10]], numeric_columns=[0])
    actual = _result(["total"], [[10]])

    comparison = compare_semantic_result(
        actual,
        expected,
        policy=_policy(("aggregate:sum:amount",), grain=SemanticRowGrain.SCALAR),
        actual_sql="SELECT AVG(amount) AS total FROM sales",
        expected_sql="SELECT SUM(amount) AS total FROM sales",
    )

    assert comparison.equivalent is False
    assert comparison.required_fields_present is False


def test_different_derived_operators_do_not_pass_on_coincidentally_equal_values() -> None:
    expected = ExpectedQueryResult(columns=["delta"], rows=[[9]], numeric_columns=[0])
    actual = _result(["delta"], [[9]])

    comparison = compare_semantic_result(
        actual,
        expected,
        policy=_policy(("expression:amount - 1",)),
        actual_sql="SELECT amount + 1 AS delta FROM sales",
        expected_sql="SELECT amount - 1 AS delta FROM sales",
    )

    assert comparison.equivalent is False
    assert comparison.required_fields_present is False


def test_default_sidecar_load_is_independent_of_cwd(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    policy = load_semantic_policy()

    assert policy.schema_version == "a5.7f2-semantic-policy-v1"


def test_failed_execution_is_not_successfully_shape_compliant() -> None:
    expected = ExpectedQueryResult(columns=["amount"], rows=[[10]])
    actual = _result(["amount"], [[10]]).model_copy(update={"state": QueryLifecycleState.FAILED})

    comparison = compare_semantic_result(
        actual,
        expected,
        policy=_policy(("amount",)),
        actual_sql="SELECT amount FROM sales",
        expected_sql="SELECT amount FROM sales",
    )

    assert comparison.assessable is True
    assert comparison.shape.status is OutputShapeStatus.UNASSESSABLE
    assert comparison.shape.shape_compliant is False


def test_default_policy_uses_duplicate_labels_to_require_entity_id() -> None:
    expected = ExpectedQueryResult(
        columns=["customer_id", "customer_name", "average_amount"],
        rows=[[1, "A", 10], [2, "A", 20]],
        numeric_columns=[0, 2],
    )
    policy = build_default_semantic_policy(
        question="每位客户的平均金额是多少",
        expected=expected,
        expected_sql=(
            "SELECT customer_id, customer_name, AVG(amount) AS average_amount "
            "FROM sales GROUP BY customer_id, customer_name"
        ),
    )

    assert "customer_id" in policy.normalized_required_fields()


def test_formal_comparator_remains_exact_and_separate() -> None:
    expected = ExpectedQueryResult(columns=["sale_id", "amount"], rows=[[1, 10]])
    actual = _result(["amount", "sale_id"], [[10, 1]])

    assert compare_normalized_result(actual, expected).equivalent is False


def test_versioned_policy_sidecar_is_frozen_to_v2_dataset() -> None:
    policy = load_semantic_policy(DEFAULT_SEMANTIC_POLICY_PATH)

    assert policy.schema_version == "a5.7f2-semantic-policy-v1"
    assert "join-01" in policy.cases
    assert Path(DEFAULT_SEMANTIC_POLICY_PATH).is_file()


def test_sidecar_records_the_five_human_review_uncertainties() -> None:
    policy = load_semantic_policy(DEFAULT_SEMANTIC_POLICY_PATH)

    assert policy.cases["join-01"].normalized_required_fields() == {
        "customer_name",
        "amount",
    }
    assert policy.cases["join-03"].normalized_required_fields() == {
        "customer_name",
        "amount",
    }
    assert "customer_id" in policy.cases["null-03"].normalized_required_fields()
    assert "customer_id" in policy.cases["derived-03"].normalized_required_fields()
    assert policy.cases["cte-set-01"].normalized_required_fields() == {
        "customer_id",
        "total_amount",
    }


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param(
            lambda schema: schema["relations"][0].update(primary_key=["customer_name"]),
            id="primary-key",
        ),
        pytest.param(
            lambda schema: schema["relations"][1]["unique_constraints"].append(["market"]),
            id="unique-constraint",
        ),
        pytest.param(
            lambda schema: schema["relations"][1]["not_null_columns"].remove("region_name"),
            id="not-null",
        ),
        pytest.param(
            lambda schema: schema["relations"][2]["foreign_keys"][0].update(
                target_columns=["customer_name"]
            ),
            id="foreign-key",
        ),
        pytest.param(
            lambda schema: schema["relations"][2]["foreign_keys"][0].update(
                source_relation="chatbi_demo.customers"
            ),
            id="foreign-key-source-relation",
        ),
        pytest.param(
            lambda schema: schema["relations"][2]["foreign_keys"][0].update(
                target_relation="customers"
            ),
            id="foreign-key-target-scope",
        ),
        pytest.param(
            lambda schema: schema["relations"][2]["columns"].__setitem__(1, "customer_key"),
            id="column-identity",
        ),
        pytest.param(
            lambda schema: schema["relations"][2]["check_constraints"][0].update(
                definition="CHECK (amount > 0::numeric)"
            ),
            id="check-definition",
        ),
    ],
)
def test_schema_metadata_mutation_with_original_claimed_fingerprint_fails_closed(
    tmp_path: Path, mutation
) -> None:
    payload = json.loads(DEFAULT_SEMANTIC_POLICY_PATH.read_text(encoding="utf-8"))
    mutation(payload["schema"])
    path = tmp_path / "mutated-policy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SemanticEvaluationError):
        load_semantic_policy(path)


def test_untouched_schema_metadata_fingerprint_is_reproducible() -> None:
    schema = load_semantic_policy().schema

    assert semantic_schema_metadata_fingerprint(schema) == schema.schema_metadata_fingerprint
    assert schema.schema_metadata_fingerprint == schema.fixture_schema_fingerprint
    assert schema.schema_metadata_fingerprint == EXPECTED_FIXTURE_SCHEMA_FINGERPRINT_V2
    assert schema.catalog_schema == "chatbi_demo"
    assert {relation.kind for relation in schema.relations} == {"table", "view"}


def test_relation_identity_mutation_fails_closed(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_SEMANTIC_POLICY_PATH.read_text(encoding="utf-8"))
    payload["schema"]["relations"][0]["relation"] = "chatbi_demo.customer_alias"

    path = tmp_path / "invalid-relation-policy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SemanticEvaluationError):
        load_semantic_policy(path)


def test_schema_primary_key_proves_dependent_grouping_column() -> None:
    schema = load_semantic_policy().schema
    expected = ExpectedQueryResult(columns=["customer_name"], rows=[["A"], ["B"]])
    actual = _result(["customer_name"], [["A"], ["B"]])

    comparison = compare_semantic_result(
        actual,
        expected,
        policy=_policy(("customer_name",), grain=SemanticRowGrain.GROUPED),
        actual_sql="SELECT customer_name FROM chatbi_demo.customers GROUP BY customer_id",
        expected_sql=(
            "SELECT customer_name FROM chatbi_demo.customers GROUP BY customer_id, customer_name"
        ),
        schema=schema,
    )

    assert comparison.equivalent is True
    assert "equivalent_by_primary_key_dependency" in comparison.proof_reasons


def test_non_unique_label_grouping_is_not_proven_equivalent() -> None:
    schema = load_semantic_policy().schema
    expected = ExpectedQueryResult(columns=["customer_name"], rows=[["A"], ["A"]])
    actual = _result(["customer_name"], [["A"]])

    comparison = compare_semantic_result(
        actual,
        expected,
        policy=_policy(("customer_name",), grain=SemanticRowGrain.GROUPED),
        actual_sql="SELECT customer_name FROM chatbi_demo.customers GROUP BY customer_name",
        expected_sql=(
            "SELECT customer_name FROM chatbi_demo.customers GROUP BY customer_id, customer_name"
        ),
        schema=schema,
    )

    assert comparison.equivalent is False
    assert comparison.assessable is True
    assert "different_grouping_grain" in comparison.proof_reasons


def test_left_join_and_inner_join_are_not_equivalent_when_unmatched_rows_are_possible() -> None:
    schema = load_semantic_policy().schema
    expected = ExpectedQueryResult(columns=["customer_id"], rows=[[1], [2]])
    actual = _result(["customer_id"], [[1]])

    comparison = compare_semantic_result(
        actual,
        expected,
        policy=_policy(("customer_id",), grain=SemanticRowGrain.GROUPED),
        actual_sql=(
            "SELECT c.customer_id FROM chatbi_demo.customers AS c "
            "JOIN chatbi_demo.sales AS s ON s.customer_id = c.customer_id "
            "GROUP BY c.customer_id"
        ),
        expected_sql=(
            "SELECT c.customer_id FROM chatbi_demo.customers AS c "
            "LEFT JOIN chatbi_demo.sales AS s ON s.customer_id = c.customer_id "
            "GROUP BY c.customer_id"
        ),
        schema=schema,
    )

    assert comparison.equivalent is False
    assert comparison.assessable is True
    assert "different_join_type" in comparison.proof_reasons


@pytest.mark.parametrize(
    ("join_sql", "expected_expression", "actual_expression"),
    [
        (
            "LEFT JOIN chatbi_demo.sales AS s ON r.region_code = s.region_code",
            "r.region_code",
            "s.region_code",
        ),
        (
            "RIGHT JOIN chatbi_demo.sales AS s ON r.region_code = s.region_code",
            "s.region_code",
            "r.region_code",
        ),
        (
            "FULL OUTER JOIN chatbi_demo.sales AS s ON r.region_code = s.region_code",
            "r.region_code",
            "s.region_code",
        ),
    ],
)
def test_outer_join_projection_substitution_is_unassessable(
    join_sql: str, expected_expression: str, actual_expression: str
) -> None:
    schema = load_semantic_policy().schema
    expected_sql = f"SELECT {expected_expression} FROM chatbi_demo.regions AS r {join_sql}"
    actual_sql = f"SELECT {actual_expression} FROM chatbi_demo.regions AS r {join_sql}"
    comparison = semantic_evaluation._structural_artifact_assessment(
        record={
            "redacted_sql": actual_sql,
            "result_status": QueryLifecycleState.SUCCEEDED.value,
            "row_count": 5,
        },
        expected=ExpectedQueryResult(columns=["region_code"], rows=[["R1"]] * 5),
        expected_sql=expected_sql,
        policy=_policy(("region_code",)),
        schema=schema,
    )

    assert comparison.equivalent is False
    assert comparison.assessable is False
    assert "outer_join_projection_equivalence_unproven" in comparison.proof_reasons


def test_inner_join_equality_projection_can_be_proven_equivalent() -> None:
    schema = load_semantic_policy().schema
    join_sql = "JOIN chatbi_demo.sales AS s ON r.region_code = s.region_code"
    expected_sql = f"SELECT r.region_code FROM chatbi_demo.regions AS r {join_sql}"
    actual_sql = f"SELECT s.region_code FROM chatbi_demo.regions AS r {join_sql}"
    comparison = semantic_evaluation._structural_artifact_assessment(
        record={
            "redacted_sql": actual_sql,
            "result_status": QueryLifecycleState.SUCCEEDED.value,
            "row_count": 5,
        },
        expected=ExpectedQueryResult(columns=["region_code"], rows=[["R1"]] * 5),
        expected_sql=expected_sql,
        policy=_policy(("region_code",)),
        schema=schema,
    )

    assert comparison.equivalent is True
    assert comparison.assessable is True


def test_available_and_redacted_literals_are_not_proven_different() -> None:
    schema = load_semantic_policy().schema
    expected_sql = "SELECT r.region_name FROM chatbi_demo.regions AS r WHERE r.region_name = '华东'"
    redacted_sql = (
        "SELECT r.region_name FROM chatbi_demo.regions AS r WHERE r.region_name = '<redacted>'"
    )
    comparison = semantic_evaluation._structural_artifact_assessment(
        record={
            "redacted_sql": redacted_sql,
            "result_status": QueryLifecycleState.SUCCEEDED.value,
            "row_count": 1,
        },
        expected=ExpectedQueryResult(columns=["region_name"], rows=[["华东"]]),
        expected_sql=expected_sql,
        policy=_policy(("region_name",)),
        schema=schema,
    )

    assert comparison.equivalent is False
    assert comparison.assessable is False
    assert "literal_value_unavailable" in comparison.proof_reasons


def test_literal_proof_distinguishes_available_values_but_not_redacted_values() -> None:
    schema = load_semantic_policy().schema
    same = semantic_evaluation._predicate_proof(
        "SELECT region_name FROM chatbi_demo.regions WHERE region_name = '华东'",
        "SELECT region_name FROM chatbi_demo.regions WHERE region_name = '华东'",
        schema,
        actual_literals_available=True,
    )
    different = semantic_evaluation._predicate_proof(
        "SELECT region_name FROM chatbi_demo.regions WHERE region_name = '华东'",
        "SELECT region_name FROM chatbi_demo.regions WHERE region_name = '华南'",
        schema,
        actual_literals_available=True,
    )
    redacted = semantic_evaluation._predicate_proof(
        "SELECT region_name FROM chatbi_demo.regions WHERE region_name = '华东'",
        "SELECT region_name FROM chatbi_demo.regions WHERE region_name = '<redacted>'",
        schema,
        actual_literals_available=False,
    )
    both_redacted = semantic_evaluation._predicate_proof(
        "SELECT region_name FROM chatbi_demo.regions WHERE region_name = '<redacted>'",
        "SELECT region_name FROM chatbi_demo.regions WHERE region_name = '<redacted>'",
        schema,
        actual_literals_available=False,
    )

    assert same.status.value == "proven_equivalent"
    assert different.status.value == "proven_different"
    assert redacted.status.value == "unknown"
    assert both_redacted.status.value == "unknown"


def test_derived_alias_lineage_proves_equivalent_predicates() -> None:
    schema = load_semantic_policy().schema
    expected_sql = (
        "SELECT s.amount FROM chatbi_demo.sales AS s "
        "JOIN (SELECT region_code, AVG(amount) AS average_amount "
        "FROM chatbi_demo.sales GROUP BY region_code) AS first_average "
        "ON s.region_code = first_average.region_code "
        "WHERE s.amount > first_average.average_amount"
    )
    actual_sql = (
        "SELECT s.amount FROM chatbi_demo.sales AS s "
        "JOIN (SELECT region_code, AVG(amount) AS avg_amount "
        "FROM chatbi_demo.sales GROUP BY region_code) AS second_average "
        "ON s.region_code = second_average.region_code "
        "WHERE s.amount > second_average.avg_amount"
    )

    proof = semantic_evaluation._predicate_proof(
        expected_sql,
        actual_sql,
        schema,
        actual_literals_available=True,
    )

    assert proof.status is semantic_evaluation.SemanticProofStatus.PROVEN_EQUIVALENT


def test_derived_alias_lineage_proves_different_expressions() -> None:
    schema = load_semantic_policy().schema
    expected_sql = (
        "SELECT s.amount FROM chatbi_demo.sales AS s "
        "JOIN (SELECT region_code, AVG(amount) AS summary_value "
        "FROM chatbi_demo.sales GROUP BY region_code) AS derived_value "
        "ON s.region_code = derived_value.region_code "
        "WHERE s.amount > derived_value.summary_value"
    )
    actual_sql = (
        "SELECT s.amount FROM chatbi_demo.sales AS s "
        "JOIN (SELECT region_code, SUM(amount) AS summary_value "
        "FROM chatbi_demo.sales GROUP BY region_code) AS derived_value "
        "ON s.region_code = derived_value.region_code "
        "WHERE s.amount > derived_value.summary_value"
    )

    proof = semantic_evaluation._predicate_proof(
        expected_sql,
        actual_sql,
        schema,
        actual_literals_available=True,
    )

    assert proof.status is semantic_evaluation.SemanticProofStatus.PROVEN_DIFFERENT


def test_unresolved_or_nested_derived_lineage_is_unknown() -> None:
    schema = load_semantic_policy().schema
    unresolved = semantic_evaluation._predicate_proof(
        "SELECT amount FROM chatbi_demo.sales WHERE amount > missing_alias.avg_amount",
        "SELECT amount FROM chatbi_demo.sales WHERE amount > other_alias.avg_amount",
        schema,
        actual_literals_available=True,
    )
    unresolved_presence = semantic_evaluation._predicate_proof(
        "SELECT amount FROM chatbi_demo.sales",
        "SELECT amount FROM chatbi_demo.sales WHERE missing_alias.avg_amount > 0",
        schema,
        actual_literals_available=True,
    )
    nested = semantic_evaluation._predicate_proof(
        "SELECT s.amount FROM chatbi_demo.sales AS s "
        "JOIN (SELECT region_code, (SELECT AVG(amount) FROM chatbi_demo.sales) AS avg_amount "
        "FROM chatbi_demo.sales) AS derived ON s.region_code = derived.region_code "
        "WHERE s.amount > derived.avg_amount",
        "SELECT s.amount FROM chatbi_demo.sales AS s "
        "JOIN (SELECT region_code, (SELECT SUM(amount) FROM chatbi_demo.sales) AS avg_amount "
        "FROM chatbi_demo.sales) AS derived ON s.region_code = derived.region_code "
        "WHERE s.amount > derived.avg_amount",
        schema,
        actual_literals_available=True,
    )

    assert unresolved.status is semantic_evaluation.SemanticProofStatus.UNKNOWN
    assert unresolved_presence.status is semantic_evaluation.SemanticProofStatus.UNKNOWN
    assert nested.status is semantic_evaluation.SemanticProofStatus.UNKNOWN


def test_same_derived_alias_with_different_expression_is_not_equivalent() -> None:
    schema = load_semantic_policy().schema
    expected_sql = (
        "SELECT amount FROM chatbi_demo.sales "
        "WHERE amount > (SELECT AVG(amount) AS value FROM chatbi_demo.sales)"
    )
    actual_sql = (
        "SELECT amount FROM chatbi_demo.sales "
        "WHERE amount > (SELECT SUM(amount) AS value FROM chatbi_demo.sales)"
    )

    proof = semantic_evaluation._predicate_proof(
        expected_sql,
        actual_sql,
        schema,
        actual_literals_available=True,
    )

    assert proof.status is semantic_evaluation.SemanticProofStatus.PROVEN_DIFFERENT


def test_cte_set_02_alias_difference_is_not_a_proven_failure() -> None:
    schema = load_semantic_policy().schema
    expected_sql = (
        "WITH region_average AS (SELECT region_code, AVG(amount) AS average_amount "
        "FROM chatbi_demo.sales GROUP BY region_code) "
        "SELECT s.sale_id, r.region_name, s.amount "
        "FROM chatbi_demo.sales AS s "
        "JOIN region_average AS a ON a.region_code = s.region_code "
        "JOIN chatbi_demo.regions AS r ON r.region_code = s.region_code "
        "WHERE s.amount > a.average_amount ORDER BY s.sale_id"
    )
    actual_sql = (
        "SELECT s.sale_id, s.amount FROM chatbi_demo.sales AS s "
        "JOIN (SELECT region_code, AVG(amount) AS avg_amount "
        "FROM chatbi_demo.sales GROUP BY region_code) AS r "
        "ON s.region_code = r.region_code WHERE s.amount > r.avg_amount"
    )
    comparison = semantic_evaluation._structural_artifact_assessment(
        record={
            "redacted_sql": actual_sql,
            "result_status": QueryLifecycleState.SUCCEEDED.value,
            "row_count": 14,
        },
        expected=ExpectedQueryResult(
            columns=["sale_id", "region_name", "amount"], rows=[[1, 2, 3]] * 14
        ),
        expected_sql=expected_sql,
        policy=_policy(("sale_id", "amount"), optional=("region_name",), identity=("sale_id",)),
        schema=schema,
    )

    assert comparison.assessable is False
    assert "different_predicate" not in comparison.proof_reasons


def test_date_literal_redaction_and_boolean_predicate_are_not_collapsed() -> None:
    schema = load_semantic_policy().schema
    date_proof = semantic_evaluation._predicate_proof(
        "SELECT sale_id FROM chatbi_demo.sales WHERE sold_on >= DATE '2026-01-01'",
        "SELECT sale_id FROM chatbi_demo.sales WHERE sold_on >= DATE '<redacted>'",
        schema,
        actual_literals_available=False,
    )
    false_proof = semantic_evaluation._predicate_proof(
        "SELECT sale_id FROM chatbi_demo.sales WHERE FALSE",
        "SELECT sale_id FROM chatbi_demo.sales WHERE sold_on >= '<redacted>'",
        schema,
        actual_literals_available=False,
    )

    assert date_proof.status is semantic_evaluation.SemanticProofStatus.UNKNOWN
    assert false_proof.status is semantic_evaluation.SemanticProofStatus.PROVEN_DIFFERENT


def test_non_null_foreign_key_proves_unused_inner_join_redundant() -> None:
    schema = load_semantic_policy().schema
    expected = ExpectedQueryResult(columns=["amount"], rows=[[10]])
    actual = _result(["amount"], [[10]])

    comparison = compare_semantic_result(
        actual,
        expected,
        policy=_policy(("amount",)),
        actual_sql="SELECT s.amount FROM chatbi_demo.sales AS s",
        expected_sql=(
            "SELECT s.amount FROM chatbi_demo.sales AS s "
            "JOIN chatbi_demo.customers AS c ON c.customer_id = s.customer_id"
        ),
        schema=schema,
    )

    assert comparison.equivalent is True
    assert "equivalent_by_foreign_key_join_redundancy" in comparison.proof_reasons


def test_predicate_and_term_reordering_is_proven_equivalent() -> None:
    expected = ExpectedQueryResult(columns=["amount"], rows=[[10]])
    actual = _result(["amount"], [[10]])

    comparison = compare_semantic_result(
        actual,
        expected,
        policy=_policy(("amount",)),
        actual_sql="SELECT amount FROM sales WHERE amount > 0 AND sale_id = 1",
        expected_sql="SELECT amount FROM sales WHERE sale_id = 1 AND amount > 0",
    )

    assert comparison.equivalent is True


def test_arbitrary_extra_projection_is_unassessable_without_policy_or_fd() -> None:
    expected = ExpectedQueryResult(columns=["amount"], rows=[[10]])
    actual = _result(["amount", "sold_on"], [[10, "2026-01-01"]])

    comparison = compare_semantic_result(
        actual,
        expected,
        policy=_policy(("amount",)),
        actual_sql="SELECT amount, sold_on FROM sales",
        expected_sql="SELECT amount FROM sales",
    )

    assert comparison.equivalent is False
    assert comparison.assessable is False
    assert "extra_projection_safety_unproven" in comparison.proof_reasons


def test_nullable_unique_is_not_treated_as_a_grouping_determinant() -> None:
    schema = _schema_variant(
        lambda raw: raw["relations"][1]["not_null_columns"].remove("region_name")
    )
    proof = semantic_evaluation._group_proof(
        "SELECT region_name FROM chatbi_demo.regions GROUP BY region_code, region_name",
        "SELECT region_name FROM chatbi_demo.regions GROUP BY region_name",
        schema,
    )

    assert proof.status is not semantic_evaluation.SemanticProofStatus.PROVEN_EQUIVALENT


def test_composite_unique_and_primary_keys_require_all_components() -> None:
    unique_schema = _schema_variant(
        lambda raw: raw["relations"][1].update(unique_constraints=[["region_name", "market"]])
    )
    primary_schema = _schema_variant(
        lambda raw: raw["relations"][0].update(primary_key=["customer_id", "customer_name"])
    )

    unique_closure = semantic_evaluation._functional_closure(
        {"chatbi_demo.regions.region_name"}, unique_schema
    )
    primary_closure = semantic_evaluation._functional_closure(
        {"chatbi_demo.customers.customer_id"}, primary_schema
    )

    assert "chatbi_demo.regions.market" not in unique_closure
    assert "chatbi_demo.customers.customer_name" not in primary_closure


def test_nullable_or_composite_foreign_key_does_not_prove_redundant_join() -> None:
    nullable_schema = _schema_variant(
        lambda raw: raw["relations"][2]["not_null_columns"].remove("customer_id")
    )
    composite_schema = _schema_variant(
        lambda raw: raw["relations"][2]["foreign_keys"][0].update(
            source_columns=["customer_id", "region_code"],
            target_columns=["customer_id", "customer_id"],
        )
    )
    expected_sql = (
        "SELECT s.amount FROM chatbi_demo.sales AS s "
        "JOIN chatbi_demo.customers AS c ON c.customer_id = s.customer_id"
    )
    actual_sql = "SELECT s.amount FROM chatbi_demo.sales AS s"

    nullable_proof = semantic_evaluation._join_proof(expected_sql, actual_sql, nullable_schema)
    composite_proof = semantic_evaluation._join_proof(expected_sql, actual_sql, composite_schema)

    assert nullable_proof.status is not semantic_evaluation.SemanticProofStatus.PROVEN_EQUIVALENT
    assert composite_proof.status is not semantic_evaluation.SemanticProofStatus.PROVEN_EQUIVALENT


def test_joined_table_predicate_or_projection_blocks_redundant_join_proof() -> None:
    schema = load_semantic_policy().schema
    actual_sql = "SELECT s.amount FROM chatbi_demo.sales AS s"
    predicate_sql = (
        "SELECT s.amount FROM chatbi_demo.sales AS s "
        "JOIN chatbi_demo.customers AS c ON c.customer_id = s.customer_id "
        "WHERE c.customer_name = '北辰制造'"
    )
    projection_sql = (
        "SELECT s.amount, c.customer_name FROM chatbi_demo.sales AS s "
        "JOIN chatbi_demo.customers AS c ON c.customer_id = s.customer_id"
    )

    predicate_proof = semantic_evaluation._join_proof(predicate_sql, actual_sql, schema)
    projection_proof = semantic_evaluation._join_proof(projection_sql, actual_sql, schema)

    assert predicate_proof.status is not semantic_evaluation.SemanticProofStatus.PROVEN_EQUIVALENT
    assert projection_proof.status is not semantic_evaluation.SemanticProofStatus.PROVEN_EQUIVALENT


@pytest.mark.parametrize(
    ("expected_sql", "actual_sql", "available", "status"),
    [
        (
            "SELECT amount FROM chatbi_demo.sales LIMIT 5",
            "SELECT amount FROM chatbi_demo.sales LIMIT 5",
            True,
            "proven_equivalent",
        ),
        (
            "SELECT amount FROM chatbi_demo.sales LIMIT 5",
            "SELECT amount FROM chatbi_demo.sales LIMIT 6",
            True,
            "proven_different",
        ),
        (
            "SELECT amount FROM chatbi_demo.sales LIMIT 5",
            "SELECT amount FROM chatbi_demo.sales LIMIT 5",
            False,
            "unknown",
        ),
        (
            "SELECT amount FROM chatbi_demo.sales OFFSET 2",
            "SELECT amount FROM chatbi_demo.sales OFFSET 2",
            False,
            "unknown",
        ),
        (
            "SELECT amount FROM chatbi_demo.sales OFFSET 2",
            "SELECT amount FROM chatbi_demo.sales OFFSET 3",
            True,
            "proven_different",
        ),
        (
            "SELECT amount FROM chatbi_demo.sales FETCH FIRST 5 ROWS ONLY",
            "SELECT amount FROM chatbi_demo.sales FETCH FIRST 5 ROWS ONLY",
            True,
            "proven_equivalent",
        ),
        (
            "SELECT amount FROM chatbi_demo.sales FETCH FIRST 5 ROWS ONLY",
            "SELECT amount FROM chatbi_demo.sales FETCH FIRST 5 ROWS ONLY",
            False,
            "unknown",
        ),
        (
            "SELECT amount FROM chatbi_demo.sales LIMIT ALL",
            "SELECT amount FROM chatbi_demo.sales LIMIT 5",
            True,
            "unknown",
        ),
    ],
)
def test_resource_bounds_are_conservative(
    expected_sql: str, actual_sql: str, available: bool, status: str
) -> None:
    proof = semantic_evaluation._order_limit_proof(
        expected_sql,
        actual_sql,
        load_semantic_policy().schema,
        order_required=False,
        actual_literals_available=available,
    )

    assert proof.status.value == status


def test_null_sensitive_count_and_distinct_count_are_not_equivalent() -> None:
    expected = ExpectedQueryResult(columns=["count"], rows=[[2]])
    actual = _result(["count"], [[2]])
    count_nullable = compare_semantic_result(
        actual,
        expected,
        policy=_policy(("aggregate:count:*",), grain=SemanticRowGrain.SCALAR),
        actual_sql="SELECT COUNT(customer_name) AS count FROM chatbi_demo.customers",
        expected_sql="SELECT COUNT(*) AS count FROM chatbi_demo.customers",
    )
    count_distinct = compare_semantic_result(
        actual,
        expected,
        policy=_policy(("aggregate:count:customer_name",), grain=SemanticRowGrain.SCALAR),
        actual_sql="SELECT COUNT(DISTINCT customer_name) AS count FROM chatbi_demo.customers",
        expected_sql="SELECT COUNT(customer_name) AS count FROM chatbi_demo.customers",
    )

    assert count_nullable.equivalent is False
    assert count_distinct.equivalent is False


def test_set_operation_duplicate_semantics_do_not_pass_on_matching_shape() -> None:
    expected_sql = (
        "SELECT customer_id FROM chatbi_demo.customers WHERE customer_id = 1 "
        "UNION SELECT customer_id FROM chatbi_demo.customers WHERE customer_id = 1"
    )
    actual_sql = (
        "SELECT customer_id FROM chatbi_demo.customers WHERE customer_id = 1 "
        "UNION ALL SELECT customer_id FROM chatbi_demo.customers WHERE customer_id = 1"
    )
    proof = semantic_evaluation._set_operation_proof(expected_sql, actual_sql)

    assert proof.status is semantic_evaluation.SemanticProofStatus.UNKNOWN


def test_set_operation_branch_rewrite_is_not_proven_equivalent() -> None:
    expected_sql = (
        "SELECT customer_id FROM chatbi_demo.customers WHERE customer_id = 1 "
        "UNION SELECT customer_id FROM chatbi_demo.customers WHERE customer_id = 2"
    )
    actual_sql = (
        "SELECT customer_id FROM chatbi_demo.customers WHERE customer_id = 1 "
        "UNION SELECT customer_id + 1 FROM chatbi_demo.customers WHERE customer_id = 2"
    )

    proof = semantic_evaluation._set_operation_proof(expected_sql, actual_sql)

    assert proof.status is semantic_evaluation.SemanticProofStatus.UNKNOWN


@pytest.mark.parametrize("operator", ["INTERSECT", "EXCEPT"])
def test_supported_set_operation_structure_is_stable(operator: str) -> None:
    sql = (
        "SELECT customer_id FROM chatbi_demo.customers WHERE customer_id = 1 "
        f"{operator} SELECT customer_id FROM chatbi_demo.customers WHERE customer_id = 1"
    )
    proof = semantic_evaluation._set_operation_proof(sql, sql)

    assert proof.status is semantic_evaluation.SemanticProofStatus.PROVEN_EQUIVALENT


def test_structural_artifact_does_not_promote_formal_pass_without_literals() -> None:
    report = evaluate_provider_artifact(
        Path("docs/benchmarks/a5-7-chatbi-eval-v2.json"),
        Path("data/evaluation/a5-7/provider/chatbi-eval-v2-dev-run-6.json"),
    )

    assert report.aggregate.formal_execution_equivalent_count == 21
    assert report.aggregate.assessable_case_count == 14
    assert report.aggregate.semantic_equivalent_count == 6
    assert report.aggregate.shape_compliant_count == 13
    assert report.aggregate.shape_non_compliant_count == 32
    assert report.aggregate.shape_unassessable_count == 2
    assert report.aggregate.presentation_only_failure_count == 6
    assert report.aggregate.semantic_failure_count == 8
    assert report.aggregate.unassessable_count == 33
    assert report.aggregate.structurally_compatible_unassessable_count == 3
    assert report.aggregate.result_unavailable_unassessable_count == 2

    assert report.cases["group-by-04"].equivalent is True
    assert "equivalent_by_unique_not_null_dependency" in report.cases["group-by-04"].proof_reasons
    assert report.cases["group-by-05"].equivalent is True
    assert "equivalent_by_unique_not_null_dependency" in report.cases["group-by-05"].proof_reasons
    assert report.cases["join-05"].equivalent is True
    assert "equivalent_by_primary_key_dependency" in report.cases["join-05"].proof_reasons
    assert report.cases["multi-join-02"].equivalent is True
    assert (
        "equivalent_by_foreign_key_join_redundancy" in report.cases["multi-join-02"].proof_reasons
    )
    assert "equivalent_by_unique_not_null_dependency" in report.cases["multi-join-02"].proof_reasons
    assert report.cases["join-03"].assessable is False
    assert "literal_value_unavailable" in report.cases["join-03"].proof_reasons
    assert report.cases["cte-set-02"].assessable is False
    assert "different_predicate" not in report.cases["cte-set-02"].proof_reasons

    unsupported_cases = (
        "simple-03",
        "simple-04",
        "top-k-02",
        "top-k-03",
        "top-k-04",
        "date-01",
        "date-03",
    )
    for case_id in unsupported_cases:
        comparison = report.cases[case_id]
        assert comparison.equivalent is False
        assert comparison.assessable is False


def test_provider_artifact_fixture_fingerprint_mismatch_fails_closed(tmp_path: Path) -> None:
    source = Path("data/evaluation/a5-7/provider/chatbi-eval-v2-dev-run-6.json")
    artifact = json.loads(source.read_text(encoding="utf-8"))
    artifact["fixture_fingerprint"] = "0" * 64
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(SemanticEvaluationError, match="fixture fingerprint mismatch"):
        evaluate_provider_artifact(
            Path("docs/benchmarks/a5-7-chatbi-eval-v2.json"),
            path,
        )


def test_provider_artifact_dataset_fingerprint_mismatch_fails_closed(tmp_path: Path) -> None:
    source = Path("data/evaluation/a5-7/provider/chatbi-eval-v2-dev-run-6.json")
    artifact = json.loads(source.read_text(encoding="utf-8"))
    artifact["dataset_fingerprint"] = "0" * 64
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(SemanticEvaluationError, match="dataset fingerprint mismatch"):
        evaluate_provider_artifact(
            Path("docs/benchmarks/a5-7-chatbi-eval-v2.json"),
            path,
        )


def test_unlisted_case_uses_declared_default_policy() -> None:
    artifact = load_semantic_policy()
    expected = ExpectedQueryResult(columns=["customer_id"], rows=[])
    policy = policy_for_case(
        artifact,
        case_id="not-human-reviewed",
        question="任意问题",
        expected=expected,
        expected_sql="SELECT customer_id FROM chatbi_demo.customers",
    )

    assert policy == artifact.defaults
    assert policy.required_fields == ()
