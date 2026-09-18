from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from knowledge_scope.chatbi.schemas import ColumnMetadata, QueryExecutionResult, QueryLifecycleState
from knowledge_scope.evaluation.chatbi_evaluation import ExpectedQueryResult
from knowledge_scope.evaluation.chatbi_evaluation_v2 import load_chatbi_evaluation_dataset_v2
from knowledge_scope.evaluation.chatbi_semantic_evaluation import (
    DEFAULT_G3_SEMANTIC_POLICY_PATH,
    G3_DEV_POSITIVE_CASE_IDS,
    SEMANTIC_POLICY_G3_VERSION,
    SEMANTIC_POLICY_VERSION,
    SemanticEvaluationError,
    SemanticField,
    SemanticPolicyAuthoringConfidence,
    SemanticPolicyDuplicateKeyError,
    SemanticRowGrain,
    compare_semantic_result,
    load_semantic_policy,
    load_semantic_policy_v3,
    load_versioned_semantic_policy,
    policy_for_case,
    semantic_field_is_supported,
    semantic_policy_v3_fingerprint,
)


def test_g3_policy_has_exact_dev_positive_case_coverage_without_test_entries() -> None:
    dataset = load_chatbi_evaluation_dataset_v2(Path("docs/benchmarks/a5-7-chatbi-eval-v2.json"))
    dev_positive_ids = {
        case.case_id for case in dataset.cases if case.split == "dev" and case.positive
    }
    policy = load_semantic_policy_v3()

    assert len(dev_positive_ids) == 47
    assert dev_positive_ids == G3_DEV_POSITIVE_CASE_IDS == set(policy.cases)
    assert not {case.case_id for case in dataset.cases if case.split == "test"} & set(policy.cases)
    assert policy.policy_fingerprint == (
        "7da8b8c4b843dd1cdb50bc56cb2f826a02b625d1c797084f3000dbdb8cdf70ee"
    )


@pytest.mark.parametrize(
    "raw",
    [
        ('{"schema_version":"a5.7g3-semantic-policy-v1","cases":{"simple-01":{},"simple-01":{}}}'),
        (
            '{"schema_version":"a5.7g3-semantic-policy-v1",'
            '"schema_version":"a5.7g3-semantic-policy-v1"}'
        ),
        (
            '{"schema_version":"a5.7g3-semantic-policy-v1",'
            '"cases":{"simple-01":{"row_grain":"detail",'
            '"row_grain":"entity"}}}'
        ),
    ],
    ids=["duplicate-case-id", "duplicate-top-level-key", "duplicate-nested-policy-key"],
)
def test_g3_loader_rejects_duplicate_json_keys_before_model_validation(
    tmp_path: Path, raw: str
) -> None:
    path = tmp_path / "duplicate-policy.json"
    path.write_text(raw, encoding="utf-8")

    with pytest.raises(SemanticPolicyDuplicateKeyError, match="duplicate JSON object key"):
        load_semantic_policy_v3(path)


def test_historical_f2_loader_uses_duplicate_key_rejection(tmp_path: Path) -> None:
    path = tmp_path / "duplicate-f2-policy.json"
    path.write_text(
        '{"schema_version":"a5.7f2-semantic-policy-v1",'
        '"schema_version":"a5.7f2-semantic-policy-v1"}',
        encoding="utf-8",
    )

    with pytest.raises(SemanticPolicyDuplicateKeyError, match="duplicate JSON object key"):
        load_versioned_semantic_policy(path)


def test_g3_policy_supports_pair_and_all_confidence_values() -> None:
    policy = load_semantic_policy_v3()

    assert policy.cases["join-02"].row_grain is SemanticRowGrain.PAIR
    assert {item.value for item in SemanticPolicyAuthoringConfidence} == {
        "explicit",
        "schema_required",
        "product_rule_derived",
        "ambiguous",
    }
    assert policy.cases["derived-01"].required_fields == (
        SemanticField.SALE_ID,
        SemanticField.AMOUNT_MINUS_OVERALL_AVERAGE,
    )


@pytest.mark.parametrize(
    "value",
    [
        "customer_id",
        "aggregate:count:distinct:customer_id",
        "derived:region_sales_share",
    ],
)
def test_g3_semantic_field_vocabulary_is_closed(value: str) -> None:
    assert semantic_field_is_supported(value)


def test_g3_unknown_semantic_field_is_rejected(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_G3_SEMANTIC_POLICY_PATH.read_text(encoding="utf-8"))
    payload["cases"]["simple-01"]["required_fields"] = ["arbitrary_sql_expression"]
    path = tmp_path / "unknown-field.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(SemanticEvaluationError):
        load_semantic_policy_v3(path)


@pytest.mark.parametrize("mutation", ["remove", "add"])
def test_g3_case_set_rejects_missing_or_extra_entries(tmp_path: Path, mutation: str) -> None:
    payload = json.loads(DEFAULT_G3_SEMANTIC_POLICY_PATH.read_text(encoding="utf-8"))
    if mutation == "remove":
        payload["cases"].pop("simple-01")
    else:
        payload["cases"]["test-only"] = payload["cases"]["simple-02"]
    path = tmp_path / f"{mutation}-case.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(SemanticEvaluationError):
        load_semantic_policy_v3(path)


def test_g3_rationale_is_bounded_and_excludes_execution_text(tmp_path: Path) -> None:
    payload = json.loads(DEFAULT_G3_SEMANTIC_POLICY_PATH.read_text(encoding="utf-8"))
    payload["cases"]["simple-01"]["human_rationale"] = "SELECT * FROM chatbi_demo.sales"
    path = tmp_path / "unsafe-rationale.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(SemanticEvaluationError):
        load_semantic_policy_v3(path)

    payload["cases"]["simple-01"]["human_rationale"] = "x" * 513
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(SemanticEvaluationError):
        load_semantic_policy_v3(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("row_grain", "pair"),
        ("required_fields", ["amount"]),
        ("optional_fields", ["amount"]),
        ("identity_fields", ["sale_id"]),
        ("semantic_order_required", False),
        ("allow_safe_extra_fields", True),
        ("authoring_confidence", "explicit"),
        ("human_rationale", "改写后的审阅理由。"),
    ],
)
def test_g3_fingerprint_rejects_mutation_of_executable_or_audit_fields(
    tmp_path: Path, field: str, value: object
) -> None:
    payload = json.loads(DEFAULT_G3_SEMANTIC_POLICY_PATH.read_text(encoding="utf-8"))
    if field == "human_rationale" or field == "authoring_confidence":
        payload["cases"]["simple-01"][field] = value
    else:
        payload["cases"]["simple-01"][field] = value
    path = tmp_path / f"mutated-{field}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(SemanticEvaluationError):
        load_semantic_policy_v3(path)


def test_g3_metadata_is_not_executable_semantics() -> None:
    artifact = load_semantic_policy_v3()
    original = artifact.cases["join-02"]
    changed = original.model_copy(
        update={
            "authoring_confidence": SemanticPolicyAuthoringConfidence.PRODUCT_RULE_DERIVED,
            "human_rationale": "只修改审阅记录,不修改题目要求。",
        }
    )

    assert changed.to_executable_policy() == original.to_executable_policy()
    assert semantic_policy_v3_fingerprint(artifact) == artifact.policy_fingerprint


def test_pair_policy_is_visible_to_evaluator_without_new_proof_rules() -> None:
    artifact = load_semantic_policy_v3()
    policy = policy_for_case(
        artifact,
        case_id="join-02",
        question="",
        expected=ExpectedQueryResult(
            columns=["customer_id", "customer_name", "sale_id"],
            rows=[[1, "A", 10]],
        ),
        expected_sql="SELECT customer_id, customer_name, sale_id FROM chatbi_demo.sales",
    )
    actual = compare_semantic_result(
        actual=_result(["customer_id", "customer_name", "sale_id"], [[1, "A", 10]]),
        expected=ExpectedQueryResult(
            columns=["customer_id", "customer_name", "sale_id"],
            rows=[[1, "A", 10]],
        ),
        policy=policy,
        actual_sql="SELECT customer_id, customer_name, sale_id FROM chatbi_demo.sales",
        expected_sql="SELECT customer_id, customer_name, sale_id FROM chatbi_demo.sales",
    )

    assert policy.row_grain is SemanticRowGrain.PAIR
    assert actual.equivalent is True


def test_old_policy_and_versioned_loader_remain_compatible() -> None:
    old = load_semantic_policy()
    assert old.schema_version == SEMANTIC_POLICY_VERSION
    assert (
        load_versioned_semantic_policy(Path("docs/benchmarks/a5-7-semantic-policy-v1.json")) == old
    )
    assert load_versioned_semantic_policy(DEFAULT_G3_SEMANTIC_POLICY_PATH).schema_version == (
        SEMANTIC_POLICY_G3_VERSION
    )


def test_versioned_loader_rejects_unknown_policy_version(tmp_path: Path) -> None:
    path = tmp_path / "unknown-version.json"
    path.write_text(json.dumps({"schema_version": "unknown"}), encoding="utf-8")

    with pytest.raises(SemanticEvaluationError, match="unsupported semantic policy version"):
        load_versioned_semantic_policy(path)


def test_g3_artifact_has_no_candidate_or_oracle_fields() -> None:
    raw = DEFAULT_G3_SEMANTIC_POLICY_PATH.read_text(encoding="utf-8")
    for forbidden in (
        "candidate_sql",
        "redacted_sql",
        "expected_sql",
        "generated_rows",
        "execution_equivalent",
        "proof_outcome",
        "provider_response",
    ):
        assert forbidden not in raw


def _result(columns: list[str], rows: list[list[object]]):
    datasource_id = UUID("00000000-0000-4000-8000-000000000057")
    return QueryExecutionResult(
        query_id=uuid4(),
        datasource_id=datasource_id,
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
