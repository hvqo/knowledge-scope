from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from pydantic import ValidationError

import knowledge_scope.evaluation.chatbi_evaluation_v2_provider as provider_module
from knowledge_scope.chatbi import (
    ChatBIAgentService,
    ChatBIResult,
    ChatBIUsageSummary,
    ColumnMetadata,
    EnvironmentCredentialResolver,
    NL2SQLService,
    PostgresExecutionAdapter,
    QueryExecutionResult,
    QueryLifecycleState,
    QueryPolicy,
    SQLExecutionService,
)
from knowledge_scope.chatbi.discovery import create_postgres_schema_discovery_service
from knowledge_scope.chatbi.errors import ChatBIErrorCategory
from knowledge_scope.chatbi.models import ChatBIDataSourceRecord
from knowledge_scope.chatbi.nl2sql import (
    LLMUsageMetadata,
    NL2SQLGenerationError,
    _RegisteredValidation,
)
from knowledge_scope.chatbi.nl2sql_models import ValidatedSQL
from knowledge_scope.chatbi.policy import SQLDialect
from knowledge_scope.chatbi.registry import DatabaseDataSourceProvider
from knowledge_scope.chatbi.result_contract import (
    OutputColumnKind,
    ResultContract,
    ResultGrain,
    ResultOutputColumn,
)
from knowledge_scope.chatbi.schema_models import (
    SchemaColumn,
    SchemaDiscoveryResult,
    SchemaObjectKind,
    SchemaRelation,
    SchemaSnapshot,
    build_semantic_schema_context,
)
from knowledge_scope.evaluation.chatbi_evaluation import (
    ChatBIEvaluationObservation,
    ChatBIStageTimings,
    EvaluationRuntimeConfiguration,
    EvaluationStageState,
    RepairOutcome,
    _TimedValidation,
    build_result_contract_diagnostic,
)
from knowledge_scope.evaluation.chatbi_evaluation_v2 import (
    ChatBIEvaluationV2Category,
    ChatBIEvaluationV2Difficulty,
    load_chatbi_evaluation_dataset_v2,
)
from knowledge_scope.evaluation.chatbi_evaluation_v2_provider import (
    _RESULT_CONTRACT_DIAGNOSTIC_CASE_IDS,
    CHATBI_EVALUATION_V2_CONNECTION_REF,
    CHATBI_EVALUATION_V2_DATABASE_NAME,
    CHATBI_EVALUATION_V2_DATASOURCE_DISPLAY_NAME,
    CHATBI_EVALUATION_V2_DATASOURCE_ID,
    CHATBI_EVALUATION_V2_PROVIDER_RUN_SCHEMA_VERSION,
    CHATBI_EVALUATION_V2_PROVIDER_RUN_SCHEMA_VERSION_V3,
    CHATBI_EVALUATION_V2_SCHEMA,
    DEFAULT_DATASET_V2,
    DEFAULT_FIXTURE_PATH_V2,
    DEFAULT_PROVIDER_OUTPUT_V2,
    EXPECTED_DATASET_FINGERPRINT_V2,
    EXPECTED_FIXTURE_DATA_FINGERPRINT_V2,
    EXPECTED_FIXTURE_FINGERPRINT_V2,
    EXPECTED_FIXTURE_SCHEMA_FINGERPRINT_V2,
    EXPECTED_SEMANTIC_POLICY_G3_FINGERPRINT,
    EXPECTED_SEMANTIC_POLICY_G3_VERSION,
    V2ProviderBenchmarkError,
    V2ProviderCaseRecord,
    V2ProviderPreflightReport,
    V2ProviderRun,
    V2ProviderRunProvenance,
    V2ProviderSplit,
    _authoritative_record_matches,
    _build_v2_aggregate,
    _compare_result_contract_diagnostic,
    _expected_result_contract_diagnostic,
    _fixture_data_fingerprint,
    _fixture_schema_fingerprint,
    _fixture_schema_payload,
    _local_postgres_urls,
    _parse_v2_contract_only_payload,
    _prepare_v2_preflight,
    _ResultContractCapture,
    _TimingState,
    _v2_configuration,
    _v2_query_policy,
    _v2_record,
    _v2_usage_aggregate,
    _V2AgentRunner,
    build_v2_contract_only_messages,
    evaluate_v2_provider_cases,
    load_frozen_semantic_policy_provenance,
    load_v2_provider_run,
    select_v2_provider_cases,
    select_v2_provider_contract_diagnostic_cases,
    select_v2_provider_diagnostic_cases,
)
from knowledge_scope.evaluation.chatbi_schema_fingerprint import normalize_check_definition
from knowledge_scope.evaluation.chatbi_schema_provenance import (
    canonical_postgres_schema_payload,
    discover_postgres_schema_provenance,
)
from knowledge_scope.evaluation.schema_provenance import EvaluationValidatedSQL
from knowledge_scope.evaluation.semantic_evidence import (
    SEMANTIC_EVIDENCE_VERSION,
    SemanticEvidenceError,
    SemanticEvidenceRecord,
    build_unavailable_semantic_evidence,
    capture_semantic_evidence,
    evidence_digest_for,
)
from knowledge_scope.llm import LLMProviderInvocation, LLMRequest, LLMResult
from knowledge_scope.llm.usage import (
    CompositeProviderInvocationRecorder,
    InMemoryProviderInvocationRecorder,
)
from knowledge_scope.shared.config import Settings


def test_v2_provider_selects_only_the_frozen_dev_split() -> None:
    dataset = load_chatbi_evaluation_dataset_v2(DEFAULT_DATASET_V2)

    selected = select_v2_provider_cases(dataset, V2ProviderSplit.DEV.value)

    assert len(selected) == 50
    assert {case.split for case in selected} == {"dev"}
    with pytest.raises(V2ProviderBenchmarkError, match="permits only"):
        select_v2_provider_cases(dataset, V2ProviderSplit.TEST.value)


def test_v2_structured_output_diagnostic_selects_exact_frozen_controls() -> None:
    dataset = load_chatbi_evaluation_dataset_v2(DEFAULT_DATASET_V2)

    selected = select_v2_provider_diagnostic_cases(dataset, V2ProviderSplit.DEV.value)

    assert [case.case_id for case in selected] == [
        "aggregation-01",
        "join-02",
        "group-by-02",
    ]
    with pytest.raises(V2ProviderBenchmarkError):
        select_v2_provider_diagnostic_cases(
            dataset,
            V2ProviderSplit.DEV.value,
            ("aggregation-01", "join-02"),
        )
    with pytest.raises(V2ProviderBenchmarkError):
        select_v2_provider_diagnostic_cases(dataset, V2ProviderSplit.TEST.value)


def test_v2_result_contract_diagnostic_selects_exact_eleven_frozen_dev_cases() -> None:
    dataset = load_chatbi_evaluation_dataset_v2(DEFAULT_DATASET_V2)

    selected = select_v2_provider_contract_diagnostic_cases(
        dataset,
        V2ProviderSplit.DEV.value,
    )

    assert [case.case_id for case in selected] == list(_RESULT_CONTRACT_DIAGNOSTIC_CASE_IDS)
    with pytest.raises(V2ProviderBenchmarkError):
        select_v2_provider_contract_diagnostic_cases(
            dataset,
            V2ProviderSplit.DEV.value,
            _RESULT_CONTRACT_DIAGNOSTIC_CASE_IDS[:-1],
        )
    with pytest.raises(V2ProviderBenchmarkError, match="only"):
        select_v2_provider_contract_diagnostic_cases(dataset, V2ProviderSplit.TEST.value)


def test_result_contract_diagnostic_omits_raw_expression_and_alias() -> None:
    contract = ResultContract(
        row_grain=ResultGrain.DETAIL,
        grain_keys=("public.sales.sale_id",),
        output_columns=(
            ResultOutputColumn(
                kind=OutputColumnKind.SOURCE,
                source="public.sales.amount",
                alias="sensitive_alias",
            ),
            ResultOutputColumn(
                kind=OutputColumnKind.DERIVED,
                expression="amount * 987654321",
                source_columns=("public.sales.amount",),
                alias="secret_total",
            ),
        ),
    )

    summary = build_result_contract_diagnostic(
        contract,
        logical_stage="generation",
        validation_status="accepted",
    )
    encoded = json.dumps(summary.model_dump(mode="json"), ensure_ascii=False)

    assert summary.output_columns[1].expression_classification == "arithmetic"
    assert "987654321" not in encoded
    assert "sensitive_alias" not in encoded
    assert "secret_total" not in encoded


def test_contract_capture_records_success_and_semantic_failure_without_raw_text() -> None:
    contract = ResultContract(
        row_grain=ResultGrain.DETAIL,
        grain_keys=("public.sales.sale_id",),
        output_columns=(
            ResultOutputColumn(kind=OutputColumnKind.SOURCE, source="public.sales.amount"),
        ),
    )
    capture = _ResultContractCapture()
    candidate = SimpleNamespace(result_contract=contract)
    capture.capture_success(candidate, is_repair=False)  # type: ignore[arg-type]
    error = NL2SQLGenerationError(
        ChatBIErrorCategory.RESULT_CONTRACT_INCONSISTENT,
        "contract inconsistent",
        usage=LLMUsageMetadata(
            provider="fake",
            model="fake",
            input_tokens=1,
            output_tokens=2,
            provider_attempts=1,
        ),
        result_contract=contract,
        boundary_observation=SimpleNamespace(semantic_contract_validation_status="passed"),
    )
    capture.capture_failure(error, is_repair=True)

    assert [item.logical_stage for item in capture.diagnostics] == [
        "generation",
        "repair_generation",
    ]
    assert all(
        "contract inconsistent" not in item.model_dump_json() for item in capture.diagnostics
    )


def test_result_contract_comparison_has_explicit_unavailable_and_exact_states() -> None:
    expected = _expected_result_contract_diagnostic("simple-04")
    assert expected is not None

    unavailable = _compare_result_contract_diagnostic(None, expected)
    assert unavailable.overall == "unavailable"
    assert unavailable.output_columns == "unavailable"

    exact = _compare_result_contract_diagnostic(expected, expected)
    assert exact.overall == "exact"
    assert all(
        getattr(exact, name) == "correct"
        for name in ("row_grain", "grain_keys", "output_columns", "group_by", "order_by", "limit")
    )


def _contract_only_schema_result() -> SchemaDiscoveryResult:
    datasource_id = uuid4()
    snapshot = SchemaSnapshot(
        datasource_id=datasource_id,
        dialect=SQLDialect.POSTGRESQL,
        database_name="chatbi_eval",
        schemas=("chatbi_demo",),
        relations=(
            SchemaRelation(
                schema_name="chatbi_demo",
                name="customers",
                kind=SchemaObjectKind.TABLE,
                columns=(
                    SchemaColumn(
                        name="customer_id",
                        normalized_type="integer",
                        nullable=False,
                        ordinal=1,
                    ),
                    SchemaColumn(
                        name="customer_name",
                        normalized_type="text",
                        nullable=False,
                        ordinal=2,
                    ),
                ),
                primary_key=("customer_id",),
            ),
        ),
    )
    return SchemaDiscoveryResult(
        snapshot=snapshot,
        fingerprint=snapshot.fingerprint,
        context=build_semantic_schema_context(snapshot, max_chars=24_000),
    )


def _contract_only_result_contract() -> ResultContract:
    return ResultContract(
        row_grain=ResultGrain.DETAIL,
        grain_keys=("chatbi_demo.customers.customer_id",),
        output_columns=(
            ResultOutputColumn(
                kind=OutputColumnKind.SOURCE,
                source="chatbi_demo.customers.customer_name",
            ),
        ),
    )


def test_contract_only_parser_accepts_only_one_structured_result_contract() -> None:
    contract_json = json.dumps(
        {"result_contract": _contract_only_result_contract().model_dump(mode="json")}
    )

    parsed, stage = _parse_v2_contract_only_payload(contract_json)

    assert stage == "parsed"
    assert parsed == _contract_only_result_contract()

    extra_sql, extra_stage = _parse_v2_contract_only_payload(
        contract_json[:-1] + ',"sql":"SELECT 1"}'
    )
    assert extra_sql is None
    assert extra_stage == "top_level_schema_invalid"

    duplicate, duplicate_stage = _parse_v2_contract_only_payload(
        '{"result_contract":{},"result_contract":{}}'
    )
    assert duplicate is None
    assert duplicate_stage == "json_decode_failed"

    non_finite, non_finite_stage = _parse_v2_contract_only_payload(
        '{"result_contract":{"row_grain":"detail","grain_keys":[],'
        '"output_columns":[{"kind":"source","source":"x"}],'
        '"group_by":[],"order_by":[],"limit":NaN}}'
    )
    assert non_finite is None
    assert non_finite_stage == "json_decode_failed"


def test_contract_only_prompt_is_schema_data_and_contains_no_sql_generation_contract() -> None:
    discovered = _contract_only_schema_result()
    policy = QueryPolicy(allowed_schemas=("chatbi_demo",))

    messages = build_v2_contract_only_messages(
        question="列出客户名称",
        discovered=discovered,
        policy=policy,
    )
    encoded = "\n".join(message.content for message in messages)

    assert [message.role for message in messages] == ["system", "user"]
    assert "result_contract" in encoded
    assert '"sql"' not in encoded.lower()
    assert "select" not in encoded.lower()
    assert "chatbi_demo" in encoded
    assert "customer_name" in encoded


@pytest.mark.parametrize(
    "artifact_name",
    (
        "chatbi-eval-v2-dev.json",
        "chatbi-eval-v2-dev-run-6.json",
        "chatbi-eval-v2-dev-run-7.json",
        "chatbi-eval-v2-dev-run-8.json",
        "chatbi-eval-v2-dev-run-9.json",
        "chatbi-eval-v2-dev-run-10.json",
        "chatbi-eval-v2-dev-run-11.json",
        "chatbi-eval-v2-dev-run-12.json",
        "chatbi-eval-v2-dev-run-13.json",
    ),
)
def test_historical_provider_artifacts_without_contract_diagnostics_remain_readable(
    artifact_name: str,
) -> None:
    artifact = DEFAULT_PROVIDER_OUTPUT_V2.parent / artifact_name
    if not artifact.is_file():
        pytest.skip("historical provider artifact is not present in the local runtime")

    run = load_v2_provider_run(artifact)

    assert len(run.records) == 50
    assert all(record.result_contract_diagnostics == [] for record in run.records)


def test_full_v2_semantic_evidence_artifact_round_trips_through_real_reader() -> None:
    artifact = DEFAULT_PROVIDER_OUTPUT_V2
    if not artifact.is_file():
        pytest.skip("historical provider artifact is not present in the local runtime")

    legacy = V2ProviderRun.model_validate_json(artifact.read_text(encoding="utf-8"))
    payload = legacy.model_dump(mode="json")
    payload["artifact_schema_version"] = "a5.7g2-provider-run-v2"
    payload["provenance"]["semantic_evidence_version"] = SEMANTIC_EVIDENCE_VERSION
    datasource_id = legacy.datasource_id
    for record in payload["records"]:
        evidence = build_unavailable_semantic_evidence(
            case_id=record["case_id"],
            split=record["split"],
            datasource_id=datasource_id,
            dataset_fingerprint=legacy.dataset_fingerprint,
            fixture_fingerprint=legacy.fixture_fingerprint,
            schema_fingerprint=legacy.fixture_schema_fingerprint,
            prompt_version="a5.3-v3",
            reasoning_mode="disabled",
            reason="roundtrip_test",
        )
        assert evidence_digest_for(evidence) == evidence.evidence_digest
        record["semantic_evidence"] = evidence.model_dump(mode="json")

    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    run = V2ProviderRun.model_validate_json(serialized)
    reloaded = V2ProviderRun.model_validate_json(run.model_dump_json())

    assert reloaded.artifact_schema_version == "a5.7g2-provider-run-v2"
    assert reloaded.provenance.semantic_evidence_version == SEMANTIC_EVIDENCE_VERSION
    assert all(record.semantic_evidence is not None for record in reloaded.records)
    assert all(
        evidence_digest_for(record.semantic_evidence) == record.semantic_evidence.evidence_digest
        for record in reloaded.records
        if record.semantic_evidence is not None
    )


def test_v3_roundtrip_retains_verified_authoritative_schema_fingerprint() -> None:
    artifact = DEFAULT_PROVIDER_OUTPUT_V2
    if not artifact.is_file():
        pytest.skip("historical provider artifact is not present in the local runtime")

    legacy = V2ProviderRun.model_validate_json(artifact.read_text(encoding="utf-8"))
    payload = legacy.model_dump(mode="json")
    payload["artifact_schema_version"] = CHATBI_EVALUATION_V2_PROVIDER_RUN_SCHEMA_VERSION_V3
    payload["provenance"]["semantic_evidence_version"] = SEMANTIC_EVIDENCE_VERSION
    payload["provenance"]["semantic_policy_version"] = EXPECTED_SEMANTIC_POLICY_G3_VERSION
    payload["provenance"]["semantic_policy_fingerprint"] = EXPECTED_SEMANTIC_POLICY_G3_FINGERPRINT
    payload["provenance"]["authoritative_schema_fingerprint"] = (
        EXPECTED_FIXTURE_SCHEMA_FINGERPRINT_V2
    )
    for record in payload["records"]:
        evidence = build_unavailable_semantic_evidence(
            case_id=record["case_id"],
            split=record["split"],
            datasource_id=legacy.datasource_id,
            dataset_fingerprint=legacy.dataset_fingerprint,
            fixture_fingerprint=legacy.fixture_fingerprint,
            schema_fingerprint=legacy.fixture_schema_fingerprint,
            prompt_version="a5.3-v3",
            reasoning_mode="disabled",
            reason="v3_roundtrip_test",
        )
        record["semantic_evidence"] = evidence.model_dump(mode="json")

    run = V2ProviderRun.model_validate(payload)
    reloaded = V2ProviderRun.model_validate_json(run.model_dump_json())

    assert (
        reloaded.provenance.authoritative_schema_fingerprint
        == EXPECTED_FIXTURE_SCHEMA_FINGERPRINT_V2
    )
    assert all(record.semantic_evidence is not None for record in reloaded.records)


def test_serialized_v2_semantic_evidence_rejects_unknown_fields() -> None:
    evidence = build_unavailable_semantic_evidence(
        case_id="unknown-field",
        split="dev",
        datasource_id=CHATBI_EVALUATION_V2_DATASOURCE_ID,
        dataset_fingerprint=EXPECTED_DATASET_FINGERPRINT_V2,
        fixture_fingerprint=EXPECTED_FIXTURE_FINGERPRINT_V2,
        schema_fingerprint=EXPECTED_FIXTURE_SCHEMA_FINGERPRINT_V2,
        prompt_version="a5.3-v3",
        reasoning_mode="disabled",
    )
    payload = evidence.model_dump(mode="json")
    payload["unknown_field"] = "must reject"

    with pytest.raises(ValidationError):
        SemanticEvidenceRecord.model_validate(payload)


def test_v2_provenance_records_retained_nl2sql_reasoning_mode() -> None:
    settings = Settings(_env_file=None)

    configuration = _v2_configuration(
        settings,
        _v2_query_policy(settings),
        max_chars=settings.chatbi_schema_context_max_chars,
    )

    assert configuration.nl2sql_max_tokens == 1024
    assert configuration.nl2sql_reasoning == "disabled"
    assert configuration.nl2sql_thinking_type == "disabled"
    assert configuration.nl2sql_reasoning_effort is None
    assert configuration.nl2sql_prompt_version == "a5.3-v3"
    assert configuration.nl2sql_result_contract_enabled is False

    diagnostic_configuration = _v2_configuration(
        settings,
        _v2_query_policy(settings),
        max_chars=settings.chatbi_schema_context_max_chars,
        result_contract_enabled=True,
    )
    assert diagnostic_configuration.nl2sql_prompt_version == "a5.3-v4"
    assert diagnostic_configuration.nl2sql_result_contract_enabled is True


def _test_validated_sql() -> ValidatedSQL:
    return ValidatedSQL._from_validator(
        validation_version="test-validation-v1",
        datasource_id=CHATBI_EVALUATION_V2_DATASOURCE_ID,
        dialect=SQLDialect.POSTGRESQL,
        provider="fake",
        model="fake-model",
        prompt_version="a5.3-v3",
        original_sql="SELECT 1",
        normalized_sql="SELECT 1",
        referenced_schemas=(CHATBI_EVALUATION_V2_SCHEMA,),
        referenced_relations=(),
        schema_fingerprint=EXPECTED_FIXTURE_SCHEMA_FINGERPRINT_V2,
        policy_fingerprint="a" * 64,
        limit_bounded=True,
        effective_limit=1,
        limit_source="test",
    )


@pytest.mark.anyio
async def test_timed_validation_unwraps_registered_authoritative_validated_sql() -> None:
    validated = _test_validated_sql()
    registered = _RegisteredValidation(
        data_source=SimpleNamespace(),
        request=SimpleNamespace(),
        validated=validated,
    )

    class _Validation:
        async def _validate_registered_candidate(self, *_args: object, **_kwargs: object):
            return registered

    timing = _TimingState()
    result = await _TimedValidation(_Validation(), timing)._validate_registered_candidate()

    assert result is registered
    assert timing.validated_sql is validated
    assert timing.stages.validation == "passed"
    assert timing.validation_ms >= 0


@pytest.mark.anyio
async def test_timed_validation_rejection_does_not_fabricate_validated_sql() -> None:
    class _Validation:
        async def _validate_registered_candidate(self, *_args: object, **_kwargs: object):
            raise ValueError("rejected")

    timing = _TimingState()
    with pytest.raises(ValueError, match="rejected"):
        await _TimedValidation(_Validation(), timing)._validate_registered_candidate()

    assert timing.validated_sql is None
    assert timing.stages.validation == "failed"


@pytest.mark.anyio
async def test_wrapped_authoritative_validation_produces_available_semantic_evidence() -> None:
    validated = _test_validated_sql()
    registered = _RegisteredValidation(
        data_source=SimpleNamespace(),
        request=SimpleNamespace(),
        validated=validated,
    )

    class _Validation:
        async def _validate_registered_candidate(self, *_args: object, **_kwargs: object):
            return registered

    timing = _TimingState()
    await _TimedValidation(_Validation(), timing)._validate_registered_candidate()
    now = datetime.now(UTC)
    invocation = LLMProviderInvocation(
        case_id="wrapper-case",
        logical_stage="generation",
        attempt_index=1,
        provider="fake",
        model="fake-model",
        started_at=now,
        completed_at=now,
        duration_ms=1,
        outcome="success",
        input_tokens=1,
        output_tokens=1,
        llm_result_returned=True,
    )
    result = ChatBIResult(
        query_id=uuid4(),
        datasource_id=CHATBI_EVALUATION_V2_DATASOURCE_ID,
        execution_status=QueryLifecycleState.SUCCEEDED,
        redacted_sql="SELECT 1",
        row_count=1,
        sql_attempts=1,
        repair_attempts=0,
        usage=ChatBIUsageSummary(llm_calls=1, provider_attempts=1),
    )
    execution = QueryExecutionResult(
        query_id=result.query_id,
        datasource_id=CHATBI_EVALUATION_V2_DATASOURCE_ID,
        state=QueryLifecycleState.SUCCEEDED,
        columns=[ColumnMetadata(name="one", data_type="integer", nullable=False, ordinal=1)],
        rows=[[1]],
        row_count=1,
        max_rows=1,
        duration_ms=1,
    )

    evidence = capture_semantic_evidence(
        case_id="wrapper-case",
        split="dev",
        datasource_id=CHATBI_EVALUATION_V2_DATASOURCE_ID,
        dataset_fingerprint=EXPECTED_DATASET_FINGERPRINT_V2,
        fixture_fingerprint=EXPECTED_FIXTURE_FINGERPRINT_V2,
        schema_fingerprint=EXPECTED_FIXTURE_SCHEMA_FINGERPRINT_V2,
        synthetic_fixture_fingerprint=EXPECTED_FIXTURE_FINGERPRINT_V2,
        validated_sql=timing.validated_sql,
        execution_result=execution,
        chatbi_result=result,
        provider_invocations=[invocation],
        prompt_version="a5.3-v3",
        reasoning_mode="disabled",
    )

    assert evidence.evidence_status.value == "available"
    assert "validated_sql_unavailable" not in evidence.unavailable_reasons


def test_evaluation_schema_binding_keeps_native_validation_fingerprint_separate() -> None:
    native_payload = _test_validated_sql().model_dump()
    native_payload["schema_fingerprint"] = (
        "bfb282adc1bb8c6e8597f02b572aa6c9484e3de0b946daa944b39f471f09a94b"
    )
    native = ValidatedSQL._from_validator(**native_payload)
    bound = EvaluationValidatedSQL(native, EXPECTED_FIXTURE_SCHEMA_FINGERPRINT_V2)

    assert bound.native_schema_fingerprint == native.schema_fingerprint
    assert native.schema_fingerprint != bound.evaluation_schema_fingerprint
    assert native.model_dump()["schema_fingerprint"] == native.schema_fingerprint

    result = ChatBIResult(
        query_id=uuid4(),
        datasource_id=CHATBI_EVALUATION_V2_DATASOURCE_ID,
        execution_status=QueryLifecycleState.SUCCEEDED,
        row_count=0,
        sql_attempts=1,
        repair_attempts=0,
        usage=ChatBIUsageSummary(llm_calls=0, provider_attempts=0),
    )
    with pytest.raises(SemanticEvidenceError):
        capture_semantic_evidence(
            case_id="native-mismatch",
            split="dev",
            datasource_id=CHATBI_EVALUATION_V2_DATASOURCE_ID,
            dataset_fingerprint=EXPECTED_DATASET_FINGERPRINT_V2,
            fixture_fingerprint=EXPECTED_FIXTURE_FINGERPRINT_V2,
            schema_fingerprint=EXPECTED_FIXTURE_SCHEMA_FINGERPRINT_V2,
            synthetic_fixture_fingerprint=EXPECTED_FIXTURE_FINGERPRINT_V2,
            validated_sql=native,
            execution_result=None,
            chatbi_result=result,
            provider_invocations=[],
            prompt_version="a5.3-v3",
            reasoning_mode="disabled",
        )

    evidence = capture_semantic_evidence(
        case_id="bound-evaluation",
        split="dev",
        datasource_id=CHATBI_EVALUATION_V2_DATASOURCE_ID,
        dataset_fingerprint=EXPECTED_DATASET_FINGERPRINT_V2,
        fixture_fingerprint=EXPECTED_FIXTURE_FINGERPRINT_V2,
        schema_fingerprint=EXPECTED_FIXTURE_SCHEMA_FINGERPRINT_V2,
        synthetic_fixture_fingerprint=EXPECTED_FIXTURE_FINGERPRINT_V2,
        validated_sql=bound,
        execution_result=None,
        chatbi_result=result,
        provider_invocations=[],
        prompt_version="a5.3-v3",
        reasoning_mode="disabled",
    )
    assert evidence.evidence_status.value == "available"


def test_v2_provenance_uses_resolved_budgets() -> None:
    settings = Settings(
        _env_file=None,
        chatbi_nl2sql_max_tokens=3072,
        chatbi_analysis_max_tokens=768,
    )

    configuration = _v2_configuration(
        settings,
        _v2_query_policy(settings),
        max_chars=settings.chatbi_schema_context_max_chars,
    )

    assert configuration.nl2sql_max_tokens == 3072
    assert configuration.agent_limits.analysis_max_tokens == 768


def test_runtime_provenance_retains_historical_low_and_high_modes() -> None:
    settings = Settings(_env_file=None)
    baseline = _v2_configuration(
        settings,
        _v2_query_policy(settings),
        max_chars=settings.chatbi_schema_context_max_chars,
    )

    for mode in ("low", "high"):
        historical_payload = baseline.model_dump(mode="json")
        historical_payload.update(
            nl2sql_reasoning=mode,
            nl2sql_thinking_type="enabled",
            nl2sql_reasoning_effort=mode,
        )
        historical = EvaluationRuntimeConfiguration.model_validate(historical_payload)
        assert historical.nl2sql_reasoning == mode
        assert historical.nl2sql_thinking_type == "enabled"
        assert historical.nl2sql_reasoning_effort == mode


def test_historical_provenance_without_result_contract_flag_remains_readable() -> None:
    settings = Settings(_env_file=None)
    payload = _v2_configuration(
        settings,
        _v2_query_policy(settings),
        max_chars=settings.chatbi_schema_context_max_chars,
    ).model_dump(mode="json")
    payload.pop("nl2sql_result_contract_enabled")

    historical = EvaluationRuntimeConfiguration.model_validate(payload)

    assert historical.nl2sql_result_contract_enabled is False


def test_v2_provider_cli_rejects_test_before_provider_setup(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from knowledge_scope import cli

    monkeypatch.setattr(cli, "get_settings", lambda: Settings(_env_file=None))

    assert cli.main(["chatbi", "eval-v2-provider", "--split", "test", "--preflight"]) == 1
    assert "permits only" in capsys.readouterr().err


class _CountingV2Runner:
    def __init__(self) -> None:
        self.calls = 0

    async def run(self, _case: object) -> ChatBIEvaluationObservation:
        self.calls += 1
        return ChatBIEvaluationObservation(
            result=ChatBIResult(
                query_id=uuid4(),
                datasource_id=uuid4(),
                execution_status=QueryLifecycleState.SUCCEEDED,
                row_count=0,
                sql_attempts=1,
                repair_attempts=0,
                usage=ChatBIUsageSummary(
                    llm_calls=0,
                    provider_attempts=0,
                ),
            )
        )


def _provider_case_record(
    *,
    case_id: str,
    positive: bool = True,
    initial_generation_parseable: bool = True,
    repair_attempted: bool = False,
    repair_generation_parseable: bool | None = None,
    repair_recovery_success: bool = False,
    provider_invocations: list[LLMProviderInvocation] | None = None,
) -> V2ProviderCaseRecord:
    if not repair_attempted:
        repair_outcome = RepairOutcome.NOT_ATTEMPTED
        repair_generation_state = "not_attempted"
    elif repair_recovery_success:
        repair_outcome = RepairOutcome.REPAIRED
        repair_generation_state = "passed"
    else:
        repair_outcome = RepairOutcome.EXHAUSTED
        repair_generation_state = "passed" if repair_generation_parseable else "failed"
    return V2ProviderCaseRecord(
        case_id=case_id,
        positive=positive,
        category=ChatBIEvaluationV2Category.SIMPLE_FILTER_PROJECTION,
        difficulty=ChatBIEvaluationV2Difficulty.EASY,
        stages=EvaluationStageState(
            generation="passed" if initial_generation_parseable else "failed",
            validation="passed" if positive else "failed",
            execution="passed" if positive else "not_attempted",
            analysis="passed" if positive else "not_attempted",
            repair_generation=repair_generation_state,
        ),
        result_status=QueryLifecycleState.SUCCEEDED if positive else QueryLifecycleState.REJECTED,
        generation_parseable=initial_generation_parseable,
        initial_generation_parseable=initial_generation_parseable,
        eventual_sql_candidate_available=positive or repair_generation_parseable is True,
        repair_generation_parseable=repair_generation_parseable if repair_attempted else None,
        repair_recovery_success=repair_recovery_success,
        validation_accepted=positive,
        execution_equivalent=True if positive else None,
        structured_result_fact_coverage=None,
        expected_failure_match=True if not positive else None,
        repair_attempted=repair_attempted,
        repair_outcome=repair_outcome,
        row_count=1 if positive else 0,
        truncated=False,
        latency=ChatBIStageTimings(
            schema_prep_ms=1,
            generation_ms=1,
            validation_ms=1,
            execution_ms=1,
            repair_ms=1 if repair_attempted else None,
            analysis_ms=1 if positive else None,
            total_ms=1,
        ),
        usage=ChatBIUsageSummary(
            llm_calls=1,
            provider_attempts=1,
            input_tokens=10,
            output_tokens=5,
        ),
        provider_invocations=provider_invocations or [],
        trace_events=["sql_executed"] if positive else [],
    )


def _minimal_v3_provider_run() -> V2ProviderRun:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    records = []
    for index in range(50):
        case_id = f"v3-case-{index:03d}"
        evidence = build_unavailable_semantic_evidence(
            case_id=case_id,
            split="dev",
            datasource_id=CHATBI_EVALUATION_V2_DATASOURCE_ID,
            dataset_fingerprint=EXPECTED_DATASET_FINGERPRINT_V2,
            fixture_fingerprint=EXPECTED_FIXTURE_FINGERPRINT_V2,
            schema_fingerprint=EXPECTED_FIXTURE_SCHEMA_FINGERPRINT_V2,
            prompt_version="a5.3-v3",
            reasoning_mode="disabled",
            reason="test_artifact",
        )
        records.append(
            _provider_case_record(case_id=case_id).model_copy(
                update={"semantic_evidence": evidence}
            )
        )
    settings = Settings(_env_file=None)
    policy = _v2_query_policy(settings)
    provenance = V2ProviderRunProvenance(
        dataset_fingerprint=EXPECTED_DATASET_FINGERPRINT_V2,
        fixture_path=DEFAULT_FIXTURE_PATH_V2.as_posix(),
        fixture_sha256=EXPECTED_FIXTURE_FINGERPRINT_V2,
        fixture_schema_fingerprint=EXPECTED_FIXTURE_SCHEMA_FINGERPRINT_V2,
        fixture_data_fingerprint=EXPECTED_FIXTURE_DATA_FINGERPRINT_V2,
        datasource_id=CHATBI_EVALUATION_V2_DATASOURCE_ID,
        case_count=50,
        git_revision="a" * 40,
        git_dirty=False,
        provider="fake",
        model="fake-model",
        temperature=0.0,
        response_format="json_object",
        configuration=_v2_configuration(settings, policy, max_chars=24_000),
        started_at=now,
        completed_at=now,
        semantic_evidence_version=SEMANTIC_EVIDENCE_VERSION,
        semantic_policy_version=EXPECTED_SEMANTIC_POLICY_G3_VERSION,
        semantic_policy_fingerprint=EXPECTED_SEMANTIC_POLICY_G3_FINGERPRINT,
    )
    return V2ProviderRun(
        artifact_schema_version=CHATBI_EVALUATION_V2_PROVIDER_RUN_SCHEMA_VERSION_V3,
        run_id=uuid4(),
        started_at=now,
        completed_at=now,
        dataset_fingerprint=EXPECTED_DATASET_FINGERPRINT_V2,
        fixture_fingerprint=EXPECTED_FIXTURE_FINGERPRINT_V2,
        fixture_schema_fingerprint=EXPECTED_FIXTURE_SCHEMA_FINGERPRINT_V2,
        datasource_id=CHATBI_EVALUATION_V2_DATASOURCE_ID,
        case_count=50,
        records=records,
        aggregates=_build_v2_aggregate(records),
        provenance=provenance,
    )


def test_frozen_semantic_policy_provenance_is_loaded_and_exact() -> None:
    assert load_frozen_semantic_policy_provenance() == (
        EXPECTED_SEMANTIC_POLICY_G3_VERSION,
        EXPECTED_SEMANTIC_POLICY_G3_FINGERPRINT,
    )


def test_v3_provider_artifact_round_trips_policy_provenance(tmp_path: Path) -> None:
    path = tmp_path / "provider-v3.json"
    run = _minimal_v3_provider_run()
    path.write_text(run.model_dump_json(), encoding="utf-8")

    reloaded = load_v2_provider_run(path)

    assert reloaded.artifact_schema_version == CHATBI_EVALUATION_V2_PROVIDER_RUN_SCHEMA_VERSION_V3
    assert reloaded.provenance.semantic_policy_version == EXPECTED_SEMANTIC_POLICY_G3_VERSION
    assert (
        reloaded.provenance.semantic_policy_fingerprint == EXPECTED_SEMANTIC_POLICY_G3_FINGERPRINT
    )
    assert reloaded.model_dump(mode="json") == run.model_dump(mode="json")


@pytest.mark.parametrize(
    "field, value",
    (
        ("semantic_policy_version", None),
        ("semantic_policy_fingerprint", None),
        ("semantic_policy_version", "a5.7f2-semantic-policy-v1"),
        ("semantic_policy_fingerprint", "0" * 64),
        ("semantic_policy_fingerprint", "not-a-fingerprint"),
    ),
)
def test_v3_provider_artifact_rejects_missing_or_mismatched_policy_provenance(
    field: str,
    value: object,
) -> None:
    payload = _minimal_v3_provider_run().model_dump(mode="json")
    payload["provenance"][field] = value

    with pytest.raises(ValidationError):
        V2ProviderRun.model_validate(payload)


def test_v2_provider_artifact_does_not_require_g3_policy_provenance() -> None:
    payload = _minimal_v3_provider_run().model_dump(mode="json")
    payload["artifact_schema_version"] = CHATBI_EVALUATION_V2_PROVIDER_RUN_SCHEMA_VERSION
    payload["provenance"].pop("semantic_policy_version")
    payload["provenance"].pop("semantic_policy_fingerprint")

    historical = V2ProviderRun.model_validate(payload)

    assert historical.artifact_schema_version == CHATBI_EVALUATION_V2_PROVIDER_RUN_SCHEMA_VERSION
    assert historical.provenance.semantic_policy_version is None
    assert historical.provenance.semantic_policy_fingerprint is None


def test_v3_provider_loader_validates_against_the_loaded_policy_file(tmp_path: Path) -> None:
    path = tmp_path / "provider-v3.json"
    path.write_text(_minimal_v3_provider_run().model_dump_json(), encoding="utf-8")

    with pytest.raises(V2ProviderBenchmarkError):
        load_v2_provider_run(
            path,
            policy_path=Path("docs/benchmarks/a5-7-semantic-policy-v1.json"),
        )


def test_unknown_future_provider_artifact_version_is_rejected() -> None:
    payload = _minimal_v3_provider_run().model_dump(mode="json")
    payload["artifact_schema_version"] = "a5.7future-provider-run-v4"

    with pytest.raises(ValidationError):
        V2ProviderRun.model_validate(payload)


def test_v2_usage_aggregate_separates_attempts_from_completed_results() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    invocations = [
        LLMProviderInvocation(
            case_id="case-001",
            logical_stage="generation",
            attempt_index=1,
            provider="fake",
            model="fake-model",
            started_at=now,
            completed_at=now,
            duration_ms=1,
            outcome="success",
            input_tokens=10,
            output_tokens=5,
            llm_result_returned=True,
        ),
        LLMProviderInvocation(
            case_id="case-001",
            logical_stage="generation",
            attempt_index=2,
            provider="fake",
            model="fake-model",
            started_at=now,
            completed_at=now,
            duration_ms=1,
            outcome="failure",
            error_category="provider_api_error",
            llm_result_returned=False,
        ),
    ]

    aggregate = _v2_usage_aggregate(
        [
            _provider_case_record(
                case_id="case-001",
                provider_invocations=invocations,
            )
        ]
    )

    assert aggregate.provider_attempt_count == 2
    assert aggregate.provider_attempts_total == 2
    assert aggregate.provider_success_count == 1
    assert aggregate.provider_failure_count == 1
    assert aggregate.llm_result_count == 1
    assert aggregate.input_tokens.total == 10
    assert aggregate.output_tokens.total == 5


def test_v2_repair_aggregate_uses_runtime_recovery_not_review_metadata() -> None:
    recovered = _provider_case_record(
        case_id="repair-recovered",
        initial_generation_parseable=False,
        repair_attempted=True,
        repair_generation_parseable=True,
        repair_recovery_success=True,
    )
    parseable_but_not_recovered = _provider_case_record(
        case_id="repair-exhausted",
        initial_generation_parseable=False,
        repair_attempted=True,
        repair_generation_parseable=True,
    )
    failed_repair = _provider_case_record(
        case_id="repair-malformed",
        initial_generation_parseable=False,
        repair_attempted=True,
        repair_generation_parseable=False,
    )

    aggregate = _build_v2_aggregate([recovered, parseable_but_not_recovered, failed_repair])

    assert aggregate.generation_parseable_count == 0
    assert aggregate.repair_attempted_count == 3
    assert aggregate.repair_generation_success_count == 2
    assert aggregate.repair_recovery_success_count == 1
    assert aggregate.repair_success_count == 1
    assert aggregate.repair_failure_count == 2
    assert aggregate.incremental_recovered_count == 1


def test_v2_record_keeps_initial_and_eventual_generation_metrics_distinct() -> None:
    case = next(
        case
        for case in load_chatbi_evaluation_dataset_v2(DEFAULT_DATASET_V2).cases
        if case.split == V2ProviderSplit.DEV.value and case.positive
    )
    result = ChatBIResult(
        query_id=uuid4(),
        datasource_id=CHATBI_EVALUATION_V2_DATASOURCE_ID,
        execution_status=QueryLifecycleState.SUCCEEDED,
        redacted_sql="SELECT 1",
        row_count=1,
        sql_attempts=2,
        repair_attempts=1,
        usage=ChatBIUsageSummary(llm_calls=2, provider_attempts=2),
    )
    record = _v2_record(
        case,
        ChatBIEvaluationObservation(
            result=result,
            stages=EvaluationStageState(
                generation="failed",
                repair_generation="passed",
                validation="passed",
                execution="passed",
            ),
        ),
    )

    assert record.initial_generation_parseable is False
    assert record.generation_parseable is False
    assert record.repair_generation_parseable is True
    assert record.eventual_sql_candidate_available is True


@pytest.mark.anyio
async def test_v2_runner_updates_durable_and_local_invocation_metadata() -> None:
    """Post-processing parse outcomes reach both evaluation and durable sinks."""
    local_recorder = InMemoryProviderInvocationRecorder()
    durable_recorder = InMemoryProviderInvocationRecorder()
    sink = CompositeProviderInvocationRecorder(durable_recorder, local_recorder)
    case = select_v2_provider_cases(
        load_chatbi_evaluation_dataset_v2(DEFAULT_DATASET_V2), V2ProviderSplit.DEV.value
    )[0]

    class _Agent:
        async def ask(self, *_args: object, **_kwargs: object) -> ChatBIResult:
            now = datetime.now(UTC)
            await sink.record_invocation(
                LLMProviderInvocation(
                    case_id=case.case_id,
                    logical_stage="analysis",
                    attempt_index=1,
                    provider="fake",
                    model="fake-model",
                    started_at=now,
                    completed_at=now,
                    duration_ms=1,
                    outcome="success",
                    input_tokens=10,
                    output_tokens=4,
                    llm_result_returned=True,
                    response_parse_outcome="provider_success",
                )
            )
            return ChatBIResult(
                query_id=uuid4(),
                datasource_id=CHATBI_EVALUATION_V2_DATASOURCE_ID,
                execution_status=QueryLifecycleState.SUCCEEDED,
                row_count=0,
                sql_attempts=1,
                repair_attempts=0,
                usage=ChatBIUsageSummary(llm_calls=1, provider_attempts=1),
                analysis_parse_outcome="structured_output_parse_error",
            )

    runner = _V2AgentRunner(
        _Agent(),
        _TimingState(),
        CHATBI_EVALUATION_V2_DATASOURCE_ID,
        QueryPolicy(),
        10_000,
        local_recorder,
        sink,
    )

    observation = await runner.run(case)

    assert observation.provider_invocations[0].response_parse_outcome == (
        "structured_output_parse_error"
    )
    assert durable_recorder.records[0].response_parse_outcome == ("structured_output_parse_error")
    assert durable_recorder.records[0].error_category == "structured_output_parse_error"


@pytest.mark.anyio
async def test_v2_evidence_capture_failure_does_not_replace_authoritative_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_recorder = InMemoryProviderInvocationRecorder()
    case = select_v2_provider_cases(
        load_chatbi_evaluation_dataset_v2(DEFAULT_DATASET_V2), V2ProviderSplit.DEV.value
    )[0]
    expected = ChatBIResult(
        query_id=uuid4(),
        datasource_id=CHATBI_EVALUATION_V2_DATASOURCE_ID,
        execution_status=QueryLifecycleState.SUCCEEDED,
        row_count=1,
        sql_attempts=1,
        repair_attempts=0,
        usage=ChatBIUsageSummary(llm_calls=1, provider_attempts=1),
    )

    class _Agent:
        async def ask(self, *_args: object, **_kwargs: object) -> ChatBIResult:
            return expected

    def _fail(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("collector-only failure")

    monkeypatch.setattr(provider_module, "capture_semantic_evidence", _fail)
    runner = _V2AgentRunner(
        _Agent(),
        _TimingState(),
        CHATBI_EVALUATION_V2_DATASOURCE_ID,
        QueryPolicy(),
        10_000,
        local_recorder,
        local_recorder,
        evidence_dataset_fingerprint=EXPECTED_DATASET_FINGERPRINT_V2,
        evidence_fixture_fingerprint=EXPECTED_FIXTURE_FINGERPRINT_V2,
        evidence_schema_fingerprint=EXPECTED_FIXTURE_SCHEMA_FINGERPRINT_V2,
        evidence_synthetic_fixture_fingerprint=EXPECTED_FIXTURE_FINGERPRINT_V2,
    )

    observation = await runner.run(case)

    assert observation.result.model_dump(mode="json") == expected.model_dump(mode="json")
    assert observation.semantic_evidence is not None
    assert observation.semantic_evidence.evidence_status.value == "unavailable"
    assert "evidence_capture_failed" in observation.semantic_evidence.unavailable_reasons


@pytest.mark.anyio
async def test_v2_provider_api_rejects_non_dev_cases_before_runner() -> None:
    dataset = load_chatbi_evaluation_dataset_v2(DEFAULT_DATASET_V2)
    test_case = next(case for case in dataset.cases if case.split == "test")
    dev_case = next(case for case in dataset.cases if case.split == "dev")
    mutated_dev_case = dev_case.model_copy(update={"question": "changed"})

    for cases in (
        [test_case],
        [dev_case, test_case],
        [dev_case, dev_case],
        [mutated_dev_case],
    ):
        runner = _CountingV2Runner()
        with pytest.raises(V2ProviderBenchmarkError):
            await evaluate_v2_provider_cases(cases, runner)
        assert runner.calls == 0


@pytest.mark.anyio
async def test_v2_provider_api_accepts_the_complete_frozen_dev_split() -> None:
    dataset = load_chatbi_evaluation_dataset_v2(DEFAULT_DATASET_V2)
    cases = select_v2_provider_cases(dataset, V2ProviderSplit.DEV.value)
    runner = _CountingV2Runner()

    records = await evaluate_v2_provider_cases(cases, runner)

    assert len(records) == 50
    assert runner.calls == 50
    assert {record.split for record in records} == {"dev"}


class _CatalogRow:
    def __init__(self, **values: object) -> None:
        self._mapping = values


def test_v2_schema_payload_is_independent_of_catalog_row_order() -> None:
    relations = [
        _CatalogRow(
            relation_name="customers",
            relation_kind="table",
            column_position=1,
            column_name="customer_id",
            data_type="bigint",
            not_null=True,
        ),
        _CatalogRow(
            relation_name="customers",
            relation_kind="table",
            column_position=2,
            column_name="customer_name",
            data_type="text",
            not_null=True,
        ),
        _CatalogRow(
            relation_name="region_sales",
            relation_kind="view",
            column_position=1,
            column_name="region_code",
            data_type="text",
            not_null=False,
        ),
    ]
    constraints = [
        _CatalogRow(
            relation_name="customers",
            constraint_type="p",
            columns=["customer_id"],
            referenced_schema=None,
            referenced_relation=None,
            referenced_columns=[],
            check_definition=None,
        )
    ]

    assert _fixture_schema_payload(relations, constraints) == _fixture_schema_payload(
        list(reversed(relations)), list(reversed(constraints))
    )


def test_v2_check_definition_normalization_preserves_frozen_contract() -> None:
    assert normalize_check_definition("CHECK (amount >= 0::numeric)") == (
        "CHECK (amount >= 0::numeric)"
    )
    assert normalize_check_definition("check ((amount >= (0)::numeric))") == (
        "CHECK (amount >= 0::numeric)"
    )


def test_v2_authoritative_payload_includes_view_and_check_without_executable_widening() -> None:
    payload = canonical_postgres_schema_payload(
        [
            _CatalogRow(
                relation_name="sales",
                relation_kind="table",
                column_position=1,
                column_name="amount",
                data_type="numeric",
                not_null=True,
            ),
            _CatalogRow(
                relation_name="region_sales",
                relation_kind="view",
                column_position=1,
                column_name="region_code",
                data_type="text",
                not_null=False,
            ),
        ],
        [
            _CatalogRow(
                relation_name="sales",
                constraint_type="c",
                columns=[],
                referenced_schema=None,
                referenced_relation=None,
                referenced_columns=[],
                check_definition="CHECK ((amount >= (0)::numeric))",
            )
        ],
        schema=CHATBI_EVALUATION_V2_SCHEMA,
    )

    relations = payload["relations"]
    assert isinstance(relations, list)
    assert {(item["kind"], item["name"]) for item in relations} == {
        ("table", "sales"),
        ("view", "region_sales"),
    }
    sales = next(item for item in relations if item["name"] == "sales")
    assert sales["constraints"][0]["check_definition"] == "CHECK (amount >= 0::numeric)"
    assert _v2_query_policy(Settings(_env_file=None)).allow_views is False


@pytest.mark.anyio
async def test_v2_schema_provenance_mismatch_stops_before_provider_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = load_chatbi_evaluation_dataset_v2(DEFAULT_DATASET_V2)
    bootstrap = SimpleNamespace(
        data_source=SimpleNamespace(id=CHATBI_EVALUATION_V2_DATASOURCE_ID),
        fixture_schema_fingerprint=EXPECTED_FIXTURE_SCHEMA_FINGERPRINT_V2,
        environment_was_present=True,
        fixture_data_fingerprint=EXPECTED_FIXTURE_DATA_FINGERPRINT_V2,
        database_created=False,
        datasource_created=False,
    )

    class _Engine:
        async def dispose(self) -> None:
            return None

    class _Registry:
        def __init__(self, _session_factory: object) -> None:
            pass

        async def get(self, _datasource_id: object) -> object:
            return SimpleNamespace(id=CHATBI_EVALUATION_V2_DATASOURCE_ID)

    async def fake_ensure(_settings: Settings, *, fixture_path: Any) -> object:
        return bootstrap

    async def fake_authoritative(_data_source: object) -> object:
        return SimpleNamespace(
            fingerprint="0" * 64,
            schema=CHATBI_EVALUATION_V2_SCHEMA,
            relation_labels=(
                "chatbi_demo.customers",
                "chatbi_demo.regions",
                "chatbi_demo.sales",
                "chatbi_demo.region_sales",
            ),
        )

    provider_created = False
    execution_started = False

    def fail_provider(_settings: Settings) -> object:
        nonlocal provider_created
        provider_created = True
        raise AssertionError("provider factory must not run")

    async def fail_execution(*_args: object, **_kwargs: object) -> object:
        nonlocal execution_started
        execution_started = True
        raise AssertionError("provider cases must not run")

    monkeypatch.setattr(provider_module, "current_git_revision", lambda: "a" * 40)
    monkeypatch.setattr(provider_module, "current_git_dirty", lambda: False)
    monkeypatch.setattr(
        provider_module,
        "_verify_v2_dataset_and_fixture",
        lambda _dataset_path, _fixture_path: (dataset, EXPECTED_FIXTURE_FINGERPRINT_V2),
    )
    monkeypatch.setattr(provider_module, "_check_provider_configuration", lambda _settings: None)
    monkeypatch.setattr(provider_module, "ensure_authoritative_v2_datasource", fake_ensure)
    monkeypatch.setattr(provider_module, "create_database_engine", lambda _settings: _Engine())
    monkeypatch.setattr(provider_module, "create_session_factory", lambda _engine: object())
    monkeypatch.setattr(provider_module, "DatabaseDataSourceProvider", _Registry)
    monkeypatch.setattr(
        provider_module,
        "_discover_v2_authoritative_schema_provenance",
        fake_authoritative,
    )
    monkeypatch.setattr(provider_module, "create_llm_provider", fail_provider)
    monkeypatch.setattr(provider_module, "_execute_v2_provider_cases", fail_execution)

    with pytest.raises(V2ProviderBenchmarkError, match="frozen fingerprint"):
        await provider_module.run_v2_provider_benchmark(
            Settings(_env_file=None),
            output_path=None,
        )
    assert provider_created is False
    assert execution_started is False


def test_v2_provider_policy_is_fixed_to_the_isolated_business_schema() -> None:
    settings = Settings(_env_file=None, chatbi_allowed_schemas=["public"])

    policy = _v2_query_policy(settings)

    assert policy.allowed_schemas == (CHATBI_EVALUATION_V2_SCHEMA,)
    assert policy.denied_schemas == ()
    assert policy.allow_views is False
    assert policy.max_rows == settings.chatbi_max_rows
    assert policy.statement_timeout_ms == settings.chatbi_statement_timeout_ms


def test_v2_database_identity_rejects_application_database_collision() -> None:
    settings = Settings(
        _env_file=None,
        database_url=(
            "postgresql+asyncpg://benchmark:super-secret@localhost:5433/"
            "knowledgescope_chatbi_eval_v2"
        ),
    )

    with pytest.raises(V2ProviderBenchmarkError) as error:
        _local_postgres_urls(settings)

    assert "super-secret" not in str(error.value)
    assert "postgresql" not in str(error.value).lower()


def test_v2_database_identity_allows_distinct_local_application_database() -> None:
    settings = Settings(
        _env_file=None,
        database_url="postgresql+asyncpg://benchmark:secret@127.0.0.1:5433/knowledgescope",
    )

    admin_url, evaluation_url = _local_postgres_urls(settings)

    assert admin_url.database == "postgres"
    assert evaluation_url.database == CHATBI_EVALUATION_V2_DATABASE_NAME


@pytest.mark.anyio
async def test_v2_database_collision_fails_before_fixture_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from knowledge_scope.evaluation import chatbi_evaluation_v2_provider as provider_module

    settings = Settings(
        _env_file=None,
        database_url="postgresql+asyncpg://benchmark:secret@127.0.0.1:5433/knowledgescope_chatbi_eval_v2",
    )

    async def unexpected_database_probe(_url: object) -> bool:
        raise AssertionError("database probe must not run on an isolation collision")

    monkeypatch.setattr(provider_module, "_database_exists", unexpected_database_probe)
    with pytest.raises(V2ProviderBenchmarkError):
        await provider_module._ensure_benchmark_database(settings, DEFAULT_FIXTURE_PATH_V2)


@pytest.mark.parametrize("dirty", [False, True])
@pytest.mark.anyio
async def test_v2_start_git_state_is_captured_before_output_directory_creation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    dirty: bool,
) -> None:
    from knowledge_scope.evaluation import chatbi_evaluation_v2_provider as provider_module

    dataset = load_chatbi_evaluation_dataset_v2(DEFAULT_DATASET_V2)
    output_path = tmp_path / "unignored" / "provider.json"
    bootstrap = SimpleNamespace(
        data_source=SimpleNamespace(id=CHATBI_EVALUATION_V2_DATASOURCE_ID),
        fixture_schema_fingerprint=EXPECTED_FIXTURE_SCHEMA_FINGERPRINT_V2,
        environment_was_present=True,
    )

    class _Engine:
        async def dispose(self) -> None:
            return None

    class _Registry:
        def __init__(self, _session_factory: object) -> None:
            pass

        async def get(self, _datasource_id: object) -> object:
            return SimpleNamespace(id=CHATBI_EVALUATION_V2_DATASOURCE_ID)

    def fake_current_git_dirty() -> bool:
        assert not output_path.parent.exists()
        return dirty

    async def fake_ensure(_settings: Settings, *, fixture_path: Any) -> object:
        return bootstrap

    async def fake_discover(
        _settings: Settings,
        _data_source: object,
        *,
        session_factory: object,
        max_chars: int,
    ) -> str:
        return "fixture-data-fingerprint"

    async def fake_authoritative(_data_source: object) -> object:
        return SimpleNamespace(
            fingerprint=EXPECTED_FIXTURE_SCHEMA_FINGERPRINT_V2,
            schema=CHATBI_EVALUATION_V2_SCHEMA,
            relation_labels=(
                "chatbi_demo.customers",
                "chatbi_demo.regions",
                "chatbi_demo.sales",
                "chatbi_demo.region_sales",
            ),
        )

    monkeypatch.setattr(provider_module, "current_git_revision", lambda: "a" * 40)
    monkeypatch.setattr(provider_module, "current_git_dirty", fake_current_git_dirty)
    monkeypatch.setattr(
        provider_module,
        "_verify_v2_dataset_and_fixture",
        lambda _dataset_path, _fixture_path: (dataset, EXPECTED_FIXTURE_FINGERPRINT_V2),
    )
    monkeypatch.setattr(provider_module, "_check_provider_configuration", lambda _settings: None)
    monkeypatch.setattr(provider_module, "ensure_authoritative_v2_datasource", fake_ensure)
    monkeypatch.setattr(provider_module, "create_database_engine", lambda _settings: _Engine())
    monkeypatch.setattr(provider_module, "create_session_factory", lambda _engine: object())
    monkeypatch.setattr(provider_module, "DatabaseDataSourceProvider", _Registry)
    monkeypatch.setattr(
        provider_module, "_discover_v2_authoritative_schema_provenance", fake_authoritative
    )
    monkeypatch.setattr(provider_module, "_discover_and_verify_v2_schema", fake_discover)

    context = await _prepare_v2_preflight(
        Settings(_env_file=None),
        split=V2ProviderSplit.DEV.value,
        dataset_path=DEFAULT_DATASET_V2,
        fixture_path=DEFAULT_FIXTURE_PATH_V2,
        output_path=output_path,
    )

    assert context.git_revision == "a" * 40
    assert context.git_dirty is dirty
    assert context.authoritative_schema_fingerprint == EXPECTED_FIXTURE_SCHEMA_FINGERPRINT_V2
    assert output_path.parent.is_dir()


def test_authoritative_datasource_identity_is_exact_and_opaque() -> None:
    record = ChatBIDataSourceRecord(
        id=CHATBI_EVALUATION_V2_DATASOURCE_ID,
        display_name=CHATBI_EVALUATION_V2_DATASOURCE_DISPLAY_NAME,
        dialect=SQLDialect.POSTGRESQL.value,
        enabled=True,
        connection_ref=CHATBI_EVALUATION_V2_CONNECTION_REF,
        default_database=CHATBI_EVALUATION_V2_DATABASE_NAME,
        default_schema=CHATBI_EVALUATION_V2_SCHEMA,
    )

    assert _authoritative_record_matches(record)
    assert "postgresql" not in record.connection_ref
    record.display_name = "different"  # type: ignore[assignment]
    assert not _authoritative_record_matches(record)


def test_preflight_report_is_provider_free_and_safe() -> None:
    report = V2ProviderPreflightReport(
        dataset_fingerprint=EXPECTED_DATASET_FINGERPRINT_V2,
        fixture_fingerprint=EXPECTED_FIXTURE_FINGERPRINT_V2,
        fixture_schema_fingerprint=EXPECTED_FIXTURE_SCHEMA_FINGERPRINT_V2,
        authoritative_schema_fingerprint=EXPECTED_FIXTURE_SCHEMA_FINGERPRINT_V2,
        fixture_data_fingerprint="0" * 64,
        datasource_id=CHATBI_EVALUATION_V2_DATASOURCE_ID,
        split="dev",
        selected_case_count=50,
        selected_test_case_count=0,
        provider="deepseek",
        model="configured-model",
        provider_configured=True,
        database_created=False,
        datasource_created=False,
        git_revision="a" * 40,
        git_dirty=False,
    )

    output = report.model_dump(mode="json")

    assert output["provider_calls"] == 0
    assert "connection_url" not in output
    assert "api_key" not in json.dumps(output).lower()


class _DeterministicV2Gateway:
    def __init__(self) -> None:
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResult:
        self.requests.append(request)
        if request.task_type == "nl2sql":
            response = (
                '{"sql":"SELECT customer_name FROM chatbi_demo.customers ORDER BY customer_id"}'
            )
            input_tokens, output_tokens = 13, 8
        elif request.task_type == "chatbi_analysis":
            response = '{"answer":"已返回客户名称。","warning":null}'
            input_tokens, output_tokens = 17, 6
        else:
            raise AssertionError(f"unexpected task type: {request.task_type}")
        return LLMResult(
            text=response,
            provider="fake-v2",
            model="fake-v2-model",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=1,
            provider_attempts=1,
        )


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("KNOWLEDGE_SCOPE_RUN_CHATBI_EVAL_V2_PROVIDER_INTEGRATION") != "1",
    reason="set KNOWLEDGE_SCOPE_RUN_CHATBI_EVAL_V2_PROVIDER_INTEGRATION=1 to run",
)
@pytest.mark.anyio
async def test_v2_schema_fingerprint_rejects_structural_mutations(
    postgres_test_engine: Any,
    postgres_test_database: str,
) -> None:
    """The frozen schema contract rejects every relevant structural mutation."""
    fixture_sql = DEFAULT_FIXTURE_PATH_V2.read_text(encoding="utf-8")
    async with postgres_test_engine.begin() as connection:
        for statement in fixture_sql.split(";"):
            if statement.strip():
                await connection.exec_driver_sql(statement)

    assert await _fixture_schema_fingerprint(postgres_test_database) == (
        EXPECTED_FIXTURE_SCHEMA_FINGERPRINT_V2
    )
    provenance = await discover_postgres_schema_provenance(
        postgres_test_database,
        schema=CHATBI_EVALUATION_V2_SCHEMA,
    )
    assert provenance.fingerprint == EXPECTED_FIXTURE_SCHEMA_FINGERPRINT_V2
    assert provenance.relation_labels == (
        "chatbi_demo.customers",
        "chatbi_demo.regions",
        "chatbi_demo.sales",
        "chatbi_demo.region_sales",
    )
    mutations = (
        (
            "added column",
            ("ALTER TABLE chatbi_demo.customers ADD COLUMN temporary_extra TEXT",),
            ("ALTER TABLE chatbi_demo.customers DROP COLUMN temporary_extra",),
        ),
        (
            "changed column type",
            ("ALTER TABLE chatbi_demo.customers ALTER COLUMN customer_name TYPE VARCHAR",),
            ("ALTER TABLE chatbi_demo.customers ALTER COLUMN customer_name TYPE TEXT",),
        ),
        (
            "changed nullability",
            ("ALTER TABLE chatbi_demo.customers ALTER COLUMN customer_name DROP NOT NULL",),
            ("ALTER TABLE chatbi_demo.customers ALTER COLUMN customer_name SET NOT NULL",),
        ),
        (
            "removed foreign key",
            ("ALTER TABLE chatbi_demo.sales DROP CONSTRAINT sales_customer_id_fkey",),
            (
                "ALTER TABLE chatbi_demo.sales ADD CONSTRAINT sales_customer_id_fkey "
                "FOREIGN KEY (customer_id) REFERENCES chatbi_demo.customers(customer_id)",
            ),
        ),
        (
            "changed foreign key target",
            (
                "ALTER TABLE chatbi_demo.sales DROP CONSTRAINT sales_region_code_fkey",
                "ALTER TABLE chatbi_demo.sales ADD CONSTRAINT sales_region_code_fkey "
                "FOREIGN KEY (region_code) REFERENCES chatbi_demo.regions(region_name) NOT VALID",
            ),
            (
                "ALTER TABLE chatbi_demo.sales DROP CONSTRAINT sales_region_code_fkey",
                "ALTER TABLE chatbi_demo.sales ADD CONSTRAINT sales_region_code_fkey "
                "FOREIGN KEY (region_code) REFERENCES chatbi_demo.regions(region_code)",
            ),
        ),
        (
            "extra table",
            ("CREATE TABLE chatbi_demo.temporary_extra (value TEXT)",),
            ("DROP TABLE chatbi_demo.temporary_extra",),
        ),
        (
            "extra view",
            ("CREATE VIEW chatbi_demo.temporary_extra AS SELECT 1 AS value",),
            ("DROP VIEW chatbi_demo.temporary_extra",),
        ),
        (
            "missing table",
            ("ALTER TABLE chatbi_demo.customers RENAME TO customers_missing",),
            ("ALTER TABLE chatbi_demo.customers_missing RENAME TO customers",),
        ),
        (
            "missing view",
            ("ALTER VIEW chatbi_demo.region_sales RENAME TO region_sales_missing",),
            ("ALTER VIEW chatbi_demo.region_sales_missing RENAME TO region_sales",),
        ),
    )
    for label, apply_sql, restore_sql in mutations:
        async with postgres_test_engine.begin() as connection:
            for statement in apply_sql:
                await connection.exec_driver_sql(statement)
        try:
            assert await _fixture_schema_fingerprint(postgres_test_database) != (
                EXPECTED_FIXTURE_SCHEMA_FINGERPRINT_V2
            ), label
        finally:
            async with postgres_test_engine.begin() as connection:
                for statement in restore_sql:
                    await connection.exec_driver_sql(statement)
        assert await _fixture_schema_fingerprint(postgres_test_database) == (
            EXPECTED_FIXTURE_SCHEMA_FINGERPRINT_V2
        ), label


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("KNOWLEDGE_SCOPE_RUN_CHATBI_EVAL_V2_PROVIDER_INTEGRATION") != "1",
    reason="set KNOWLEDGE_SCOPE_RUN_CHATBI_EVAL_V2_PROVIDER_INTEGRATION=1 to run",
)
@pytest.mark.anyio
async def test_v2_authoritative_schema_provenance_is_repeatably_deterministic(
    postgres_test_engine: Any,
    postgres_test_database: str,
) -> None:
    """The same catalog state produces the same complete provenance twice."""
    fixture_sql = DEFAULT_FIXTURE_PATH_V2.read_text(encoding="utf-8")
    async with postgres_test_engine.begin() as connection:
        for statement in fixture_sql.split(";"):
            if statement.strip():
                await connection.exec_driver_sql(statement)

    first = await discover_postgres_schema_provenance(
        postgres_test_database,
        schema=CHATBI_EVALUATION_V2_SCHEMA,
    )
    second = await discover_postgres_schema_provenance(
        postgres_test_database,
        schema=CHATBI_EVALUATION_V2_SCHEMA,
    )

    assert first.payload == second.payload
    assert first.fingerprint == second.fingerprint == EXPECTED_FIXTURE_SCHEMA_FINGERPRINT_V2
    assert "chatbi_demo.region_sales" in first.relation_labels


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("KNOWLEDGE_SCOPE_RUN_CHATBI_EVAL_V2_PROVIDER_INTEGRATION") != "1",
    reason="set KNOWLEDGE_SCOPE_RUN_CHATBI_EVAL_V2_PROVIDER_INTEGRATION=1 to run",
)
@pytest.mark.anyio
async def test_v2_provider_path_uses_real_discovery_validation_and_execution(
    postgres_test_engine: Any,
    postgres_test_database: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the registered-datasource production chain without a provider."""
    fixture_sql = DEFAULT_FIXTURE_PATH_V2.read_text(encoding="utf-8")
    async with postgres_test_engine.begin() as connection:
        for statement in fixture_sql.split(";"):
            if statement.strip():
                await connection.exec_driver_sql(statement)
    assert await _fixture_data_fingerprint(postgres_test_database) == (
        EXPECTED_FIXTURE_DATA_FINGERPRINT_V2
    )
    assert await _fixture_schema_fingerprint(postgres_test_database) == (
        EXPECTED_FIXTURE_SCHEMA_FINGERPRINT_V2
    )

    credential_name = "KNOWLEDGE_SCOPE_CHATBI_EVAL_V2_TEST_DATABASE_URL"
    monkeypatch.setenv(credential_name, postgres_test_database)
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    record = ChatBIDataSourceRecord(
        id=CHATBI_EVALUATION_V2_DATASOURCE_ID,
        display_name=CHATBI_EVALUATION_V2_DATASOURCE_DISPLAY_NAME,
        dialect=SQLDialect.POSTGRESQL.value,
        enabled=True,
        connection_ref=f"env:{credential_name}",
        default_database=None,
        default_schema=CHATBI_EVALUATION_V2_SCHEMA,
    )
    record.created_at = timestamp
    record.updated_at = timestamp

    from sqlalchemy.ext.asyncio import async_sessionmaker

    session_factory = async_sessionmaker(postgres_test_engine, expire_on_commit=False)
    async with session_factory() as session:
        session.add(record)
        await session.commit()

    try:
        registry = DatabaseDataSourceProvider(session_factory)
        discovery = create_postgres_schema_discovery_service(
            connection_timeout_seconds=5,
            statement_timeout_ms=30_000,
        )
        gateway = _DeterministicV2Gateway()
        generation = NL2SQLService(
            gateway,
            schema_discovery=discovery,
            data_source_provider=registry,
        )
        execution = SQLExecutionService(
            generation,
            EnvironmentCredentialResolver(),
            PostgresExecutionAdapter(
                connection_timeout_seconds=5,
                statement_timeout_ms=30_000,
            ),
        )
        result = await ChatBIAgentService(generation, execution, gateway).ask(
            CHATBI_EVALUATION_V2_DATASOURCE_ID,
            "列出客户名称。",
            policy=QueryPolicy(allowed_schemas=(CHATBI_EVALUATION_V2_SCHEMA,), max_rows=10),
            max_chars=20_000,
        )

        assert result.execution_status is QueryLifecycleState.SUCCEEDED
        assert result.datasource_id == CHATBI_EVALUATION_V2_DATASOURCE_ID
        assert result.answer == "已返回客户名称。"
        assert result.row_count == 10
        assert result.truncated is False
        assert result.truncation_reason is None
        assert result.redacted_sql is not None
        assert "chatbi_demo.customers" in result.redacted_sql
        assert result.usage.llm_calls == 2
        assert result.usage.provider_attempts == 2
        assert result.usage.input_tokens == 30
        assert result.usage.output_tokens == 14
        assert [request.task_type for request in gateway.requests] == [
            "nl2sql",
            "chatbi_analysis",
        ]
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
        assert credential_name not in safe_result
        assert postgres_test_database not in safe_result
        assert "password" not in safe_result.lower()
    finally:
        async with session_factory() as session:
            existing = await session.get(ChatBIDataSourceRecord, CHATBI_EVALUATION_V2_DATASOURCE_ID)
            if existing is not None:
                await session.delete(existing)
                await session.commit()
