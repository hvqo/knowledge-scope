from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from knowledge_scope.chatbi import (
    ChatBIResult,
    ChatBIUsageSummary,
    ColumnMetadata,
    QueryExecutionResult,
    QueryLifecycleState,
    QueryTruncationReason,
)
from knowledge_scope.chatbi.eligibility import EligibilityAssessment
from knowledge_scope.chatbi.nl2sql import LLMUsageMetadata
from knowledge_scope.chatbi.schemas import QueryEligibilityReasonCode
from knowledge_scope.evaluation import chatbi_evaluation_v2_gated as gated_module
from knowledge_scope.evaluation.chatbi_evaluation import (
    ChatBIEvaluationObservation,
    ChatBIStageTimings,
    EvaluationStageState,
)
from knowledge_scope.evaluation.chatbi_evaluation_v2 import (
    DEFAULT_DATASET_V2,
    ChatBIEvaluationCaseV2,
    load_chatbi_evaluation_dataset_v2,
)
from knowledge_scope.evaluation.chatbi_evaluation_v2_gated import (
    GatedEvaluationIncompleteError,
    GatedEvaluationMetricError,
    GatedEvaluationPreflightError,
    GatedRunProvenance,
    aggregate_gated_provider_usage,
    aggregate_gated_records,
    build_gated_artifact,
    collect_gated_case_record,
    collect_gated_dev_cases,
    construct_gated_provider,
)
from knowledge_scope.evaluation.chatbi_evaluation_v2_provider import (
    CHATBI_EVALUATION_V2_DATASOURCE_ID,
    V2ProviderCaseRecord,
)
from knowledge_scope.llm import LLMProviderInvocation

DATASET = load_chatbi_evaluation_dataset_v2(DEFAULT_DATASET_V2)
DATASOURCE_ID = CHATBI_EVALUATION_V2_DATASOURCE_ID


def _case(case_id: str) -> ChatBIEvaluationCaseV2:
    return next(case for case in DATASET.cases if case.case_id == case_id)


def _decision(decision: str, reason_code: QueryEligibilityReasonCode):
    from knowledge_scope.chatbi.schemas import QueryEligibilityDecision

    return QueryEligibilityDecision(
        decision=decision,
        reason_code=reason_code,
        user_message="safe application-owned message",
        clarification_question="please clarify the metric" if decision == "clarify" else None,
        method="llm" if decision == "eligible" else "deterministic",
        schema_fingerprint="a" * 64 if decision != "unavailable" else None,
    )


def _assessment(
    decision: str,
    reason_code: QueryEligibilityReasonCode,
    *,
    llm_call_made: bool,
) -> EligibilityAssessment:
    usage = (
        LLMUsageMetadata(
            provider="fake-provider",
            model="fake-model",
            input_tokens=10,
            output_tokens=2,
            provider_attempts=1,
            task_type="chatbi_eligibility",
        )
        if llm_call_made
        else None
    )
    return EligibilityAssessment(
        decision=_decision(decision, reason_code),
        usage=usage,
        llm_call_made=llm_call_made,
    )


def _invocation(
    case_id: str,
    stage: str,
    *,
    attempt_index: int = 1,
    input_tokens: int = 10,
    output_tokens: int = 5,
    outcome: str = "success",
    response_parse_outcome: str | None = "parsed",
    token_limit_status: str = "not_reached",
    finish_reason: str | None = "stop",
) -> LLMProviderInvocation:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return LLMProviderInvocation(
        case_id=case_id,
        logical_stage=stage,
        attempt_index=attempt_index,
        provider="fake-provider",
        model="fake-model",
        started_at=now,
        completed_at=now,
        duration_ms=10,
        outcome=outcome,
        finish_reason=finish_reason,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        llm_result_returned=outcome == "success",
        response_parse_outcome=response_parse_outcome,
        token_limit_status=token_limit_status,
        error_category="structured_output_parse_error" if outcome == "failure" else None,
    )


def _positive_observation(
    case: ChatBIEvaluationCaseV2,
    *,
    sql_attempts: int = 1,
    repair_attempts: int = 0,
    stages: EvaluationStageState | None = None,
    invocations: list[LLMProviderInvocation] | None = None,
) -> ChatBIEvaluationObservation:
    assert case.expected_result is not None
    query_id = uuid4()
    expected = case.expected_result
    columns = [
        ColumnMetadata(name=name, data_type="text", nullable=True, ordinal=index)
        for index, name in enumerate(expected.columns)
    ]
    truncation_reason = (
        QueryTruncationReason(expected.truncation_reason)
        if expected.truncation_reason is not None
        else None
    )
    execution = QueryExecutionResult(
        query_id=query_id,
        datasource_id=DATASOURCE_ID,
        state=QueryLifecycleState.SUCCEEDED,
        columns=columns,
        rows=expected.rows,
        row_count=len(expected.rows),
        truncated=expected.expected_truncated,
        truncation_reason=truncation_reason,
        max_rows=max(1, len(expected.rows)),
        duration_ms=1,
    )
    usage_counts = {"chatbi_eligibility": 1, "nl2sql": sql_attempts, "chatbi_analysis": 1}
    result = ChatBIResult(
        query_id=query_id,
        datasource_id=DATASOURCE_ID,
        answer="deterministic analysis",
        execution_status=QueryLifecycleState.SUCCEEDED,
        redacted_sql="SELECT ...",
        columns=columns,
        row_count=execution.row_count,
        truncated=execution.truncated,
        truncation_reason=truncation_reason,
        sql_attempts=sql_attempts,
        repair_attempts=repair_attempts,
        usage=ChatBIUsageSummary(
            llm_calls=sum(usage_counts.values()),
            provider_attempts=len(invocations or []),
            provider="fake-provider",
            model="fake-model",
            input_tokens=100,
            output_tokens=30,
            task_type_counts=usage_counts,
        ),
        analysis_parse_outcome="parsed",
    )
    return ChatBIEvaluationObservation(
        result=result,
        execution_result=execution,
        timings=ChatBIStageTimings(
            schema_prep_ms=1,
            generation_ms=2,
            validation_ms=3,
            execution_ms=4,
            repair_ms=5 if repair_attempts else None,
            analysis_ms=6,
            total_ms=21,
        ),
        stages=stages
        or EvaluationStageState(
            schema_preparation="passed",
            generation="passed",
            validation="passed",
            execution="passed",
            analysis="passed",
            repair_generation="not_attempted",
        ),
        provider_invocations=invocations or [],
    )


def _negative_observation(
    case: ChatBIEvaluationCaseV2,
    decision: str,
    reason_code: QueryEligibilityReasonCode,
) -> ChatBIEvaluationObservation:
    query_id = uuid4()
    eligibility = _decision(decision, reason_code)
    unavailable = decision == "unavailable"
    result = ChatBIResult(
        query_id=query_id,
        datasource_id=DATASOURCE_ID,
        answer=None if unavailable else eligibility.user_message,
        execution_status=(
            QueryLifecycleState.ELIGIBILITY_UNAVAILABLE
            if unavailable
            else QueryLifecycleState(decision)
        ),
        row_count=0,
        sql_attempts=0,
        repair_attempts=0,
        usage=ChatBIUsageSummary(llm_calls=0, provider_attempts=0),
        eligibility=eligibility,
        error_category=(
            "eligibility_unavailable" if unavailable else None  # type: ignore[arg-type]
        ),
        error_message="safe unavailable message" if unavailable else None,
    )
    return ChatBIEvaluationObservation(
        result=result,
        stages=EvaluationStageState(),
    )


def test_gated_positive_uses_authoritative_result_attempts_without_stale_record_field() -> None:
    case = _case("simple-01")
    invocations = [
        _invocation(case.case_id, "eligibility", input_tokens=10, output_tokens=2),
        _invocation(case.case_id, "generation", input_tokens=20, output_tokens=8),
        _invocation(case.case_id, "analysis", input_tokens=30, output_tokens=12),
    ]
    record = collect_gated_case_record(
        case,
        _positive_observation(case, invocations=invocations),
        _assessment("eligible", QueryEligibilityReasonCode.ELIGIBLE_ANALYTICAL, llm_call_made=True),
    )

    assert "sql_attempts" not in V2ProviderCaseRecord.model_fields
    assert record.sql_pipeline.nl2sql_calls == 1
    assert record.sql_pipeline.initial_generation_attempts == 1
    assert record.sql_pipeline.repair_attempts == 0
    assert record.sql_pipeline.analysis_attempts == 1
    assert record.analysis_usage.logical_calls == 1
    assert record.sql_pipeline.formal_execution_equivalent is True
    assert record.negative_short_circuit_safe is False


@pytest.mark.parametrize(
    ("case_id", "decision", "reason_code"),
    [
        ("ambiguous-01", "clarify", QueryEligibilityReasonCode.AMBIGUOUS_INTENT),
        (
            "unsupported-01",
            "refuse",
            QueryEligibilityReasonCode.UNSUPPORTED_CAPABILITY,
        ),
        (
            "unsupported-02",
            "unavailable",
            QueryEligibilityReasonCode.ELIGIBILITY_CHECK_UNAVAILABLE,
        ),
    ],
)
def test_gated_noneligible_cases_hard_stop_all_downstream_metrics(
    case_id: str,
    decision: str,
    reason_code: QueryEligibilityReasonCode,
) -> None:
    case = _case(case_id)
    assessment = _assessment(decision, reason_code, llm_call_made=False)
    record = collect_gated_case_record(
        case,
        _negative_observation(case, decision, reason_code),
        assessment,
    )

    assert record.eligibility.usage.logical_calls == 0
    assert record.sql_pipeline.nl2sql_calls == 0
    assert record.sql_pipeline.initial_generation_attempts == 0
    assert record.sql_pipeline.repair_attempts == 0
    assert record.sql_pipeline.analysis_attempts == 0
    assert record.sql_pipeline.validation_attempted is False
    assert record.sql_pipeline.execution_attempted is False
    assert record.negative_short_circuit_safe is True
    assert record.negative_gate_violation is False


def test_gated_repair_and_analysis_attempts_are_separate() -> None:
    case = _case("simple-02")
    invocations = [
        _invocation(case.case_id, "eligibility", input_tokens=10, output_tokens=2),
        _invocation(
            case.case_id,
            "generation",
            input_tokens=20,
            output_tokens=8,
            response_parse_outcome="structured_output_parse_error",
        ),
        _invocation(case.case_id, "repair_generation", input_tokens=21, output_tokens=9),
        _invocation(case.case_id, "analysis", input_tokens=30, output_tokens=12),
    ]
    observation = _positive_observation(
        case,
        sql_attempts=2,
        repair_attempts=1,
        stages=EvaluationStageState(
            schema_preparation="passed",
            generation="failed",
            validation="passed",
            execution="passed",
            analysis="passed",
            repair_generation="passed",
        ),
        invocations=invocations,
    )
    record = collect_gated_case_record(
        case,
        observation,
        _assessment("eligible", QueryEligibilityReasonCode.ELIGIBLE_ANALYTICAL, llm_call_made=True),
    )

    assert record.sql_pipeline.nl2sql_calls == 2
    assert record.sql_pipeline.initial_generation_attempts == 1
    assert record.sql_pipeline.repair_attempts == 1
    assert record.sql_pipeline.repair_generation_parseable is True
    assert record.sql_pipeline.analysis_attempts == 1
    assert record.provider_record.repair_recovery_success is True

    usage = aggregate_gated_provider_usage([record])
    assert usage.eligibility.logical_calls == 1
    assert usage.nl2sql.logical_calls == 2
    assert usage.nl2sql.provider_attempts == 2
    assert usage.analysis.logical_calls == 1
    assert usage.analysis.provider_attempts == 1
    assert usage.nl2sql.output_tokens == 17
    assert usage.analysis.output_tokens == 12


def test_provider_retries_do_not_become_extra_logical_nl2sql_attempts() -> None:
    case = _case("simple-03")
    invocations = [
        _invocation(case.case_id, "eligibility"),
        _invocation(
            case.case_id,
            "generation",
            attempt_index=1,
            outcome="failure",
            response_parse_outcome=None,
        ),
        _invocation(case.case_id, "generation", attempt_index=2),
        _invocation(case.case_id, "analysis"),
    ]
    observation = _positive_observation(case, invocations=invocations)
    record = collect_gated_case_record(
        case,
        observation,
        _assessment("eligible", QueryEligibilityReasonCode.ELIGIBLE_ANALYTICAL, llm_call_made=True),
    )

    assert record.sql_pipeline.nl2sql_calls == 1
    assert record.sql_pipeline.initial_generation_attempts == 1
    assert record.sql_pipeline.usage.provider_attempts == 2


def test_gated_usage_does_not_silently_invent_missing_invocations() -> None:
    case = _case("simple-01")
    observation = _positive_observation(case, invocations=[])

    with pytest.raises(GatedEvaluationMetricError, match="no provider invocation"):
        collect_gated_case_record(
            case,
            observation,
            _assessment(
                "eligible", QueryEligibilityReasonCode.ELIGIBLE_ANALYTICAL, llm_call_made=True
            ),
        )


def test_dirty_worktree_fails_before_provider_factory_and_artifact_creation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(gated_module, "current_git_dirty", lambda _repo_root: True)
    factory_calls = 0

    def provider_factory():
        nonlocal factory_calls
        factory_calls += 1
        raise AssertionError("dirty-worktree preflight must block provider construction")

    artifact_path = tmp_path / "chatbi-gated-dev-v1.json"
    with pytest.raises(GatedEvaluationPreflightError, match="clean Git worktree"):
        construct_gated_provider(provider_factory, repo_root=tmp_path)

    assert factory_calls == 0
    assert not artifact_path.exists()


def test_clean_worktree_preflight_reaches_provider_construction_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(gated_module, "current_git_dirty", lambda _repo_root: False)
    monkeypatch.setattr(gated_module, "current_git_revision", lambda _repo_root: "a" * 40)
    factory_calls = 0

    class _StubRunner:
        async def run(self, _case: ChatBIEvaluationCaseV2):
            raise AssertionError("the provider runner must not execute in this preflight test")

    def provider_factory() -> _StubRunner:
        nonlocal factory_calls
        factory_calls += 1
        return _StubRunner()

    preflight, runner = construct_gated_provider(provider_factory, repo_root=tmp_path)

    assert factory_calls == 1
    assert preflight.git_revision == "a" * 40
    assert preflight.git_dirty is False
    assert isinstance(runner, _StubRunner)


def test_gated_aggregate_keeps_formal_accuracy_denominator_on_positive_cases() -> None:
    case = _case("simple-01")
    record = collect_gated_case_record(
        case,
        _positive_observation(
            case,
            invocations=[
                _invocation(case.case_id, "eligibility"),
                _invocation(case.case_id, "generation"),
                _invocation(case.case_id, "analysis"),
            ],
        ),
        _assessment("eligible", QueryEligibilityReasonCode.ELIGIBLE_ANALYTICAL, llm_call_made=True),
    )
    aggregate = aggregate_gated_records([record])

    assert aggregate.formal_execution_accuracy_denominator == 1
    assert aggregate.formal_execution_equivalent_count == 1
    assert aggregate.negative_safe_success_count == 0


def test_gated_artifact_rejects_partial_run_before_serialization() -> None:
    case = _case("simple-01")
    record = collect_gated_case_record(
        case,
        _positive_observation(
            case,
            invocations=[
                _invocation(case.case_id, "eligibility"),
                _invocation(case.case_id, "generation"),
                _invocation(case.case_id, "analysis"),
            ],
        ),
        _assessment("eligible", QueryEligibilityReasonCode.ELIGIBLE_ANALYTICAL, llm_call_made=True),
    )

    with pytest.raises(GatedEvaluationIncompleteError, match="all 50 DEV cases"):
        build_gated_artifact(
            GatedRunProvenance(
                dataset_fingerprint="a" * 64,
                fixture_fingerprint="b" * 64,
                fixture_schema_fingerprint="c" * 64,
                fixture_data_fingerprint="d" * 64,
                datasource_id=DATASOURCE_ID,
                datasource_identity="chatbi-demo-v2-authoritative",
                case_ids=tuple(f"case-{index:02d}" for index in range(50)),
                provider="fake-provider",
                model="fake-model",
                git_revision="e" * 40,
                git_dirty=False,
            ),
            [record],
            expected_case_ids=[case.case_id],
        )


@pytest.mark.anyio
async def test_gated_collection_rejects_partial_selection_before_runner_call() -> None:
    class _NeverRunner:
        async def run(self, _case: ChatBIEvaluationCaseV2):
            raise AssertionError("partial selection must not call the provider runner")

    with pytest.raises(GatedEvaluationIncompleteError, match="all 50 cases"):
        await collect_gated_dev_cases([_case("simple-01")], _NeverRunner())


@pytest.mark.anyio
async def test_fake_gated_runner_covers_positive_and_negative_paths() -> None:
    class _FakeGatedRunner:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def run(
            self, case: ChatBIEvaluationCaseV2
        ) -> tuple[ChatBIEvaluationObservation, EligibilityAssessment]:
            self.calls.append(case.case_id)
            if case.positive:
                invocations = [
                    _invocation(case.case_id, "eligibility"),
                    _invocation(case.case_id, "generation"),
                    _invocation(case.case_id, "analysis"),
                ]
                return (
                    _positive_observation(case, invocations=invocations),
                    _assessment(
                        "eligible",
                        QueryEligibilityReasonCode.ELIGIBLE_ANALYTICAL,
                        llm_call_made=True,
                    ),
                )
            if case.case_id == "ambiguous-01":
                decision, reason = "clarify", QueryEligibilityReasonCode.AMBIGUOUS_INTENT
            elif case.case_id == "unsupported-01":
                decision, reason = "refuse", QueryEligibilityReasonCode.UNSUPPORTED_CAPABILITY
            else:
                decision, reason = (
                    "unavailable",
                    QueryEligibilityReasonCode.ELIGIBILITY_CHECK_UNAVAILABLE,
                )
            return (
                _negative_observation(case, decision, reason),
                _assessment(decision, reason, llm_call_made=False),
            )

    runner = _FakeGatedRunner()
    selected = [
        _case(case_id)
        for case_id in ("simple-01", "ambiguous-01", "unsupported-01", "unsupported-02")
    ]
    records = [collect_gated_case_record(case, *(await runner.run(case))) for case in selected]

    assert runner.calls == [case.case_id for case in selected]
    assert len(records) == 4
    assert records[0].sql_pipeline.nl2sql_calls == 1
    assert all(record.negative_short_circuit_safe for record in records[1:])
    assert all(record.sql_pipeline.nl2sql_calls == 0 for record in records[1:])
    assert aggregate_gated_records(records).negative_safe_success_count == 3
