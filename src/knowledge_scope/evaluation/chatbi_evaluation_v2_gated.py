"""Eligibility-gated provider evaluation contracts for ChatBI v2.

This module is evaluation infrastructure only.  It adapts the production
``ChatBIResult`` and the existing v2 provider observation into a separate,
eligibility-aware artifact contract.  It deliberately does not change the
production Agent, eligibility gate, SQL validator, executor, or analysis
service.

The adapter is intentionally strict: missing stage or invocation metadata is
an evaluation error, not an implicit zero.  A future provider runner can use
``collect_gated_case_record`` after one production Agent request and then
publish an artifact only through ``build_gated_artifact`` once all 50 DEV
cases are present.
"""

from __future__ import annotations

import os
import tempfile
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)

from knowledge_scope.chatbi.agent import ChatBIResult
from knowledge_scope.chatbi.eligibility import (
    ELIGIBILITY_GATE_VERSION,
    EligibilityAssessment,
    EligibilityStructuredErrorCategory,
)
from knowledge_scope.chatbi.nl2sql_models import NL2SQL_PROMPT_VERSION
from knowledge_scope.chatbi.schemas import QueryEligibilityReasonCode, QueryLifecycleState
from knowledge_scope.evaluation.chatbi_evaluation import (
    ChatBIEvaluationError,
    ChatBIEvaluationObservation,
    EvaluationStageState,
    current_git_dirty,
    current_git_revision,
)
from knowledge_scope.evaluation.chatbi_evaluation_v2 import (
    CHATBI_DEMO_FIXTURE_VERSION_V2,
    CHATBI_EVALUATION_V2_DATASET_VERSION,
    CHATBI_EVALUATION_V2_STATUS,
    ChatBIEvaluationCaseV2,
    ChatBIEvaluationV2Category,
    ChatBIEvaluationV2Difficulty,
)
from knowledge_scope.evaluation.chatbi_evaluation_v2_provider import (
    EXPECTED_SEMANTIC_POLICY_G3_FINGERPRINT,
    EXPECTED_SEMANTIC_POLICY_G3_VERSION,
    V2ProviderCaseRecord,
    V2ProviderSplit,
    _v2_record,
)
from knowledge_scope.evaluation.chatbi_semantic_evaluation import SemanticComparison
from knowledge_scope.evaluation.semantic_evidence import SEMANTIC_EVIDENCE_VERSION
from knowledge_scope.llm.schemas import LLMProviderInvocation

GATED_PROVIDER_ARTIFACT_VERSION = "a5.7i6-gated-chatbi-dev-v1"
GATED_PROVIDER_BENCHMARK_VERSION = "a5.7i6a"
GATED_DEV_CASE_COUNT = 50
GATED_DEV_POSITIVE_COUNT = 47
GATED_DEV_NEGATIVE_COUNT = 3
GATED_TASK_TYPES = ("chatbi_eligibility", "nl2sql", "chatbi_analysis")
GATED_STAGE_TO_TASK = {
    "eligibility": "chatbi_eligibility",
    "generation": "nl2sql",
    "repair_generation": "nl2sql",
    "analysis": "chatbi_analysis",
}


class GatedEvaluationMetricError(ValueError):
    """Raised when a gated artifact cannot be measured without guessing."""


class GatedEvaluationIncompleteError(ValueError):
    """Raised when a partial run is presented as a final artifact."""


class GatedEvaluationPreflightError(ValueError):
    """Raised before a gated provider can be constructed."""


class GatedCaseRunner(Protocol):
    """Provider-runner boundary for one production Agent request."""

    async def run(
        self, case: ChatBIEvaluationCaseV2
    ) -> tuple[ChatBIEvaluationObservation, EligibilityAssessment]:
        """Return the observation and the assessment captured for one case."""


class _GatedModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        hide_input_in_errors=True,
    )


class GatedPreflight(_GatedModel):
    """Provider-free provenance captured only after a clean-worktree check."""

    git_revision: StrictStr = Field(pattern=r"^[0-9a-f]{40}$")
    git_dirty: Literal[False] = False


class GatedSemanticDiagnostic(_GatedModel):
    """Bounded semantic-proof status; no SQL, prompts, or provider text."""

    status: Literal["PROVEN_EQUIVALENT", "PROVEN_DIFFERENT", "UNASSESSABLE"]
    assessable: StrictBool
    proof_reasons: tuple[StrictStr, ...] = Field(default=(), max_length=32)
    failure_reason: StrictStr | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def validate_status(self) -> GatedSemanticDiagnostic:
        if self.status == "UNASSESSABLE" and self.assessable:
            raise ValueError("unassessable semantic diagnostics cannot be assessable")
        if self.status != "UNASSESSABLE" and not self.assessable:
            raise ValueError("proven semantic diagnostics must be assessable")
        return self

    @classmethod
    def from_comparison(cls, comparison: SemanticComparison) -> GatedSemanticDiagnostic:
        """Adapt the existing proof result without reimplementing proof rules."""
        if not comparison.assessable:
            return cls(
                status="UNASSESSABLE",
                assessable=False,
                proof_reasons=comparison.proof_reasons,
                failure_reason=comparison.failure_reason or "semantic_result_unassessable",
            )
        return cls(
            status="PROVEN_EQUIVALENT" if comparison.equivalent else "PROVEN_DIFFERENT",
            assessable=True,
            proof_reasons=comparison.proof_reasons,
            failure_reason=comparison.failure_reason,
        )

    @classmethod
    def unavailable(cls) -> GatedSemanticDiagnostic:
        return cls(
            status="UNASSESSABLE",
            assessable=False,
            failure_reason="semantic_diagnostic_not_provided",
        )


class GatedTaskUsage(_GatedModel):
    """Usage for one logical task, separated from other ChatBI tasks."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        hide_input_in_errors=True,
    )

    task_type: Literal["chatbi_eligibility", "nl2sql", "chatbi_analysis"]
    logical_calls: StrictInt = Field(ge=0)
    provider_attempts: StrictInt = Field(ge=0)
    input_tokens: StrictInt | None = Field(default=None, ge=0)
    output_tokens: StrictInt | None = Field(default=None, ge=0)
    latency_ms: StrictFloat = Field(ge=0)
    provider: StrictStr | None = Field(default=None, min_length=1, max_length=64)
    model: StrictStr | None = Field(default=None, min_length=1, max_length=255)


class GatedEligibilityMetrics(_GatedModel):
    """Eligibility decision and its isolated provider usage."""

    decision: Literal["eligible", "clarify", "refuse", "unavailable"]
    reason_code: QueryEligibilityReasonCode
    method: Literal["deterministic", "llm"]
    structured_error_category: EligibilityStructuredErrorCategory | None = None
    usage: GatedTaskUsage


class GatedSQLPipelineMetrics(_GatedModel):
    """SQL-side metrics, valid only after an eligibility gate decision."""

    nl2sql_calls: StrictInt = Field(ge=0)
    initial_generation_attempts: StrictInt = Field(ge=0)
    initial_generation_parseable: StrictBool
    initial_truncation_attempts: StrictInt = Field(ge=0)
    validation_attempted: StrictBool
    validation_accepted: StrictBool
    execution_attempted: StrictBool
    execution_succeeded: StrictBool
    repair_attempts: StrictInt = Field(ge=0)
    repair_generation_parseable: StrictBool | None = None
    analysis_attempts: StrictInt = Field(ge=0)
    analysis_succeeded: StrictBool
    formal_execution_equivalent: StrictBool | None = None
    result_status: QueryLifecycleState
    failure_category: StrictStr | None = Field(default=None, max_length=64)
    row_count: StrictInt = Field(ge=0)
    truncated: StrictBool
    truncation_reason: StrictStr | None = Field(default=None, max_length=32)
    usage: GatedTaskUsage


class GatedCaseRecord(_GatedModel):
    """Safe gated observation for one DEV case."""

    case_id: StrictStr
    split: Literal["dev"] = "dev"
    positive: StrictBool
    category: ChatBIEvaluationV2Category
    difficulty: ChatBIEvaluationV2Difficulty
    eligibility: GatedEligibilityMetrics
    sql_pipeline: GatedSQLPipelineMetrics
    analysis_usage: GatedTaskUsage
    negative_short_circuit_safe: StrictBool
    negative_gate_violation: StrictBool
    provider_record: V2ProviderCaseRecord
    semantic_diagnostic: GatedSemanticDiagnostic

    @model_validator(mode="after")
    def validate_identity(self) -> GatedCaseRecord:
        if self.provider_record.case_id != self.case_id:
            raise ValueError("gated record and provider record case IDs must match")
        if self.provider_record.positive != self.positive:
            raise ValueError("gated record and provider record polarity must match")
        if self.provider_record.split != self.split:
            raise ValueError("gated record and provider record split must match")
        return self


class GatedUsageAggregate(_GatedModel):
    """Run-level task usage with no cross-task token merging."""

    eligibility: GatedTaskUsage
    nl2sql: GatedTaskUsage
    analysis: GatedTaskUsage


class GatedAggregate(_GatedModel):
    """Bounded aggregate metrics for a complete or diagnostic record set."""

    case_count: StrictInt = Field(ge=0)
    positive_case_count: StrictInt = Field(ge=0)
    negative_case_count: StrictInt = Field(ge=0)
    decision_counts: dict[str, StrictInt] = Field(default_factory=dict)
    eligible_positive_count: StrictInt = Field(ge=0)
    initial_generation_attempted_count: StrictInt = Field(ge=0)
    initial_generation_parseable_count: StrictInt = Field(ge=0)
    initial_truncation_attempt_count: StrictInt = Field(ge=0)
    validation_attempted_count: StrictInt = Field(ge=0)
    validation_accepted_count: StrictInt = Field(ge=0)
    execution_attempted_count: StrictInt = Field(ge=0)
    execution_success_count: StrictInt = Field(ge=0)
    repair_attempt_count: StrictInt = Field(ge=0)
    repair_recovery_success_count: StrictInt = Field(ge=0)
    analysis_attempt_count: StrictInt = Field(ge=0)
    analysis_failure_count: StrictInt = Field(ge=0)
    formal_execution_equivalent_count: StrictInt = Field(ge=0)
    formal_execution_accuracy_denominator: StrictInt = Field(ge=0)
    negative_safe_success_count: StrictInt = Field(ge=0)
    negative_gate_violation_count: StrictInt = Field(ge=0)
    semantic_status_counts: dict[str, StrictInt] = Field(default_factory=dict)
    usage: GatedUsageAggregate


class GatedRunProvenance(_GatedModel):
    """Frozen provenance/configuration for the future gated DEV artifact."""

    benchmark_version: Literal["a5.7i6a"] = GATED_PROVIDER_BENCHMARK_VERSION
    artifact_schema_version: Literal["a5.7i6-gated-chatbi-dev-v1"] = GATED_PROVIDER_ARTIFACT_VERSION
    dataset_version: Literal["a5.7-v2"] = CHATBI_EVALUATION_V2_DATASET_VERSION
    dataset_status: Literal["human_reviewed_frozen"] = CHATBI_EVALUATION_V2_STATUS
    dataset_fingerprint: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    fixture_version: Literal["chatbi-demo-v2"] = CHATBI_DEMO_FIXTURE_VERSION_V2
    fixture_fingerprint: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    fixture_schema_fingerprint: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    fixture_data_fingerprint: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    datasource_id: UUID
    datasource_identity: StrictStr = Field(min_length=1, max_length=128)
    split: Literal["dev"] = "dev"
    selected_case_count: Literal[50] = GATED_DEV_CASE_COUNT
    positive_case_count: Literal[47] = GATED_DEV_POSITIVE_COUNT
    negative_case_count: Literal[3] = GATED_DEV_NEGATIVE_COUNT
    test_selected_count: Literal[0] = 0
    case_ids: tuple[StrictStr, ...] = Field(
        min_length=GATED_DEV_CASE_COUNT,
        max_length=GATED_DEV_CASE_COUNT,
    )
    provider: StrictStr = Field(min_length=1, max_length=64)
    model: StrictStr = Field(min_length=1, max_length=255)
    eligibility_gate_version: Literal["a5.7i2-v1"] = ELIGIBILITY_GATE_VERSION
    eligibility_task_type: Literal["chatbi_eligibility"] = "chatbi_eligibility"
    # Keep historical a5.3-v3 gated artifacts readable while recording the
    # current prompt experiment for newly built artifacts.
    nl2sql_prompt_version: Literal["a5.3-v3", "a5.7j2-v1"] = NL2SQL_PROMPT_VERSION
    nl2sql_reasoning: Literal["disabled"] = "disabled"
    nl2sql_max_tokens: Literal[1024] = 1024
    repair_reasoning: Literal["disabled"] = "disabled"
    repair_max_tokens: Literal[1024] = 1024
    analysis_reasoning: Literal["disabled"] = "disabled"
    analysis_max_tokens: Literal[1024] = 1024
    result_contract_enabled: Literal[False] = False
    semantic_evidence_version: Literal["a5.7g2-semantic-evidence-v1"] = SEMANTIC_EVIDENCE_VERSION
    semantic_policy_version: Literal["a5.7g3-semantic-policy-v1"] = (
        EXPECTED_SEMANTIC_POLICY_G3_VERSION
    )
    semantic_policy_fingerprint: StrictStr = Field(
        default=EXPECTED_SEMANTIC_POLICY_G3_FINGERPRINT,
        pattern=r"^[0-9a-f]{64}$",
    )
    git_revision: StrictStr = Field(pattern=r"^[0-9a-f]{40}$")
    git_dirty: Literal[False] = False


class GatedProviderRun(_GatedModel):
    """Final-only artifact; no model exists for a partial run."""

    artifact_schema_version: Literal["a5.7i6-gated-chatbi-dev-v1"] = GATED_PROVIDER_ARTIFACT_VERSION
    run_id: UUID = Field(default_factory=uuid4)
    started_at: datetime
    completed_at: datetime
    provenance: GatedRunProvenance
    records: tuple[GatedCaseRecord, ...] = Field(
        min_length=GATED_DEV_CASE_COUNT,
        max_length=GATED_DEV_CASE_COUNT,
    )
    aggregate: GatedAggregate

    @model_validator(mode="after")
    def validate_complete_run(self) -> GatedProviderRun:
        if self.completed_at < self.started_at:
            raise ValueError("gated run completed_at cannot precede started_at")
        if self.provenance.artifact_schema_version != self.artifact_schema_version:
            raise ValueError("gated artifact and provenance versions must match")
        record_ids = tuple(record.case_id for record in self.records)
        if len(set(record_ids)) != GATED_DEV_CASE_COUNT:
            raise ValueError("gated artifact case IDs must be unique")
        if record_ids != self.provenance.case_ids:
            raise ValueError("gated record order must match provenance case_ids")
        if self.aggregate.case_count != GATED_DEV_CASE_COUNT:
            raise ValueError("gated aggregate must contain exactly 50 DEV cases")
        if self.aggregate.positive_case_count != GATED_DEV_POSITIVE_COUNT:
            raise ValueError("gated aggregate must contain exactly 47 positive cases")
        if self.aggregate.negative_case_count != GATED_DEV_NEGATIVE_COUNT:
            raise ValueError("gated aggregate must contain exactly 3 negative cases")
        return self


def _sum_optional(values: Sequence[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def preflight_gated_benchmark(repo_root: Path | None = None) -> GatedPreflight:
    """Reject dirty repositories before any gated provider is constructed.

    ``current_git_dirty`` delegates to Git's porcelain status output, so
    ignored runtime/evaluation artifacts do not cause a false rejection.
    Only bounded status errors are exposed to callers; Git output is never
    included in the exception.
    """
    try:
        dirty = current_git_dirty(repo_root)
    except ChatBIEvaluationError as error:
        raise GatedEvaluationPreflightError(
            "could not verify the gated benchmark Git worktree"
        ) from error
    if dirty:
        raise GatedEvaluationPreflightError(
            "gated benchmark requires a clean Git worktree before provider construction"
        )
    revision = current_git_revision(repo_root)
    if revision is None:
        raise GatedEvaluationPreflightError("could not determine the gated benchmark Git revision")
    return GatedPreflight(git_revision=revision)


def construct_gated_provider(
    provider_factory: Callable[[], GatedCaseRunner],
    *,
    repo_root: Path | None = None,
) -> tuple[GatedPreflight, GatedCaseRunner]:
    """Run gated preflight before invoking the provider factory.

    This is the construction boundary for a future provider-backed gated run.
    Keeping factory invocation here prevents a dirty-worktree preflight error
    from constructing a gateway, starting cases, or publishing an artifact.
    """
    preflight = preflight_gated_benchmark(repo_root)
    return preflight, provider_factory()


def _single_value(values: Sequence[str]) -> str | None:
    unique = set(values)
    if len(unique) > 1:
        raise GatedEvaluationMetricError("one task used multiple provider/model identities")
    return next(iter(unique)) if unique else None


def _validate_task_counts(result: ChatBIResult) -> dict[str, int]:
    counts = dict(result.usage.task_type_counts)
    unknown = set(counts).difference(GATED_TASK_TYPES)
    if unknown:
        raise GatedEvaluationMetricError("unsupported task type in gated usage")
    if any(count < 0 for count in counts.values()):
        raise GatedEvaluationMetricError("negative logical task count")
    if sum(counts.values()) != result.usage.llm_calls:
        raise GatedEvaluationMetricError("logical task counts do not match llm_calls")
    return counts


def _task_usage(
    task_type: Literal["chatbi_eligibility", "nl2sql", "chatbi_analysis"],
    logical_calls: int,
    invocations: Sequence[LLMProviderInvocation],
    *,
    case_id: str,
) -> GatedTaskUsage:
    if logical_calls == 0 and invocations:
        raise GatedEvaluationMetricError(
            f"{case_id}: provider invocation exists without a {task_type} logical call"
        )
    if logical_calls > 0 and not invocations:
        raise GatedEvaluationMetricError(
            f"{case_id}: {task_type} logical call has no provider invocation metadata"
        )
    for invocation in invocations:
        if invocation.case_id not in {None, case_id}:
            raise GatedEvaluationMetricError(f"{case_id}: invocation belongs to another case")
    ids = [invocation.id for invocation in invocations]
    if len(ids) != len(set(ids)):
        raise GatedEvaluationMetricError(f"{case_id}: duplicate provider invocation metadata")
    return GatedTaskUsage(
        task_type=task_type,
        logical_calls=logical_calls,
        provider_attempts=len(invocations),
        input_tokens=_sum_optional([invocation.input_tokens for invocation in invocations]),
        output_tokens=_sum_optional([invocation.output_tokens for invocation in invocations]),
        latency_ms=sum(invocation.duration_ms for invocation in invocations),
        provider=_single_value([invocation.provider for invocation in invocations]),
        model=_single_value([invocation.model for invocation in invocations]),
    )


def _invocation_map(
    provider_invocations: Sequence[LLMProviderInvocation],
) -> dict[str, list[LLMProviderInvocation]]:
    grouped: dict[str, list[LLMProviderInvocation]] = {task: [] for task in GATED_TASK_TYPES}
    for invocation in provider_invocations:
        task_type = GATED_STAGE_TO_TASK.get(invocation.logical_stage)
        if task_type is None:
            raise GatedEvaluationMetricError(
                f"unsupported provider logical stage: {invocation.logical_stage}"
            )
        grouped[task_type].append(invocation)
    return grouped


def _truncation_attempt(invocation: LLMProviderInvocation) -> bool:
    if invocation.token_limit_status == "confirmed":
        return True
    return (
        invocation.finish_reason == "length"
        and invocation.output_tokens is not None
        and invocation.output_token_budget is not None
        and invocation.output_tokens >= invocation.output_token_budget
    )


def _validate_eligibility_assessment(assessment: EligibilityAssessment) -> None:
    decision = assessment.decision
    expected_reasons = {
        "eligible": {QueryEligibilityReasonCode.ELIGIBLE_ANALYTICAL},
        "clarify": {QueryEligibilityReasonCode.AMBIGUOUS_INTENT},
        "refuse": {
            QueryEligibilityReasonCode.UNSUPPORTED_WRITE_OPERATION,
            QueryEligibilityReasonCode.UNSUPPORTED_CAPABILITY,
            QueryEligibilityReasonCode.NOT_GROUNDED_IN_SCHEMA,
        },
        "unavailable": {QueryEligibilityReasonCode.ELIGIBILITY_CHECK_UNAVAILABLE},
    }
    if decision.reason_code not in expected_reasons[decision.decision]:
        raise GatedEvaluationMetricError("eligibility decision/reason mismatch")
    if assessment.structured_error_category is not None and decision.decision != "unavailable":
        raise GatedEvaluationMetricError(
            "structured eligibility errors must produce unavailable decisions"
        )
    if assessment.llm_call_made and assessment.usage is None:
        raise GatedEvaluationMetricError("eligibility call has no completed usage metadata")
    if not assessment.llm_call_made and assessment.usage is not None:
        raise GatedEvaluationMetricError("deterministic eligibility must not carry LLM usage")
    expected_method = "llm" if assessment.llm_call_made else "deterministic"
    if decision.method != expected_method:
        raise GatedEvaluationMetricError("eligibility method does not match call provenance")


def _stage_is_attempted(stages: EvaluationStageState, name: str) -> bool:
    status = {
        "generation": stages.generation,
        "validation": stages.validation,
        "execution": stages.execution,
        "analysis": stages.analysis,
        "repair_generation": stages.repair_generation,
    }[name]
    return status != "not_attempted"


def collect_gated_case_record(
    case: ChatBIEvaluationCaseV2,
    observation: ChatBIEvaluationObservation,
    eligibility: EligibilityAssessment,
    *,
    semantic_diagnostic: GatedSemanticDiagnostic | None = None,
) -> GatedCaseRecord:
    """Collect one gated observation using only authoritative current models."""
    if case.split != V2ProviderSplit.DEV.value:
        raise GatedEvaluationMetricError("gated provider records accept DEV cases only")
    _validate_eligibility_assessment(eligibility)
    result = observation.result
    counts = _validate_task_counts(result)
    grouped = _invocation_map(observation.provider_invocations)
    task_usage = {
        task: _task_usage(task, counts.get(task, 0), grouped[task], case_id=case.case_id)
        for task in GATED_TASK_TYPES
    }
    expected_eligibility_calls = 1 if eligibility.llm_call_made else 0
    if counts.get("chatbi_eligibility", 0) != expected_eligibility_calls:
        raise GatedEvaluationMetricError(
            f"{case.case_id}: eligibility logical call count does not match assessment"
        )
    if counts.get("nl2sql", 0) != result.sql_attempts:
        raise GatedEvaluationMetricError(
            f"{case.case_id}: nl2sql logical calls do not match ChatBIResult.sql_attempts"
        )
    if result.usage.provider_attempts != len(observation.provider_invocations):
        raise GatedEvaluationMetricError(
            f"{case.case_id}: provider attempt count does not match invocation records"
        )

    provider_record = _v2_record(case, observation)
    noneligible = eligibility.decision.decision != "eligible"
    downstream_stages = (
        "generation",
        "validation",
        "execution",
        "analysis",
        "repair_generation",
    )
    downstream_attempted = any(
        _stage_is_attempted(observation.stages, stage) for stage in downstream_stages
    )
    downstream_counts = counts.get("nl2sql", 0) or counts.get("chatbi_analysis", 0)
    if noneligible and (
        downstream_attempted or downstream_counts or result.sql_attempts or result.repair_attempts
    ):
        raise GatedEvaluationMetricError(
            f"{case.case_id}: noneligible result crossed the downstream gate"
        )
    if noneligible and result.eligibility is None:
        raise GatedEvaluationMetricError(
            f"{case.case_id}: noneligible result does not expose its eligibility decision"
        )
    if not noneligible and result.eligibility is not None:
        raise GatedEvaluationMetricError(
            f"{case.case_id}: eligible execution unexpectedly exposes "
            "a terminal eligibility decision"
        )

    initial_attempts = result.sql_attempts - result.repair_attempts
    if initial_attempts < 0:
        raise GatedEvaluationMetricError(f"{case.case_id}: repair attempts exceed SQL attempts")
    generation_invocations = grouped["nl2sql"]
    initial_generation_invocations = [
        invocation
        for invocation in generation_invocations
        if invocation.logical_stage == "generation"
    ]
    repair_generation_invocations = [
        invocation
        for invocation in generation_invocations
        if invocation.logical_stage == "repair_generation"
    ]
    if initial_attempts == 0 and initial_generation_invocations:
        raise GatedEvaluationMetricError(
            f"{case.case_id}: generation invocation exists without an initial attempt"
        )
    if initial_attempts > 0 and not initial_generation_invocations:
        raise GatedEvaluationMetricError(
            f"{case.case_id}: initial attempt has no generation invocation metadata"
        )
    if result.repair_attempts == 0 and repair_generation_invocations:
        raise GatedEvaluationMetricError(
            f"{case.case_id}: repair invocation exists without a repair attempt"
        )
    if result.repair_attempts > 0 and not repair_generation_invocations:
        raise GatedEvaluationMetricError(
            f"{case.case_id}: repair attempt has no repair invocation metadata"
        )
    sql_pipeline = GatedSQLPipelineMetrics(
        nl2sql_calls=counts.get("nl2sql", 0),
        initial_generation_attempts=initial_attempts,
        initial_generation_parseable=provider_record.initial_generation_parseable is True,
        initial_truncation_attempts=sum(
            _truncation_attempt(invocation) for invocation in initial_generation_invocations
        ),
        validation_attempted=_stage_is_attempted(observation.stages, "validation"),
        validation_accepted=provider_record.validation_accepted,
        execution_attempted=_stage_is_attempted(observation.stages, "execution"),
        execution_succeeded=(
            observation.stages.execution == "passed"
            and result.execution_status is QueryLifecycleState.SUCCEEDED
        ),
        repair_attempts=result.repair_attempts,
        repair_generation_parseable=provider_record.repair_generation_parseable,
        analysis_attempts=counts.get("chatbi_analysis", 0),
        analysis_succeeded=(
            observation.stages.analysis == "passed"
            and result.error_category is None
            and result.analysis_parse_outcome in {None, "parsed"}
        ),
        formal_execution_equivalent=provider_record.execution_equivalent,
        result_status=result.execution_status,
        failure_category=(
            provider_record.failure_category.value
            if provider_record.failure_category is not None
            else None
        ),
        row_count=result.row_count,
        truncated=result.truncated,
        truncation_reason=(
            result.truncation_reason.value if result.truncation_reason is not None else None
        ),
        usage=task_usage["nl2sql"],
    )
    negative_gate_violation = not case.positive and eligibility.decision.decision == "eligible"
    negative_short_circuit_safe = (
        not case.positive
        and noneligible
        and not downstream_attempted
        and not downstream_counts
        and result.sql_attempts == 0
        and result.repair_attempts == 0
    )
    eligibility_metrics = GatedEligibilityMetrics(
        decision=eligibility.decision.decision,
        reason_code=eligibility.decision.reason_code,
        method=eligibility.decision.method,
        structured_error_category=eligibility.structured_error_category,
        usage=task_usage["chatbi_eligibility"],
    )
    return GatedCaseRecord(
        case_id=case.case_id,
        split="dev",
        positive=case.positive,
        category=case.category,
        difficulty=case.difficulty,
        eligibility=eligibility_metrics,
        sql_pipeline=sql_pipeline,
        analysis_usage=task_usage["chatbi_analysis"],
        negative_short_circuit_safe=negative_short_circuit_safe,
        negative_gate_violation=negative_gate_violation,
        provider_record=provider_record,
        semantic_diagnostic=semantic_diagnostic or GatedSemanticDiagnostic.unavailable(),
    )


def _merge_task_usage(
    task_type: Literal["chatbi_eligibility", "nl2sql", "chatbi_analysis"],
    usages: Sequence[GatedTaskUsage],
) -> GatedTaskUsage:
    providers = [usage.provider for usage in usages if usage.provider is not None]
    models = [usage.model for usage in usages if usage.model is not None]
    return GatedTaskUsage(
        task_type=task_type,
        logical_calls=sum(usage.logical_calls for usage in usages),
        provider_attempts=sum(usage.provider_attempts for usage in usages),
        input_tokens=_sum_optional([usage.input_tokens for usage in usages]),
        output_tokens=_sum_optional([usage.output_tokens for usage in usages]),
        latency_ms=sum(usage.latency_ms for usage in usages),
        provider=_single_value(providers),
        model=_single_value(models),
    )


def aggregate_gated_provider_usage(records: Sequence[GatedCaseRecord]) -> GatedUsageAggregate:
    """Aggregate task usage without combining eligibility, SQL, and analysis."""
    return GatedUsageAggregate(
        eligibility=_merge_task_usage(
            "chatbi_eligibility", [record.eligibility.usage for record in records]
        ),
        nl2sql=_merge_task_usage("nl2sql", [record.sql_pipeline.usage for record in records]),
        analysis=_merge_task_usage(
            "chatbi_analysis",
            [record.analysis_usage for record in records],
        ),
    )


def aggregate_gated_records(records: Sequence[GatedCaseRecord]) -> GatedAggregate:
    """Build gated counters from collected records and existing v2 facts."""
    positive = [record for record in records if record.positive]
    negative = [record for record in records if not record.positive]
    usage = aggregate_gated_provider_usage(records)
    decision_counts = Counter(record.eligibility.decision for record in records)
    semantic_counts = Counter(record.semantic_diagnostic.status for record in records)
    eligible_positive = [record for record in positive if record.eligibility.decision == "eligible"]
    return GatedAggregate(
        case_count=len(records),
        positive_case_count=len(positive),
        negative_case_count=len(negative),
        decision_counts=dict(sorted(decision_counts.items())),
        eligible_positive_count=len(eligible_positive),
        initial_generation_attempted_count=sum(
            record.sql_pipeline.initial_generation_attempts > 0 for record in records
        ),
        initial_generation_parseable_count=sum(
            record.sql_pipeline.initial_generation_parseable
            for record in records
            if record.sql_pipeline.initial_generation_attempts > 0
        ),
        initial_truncation_attempt_count=sum(
            record.sql_pipeline.initial_truncation_attempts for record in records
        ),
        validation_attempted_count=sum(
            record.sql_pipeline.validation_attempted for record in records
        ),
        validation_accepted_count=sum(
            record.sql_pipeline.validation_accepted for record in records
        ),
        execution_attempted_count=sum(
            record.sql_pipeline.execution_attempted for record in records
        ),
        execution_success_count=sum(record.sql_pipeline.execution_succeeded for record in records),
        repair_attempt_count=sum(record.sql_pipeline.repair_attempts for record in records),
        repair_recovery_success_count=sum(
            record.provider_record.repair_recovery_success for record in records
        ),
        analysis_attempt_count=sum(record.sql_pipeline.analysis_attempts for record in records),
        analysis_failure_count=sum(
            record.sql_pipeline.analysis_attempts > 0 and not record.sql_pipeline.analysis_succeeded
            for record in records
        ),
        formal_execution_equivalent_count=sum(
            record.sql_pipeline.formal_execution_equivalent is True for record in positive
        ),
        formal_execution_accuracy_denominator=len(positive),
        negative_safe_success_count=sum(record.negative_short_circuit_safe for record in negative),
        negative_gate_violation_count=sum(record.negative_gate_violation for record in negative),
        semantic_status_counts=dict(sorted(semantic_counts.items())),
        usage=usage,
    )


async def collect_gated_dev_cases(
    cases: Sequence[ChatBIEvaluationCaseV2],
    runner: GatedCaseRunner,
) -> list[GatedCaseRecord]:
    """Run a fresh complete DEV selection in order, without resume semantics."""
    selected = tuple(cases)
    if len(selected) != GATED_DEV_CASE_COUNT:
        raise GatedEvaluationIncompleteError("a gated DEV collection must start with all 50 cases")
    case_ids = tuple(case.case_id for case in selected)
    if len(set(case_ids)) != GATED_DEV_CASE_COUNT:
        raise GatedEvaluationMetricError("gated DEV case IDs must be unique")
    if any(case.split != V2ProviderSplit.DEV.value for case in selected):
        raise GatedEvaluationMetricError("gated DEV collection cannot include TEST cases")
    if sum(case.positive for case in selected) != GATED_DEV_POSITIVE_COUNT:
        raise GatedEvaluationMetricError("gated DEV selection must contain 47 positive cases")
    records: list[GatedCaseRecord] = []
    for case in selected:
        observation, eligibility = await runner.run(case)
        records.append(collect_gated_case_record(case, observation, eligibility))
    return records


def build_gated_artifact(
    provenance: GatedRunProvenance,
    records: Sequence[GatedCaseRecord],
    *,
    expected_case_ids: Sequence[str],
    aggregate: GatedAggregate | None = None,
    run_id: UUID | None = None,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
) -> GatedProviderRun:
    """Finalize a complete 50-case DEV artifact; partial runs are rejected."""
    expected = tuple(expected_case_ids)
    actual = tuple(record.case_id for record in records)
    if len(expected) != GATED_DEV_CASE_COUNT or len(records) != GATED_DEV_CASE_COUNT:
        raise GatedEvaluationIncompleteError(
            "gated artifact requires all 50 DEV cases; partial runs are not final artifacts"
        )
    if len(set(expected)) != len(expected) or actual != expected:
        raise GatedEvaluationMetricError("gated records must match expected DEV case order exactly")
    if provenance.case_ids != expected:
        raise GatedEvaluationMetricError("provenance case IDs do not match expected DEV order")
    calculated = aggregate_gated_records(records)
    if aggregate is not None and aggregate != calculated:
        raise GatedEvaluationMetricError("supplied gated aggregate does not match records")
    if calculated.positive_case_count != GATED_DEV_POSITIVE_COUNT:
        raise GatedEvaluationMetricError("gated DEV records must contain 47 positive cases")
    if calculated.negative_case_count != GATED_DEV_NEGATIVE_COUNT:
        raise GatedEvaluationMetricError("gated DEV records must contain 3 negative cases")
    validated_records = tuple(records)
    return GatedProviderRun(
        run_id=run_id or uuid4(),
        started_at=started_at or datetime.now(UTC),
        completed_at=completed_at or datetime.now(UTC),
        provenance=provenance,
        records=validated_records,
        aggregate=calculated,
    )


def write_gated_artifact(path: Path, artifact: GatedProviderRun) -> None:
    """Atomically write only an already validated complete artifact."""
    if not isinstance(artifact, GatedProviderRun):
        raise TypeError("artifact must be a validated complete gated provider run")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(artifact.model_dump_json(indent=2))
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


__all__ = [
    "GATED_DEV_CASE_COUNT",
    "GATED_DEV_NEGATIVE_COUNT",
    "GATED_DEV_POSITIVE_COUNT",
    "GATED_PROVIDER_ARTIFACT_VERSION",
    "GatedAggregate",
    "GatedCaseRecord",
    "GatedCaseRunner",
    "GatedEligibilityMetrics",
    "GatedEvaluationIncompleteError",
    "GatedEvaluationMetricError",
    "GatedEvaluationPreflightError",
    "GatedPreflight",
    "GatedProviderRun",
    "GatedRunProvenance",
    "GatedSQLPipelineMetrics",
    "GatedSemanticDiagnostic",
    "GatedTaskUsage",
    "GatedUsageAggregate",
    "aggregate_gated_provider_usage",
    "aggregate_gated_records",
    "build_gated_artifact",
    "collect_gated_case_record",
    "collect_gated_dev_cases",
    "construct_gated_provider",
    "preflight_gated_benchmark",
    "write_gated_artifact",
]
