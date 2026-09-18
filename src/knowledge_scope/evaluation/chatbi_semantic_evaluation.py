"""Evaluation-only semantic comparison for ChatBI result shapes.

The frozen ChatBI comparator intentionally compares the complete normalized
result shape.  This module provides a separate, conservative diagnostic for
product semantics.  It never participates in prompting, SQL validation,
execution, or the production Agent.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import (
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)
from sqlglot import exp, parse_one
from sqlglot.errors import SqlglotError

from knowledge_scope.chatbi.schemas import QueryExecutionResult, QueryLifecycleState
from knowledge_scope.evaluation.chatbi_evaluation import (
    ExpectedQueryResult,
    _EvaluationModel,
    _values_equal,
)
from knowledge_scope.evaluation.chatbi_schema_fingerprint import (
    SchemaFingerprintError,
    canonical_schema_payload,
    schema_fingerprint,
)

SEMANTIC_POLICY_VERSION = "a5.7f2-semantic-policy-v1"
SEMANTIC_POLICY_G3_VERSION = "a5.7g3-semantic-policy-v1"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SEMANTIC_POLICY_PATH = _REPOSITORY_ROOT / "docs/benchmarks/a5-7-semantic-policy-v1.json"
DEFAULT_G3_SEMANTIC_POLICY_PATH = (
    _REPOSITORY_ROOT / "docs/benchmarks/a5-7g3-semantic-policy-v1.json"
)


class SemanticEvaluationError(ValueError):
    """Raised when evaluation-only semantic policy input is invalid."""


class SemanticPolicyDuplicateKeyError(SemanticEvaluationError):
    """Raised when a JSON policy artifact contains an ambiguous object key."""


class SemanticRowGrain(StrEnum):
    SCALAR = "scalar"
    DETAIL = "detail"
    ENTITY = "entity"
    GROUPED = "grouped"
    PAIR = "pair"


class SemanticField(StrEnum):
    """Closed vocabulary for the reviewed DEV semantic policy."""

    CUSTOMER_ID = "customer_id"
    CUSTOMER_NAME = "customer_name"
    SALE_ID = "sale_id"
    REGION_NAME = "region_name"
    REGION_CODE = "region_code"
    MARKET = "market"
    AMOUNT = "amount"
    SOLD_ON = "sold_on"
    COUNT_SALE_ID = "aggregate:count:sale_id"
    COUNT_CUSTOMER_ID = "aggregate:count:customer_id"
    COUNT_DISTINCT_CUSTOMER_ID = "aggregate:count:distinct:customer_id"
    SUM_AMOUNT = "aggregate:sum:amount"
    AVG_AMOUNT = "aggregate:avg:amount"
    MAX_AMOUNT = "aggregate:max:amount"
    MIN_AMOUNT = "aggregate:min:amount"
    MIN_SOLD_ON = "aggregate:min:sold_on"
    MAX_SOLD_ON = "aggregate:max:sold_on"
    AMOUNT_MINUS_OVERALL_AVERAGE = "derived:amount_minus_overall_average"
    REGION_SALES_SHARE = "derived:region_sales_share"


SEMANTIC_FIELD_VOCABULARY = frozenset(field.value for field in SemanticField)
_SEMANTIC_IDENTITY_VOCABULARY = frozenset(
    {
        SemanticField.CUSTOMER_ID.value,
        SemanticField.SALE_ID.value,
        SemanticField.REGION_CODE.value,
        SemanticField.REGION_NAME.value,
    }
)


class SemanticPolicyAuthoringConfidence(StrEnum):
    """Audit metadata; it never changes executable semantic policy."""

    EXPLICIT = "explicit"
    SCHEMA_REQUIRED = "schema_required"
    PRODUCT_RULE_DERIVED = "product_rule_derived"
    AMBIGUOUS = "ambiguous"


G3_DEV_POSITIVE_CASE_IDS = frozenset(
    {
        "simple-01",
        "simple-02",
        "simple-03",
        "simple-04",
        "aggregation-01",
        "aggregation-02",
        "aggregation-03",
        "aggregation-04",
        "aggregation-05",
        "aggregation-06",
        "group-by-01",
        "group-by-02",
        "group-by-03",
        "group-by-04",
        "group-by-05",
        "top-k-01",
        "top-k-02",
        "top-k-03",
        "top-k-04",
        "predicate-01",
        "predicate-02",
        "predicate-03",
        "predicate-04",
        "join-01",
        "join-02",
        "join-03",
        "join-04",
        "join-05",
        "join-06",
        "multi-join-01",
        "multi-join-02",
        "multi-join-03",
        "multi-join-04",
        "date-01",
        "date-02",
        "date-03",
        "null-01",
        "null-02",
        "null-03",
        "empty-01",
        "empty-02",
        "derived-01",
        "derived-02",
        "derived-03",
        "cte-set-01",
        "cte-set-02",
        "cte-set-03",
    }
)


class OutputShapeStatus(StrEnum):
    """Whether output shape was successfully observed and matched."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    UNASSESSABLE = "unassessable"


class SemanticProofStatus(StrEnum):
    """Bounded proof result used by the diagnostic evaluator."""

    PROVEN_EQUIVALENT = "proven_equivalent"
    PROVEN_DIFFERENT = "proven_different"
    UNKNOWN = "unknown"


class SemanticForeignKey(_EvaluationModel):
    """Trusted, repository-reviewed foreign-key metadata for proof only."""

    source_relation: StrictStr
    source_columns: tuple[StrictStr, ...]
    target_relation: StrictStr
    target_columns: tuple[StrictStr, ...]

    @model_validator(mode="after")
    def validate_column_arity(self) -> SemanticForeignKey:
        if not self.source_columns or len(self.source_columns) != len(self.target_columns):
            raise ValueError("semantic foreign-key column arity is invalid")
        return self


class SemanticSchemaColumn(_EvaluationModel):
    """Complete catalog column metadata needed by the shared fingerprint."""

    position: StrictInt = Field(ge=1)
    name: StrictStr
    type: StrictStr
    nullable: StrictBool


class SemanticCheckConstraint(_EvaluationModel):
    """A provider-catalog CHECK constraint preserved in the sidecar."""

    columns: tuple[StrictStr, ...]
    definition: StrictStr


class SemanticRelationMetadata(_EvaluationModel):
    """The small constraint subset needed for conservative semantic proofs."""

    relation: StrictStr
    kind: Literal["table", "view"] = "table"
    columns: tuple[StrictStr, ...] = ()
    column_metadata: tuple[SemanticSchemaColumn, ...]
    primary_key: tuple[StrictStr, ...] = ()
    unique_constraints: tuple[tuple[StrictStr, ...], ...] = ()
    not_null_columns: tuple[StrictStr, ...] = ()
    foreign_keys: tuple[SemanticForeignKey, ...] = ()
    check_constraints: tuple[SemanticCheckConstraint, ...] = ()

    @model_validator(mode="after")
    def validate_catalog_columns(self) -> SemanticRelationMetadata:
        metadata_names = tuple(column.name for column in self.column_metadata)
        if metadata_names != self.columns:
            raise ValueError("semantic schema column metadata does not match columns")
        if tuple(column.position for column in self.column_metadata) != tuple(
            range(1, len(self.column_metadata) + 1)
        ):
            raise ValueError("semantic schema column positions are not contiguous")
        metadata_not_null = {column.name for column in self.column_metadata if not column.nullable}
        if metadata_not_null != set(self.not_null_columns):
            raise ValueError("semantic schema NOT NULL metadata does not match columns")
        column_names = set(self.columns)
        if any(column not in column_names for column in self.primary_key):
            raise ValueError("semantic schema primary key references an unknown column")
        if any(
            column not in column_names
            for constraint in self.unique_constraints
            for column in constraint
        ):
            raise ValueError("semantic schema unique constraint references an unknown column")
        if any(
            column not in column_names
            for foreign_key in self.foreign_keys
            for column in foreign_key.source_columns
        ):
            raise ValueError("semantic schema foreign key references an unknown column")
        if any(
            _normalize_relation_identity(foreign_key.source_relation)
            != _normalize_relation_identity(self.relation)
            for foreign_key in self.foreign_keys
        ):
            raise ValueError("semantic schema foreign key source relation is inconsistent")
        if any(
            column not in column_names
            for constraint in self.check_constraints
            for column in constraint.columns
        ):
            raise ValueError("semantic schema CHECK references an unknown column")
        return self


class SemanticSchemaMetadata(_EvaluationModel):
    """Frozen trusted schema contract; never populated from candidate output."""

    schema_version: Literal["a5.7f2c-schema-v1"] = "a5.7f2c-schema-v1"
    fixture_version: Literal["chatbi-demo-v2"] = "chatbi-demo-v2"
    catalog_schema: StrictStr
    fixture_fingerprint: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    fixture_schema_fingerprint: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    schema_metadata_fingerprint: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    relations: tuple[SemanticRelationMetadata, ...]

    @model_validator(mode="after")
    def validate_metadata_fingerprint(self) -> SemanticSchemaMetadata:
        prefix = f"{self.catalog_schema}."
        relation_map = {
            _normalize_relation_identity(relation.relation): relation for relation in self.relations
        }
        if any(
            not relation.relation.startswith(prefix) or relation.relation.count(".") != 1
            for relation in self.relations
        ):
            raise ValueError("semantic schema relation is outside the catalog schema")
        for relation in self.relations:
            for foreign_key in relation.foreign_keys:
                target_name = _normalize_relation_identity(foreign_key.target_relation)
                if not target_name.startswith(_normalize_relation_identity(prefix)):
                    raise ValueError(
                        "semantic schema foreign key target is outside the catalog schema"
                    )
                target = relation_map.get(target_name)
                if target is None or any(
                    column not in target.columns for column in foreign_key.target_columns
                ):
                    raise ValueError("semantic schema foreign key target is invalid")
        if semantic_schema_metadata_fingerprint(self) != self.schema_metadata_fingerprint:
            raise ValueError("semantic schema metadata fingerprint does not match metadata")
        if self.schema_metadata_fingerprint != self.fixture_schema_fingerprint:
            raise ValueError("semantic schema metadata fingerprint is not authoritative")
        return self


class SemanticProof(_EvaluationModel):
    """One bounded semantic proof result with safe diagnostic reasons."""

    status: SemanticProofStatus
    reasons: tuple[StrictStr, ...] = ()


class SemanticCasePolicy(_EvaluationModel):
    """Product-semantic rules for one evaluation case.

    The policy is evaluation metadata derived from human product review.  It
    is not an oracle prompt, a validator rule, or a production constraint.
    """

    row_grain: SemanticRowGrain
    required_fields: tuple[StrictStr, ...] = Field(default=(), max_length=64)
    optional_fields: tuple[StrictStr, ...] = Field(default=(), max_length=64)
    identity_fields: tuple[StrictStr, ...] = Field(default=(), max_length=32)
    semantic_order_required: StrictBool = False
    # Retained for sidecar compatibility, but it is never sufficient by
    # itself to authorize an arbitrary extra projection.
    allow_safe_extra_fields: StrictBool = False

    def normalized_required_fields(self) -> frozenset[str]:
        return frozenset(_normalize_field_identity(field) for field in self.required_fields)

    def normalized_optional_fields(self) -> frozenset[str]:
        return frozenset(_normalize_field_identity(field) for field in self.optional_fields)

    def normalized_identity_fields(self) -> frozenset[str]:
        return frozenset(_normalize_field_identity(field) for field in self.identity_fields)


_RATIONALE_UNSAFE = re.compile(
    r"(?:```|;|--|/\*|\b(?:select|from|where|join|group\s+by|order\s+by|union|with)\b|"
    r"\b(?:candidate|expected_sql|redacted_sql|provider|model|proof|unassessable|"
    r"execution\s+accuracy|run\s*#)\b)",
    re.IGNORECASE,
)


class SemanticCasePolicyV3(_EvaluationModel):
    """Explicit DEV policy with a closed semantic-field vocabulary.

    ``authoring_confidence`` and ``human_rationale`` are review metadata only.
    ``to_executable_policy`` intentionally drops them before the evaluator sees
    the semantic contract.
    """

    row_grain: SemanticRowGrain
    required_fields: tuple[SemanticField, ...] = Field(default=(), max_length=64)
    optional_fields: tuple[SemanticField, ...] = Field(default=(), max_length=64)
    identity_fields: tuple[SemanticField, ...] = Field(default=(), max_length=32)
    semantic_order_required: StrictBool = False
    allow_safe_extra_fields: StrictBool = False
    authoring_confidence: SemanticPolicyAuthoringConfidence
    human_rationale: StrictStr = Field(min_length=1, max_length=512)

    @field_validator("human_rationale")
    @classmethod
    def validate_human_rationale(cls, value: str) -> str:
        if _RATIONALE_UNSAFE.search(value):
            raise ValueError("human rationale contains execution or provider output text")
        return value

    @model_validator(mode="after")
    def validate_field_sets(self) -> SemanticCasePolicyV3:
        required = tuple(field.value for field in self.required_fields)
        optional = tuple(field.value for field in self.optional_fields)
        identity = tuple(field.value for field in self.identity_fields)
        if len(set(required)) != len(required):
            raise ValueError("semantic policy required fields contain duplicates")
        if len(set(optional)) != len(optional):
            raise ValueError("semantic policy optional fields contain duplicates")
        if len(set(identity)) != len(identity):
            raise ValueError("semantic policy identity fields contain duplicates")
        if set(required) & set(optional):
            raise ValueError("semantic policy required and optional fields overlap")
        if set(identity) - _SEMANTIC_IDENTITY_VOCABULARY:
            raise ValueError("semantic policy identity fields are not stable identifiers")
        if self.row_grain is SemanticRowGrain.SCALAR and identity:
            raise ValueError("scalar semantic policy cannot declare row identity fields")
        return self

    def normalized_required_fields(self) -> frozenset[str]:
        return frozenset(field.value for field in self.required_fields)

    def normalized_optional_fields(self) -> frozenset[str]:
        return frozenset(field.value for field in self.optional_fields)

    def normalized_identity_fields(self) -> frozenset[str]:
        return frozenset(field.value for field in self.identity_fields)

    def to_executable_policy(self) -> SemanticCasePolicy:
        """Drop authoring metadata before semantic evaluation."""

        return SemanticCasePolicy(
            row_grain=self.row_grain,
            required_fields=tuple(field.value for field in self.required_fields),
            optional_fields=tuple(field.value for field in self.optional_fields),
            identity_fields=tuple(field.value for field in self.identity_fields),
            semantic_order_required=self.semantic_order_required,
            allow_safe_extra_fields=self.allow_safe_extra_fields,
        )


class SemanticPolicyArtifact(_EvaluationModel):
    """Versioned, repository-safe sidecar for semantic review metadata."""

    schema_version: Literal["a5.7f2-semantic-policy-v1"] = SEMANTIC_POLICY_VERSION
    provenance: StrictStr = Field(min_length=1, max_length=500)
    dataset_fingerprint: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    fixture_version: Literal["chatbi-demo-v2"] = "chatbi-demo-v2"
    fixture_fingerprint: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    fixture_schema_fingerprint: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        frozen=True,
        populate_by_name=True,
    )
    schema_metadata: SemanticSchemaMetadata = Field(alias="schema")
    defaults: SemanticCasePolicy
    cases: dict[StrictStr, SemanticCasePolicy] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_schema_binding(self) -> SemanticPolicyArtifact:
        if semantic_schema_metadata_fingerprint(self.schema_metadata) != (
            self.schema_metadata.schema_metadata_fingerprint
        ):
            raise ValueError("semantic schema metadata fingerprint does not match metadata")
        if self.schema_metadata.fixture_version != self.fixture_version:
            raise ValueError("semantic policy fixture version does not match schema metadata")
        if self.schema_metadata.fixture_fingerprint != self.fixture_fingerprint:
            raise ValueError("semantic policy fixture fingerprint does not match schema metadata")
        if self.schema_metadata.fixture_schema_fingerprint != self.fixture_schema_fingerprint:
            raise ValueError(
                "semantic policy fixture schema fingerprint does not match schema metadata"
            )
        return self

    @property
    def schema(self) -> SemanticSchemaMetadata:
        """Return the aliased trusted schema contract without shadow warnings."""

        return self.schema_metadata


class SemanticPolicyArtifactV3(_EvaluationModel):
    """Frozen A5.7g3 DEV-only policy artifact.

    The exact case-set check is deliberately independent of candidate SQL,
    provider artifacts, or result rows.  TEST cases cannot enter this artifact.
    """

    schema_version: Literal["a5.7g3-semantic-policy-v1"] = SEMANTIC_POLICY_G3_VERSION
    provenance: StrictStr = Field(min_length=1, max_length=500)
    dataset_fingerprint: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    fixture_version: Literal["chatbi-demo-v2"] = "chatbi-demo-v2"
    fixture_fingerprint: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    fixture_schema_fingerprint: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    policy_fingerprint: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        frozen=True,
        populate_by_name=True,
    )
    schema_metadata: SemanticSchemaMetadata = Field(alias="schema")
    defaults: SemanticCasePolicyV3
    cases: dict[StrictStr, SemanticCasePolicyV3] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_schema_binding_and_fingerprint(self) -> SemanticPolicyArtifactV3:
        if semantic_schema_metadata_fingerprint(self.schema_metadata) != (
            self.schema_metadata.schema_metadata_fingerprint
        ):
            raise ValueError("semantic schema metadata fingerprint does not match metadata")
        if self.schema_metadata.fixture_version != self.fixture_version:
            raise ValueError("semantic policy fixture version does not match schema metadata")
        if self.schema_metadata.fixture_fingerprint != self.fixture_fingerprint:
            raise ValueError("semantic policy fixture fingerprint does not match schema metadata")
        if self.schema_metadata.fixture_schema_fingerprint != self.fixture_schema_fingerprint:
            raise ValueError(
                "semantic policy fixture schema fingerprint does not match schema metadata"
            )
        if frozenset(self.cases) != G3_DEV_POSITIVE_CASE_IDS:
            raise ValueError("A5.7g3 policy must contain exactly the 47 positive DEV cases")
        if self.defaults.allow_safe_extra_fields:
            raise ValueError("A5.7g3 default policy cannot allow safe extra fields")
        if any(case.allow_safe_extra_fields for case in self.cases.values()):
            raise ValueError("A5.7g3 DEV policies cannot allow safe extra fields")
        if semantic_policy_v3_fingerprint(self) != self.policy_fingerprint:
            raise ValueError("semantic policy fingerprint does not match artifact")
        return self

    @property
    def schema(self) -> SemanticSchemaMetadata:
        return self.schema_metadata


class OutputShapeCompliance(_EvaluationModel):
    """Presentation/schema-shape diagnostics kept separate from semantics."""

    status: OutputShapeStatus
    shape_compliant: StrictBool
    exact_column_sequence: StrictBool
    structural_shape_compliant: StrictBool | None = None
    extra_fields: tuple[StrictStr, ...] = ()
    omitted_fields: tuple[StrictStr, ...] = ()
    optional_omitted_fields: tuple[StrictStr, ...] = ()
    column_order_difference: StrictBool = False
    alias_difference: StrictBool = False
    presentation_differences: tuple[StrictStr, ...] = ()


class SemanticComparison(_EvaluationModel):
    """Structured diagnostic result; ``equivalent`` is never a formal score."""

    equivalent: StrictBool
    assessable: StrictBool
    required_fields_present: StrictBool
    required_values_match: StrictBool
    row_grain_match: StrictBool
    identity_semantics_match: StrictBool
    row_set_match: StrictBool
    ordering_semantics_match: StrictBool
    actual_row_count: StrictInt = Field(ge=0)
    expected_row_count: StrictInt = Field(ge=0)
    optional_extra_fields: tuple[StrictStr, ...] = ()
    presentation_differences: tuple[StrictStr, ...] = ()
    proof_reasons: tuple[StrictStr, ...] = ()
    failure_reason: StrictStr | None = None
    shape: OutputShapeCompliance


class SemanticRunAggregate(_EvaluationModel):
    """Aggregate diagnostic counts for one local evaluation artifact."""

    positive_case_count: StrictInt = Field(ge=0)
    formal_execution_equivalent_count: StrictInt = Field(ge=0)
    assessable_case_count: StrictInt = Field(ge=0)
    semantic_equivalent_count: StrictInt = Field(ge=0)
    shape_compliant_count: StrictInt = Field(ge=0)
    shape_non_compliant_count: StrictInt = Field(ge=0)
    shape_unassessable_count: StrictInt = Field(ge=0)
    presentation_only_failure_count: StrictInt = Field(ge=0)
    semantic_failure_count: StrictInt = Field(ge=0)
    unassessable_count: StrictInt = Field(ge=0)
    structurally_compatible_unassessable_count: StrictInt = Field(ge=0)
    result_unavailable_unassessable_count: StrictInt = Field(ge=0)


class SemanticRunReport(_EvaluationModel):
    """Safe report for a historical provider artifact."""

    schema_version: Literal["a5.7f2-semantic-run-v1"] = "a5.7f2-semantic-run-v1"
    run_id: StrictStr
    dataset_fingerprint: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    fixture_version: Literal["chatbi-demo-v2"] = "chatbi-demo-v2"
    fixture_fingerprint: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    fixture_schema_fingerprint: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    policy_version: Literal["a5.7f2-semantic-policy-v1", "a5.7g3-semantic-policy-v1"] = (
        SEMANTIC_POLICY_VERSION
    )
    positive_case_count: StrictInt = Field(ge=0)
    aggregate: SemanticRunAggregate
    cases: dict[StrictStr, SemanticComparison]


def _normalize_field_identity(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    normalized = normalized.removeprefix("column:")
    return re.sub(r"\s+", " ", normalized)


def _normalize_relation_identity(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip().casefold()


def semantic_field_is_supported(value: str) -> bool:
    """Return whether ``value`` is one of the reviewed semantic identifiers."""

    return value in SEMANTIC_FIELD_VOCABULARY


def semantic_policy_v3_fingerprint(policy: SemanticPolicyArtifactV3) -> str:
    """Hash the complete immutable A5.7g3 contract deterministically."""

    payload = policy.model_dump(
        mode="json",
        by_alias=True,
        exclude={"policy_fingerprint"},
    )
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _semantic_schema_fingerprint_payload(schema: SemanticSchemaMetadata) -> dict[str, object]:
    """Return the provider-compatible canonical catalog payload.

    ``SemanticRelationMetadata`` keeps proof-friendly qualified relation names
    and constraint fields.  This adapter converts those fields into the exact
    provider catalog shape; ordering and hashing are then owned exclusively by
    :mod:`chatbi_schema_fingerprint`.
    """

    prefix = f"{schema.catalog_schema}."
    relations: list[dict[str, object]] = []
    for relation in schema.relations:
        qualified_name = relation.relation
        if not qualified_name.startswith(prefix) or qualified_name.count(".") != 1:
            raise SemanticEvaluationError("semantic schema relation is outside the catalog schema")
        relation_name = qualified_name.removeprefix(prefix)
        not_null_columns = set(relation.not_null_columns)
        columns = [
            {
                "position": column.position,
                "name": column.name,
                "type": column.type,
                "nullable": column.name not in not_null_columns,
            }
            for column in relation.column_metadata
        ]
        constraints: list[dict[str, object]] = []
        if relation.primary_key:
            constraints.append(
                {
                    "type": "primary_key",
                    "columns": list(relation.primary_key),
                    "referenced_schema": None,
                    "referenced_relation": None,
                    "referenced_columns": [],
                    "check_definition": None,
                }
            )
        constraints.extend(
            {
                "type": "unique",
                "columns": list(unique_constraint),
                "referenced_schema": None,
                "referenced_relation": None,
                "referenced_columns": [],
                "check_definition": None,
            }
            for unique_constraint in relation.unique_constraints
        )
        for foreign_key in relation.foreign_keys:
            target_prefix, separator, target_name = foreign_key.target_relation.partition(".")
            referenced_schema = target_prefix if separator else schema.catalog_schema
            referenced_relation = target_name if separator else target_prefix
            constraints.append(
                {
                    "type": "foreign_key",
                    "columns": list(foreign_key.source_columns),
                    "referenced_schema": referenced_schema,
                    "referenced_relation": referenced_relation,
                    "referenced_columns": list(foreign_key.target_columns),
                    "check_definition": None,
                }
            )
        constraints.extend(
            {
                "type": "check",
                "columns": list(check_constraint.columns),
                "referenced_schema": None,
                "referenced_relation": None,
                "referenced_columns": [],
                "check_definition": check_constraint.definition,
            }
            for check_constraint in relation.check_constraints
        )
        relations.append(
            {
                "name": relation_name,
                "kind": relation.kind,
                "columns": columns,
                "constraints": constraints,
            }
        )
    try:
        return canonical_schema_payload(schema=schema.catalog_schema, relations=relations)
    except SchemaFingerprintError as error:
        raise SemanticEvaluationError("semantic schema metadata is invalid") from error


def semantic_schema_metadata_fingerprint(schema: SemanticSchemaMetadata) -> str:
    """Return the authoritative provider-compatible schema fingerprint."""

    return schema_fingerprint(_semantic_schema_fingerprint_payload(schema))


def _ensure_schema_metadata_integrity(schema: SemanticSchemaMetadata) -> None:
    """Reject model copies or deserialized objects whose metadata was mutated."""

    recomputed = semantic_schema_metadata_fingerprint(schema)
    if recomputed != schema.schema_metadata_fingerprint:
        raise SemanticEvaluationError("semantic schema metadata fingerprint mismatch")
    if recomputed != schema.fixture_schema_fingerprint:
        raise SemanticEvaluationError("semantic schema metadata fingerprint is not authoritative")


def _ensure_policy_integrity(policy: SemanticPolicyArtifact) -> None:
    _ensure_schema_metadata_integrity(policy.schema)
    if policy.schema.fixture_version != policy.fixture_version:
        raise SemanticEvaluationError("semantic policy fixture version mismatch")
    if policy.schema.fixture_fingerprint != policy.fixture_fingerprint:
        raise SemanticEvaluationError("semantic policy fixture fingerprint mismatch")
    if policy.schema.fixture_schema_fingerprint != policy.fixture_schema_fingerprint:
        raise SemanticEvaluationError("semantic policy fixture schema fingerprint mismatch")


def _ensure_versioned_policy_integrity(
    policy: SemanticPolicyArtifact | SemanticPolicyArtifactV3,
) -> None:
    if isinstance(policy, SemanticPolicyArtifactV3):
        if semantic_policy_v3_fingerprint(policy) != policy.policy_fingerprint:
            raise SemanticEvaluationError("semantic policy fingerprint does not match artifact")
        _ensure_schema_metadata_integrity(policy.schema)
        if policy.schema.fixture_version != policy.fixture_version:
            raise SemanticEvaluationError("semantic policy fixture version mismatch")
        if policy.schema.fixture_fingerprint != policy.fixture_fingerprint:
            raise SemanticEvaluationError("semantic policy fixture fingerprint mismatch")
        if policy.schema.fixture_schema_fingerprint != policy.fixture_schema_fingerprint:
            raise SemanticEvaluationError("semantic policy fixture schema fingerprint mismatch")
        return
    _ensure_policy_integrity(policy)


def _schema_relation(
    schema: SemanticSchemaMetadata | None,
    relation: str,
) -> SemanticRelationMetadata | None:
    if schema is None:
        return None
    normalized = _normalize_relation_identity(relation)
    exact = next(
        (
            item
            for item in schema.relations
            if _normalize_relation_identity(item.relation) == normalized
        ),
        None,
    )
    if exact is not None:
        return exact
    suffix = [
        item
        for item in schema.relations
        if _normalize_relation_identity(item.relation).endswith(f".{normalized}")
    ]
    return suffix[0] if len(suffix) == 1 else None


def _relation_key_from_table(table: exp.Table) -> str:
    name = _normalize_relation_identity(table.name)
    database = _normalize_relation_identity(table.db) if table.db else ""
    return f"{database}.{name}" if database else name


def _table_aliases(
    select: exp.Select,
    schema: SemanticSchemaMetadata | None,
) -> dict[str, str]:
    aliases: dict[str, str] = {}
    cte_names = {
        _normalize_relation_identity(cte.alias_or_name)
        for cte in select.find_all(exp.CTE)
        if cte.alias_or_name
    }
    for table in select.find_all(exp.Table):
        relation = _relation_key_from_table(table)
        if not table.db and relation in cte_names:
            continue
        if _schema_relation(schema, relation) is None and schema is not None:
            continue
        aliases[_normalize_relation_identity(table.name)] = relation
        if table.alias:
            aliases[_normalize_relation_identity(table.alias)] = relation
    return aliases


def _column_identity(
    column: exp.Column,
    aliases: dict[str, str],
    schema: SemanticSchemaMetadata | None,
) -> str | None:
    name = _normalize_field_identity(column.name)
    if column.table:
        relation = aliases.get(_normalize_relation_identity(column.table))
        if relation is None:
            relation = _normalize_relation_identity(column.table)
        return f"{relation}.{name}"
    if schema is None:
        return name
    candidates = [
        relation.relation
        for relation in schema.relations
        if name in {_normalize_field_identity(item) for item in relation.columns}
    ]
    visible_relations = {_normalize_relation_identity(relation) for relation in aliases.values()}
    visible_candidates = [
        relation
        for relation in candidates
        if _normalize_relation_identity(relation) in visible_relations
    ]
    if len(visible_candidates) == 1:
        candidates = visible_candidates
    if len(candidates) != 1:
        return None
    return f"{_normalize_relation_identity(candidates[0])}.{name}"


def _flatten_and(node: exp.Expression) -> list[exp.Expression]:
    if isinstance(node, exp.And):
        return _flatten_and(node.this) + _flatten_and(node.expression)
    return [node]


def _direct_select_lineage(
    select: exp.Select,
    schema: SemanticSchemaMetadata | None,
) -> dict[str, tuple[object, ...]] | None:
    """Resolve a simple derived SELECT's output names to source signatures.

    This is deliberately narrow: nested subqueries/CTEs and any expression
    whose source cannot be resolved are left unknown rather than being
    treated as equivalent by alias spelling.
    """

    if any(
        next(select.find_all(node_type), None) is not None for node_type in (exp.CTE, exp.Subquery)
    ):
        return None
    aliases = _table_aliases(select, schema)
    lineage: dict[str, tuple[object, ...]] = {}
    for expression in select.expressions:
        output_name = expression.alias_or_name or expression.name
        if not output_name:
            return None
        signature = _expression_signature(
            expression,
            aliases=aliases,
            schema=schema,
            literals_available=True,
        )
        if signature is None:
            return None
        lineage[_normalize_field_identity(output_name)] = signature
    return lineage


def _derived_alias_lineage(
    sql: str | None,
    schema: SemanticSchemaMetadata | None,
) -> dict[str, Mapping[str, tuple[object, ...]]]:
    """Return only directly provable CTE/derived-table output lineage."""

    tree = _parse_query(sql)
    if not isinstance(tree, exp.Select):
        return {}
    lineage: dict[str, Mapping[str, tuple[object, ...]]] = {}
    sources: list[exp.Expression] = []
    from_clause = tree.args.get("from_")
    if isinstance(from_clause, exp.From) and isinstance(from_clause.this, exp.Expression):
        sources.append(from_clause.this)
    for join in tree.args.get("joins") or []:
        if isinstance(join, exp.Join) and isinstance(join.this, exp.Expression):
            sources.append(join.this)
    with_clause = tree.args.get("with") or tree.args.get("with_")
    if isinstance(with_clause, exp.With):
        for cte in with_clause.expressions:
            if not isinstance(cte, exp.CTE) or not isinstance(cte.this, exp.Select):
                continue
            alias = _normalize_relation_identity(cte.alias_or_name)
            if not alias:
                continue
            resolved = _direct_select_lineage(cte.this, schema)
            if resolved is not None:
                lineage[alias] = resolved
        for source in sources:
            if not isinstance(source, exp.Table):
                continue
            cte_lineage = lineage.get(_normalize_relation_identity(source.name))
            if cte_lineage is not None:
                lineage[_normalize_relation_identity(source.alias_or_name)] = cte_lineage
    for source in sources:
        if not isinstance(source, exp.Subquery):
            continue
        alias = _normalize_relation_identity(source.alias_or_name)
        if not alias or not isinstance(source.this, exp.Select):
            continue
        resolved = _direct_select_lineage(source.this, schema)
        if resolved is not None:
            lineage[alias] = resolved
    return lineage


def _expression_signature(
    node: exp.Expression,
    *,
    aliases: dict[str, str],
    schema: SemanticSchemaMetadata | None,
    literals_available: bool,
    lineage: Mapping[str, Mapping[str, tuple[object, ...]]] | None = None,
) -> tuple[object, ...] | None:
    """Return a bounded canonical signature, or None when proof is unsafe."""

    if isinstance(node, exp.Alias):
        return _expression_signature(
            node.this,
            aliases=aliases,
            schema=schema,
            literals_available=literals_available,
            lineage=lineage,
        )
    if isinstance(node, exp.Paren):
        return _expression_signature(
            node.this,
            aliases=aliases,
            schema=schema,
            literals_available=literals_available,
            lineage=lineage,
        )
    if isinstance(node, exp.Column):
        identity = None
        if node.table:
            table_name = _normalize_relation_identity(node.table)
            relation = aliases.get(table_name)
            if relation is not None:
                identity = f"{relation}.{_normalize_field_identity(node.name)}"
            elif lineage is not None:
                identity_signature = lineage.get(table_name, {}).get(
                    _normalize_field_identity(node.name)
                )
                if identity_signature is not None:
                    return identity_signature
        else:
            identity = _column_identity(node, aliases, schema)
        return ("column", identity) if identity is not None else None
    if isinstance(node, exp.Boolean):
        return ("boolean", bool(node.this))
    if isinstance(node, exp.Null):
        return ("null",)
    if isinstance(node, exp.Literal):
        if not literals_available:
            return ("literal", "unknown")
        return ("literal", bool(node.is_string), str(node.this))
    if isinstance(node, exp.And):
        signatures = [
            _expression_signature(
                term,
                aliases=aliases,
                schema=schema,
                literals_available=literals_available,
                lineage=lineage,
            )
            for term in _flatten_and(node)
        ]
        if any(signature is None for signature in signatures):
            return None
        return ("and", tuple(sorted(signatures, key=repr)))
    if isinstance(node, exp.Or):
        signatures = [
            _expression_signature(
                child,
                aliases=aliases,
                schema=schema,
                literals_available=literals_available,
                lineage=lineage,
            )
            for child in (node.this, node.expression)
        ]
        return (
            (
                "or",
                tuple(
                    sorted(
                        (signature for signature in signatures if signature is not None), key=repr
                    )
                ),
            )
            if all(signature is not None for signature in signatures)
            else None
        )
    if isinstance(node, exp.EQ):
        signatures = [
            _expression_signature(
                child,
                aliases=aliases,
                schema=schema,
                literals_available=literals_available,
                lineage=lineage,
            )
            for child in (node.this, node.expression)
        ]
        return (
            (
                "eq",
                tuple(
                    sorted(
                        (signature for signature in signatures if signature is not None), key=repr
                    )
                ),
            )
            if all(signature is not None for signature in signatures)
            else None
        )

    values: list[object] = []
    for key, value in sorted(node.args.items()):
        if key in {"comments", "alias"} or value is None:
            continue
        if isinstance(value, exp.Expression):
            signature = _expression_signature(
                value,
                aliases=aliases,
                schema=schema,
                literals_available=literals_available,
                lineage=lineage,
            )
            if signature is None:
                return None
            values.append((key, signature))
        elif isinstance(value, list):
            signatures: list[tuple[object, ...]] = []
            for item in value:
                if not isinstance(item, exp.Expression):
                    return None
                signature = _expression_signature(
                    item,
                    aliases=aliases,
                    schema=schema,
                    literals_available=literals_available,
                    lineage=lineage,
                )
                if signature is None:
                    return None
                signatures.append(signature)
            values.append((key, tuple(signatures)))
        elif isinstance(value, (str, bool, int, float)):
            values.append((key, value))
        else:
            return None
    return (node.key.casefold(), tuple(values))


def _signature_contains_unknown_literal(signature: object) -> bool:
    if isinstance(signature, tuple):
        if len(signature) == 2 and signature[0] == "literal" and signature[1] == "unknown":
            return True
        return any(_signature_contains_unknown_literal(item) for item in signature)
    return False


def _signature_structure(signature: object) -> object:
    if isinstance(signature, tuple):
        # Available literals carry (kind, is_string, value), while redacted
        # literals carry (kind, "unknown").  Both describe the same
        # structural slot; the value is simply unavailable in a historical
        # artifact.  Normalizing every literal-shaped tuple avoids turning
        # redaction into a false semantic difference.
        if signature and signature[0] == "literal":
            return ("literal", "unknown")
        return tuple(_signature_structure(item) for item in signature)
    return signature


def _compare_signatures(
    expected: tuple[object, ...] | None,
    actual: tuple[object, ...] | None,
    *,
    equivalent_reason: str,
    different_reason: str,
) -> SemanticProof:
    if expected is None or actual is None:
        return SemanticProof(status=SemanticProofStatus.UNKNOWN, reasons=("insufficient_lineage",))
    if expected == actual and not _signature_contains_unknown_literal(expected):
        return SemanticProof(
            status=SemanticProofStatus.PROVEN_EQUIVALENT,
            reasons=(equivalent_reason,),
        )
    if expected != actual and not (
        _signature_contains_unknown_literal(expected) or _signature_contains_unknown_literal(actual)
    ):
        return SemanticProof(
            status=SemanticProofStatus.PROVEN_DIFFERENT,
            reasons=(different_reason,),
        )
    if _signature_structure(expected) != _signature_structure(actual):
        return SemanticProof(
            status=SemanticProofStatus.PROVEN_DIFFERENT,
            reasons=(different_reason,),
        )
    return SemanticProof(status=SemanticProofStatus.UNKNOWN, reasons=("literal_value_unavailable",))


def _select_group_columns(
    sql: str | None,
    schema: SemanticSchemaMetadata | None,
) -> tuple[str, ...] | None:
    select = _select_expression(sql)
    if select is None:
        return None
    group = select.args.get("group")
    if group is None:
        return ()
    aliases = _table_aliases(select, schema)
    identities: list[str] = []
    for expression in group.expressions:
        if not isinstance(expression, exp.Column):
            return None
        identity = _column_identity(expression, aliases, schema)
        if identity is None:
            return None
        identities.append(identity)
    return tuple(sorted(set(identities)))


def _relation_keys(relation: SemanticRelationMetadata) -> tuple[tuple[str, ...], ...]:
    qualified = _normalize_relation_identity(relation.relation)
    keys = [
        tuple(f"{qualified}.{_normalize_field_identity(column)}" for column in relation.primary_key)
    ]
    keys.extend(
        tuple(f"{qualified}.{_normalize_field_identity(column)}" for column in constraint)
        for constraint in relation.unique_constraints
    )
    not_null = {_normalize_field_identity(column) for column in relation.not_null_columns}
    return tuple(
        key for key in keys if key and all(item.rsplit(".", 1)[-1] in not_null for item in key)
    )


def _functional_closure(
    identities: set[str],
    schema: SemanticSchemaMetadata | None,
) -> set[str]:
    closure = set(identities)
    if schema is None:
        return closure
    changed = True
    while changed:
        changed = False
        for relation in schema.relations:
            relation_key = _normalize_relation_identity(relation.relation)
            all_columns = {
                f"{relation_key}.{_normalize_field_identity(column)}" for column in relation.columns
            }
            for key in _relation_keys(relation):
                if set(key) <= closure and not all_columns <= closure:
                    closure.update(all_columns)
                    changed = True
    return closure


def _group_proof(
    expected_sql: str | None,
    actual_sql: str | None,
    schema: SemanticSchemaMetadata | None,
) -> SemanticProof:
    expected = _select_group_columns(expected_sql, schema)
    actual = _select_group_columns(actual_sql, schema)
    if expected is None or actual is None:
        return SemanticProof(
            status=SemanticProofStatus.UNKNOWN, reasons=("unsupported_group_expression",)
        )
    if expected == actual:
        return SemanticProof(
            status=SemanticProofStatus.PROVEN_EQUIVALENT, reasons=("grouping_keys_match",)
        )
    if not expected or not actual or schema is None:
        return SemanticProof(
            status=SemanticProofStatus.UNKNOWN, reasons=("grouping_dependency_unavailable",)
        )
    expected_closure = _functional_closure(set(expected), schema)
    actual_closure = _functional_closure(set(actual), schema)
    if set(expected) <= actual_closure and set(actual) <= expected_closure:
        reason = _group_dependency_reason(expected, actual, schema)
        return SemanticProof(status=SemanticProofStatus.PROVEN_EQUIVALENT, reasons=(reason,))
    if not (set(expected) <= actual_closure and set(actual) <= expected_closure):
        return SemanticProof(
            status=SemanticProofStatus.PROVEN_DIFFERENT,
            reasons=("different_grouping_grain",),
        )
    return SemanticProof(
        status=SemanticProofStatus.UNKNOWN, reasons=("grouping_dependency_unavailable",)
    )


def _group_dependency_reason(
    expected: tuple[str, ...],
    actual: tuple[str, ...],
    schema: SemanticSchemaMetadata,
) -> str:
    """Name the determinant that actually proves two grouping keys equivalent."""

    candidates: list[tuple[int, int, str]] = []
    for relation in schema.relations:
        relation_id = _normalize_relation_identity(relation.relation)
        primary_key = tuple(
            f"{relation_id}.{_normalize_field_identity(column)}" for column in relation.primary_key
        )
        keys: list[tuple[str, tuple[str, ...]]] = []
        if primary_key:
            keys.append(("equivalent_by_primary_key_dependency", primary_key))
        keys.extend(
            (
                "equivalent_by_unique_not_null_dependency",
                tuple(
                    f"{relation_id}.{_normalize_field_identity(column)}" for column in constraint
                ),
            )
            for constraint in relation.unique_constraints
        )
        for reason, key in keys:
            key_set = set(key)
            if not key_set:
                continue
            for determinant, dependent in (
                (set(expected), set(actual)),
                (set(actual), set(expected)),
            ):
                if key_set <= determinant and dependent <= _functional_closure(determinant, schema):
                    # Prefer the smaller determinant; for equal determinants a
                    # primary key is the more specific proof.
                    candidates.append((len(determinant), 0 if "primary" in reason else 1, reason))
    if candidates:
        candidates.sort()
        return candidates[0][2]
    return "equivalent_by_functional_dependency"


def _select_expression(sql: str) -> exp.Select | None:
    try:
        tree = parse_one(sql, dialect="postgres")
    except (SqlglotError, RecursionError, ValueError):
        return None
    if isinstance(tree, exp.Select):
        return tree
    if isinstance(tree, (exp.Union, exp.Intersect, exp.Except)):
        left = tree.left
        return left if isinstance(left, exp.Select) else left.find(exp.Select)
    return tree.find(exp.Select)


def _normalized_expression_sql(node: exp.Expression) -> str:
    return " ".join(node.sql(dialect="postgres", pretty=False).casefold().split())


def _parse_query(sql: str | None) -> exp.Expression | None:
    if sql is None:
        return None
    try:
        return parse_one(sql, dialect="postgres")
    except (SqlglotError, RecursionError, ValueError):
        return None


def _strip_outer_projection(node: exp.Expression) -> exp.Expression | None:
    """Remove only the result projection while retaining query semantics.

    This is used for the empty-result guard.  Nested SELECTs remain intact, so
    a caller cannot prove two different subqueries equivalent merely because
    their outer result sets are both empty.
    """

    copied = node.copy()
    if isinstance(copied, exp.Select):
        copied.set("expressions", [])
        return copied
    if isinstance(copied, (exp.Union, exp.Intersect, exp.Except)):
        left = _strip_outer_projection(copied.left)
        right = _strip_outer_projection(copied.right)
        if left is None or right is None:
            return None
        copied.set("this", left)
        copied.set("expression", right)
        return copied
    return None


def _empty_query_signature(sql: str | None) -> str | None:
    tree = _parse_query(sql)
    if tree is None:
        return None
    stripped = _strip_outer_projection(tree)
    if stripped is None:
        return None
    # Parentheses around an already parsed expression do not carry additional
    # semantics.  Removing only those AST wrappers makes harmless rewrites
    # comparable without doing value-based or textual query matching.
    stripped = stripped.transform(
        lambda node: node.this if isinstance(node, exp.Paren) else node,
    )
    return _normalized_expression_sql(stripped)


def _group_key_identities(sql: str | None) -> tuple[str, ...] | None:
    select = _select_expression(sql)
    if select is None:
        return None
    group = select.args.get("group")
    if group is None:
        return ()
    identities: list[str] = []
    for expression in group.expressions:
        identity = _projection_identity(expression)
        identities.append(identity or _normalized_expression_sql(expression))
    # GROUP BY key order does not change row grain.  Sorting the bounded key
    # identities avoids rejecting an equivalent key permutation while keeping
    # duplicate keys visible to the comparison.
    return tuple(sorted(identities))


def _top_level_select(sql: str | None) -> exp.Select | None:
    tree = _parse_query(sql)
    return tree if isinstance(tree, exp.Select) else None


def _join_kind(join: exp.Join) -> str:
    side = str(join.side or "").casefold()
    kind = str(join.kind or "").casefold()
    if side:
        return side
    return kind or "inner"


def _source_and_join_descriptors(
    sql: str | None,
    schema: SemanticSchemaMetadata | None,
) -> tuple[str | None, tuple[tuple[str, str, str, tuple[object, ...] | None], ...]] | None:
    select = _top_level_select(sql)
    if select is None:
        return None
    aliases = _table_aliases(select, schema)
    from_clause = select.args.get("from_")
    if from_clause is None or not isinstance(from_clause.this, exp.Table):
        return None
    base_table = from_clause.this
    base_relation = aliases.get(_normalize_relation_identity(base_table.alias or base_table.name))
    if base_relation is None:
        base_relation = _relation_key_from_table(base_table)
    descriptors: list[tuple[str, str, str, tuple[object, ...] | None]] = []
    left_relation = base_relation
    for join in select.args.get("joins") or []:
        if not isinstance(join, exp.Join) or not isinstance(join.this, exp.Table):
            return None
        target_table = join.this
        target_relation = aliases.get(
            _normalize_relation_identity(target_table.alias or target_table.name)
        )
        if target_relation is None:
            target_relation = _relation_key_from_table(target_table)
        on = join.args.get("on")
        on_signature = (
            _expression_signature(
                on,
                aliases=aliases,
                schema=schema,
                literals_available=True,
            )
            if isinstance(on, exp.Expression)
            else None
        )
        join_kind = _join_kind(join)
        edge_left = left_relation
        edge_right = target_relation
        if join_kind == "inner" and on_signature is not None:
            relation_identities = sorted(
                identity.rsplit(".", 1)[0]
                for identity in _signature_column_identities(on_signature)
            )
            if len(set(relation_identities)) == 2:
                edge_left, edge_right = tuple(dict.fromkeys(relation_identities))
        descriptors.append((edge_left, edge_right, join_kind, on_signature))
        left_relation = target_relation
    return base_relation, tuple(descriptors)


def _join_edge_matches(
    left: str,
    right: str,
    on: tuple[object, ...] | None,
    candidate: tuple[str, str, str, tuple[object, ...] | None],
) -> bool:
    candidate_left, candidate_right, _kind, candidate_on = candidate
    same_direction = candidate_left == left and candidate_right == right
    reverse_direction = candidate_left == right and candidate_right == left
    return (same_direction or reverse_direction) and candidate_on == on


def _foreign_key_guarantees_match(
    *,
    source_relation: str,
    target_relation: str,
    on_signature: tuple[object, ...] | None,
    schema: SemanticSchemaMetadata | None,
) -> bool:
    """Prove all source rows match a target using only a non-null FK."""

    if schema is None:
        return False
    source = _schema_relation(schema, source_relation)
    if source is None:
        return False
    for foreign_key in source.foreign_keys:
        if _normalize_relation_identity(
            foreign_key.target_relation
        ) != _normalize_relation_identity(target_relation):
            continue
        source_columns = {
            _normalize_field_identity(column) for column in foreign_key.source_columns
        }
        not_null = {_normalize_field_identity(column) for column in source.not_null_columns}
        target = _schema_relation(schema, foreign_key.target_relation)
        if target is None:
            continue
        target_keys = _relation_keys(target)
        if tuple(
            _normalize_field_identity(column) for column in foreign_key.target_columns
        ) not in {tuple(item.rsplit(".", 1)[-1] for item in key) for key in target_keys}:
            continue
        if not source_columns <= not_null:
            continue
        expected_pairs = {
            frozenset(
                {
                    f"{_normalize_relation_identity(source_relation)}.{_normalize_field_identity(source_column)}",
                    f"{_normalize_relation_identity(target_relation)}.{_normalize_field_identity(target_column)}",
                }
            )
            for source_column, target_column in zip(
                foreign_key.source_columns,
                foreign_key.target_columns,
                strict=True,
            )
        }
        if on_signature is not None and _equality_pairs(on_signature) == expected_pairs:
            return True
    return False


def _equality_pairs(signature: object) -> set[frozenset[str]]:
    """Extract only direct equality column pairs from a canonical signature."""

    if not isinstance(signature, tuple) or not signature:
        return set()
    if signature[0] == "eq" and len(signature) == 2 and isinstance(signature[1], tuple):
        columns = [
            item[1]
            for item in signature[1]
            if isinstance(item, tuple) and len(item) == 2 and item[0] == "column"
        ]
        return {frozenset(columns)} if len(columns) == 2 else set()
    if signature[0] == "and" and len(signature) == 2 and isinstance(signature[1], tuple):
        pairs: set[frozenset[str]] = set()
        for child in signature[1]:
            pairs.update(_equality_pairs(child))
        return pairs
    return set()


def _signature_column_identities(signature: object) -> set[str]:
    if isinstance(signature, tuple):
        if len(signature) == 2 and signature[0] == "column":
            return {str(signature[1])}
        identities: set[str] = set()
        for item in signature:
            identities.update(_signature_column_identities(item))
        return identities
    return set()


def _target_usage_is_redundant(
    select: exp.Select,
    *,
    target_relation: str,
    target_key_columns: frozenset[str],
    aliases: dict[str, str],
) -> bool:
    """Allow only a target-key COUNT(DISTINCT ...) use for redundant joins."""

    expressions: list[exp.Expression] = list(select.expressions)
    group = select.args.get("group")
    if isinstance(group, exp.Group):
        expressions.extend(group.expressions)
    order = select.args.get("order")
    if isinstance(order, exp.Order):
        expressions.extend(item.this for item in order.expressions)
    for key in ("where", "having"):
        clause = select.args.get(key)
        if isinstance(clause, exp.Expression):
            expressions.append(clause)
    target_columns = [
        column
        for expression in expressions
        for column in expression.find_all(exp.Column)
        if aliases.get(_normalize_relation_identity(column.table or ""))
        == _normalize_relation_identity(target_relation)
    ]
    if not target_columns:
        return True
    allowed = set(target_key_columns)
    for column in target_columns:
        if _normalize_field_identity(column.name) not in allowed:
            return False
        parent = column.parent
        if not isinstance(parent, exp.Distinct) or not isinstance(parent.parent, exp.Count):
            return False
    return True


def _prove_redundant_inner_join(
    sql: str | None,
    descriptor: tuple[str, str, str, tuple[object, ...] | None],
    schema: SemanticSchemaMetadata | None,
) -> bool:
    select = _top_level_select(sql)
    if select is None or descriptor[2] != "inner" or schema is None:
        return False
    aliases = _table_aliases(select, schema)
    left_relation, right_relation, _kind, on_signature = descriptor
    for source_relation, target_relation in (
        (left_relation, right_relation),
        (right_relation, left_relation),
    ):
        target = _schema_relation(schema, target_relation)
        if target is None or not target.primary_key:
            continue
        if not _target_usage_is_redundant(
            select,
            target_relation=target_relation,
            target_key_columns=frozenset(target.primary_key),
            aliases=aliases,
        ):
            continue
        if _foreign_key_guarantees_match(
            source_relation=source_relation,
            target_relation=target_relation,
            on_signature=on_signature,
            schema=schema,
        ):
            return True
    return False


def _join_proof(
    expected_sql: str | None,
    actual_sql: str | None,
    schema: SemanticSchemaMetadata | None,
) -> SemanticProof:
    expected = _source_and_join_descriptors(expected_sql, schema)
    actual = _source_and_join_descriptors(actual_sql, schema)
    if expected is None or actual is None:
        return SemanticProof(status=SemanticProofStatus.UNKNOWN, reasons=("insufficient_lineage",))
    if expected == actual:
        return SemanticProof(
            status=SemanticProofStatus.PROVEN_EQUIVALENT, reasons=("join_semantics_match",)
        )

    _expected_base, expected_joins = expected
    _actual_base, actual_joins = actual
    # A LEFT JOIN and INNER JOIN over the same edge are different unless a
    # trusted non-null FK proves that every left row has a target.
    for expected_join in expected_joins:
        for actual_join in actual_joins:
            if expected_join[2] == actual_join[2]:
                continue
            if not _join_edge_matches(
                expected_join[0], expected_join[1], expected_join[3], actual_join
            ):
                continue
            if {expected_join[2], actual_join[2]} == {"left", "inner"}:
                if expected_join[3] is None or actual_join[3] is None:
                    continue
                if _foreign_key_guarantees_match(
                    source_relation=expected_join[0],
                    target_relation=expected_join[1],
                    on_signature=expected_join[3],
                    schema=schema,
                ):
                    continue
                return SemanticProof(
                    status=SemanticProofStatus.PROVEN_DIFFERENT,
                    reasons=("different_join_type",),
                )

    expected_by_edge = {(item[0], item[1], item[3]): item for item in expected_joins}
    actual_by_edge = {(item[0], item[1], item[3]): item for item in actual_joins}
    missing_from_actual = [
        item for key, item in expected_by_edge.items() if key not in actual_by_edge
    ]
    missing_from_expected = [
        item for key, item in actual_by_edge.items() if key not in expected_by_edge
    ]
    if (
        len(missing_from_actual) == 1
        and not missing_from_expected
        and _prove_redundant_inner_join(expected_sql, missing_from_actual[0], schema)
    ):
        return SemanticProof(
            status=SemanticProofStatus.PROVEN_EQUIVALENT,
            reasons=("equivalent_by_foreign_key_join_redundancy",),
        )
    if (
        len(missing_from_expected) == 1
        and not missing_from_actual
        and _prove_redundant_inner_join(actual_sql, missing_from_expected[0], schema)
    ):
        return SemanticProof(
            status=SemanticProofStatus.PROVEN_EQUIVALENT,
            reasons=("equivalent_by_foreign_key_join_redundancy",),
        )
    return SemanticProof(status=SemanticProofStatus.UNKNOWN, reasons=("join_equivalence_unproven",))


def _source_relation_proof(
    expected_sql: str | None,
    actual_sql: str | None,
    schema: SemanticSchemaMetadata | None,
) -> SemanticProof:
    expected = _source_and_join_descriptors(expected_sql, schema)
    actual = _source_and_join_descriptors(actual_sql, schema)
    if expected is None or actual is None:
        return SemanticProof(status=SemanticProofStatus.UNKNOWN, reasons=("insufficient_lineage",))
    expected_relations = {
        expected[0],
        *(item[0] for item in expected[1]),
        *(item[1] for item in expected[1]),
    }
    actual_relations = {
        actual[0],
        *(item[0] for item in actual[1]),
        *(item[1] for item in actual[1]),
    }
    if expected_relations == actual_relations:
        return SemanticProof(
            status=SemanticProofStatus.PROVEN_EQUIVALENT,
            reasons=("source_relations_match",),
        )
    if (
        actual_relations < expected_relations or expected_relations < actual_relations
    ) and _join_proof(
        expected_sql, actual_sql, schema
    ).status is SemanticProofStatus.PROVEN_EQUIVALENT:
        return SemanticProof(
            status=SemanticProofStatus.PROVEN_EQUIVALENT,
            reasons=("source_relation_redundancy_proven",),
        )
    return SemanticProof(
        status=SemanticProofStatus.UNKNOWN, reasons=("source_relation_equivalence_unproven",)
    )


def _predicate_expression(select: exp.Select | None) -> exp.Expression | None:
    if select is None:
        return None
    where = select.args.get("where")
    having = select.args.get("having")
    expressions = [
        item.this if isinstance(item, exp.Where) else item
        for item in (where, having)
        if isinstance(item, exp.Expression)
    ]
    if not expressions:
        return None
    if len(expressions) == 1:
        return expressions[0]
    return exp.and_(*expressions)


def _predicate_proof(
    expected_sql: str | None,
    actual_sql: str | None,
    schema: SemanticSchemaMetadata | None,
    *,
    actual_literals_available: bool,
) -> SemanticProof:
    expected_select = _top_level_select(expected_sql)
    actual_select = _top_level_select(actual_sql)
    if expected_select is None or actual_select is None:
        return SemanticProof(status=SemanticProofStatus.UNKNOWN, reasons=("insufficient_lineage",))
    expected_aliases = _table_aliases(expected_select, schema)
    actual_aliases = _table_aliases(actual_select, schema)
    expected_lineage = _derived_alias_lineage(expected_sql, schema)
    actual_lineage = _derived_alias_lineage(actual_sql, schema)
    expected_expression = _predicate_expression(expected_select)
    actual_expression = _predicate_expression(actual_select)
    if expected_expression is None and actual_expression is None:
        return SemanticProof(
            status=SemanticProofStatus.PROVEN_EQUIVALENT, reasons=("no_predicate",)
        )
    if (expected_expression is None) != (actual_expression is None):
        present_expression = expected_expression or actual_expression
        present_aliases = expected_aliases if expected_expression is not None else actual_aliases
        present_lineage = expected_lineage if expected_expression is not None else actual_lineage
        present_signature = _expression_signature(
            present_expression,
            aliases=present_aliases,
            schema=schema,
            literals_available=(
                True if expected_expression is not None else actual_literals_available
            ),
            lineage=present_lineage,
        )
        if present_signature is None or _signature_contains_unknown_literal(present_signature):
            return SemanticProof(
                status=SemanticProofStatus.UNKNOWN, reasons=("insufficient_lineage",)
            )
        return SemanticProof(
            status=SemanticProofStatus.PROVEN_DIFFERENT, reasons=("predicate_presence_diff",)
        )
    expected_signature = (
        _expression_signature(
            expected_expression,
            aliases=expected_aliases,
            schema=schema,
            literals_available=True,
            lineage=expected_lineage,
        )
        if expected_expression is not None
        else None
    )
    actual_signature = (
        _expression_signature(
            actual_expression,
            aliases=actual_aliases,
            schema=schema,
            literals_available=actual_literals_available,
            lineage=actual_lineage,
        )
        if actual_expression is not None
        else None
    )
    return _compare_signatures(
        expected_signature,
        actual_signature,
        equivalent_reason="predicate_semantics_match",
        different_reason="different_predicate",
    )


def _limit_signature(
    select: exp.Select | None,
    *,
    aliases: dict[str, str],
    schema: SemanticSchemaMetadata | None,
    literals_available: bool,
) -> tuple[object, ...] | None:
    if select is None:
        return None
    limit = select.args.get("limit")
    if limit is None:
        limit_signature: tuple[object, ...] = ("no_limit",)
    else:
        expression = limit.args.get("expression") or limit.args.get("count")
        if not isinstance(expression, exp.Expression):
            return None
        value = _expression_signature(
            expression,
            aliases=aliases,
            schema=schema,
            literals_available=literals_available,
        )
        if value is None:
            return None
        limit_signature = (limit.key.casefold(), value)
    offset = select.args.get("offset")
    if offset is None:
        offset_signature: tuple[object, ...] = ("no_offset",)
    else:
        expression = offset.args.get("expression")
        if not isinstance(expression, exp.Expression):
            return None
        value = _expression_signature(
            expression,
            aliases=aliases,
            schema=schema,
            literals_available=literals_available,
        )
        if value is None:
            return None
        offset_signature = ("offset", value)
    return ("resource_bounds", limit_signature, offset_signature)


def _order_signature(
    select: exp.Select | None,
    *,
    aliases: dict[str, str],
    schema: SemanticSchemaMetadata | None,
    literals_available: bool,
) -> tuple[tuple[object, ...], ...] | None:
    if select is None:
        return None
    order = select.args.get("order")
    if order is None:
        return ()
    signatures: list[tuple[object, ...]] = []
    for ordered in order.expressions:
        signature = _expression_signature(
            ordered.this,
            aliases=aliases,
            schema=schema,
            literals_available=literals_available,
        )
        if signature is None:
            return None
        signatures.append(
            (signature, bool(ordered.args.get("desc")), ordered.args.get("nulls_first"))
        )
    return tuple(signatures)


def _order_limit_proof(
    expected_sql: str | None,
    actual_sql: str | None,
    schema: SemanticSchemaMetadata | None,
    *,
    order_required: bool,
    actual_literals_available: bool,
) -> SemanticProof:
    expected_select = _top_level_select(expected_sql)
    actual_select = _top_level_select(actual_sql)
    if expected_select is None or actual_select is None:
        return SemanticProof(status=SemanticProofStatus.UNKNOWN, reasons=("insufficient_lineage",))
    expected_aliases = _table_aliases(expected_select, schema)
    actual_aliases = _table_aliases(actual_select, schema)
    expected_limit = _limit_signature(
        expected_select,
        aliases=expected_aliases,
        schema=schema,
        literals_available=True,
    )
    actual_limit = _limit_signature(
        actual_select,
        aliases=actual_aliases,
        schema=schema,
        literals_available=actual_literals_available,
    )
    if expected_limit is None or actual_limit is None:
        return SemanticProof(
            status=SemanticProofStatus.UNKNOWN, reasons=("limit_value_unavailable",)
        )
    no_resource_bound = ("resource_bounds", ("no_limit",), ("no_offset",))
    has_top_k = expected_limit != no_resource_bound or actual_limit != no_resource_bound
    if has_top_k:
        if expected_limit == actual_limit:
            limit_proof = SemanticProof(
                status=SemanticProofStatus.PROVEN_EQUIVALENT,
                reasons=("limit_semantics_match",),
            )
        elif _signature_contains_unknown_literal(actual_limit):
            return SemanticProof(
                status=SemanticProofStatus.UNKNOWN, reasons=("limit_value_unavailable",)
            )
        else:
            return SemanticProof(
                status=SemanticProofStatus.PROVEN_DIFFERENT, reasons=("different_limit",)
            )
    else:
        limit_proof = SemanticProof(
            status=SemanticProofStatus.PROVEN_EQUIVALENT,
            reasons=("no_limit_semantics",),
        )
    if not order_required and not has_top_k:
        return limit_proof
    expected_order = _order_signature(
        expected_select,
        aliases=expected_aliases,
        schema=schema,
        literals_available=True,
    )
    actual_order = _order_signature(
        actual_select,
        aliases=actual_aliases,
        schema=schema,
        literals_available=actual_literals_available,
    )
    order_proof = _compare_signatures(
        ("order", expected_order) if expected_order is not None else None,
        ("order", actual_order) if actual_order is not None else None,
        equivalent_reason="ordering_semantics_match",
        different_reason="different_ordering_semantics",
    )
    return _combine_proofs((limit_proof, order_proof))


def _distinct_proof(expected_sql: str | None, actual_sql: str | None) -> SemanticProof:
    _, expected_distinct = _query_shape(expected_sql)
    _, actual_distinct = _query_shape(actual_sql)
    if expected_distinct == actual_distinct:
        return SemanticProof(
            status=SemanticProofStatus.PROVEN_EQUIVALENT, reasons=("distinct_semantics_match",)
        )
    return SemanticProof(
        status=SemanticProofStatus.PROVEN_DIFFERENT,
        reasons=("distinct_semantics_differ",),
    )


def _set_operation_proof(expected_sql: str | None, actual_sql: str | None) -> SemanticProof:
    expected = _parse_query(expected_sql)
    actual = _parse_query(actual_sql)
    set_types = (exp.Union, exp.Intersect, exp.Except)
    expected_is_set = isinstance(expected, set_types)
    actual_is_set = isinstance(actual, set_types)
    if not expected_is_set and not actual_is_set:
        return SemanticProof(
            status=SemanticProofStatus.PROVEN_EQUIVALENT, reasons=("set_operation_absent",)
        )
    if type(expected) is not type(actual):
        return SemanticProof(
            status=SemanticProofStatus.PROVEN_DIFFERENT, reasons=("different_set_operation",)
        )
    if expected is None or actual is None:
        return SemanticProof(status=SemanticProofStatus.UNKNOWN, reasons=("insufficient_lineage",))
    if _normalized_expression_sql(expected) == _normalized_expression_sql(actual):
        return SemanticProof(
            status=SemanticProofStatus.PROVEN_EQUIVALENT, reasons=("set_operation_semantics_match",)
        )
    return SemanticProof(
        status=SemanticProofStatus.UNKNOWN, reasons=("set_operation_equivalence_unproven",)
    )


def _combine_proofs(proofs: Sequence[SemanticProof]) -> SemanticProof:
    reasons: list[str] = []
    for proof in proofs:
        reasons.extend(proof.reasons)
    if any(proof.status is SemanticProofStatus.PROVEN_DIFFERENT for proof in proofs):
        status = SemanticProofStatus.PROVEN_DIFFERENT
    elif all(proof.status is SemanticProofStatus.PROVEN_EQUIVALENT for proof in proofs):
        status = SemanticProofStatus.PROVEN_EQUIVALENT
    else:
        status = SemanticProofStatus.UNKNOWN
    return SemanticProof(status=status, reasons=tuple(dict.fromkeys(reasons)))


def _query_semantic_proof(
    expected_sql: str | None,
    actual_sql: str | None,
    *,
    schema: SemanticSchemaMetadata | None,
    order_required: bool,
    actual_literals_available: bool,
) -> SemanticProof:
    """Combine independent proof dimensions without treating AST inequality as failure."""

    if expected_sql is None or actual_sql is None:
        return SemanticProof(status=SemanticProofStatus.UNKNOWN, reasons=("insufficient_lineage",))
    return _combine_proofs(
        (
            _source_relation_proof(expected_sql, actual_sql, schema),
            _set_operation_proof(expected_sql, actual_sql),
            _join_proof(expected_sql, actual_sql, schema),
            _predicate_proof(
                expected_sql,
                actual_sql,
                schema,
                actual_literals_available=actual_literals_available,
            ),
            _group_proof(expected_sql, actual_sql, schema),
            _distinct_proof(expected_sql, actual_sql),
            _order_limit_proof(
                expected_sql,
                actual_sql,
                schema,
                order_required=order_required,
                actual_literals_available=actual_literals_available,
            ),
        )
    )


def _mark_shape_unassessable(shape: OutputShapeCompliance) -> OutputShapeCompliance:
    return shape.model_copy(
        update={
            "status": OutputShapeStatus.UNASSESSABLE,
            "shape_compliant": False,
            "structural_shape_compliant": shape.shape_compliant,
        }
    )


def _projection_identity(node: exp.Expression) -> str | None:
    if isinstance(node, exp.Alias):
        node = node.this
    if isinstance(node, exp.Paren):
        return _projection_identity(node.this)
    if isinstance(node, exp.Column):
        return _normalize_field_identity(node.name)
    if isinstance(node, exp.Cast):
        return _projection_identity(node.this)
    if isinstance(node, exp.Distinct):
        # sqlglot stores DISTINCT arguments in ``expressions``.  The
        # ``this`` slot is empty for forms such as COUNT(DISTINCT key), so do
        # not accidentally classify a valid projection as unassessable.
        expressions = list(node.expressions)
        if len(expressions) != 1:
            return None
        inner = _projection_identity(expressions[0])
        return f"distinct:{inner}" if inner is not None else None
    if isinstance(node, exp.AggFunc):
        function = node.key.casefold()
        argument = node.this
        if isinstance(argument, exp.Star):
            argument_identity = "*"
        elif isinstance(argument, exp.Expression):
            argument_identity = _projection_identity(argument)
        else:
            argument_identity = None
        if argument_identity is None:
            return None
        return f"aggregate:{function}:{argument_identity}"
    if isinstance(node, exp.Window):
        inner = _projection_identity(node.this)
        return f"window:{inner}" if inner is not None else None
    if isinstance(node, exp.Literal):
        return "literal"
    if isinstance(node, exp.Star):
        return None
    if isinstance(node, exp.Expression):
        return f"expression:{_normalized_expression_sql(node)}"
    return None


def projected_field_identities(
    sql: str | None,
    output_columns: Sequence[str],
) -> tuple[str, ...] | None:
    """Derive bounded projection identities from a PostgreSQL AST.

    Aliases are deliberately discarded.  If an expression cannot be mapped
    safely, ``None`` is returned instead of guessing its business meaning.
    """

    if sql is None:
        # A result column name may be an arbitrary alias.  Without the
        # validated SQL AST (or a separately trusted lineage map), treating it
        # as a business field would manufacture semantic evidence.
        return None
    select = _select_expression(sql)
    if select is None or len(select.expressions) != len(output_columns):
        return None
    identities = tuple(_projection_identity(expression) for expression in select.expressions)
    if any(identity is None for identity in identities):
        return None
    return tuple(identity for identity in identities if identity is not None)


def _query_shape(sql: str | None) -> tuple[SemanticRowGrain | None, bool]:
    if sql is None:
        return None, False
    select = _select_expression(sql)
    if select is None:
        return None, False
    has_group = bool(select.args.get("group"))

    def contains_outer_aggregate(node: exp.Expression) -> bool:
        if isinstance(node, exp.Subquery):
            return False
        if isinstance(node, exp.AggFunc):
            return True
        return any(contains_outer_aggregate(child) for child in node.iter_expressions())

    has_aggregate = any(contains_outer_aggregate(expression) for expression in select.expressions)
    if has_group:
        return SemanticRowGrain.GROUPED, bool(select.args.get("distinct"))
    if has_aggregate:
        return SemanticRowGrain.SCALAR, bool(select.args.get("distinct"))
    return SemanticRowGrain.DETAIL, bool(select.args.get("distinct"))


def _actual_shape_matches_policy(
    actual_shape: SemanticRowGrain | None,
    policy: SemanticCasePolicy,
) -> bool:
    if actual_shape is None:
        return False
    if policy.row_grain is SemanticRowGrain.ENTITY:
        # Entity results may be implemented as an explicit GROUP BY.
        return actual_shape in {SemanticRowGrain.ENTITY, SemanticRowGrain.GROUPED}
    if policy.row_grain is SemanticRowGrain.PAIR:
        # Pair is an authoring contract for one relationship between two
        # identities.  SQL lineage may expose that relationship as detail rows
        # or as a grouped customer/region result; no new equivalence theorem
        # is implied by accepting either structural shape.
        return actual_shape in {SemanticRowGrain.DETAIL, SemanticRowGrain.GROUPED}
    return actual_shape is policy.row_grain


def _stable_identity(identity: str) -> bool:
    field = _normalize_field_identity(identity)
    return field == "id" or field.endswith("_id") or field.endswith("id")


def _duplicate_entity_identity(
    expected: ExpectedQueryResult,
    identities: Sequence[str],
) -> bool:
    stable_indices = [
        index for index, identity in enumerate(identities) if _stable_identity(identity)
    ]
    display_indices = [
        index
        for index, identity in enumerate(identities)
        if not _stable_identity(identity)
        and (
            identity.endswith("_name")
            or identity.endswith("_label")
            or identity in {"region", "region_name"}
        )
    ]
    for display_index in display_indices:
        by_display: dict[object, set[object]] = {}
        for row in expected.rows:
            display = row[display_index]
            stable_values = {row[index] for index in stable_indices}
            by_display.setdefault(display, set()).update(stable_values)
        if any(len(values) > 1 for values in by_display.values()):
            return True
    return False


def build_default_semantic_policy(
    *,
    question: str,
    expected: ExpectedQueryResult,
    expected_sql: str | None,
) -> SemanticCasePolicy:
    """Build generic policy metadata without case-specific code branches."""

    identities = projected_field_identities(expected_sql, expected.columns)
    if identities is None:
        identities = tuple(_normalize_field_identity(column) for column in expected.columns)
    row_grain, _ = _query_shape(expected_sql)
    row_grain = row_grain or SemanticRowGrain.DETAIL
    normalized_question = unicodedata.normalize("NFKC", question).casefold()
    duplicate_identity = _duplicate_entity_identity(expected, identities)
    stable = {
        _normalize_field_identity(identity) for identity in identities if _stable_identity(identity)
    }
    required = [identity for identity in identities if not _stable_identity(identity)]
    explicit_identity_fields = {
        identity
        for identity in identities
        if _stable_identity(identity)
        and (
            (
                identity == "sale_id"
                and any(term in normalized_question for term in ("销售编号", "销售id", "销售号"))
            )
            or (
                identity == "customer_id"
                and any(term in normalized_question for term in ("客户编号", "客户id"))
            )
            or identity in normalized_question
        )
    }
    has_display_field = any(
        identity.endswith("_name")
        or identity.endswith("_label")
        or identity in {"region", "region_name"}
        for identity in identities
        if not _stable_identity(identity)
    )
    if explicit_identity_fields:
        required.extend(sorted(explicit_identity_fields))
    elif duplicate_identity or (
        row_grain in {SemanticRowGrain.ENTITY, SemanticRowGrain.GROUPED} and not has_display_field
    ):
        required.extend(sorted(stable))
    optional = [identity for identity in identities if identity not in required]
    identity_fields = tuple(sorted(stable))
    return SemanticCasePolicy(
        row_grain=row_grain,
        required_fields=tuple(dict.fromkeys(required)),
        optional_fields=tuple(dict.fromkeys(optional)),
        identity_fields=identity_fields,
        semantic_order_required=expected.row_order_sensitive,
    )


def _reject_duplicate_json_object_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    """Build JSON objects without allowing later keys to overwrite earlier keys."""

    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SemanticPolicyDuplicateKeyError(
                "duplicate JSON object key in semantic policy artifact"
            )
        result[key] = value
    return result


def _load_semantic_policy_payload(path: Path) -> dict[str, object]:
    """Parse a policy artifact with duplicate-key rejection at every object level."""

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_object_keys,
        )
    except SemanticPolicyDuplicateKeyError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise SemanticEvaluationError(f"invalid semantic policy artifact: {path}") from error
    if not isinstance(payload, dict):
        raise SemanticEvaluationError(f"invalid semantic policy artifact: {path}")
    return payload


def load_semantic_policy(path: Path = DEFAULT_SEMANTIC_POLICY_PATH) -> SemanticPolicyArtifact:
    try:
        payload = _load_semantic_policy_payload(path)
        return SemanticPolicyArtifact.model_validate(payload)
    except ValidationError as error:
        if "semantic schema metadata fingerprint" in str(error):
            raise SemanticEvaluationError(
                "semantic schema metadata fingerprint mismatch"
            ) from error
        raise SemanticEvaluationError(f"invalid semantic policy artifact: {path}") from error


def load_semantic_policy_v3(
    path: Path = DEFAULT_G3_SEMANTIC_POLICY_PATH,
) -> SemanticPolicyArtifactV3:
    """Load the explicit DEV-only A5.7g3 policy with strict JSON object parsing."""

    try:
        payload = _load_semantic_policy_payload(path)
        return SemanticPolicyArtifactV3.model_validate(payload)
    except ValidationError as error:
        if "semantic schema metadata fingerprint" in str(error):
            raise SemanticEvaluationError(
                "semantic schema metadata fingerprint mismatch"
            ) from error
        raise SemanticEvaluationError(f"invalid semantic policy artifact: {path}") from error


def load_versioned_semantic_policy(
    path: Path,
) -> SemanticPolicyArtifact | SemanticPolicyArtifactV3:
    """Load either immutable policy generation without silently migrating it."""

    payload = _load_semantic_policy_payload(path)
    version = payload.get("schema_version") if isinstance(payload, dict) else None
    try:
        if version == SEMANTIC_POLICY_VERSION:
            return SemanticPolicyArtifact.model_validate(payload)
        if version == SEMANTIC_POLICY_G3_VERSION:
            return SemanticPolicyArtifactV3.model_validate(payload)
    except ValidationError as error:
        raise SemanticEvaluationError(f"invalid semantic policy artifact: {path}") from error
    raise SemanticEvaluationError(f"unsupported semantic policy version: {version!r}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise SemanticEvaluationError(
            f"semantic evaluation fixture is unavailable: {path}"
        ) from error
    return digest.hexdigest()


def policy_for_case(
    artifact: SemanticPolicyArtifact | SemanticPolicyArtifactV3,
    *,
    case_id: str,
    question: str,
    expected: ExpectedQueryResult,
    expected_sql: str | None,
) -> SemanticCasePolicy:
    if isinstance(artifact, SemanticPolicyArtifactV3):
        _ensure_versioned_policy_integrity(artifact)
        explicit_v3 = artifact.cases.get(case_id)
        return (explicit_v3 or artifact.defaults).to_executable_policy()
    _ensure_policy_integrity(artifact)
    explicit = artifact.cases.get(case_id)
    # The sidecar is the review contract.  Runtime-derived policies would
    # turn the oracle SQL/result back into an implicit policy and could make
    # an unreviewed case appear assessable.
    return explicit or artifact.defaults


def _shape_compliance(
    expected_columns: Sequence[str],
    actual_columns: Sequence[str],
    expected_ids: Sequence[str],
    actual_ids: Sequence[str],
    policy: SemanticCasePolicy,
) -> OutputShapeCompliance:
    expected_set = set(expected_ids)
    actual_set = set(actual_ids)
    extra = tuple(sorted(actual_set - expected_set))
    omitted = tuple(sorted(expected_set - actual_set))
    optional = tuple(sorted(set(omitted) & policy.normalized_optional_fields()))
    exact_sequence = tuple(actual_columns) == tuple(expected_columns)
    same_identity_sequence = tuple(actual_ids) == tuple(expected_ids)
    order_difference = (
        not exact_sequence and actual_set == expected_set and not same_identity_sequence
    )
    alias_difference = not exact_sequence and same_identity_sequence
    differences: list[str] = []
    if extra:
        differences.append("extra_optional_or_unexpected_fields")
    if omitted:
        differences.append("omitted_optional_or_unexpected_fields")
    if order_difference:
        differences.append("column_order")
    if alias_difference:
        differences.append("column_alias")
    return OutputShapeCompliance(
        status=(OutputShapeStatus.COMPLIANT if exact_sequence else OutputShapeStatus.NON_COMPLIANT),
        shape_compliant=exact_sequence,
        exact_column_sequence=exact_sequence,
        structural_shape_compliant=exact_sequence,
        extra_fields=extra,
        omitted_fields=omitted,
        optional_omitted_fields=optional,
        column_order_difference=order_difference,
        alias_difference=alias_difference,
        presentation_differences=tuple(differences),
    )


def _row_matches(
    actual: Sequence[object],
    expected: Sequence[object],
    actual_indices: Sequence[int],
    expected_indices: Sequence[int],
    numeric_expected_indices: frozenset[int],
) -> bool:
    for actual_index, expected_index in zip(actual_indices, expected_indices, strict=True):
        if not _values_equal(
            actual[actual_index],
            expected[expected_index],
            numeric=expected_index in numeric_expected_indices,
        ):
            return False
    return True


def _select_projection_map(
    sql: str | None,
) -> tuple[exp.Select, dict[str, exp.Expression]] | None:
    select = _top_level_select(sql)
    if select is None:
        return None
    result: dict[str, exp.Expression] = {}
    for expression in select.expressions:
        identity = _projection_identity(expression)
        if identity is None or identity in result:
            return None
        result[identity] = expression
    return select, result


def _has_outer_join(sql: str | None) -> bool:
    tree = _parse_query(sql)
    return bool(
        tree is not None
        and any(_join_kind(join) in {"left", "right", "full"} for join in tree.find_all(exp.Join))
    )


def _single_column_signature_identity(signature: object) -> str | None:
    if (
        isinstance(signature, tuple)
        and len(signature) == 2
        and signature[0] == "column"
        and isinstance(signature[1], str)
    ):
        return signature[1]
    return None


def _schema_column_equivalence(
    sql: str | None,
    schema: SemanticSchemaMetadata | None,
    *,
    allow_schema_foreign_keys: bool = False,
) -> dict[str, str]:
    """Return only join-context-safe equality substitutions for projections."""

    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    tree = _parse_query(sql)
    if tree is not None:
        for join in tree.find_all(exp.Join):
            # Equality in an outer join only constrains matched rows.  The
            # null-extended side is not projection-equivalent to the
            # preserved side, so only an explicit INNER JOIN can contribute
            # cross-relation substitutions here.
            if _join_kind(join) != "inner":
                continue
            on = join.args.get("on")
            if isinstance(on, exp.Expression):
                select = tree if isinstance(tree, exp.Select) else _select_expression(sql)
                if select is None:
                    continue
                aliases = _table_aliases(select, schema)
                signature = _expression_signature(
                    on,
                    aliases=aliases,
                    schema=schema,
                    literals_available=True,
                )
                for pair in _equality_pairs(signature):
                    columns = tuple(pair)
                    if len(columns) == 2:
                        union(columns[0], columns[1])
    if schema is not None and allow_schema_foreign_keys:
        relation_names = (
            {_relation_key_from_table(table) for table in tree.find_all(exp.Table)}
            if tree is not None
            else set()
        )
        for relation in schema.relations:
            source_relation = _normalize_relation_identity(relation.relation)
            if source_relation not in relation_names:
                continue
            for foreign_key in relation.foreign_keys:
                target_relation = _normalize_relation_identity(foreign_key.target_relation)
                if not set(
                    _normalize_field_identity(column) for column in foreign_key.source_columns
                ) <= {_normalize_field_identity(column) for column in relation.not_null_columns}:
                    continue
                for source_column, target_column in zip(
                    foreign_key.source_columns,
                    foreign_key.target_columns,
                    strict=True,
                ):
                    union(
                        f"{source_relation}.{_normalize_field_identity(source_column)}",
                        f"{target_relation}.{_normalize_field_identity(target_column)}",
                    )
    return {key: find(key) for key in parent}


def _canonicalize_signature(signature: object, substitutions: dict[str, str]) -> object:
    if isinstance(signature, tuple):
        if len(signature) == 2 and signature[0] == "column":
            return ("column", substitutions.get(str(signature[1]), str(signature[1])))
        return tuple(_canonicalize_signature(item, substitutions) for item in signature)
    return signature


def _projection_proof(
    expected_sql: str | None,
    actual_sql: str | None,
    required: frozenset[str],
    schema: SemanticSchemaMetadata | None,
    *,
    actual_literals_available: bool,
) -> SemanticProof:
    if not required:
        return SemanticProof(
            status=SemanticProofStatus.UNKNOWN, reasons=("insufficient_semantic_policy",)
        )
    expected_map = _select_projection_map(expected_sql)
    actual_map = _select_projection_map(actual_sql)
    if expected_map is None or actual_map is None:
        return SemanticProof(
            status=SemanticProofStatus.UNKNOWN, reasons=("projection_lineage_unavailable",)
        )
    expected_select, expected_projections = expected_map
    actual_select, actual_projections = actual_map
    expected_aliases = _table_aliases(expected_select, schema)
    actual_aliases = _table_aliases(actual_select, schema)
    join_proof = _join_proof(expected_sql, actual_sql, schema)
    foreign_key_redundancy_proven = (
        join_proof.status is SemanticProofStatus.PROVEN_EQUIVALENT
        and "equivalent_by_foreign_key_join_redundancy" in join_proof.reasons
    )
    expected_substitutions = _schema_column_equivalence(
        expected_sql,
        schema,
        allow_schema_foreign_keys=foreign_key_redundancy_proven,
    )
    actual_substitutions = _schema_column_equivalence(
        actual_sql,
        schema,
        allow_schema_foreign_keys=foreign_key_redundancy_proven,
    )
    proofs: list[SemanticProof] = []
    for identity in sorted(required):
        expected_expression = expected_projections.get(identity)
        actual_expression = actual_projections.get(identity)
        if expected_expression is None or actual_expression is None:
            proofs.append(
                SemanticProof(
                    status=SemanticProofStatus.PROVEN_DIFFERENT,
                    reasons=("required_semantic_projection_missing",),
                )
            )
            continue
        expected_signature = _expression_signature(
            expected_expression,
            aliases=expected_aliases,
            schema=schema,
            literals_available=True,
        )
        actual_signature = _expression_signature(
            actual_expression,
            aliases=actual_aliases,
            schema=schema,
            literals_available=actual_literals_available,
        )
        expected_column = _single_column_signature_identity(expected_signature)
        actual_column = _single_column_signature_identity(actual_signature)
        if (
            expected_column is not None
            and actual_column is not None
            and expected_column != actual_column
            and (_has_outer_join(expected_sql) or _has_outer_join(actual_sql))
        ):
            proofs.append(
                SemanticProof(
                    status=SemanticProofStatus.UNKNOWN,
                    reasons=("outer_join_projection_equivalence_unproven",),
                )
            )
            continue
        proofs.append(
            _compare_signatures(
                _canonicalize_signature(expected_signature, expected_substitutions)
                if expected_signature is not None
                else None,
                _canonicalize_signature(actual_signature, actual_substitutions)
                if actual_signature is not None
                else None,
                equivalent_reason="required_projection_semantics_match",
                different_reason="required_projection_semantics_differ",
            )
        )
    return _combine_proofs(proofs)


def _extra_projection_proof(
    expected_sql: str | None,
    actual_sql: str | None,
    shape: OutputShapeCompliance,
    policy: SemanticCasePolicy,
    schema: SemanticSchemaMetadata | None,
) -> SemanticProof:
    if not shape.extra_fields:
        return SemanticProof(
            status=SemanticProofStatus.PROVEN_EQUIVALENT, reasons=("no_extra_projection",)
        )
    optional = policy.normalized_optional_fields()
    actual_map = _select_projection_map(actual_sql)
    if actual_map is None:
        return SemanticProof(
            status=SemanticProofStatus.UNKNOWN, reasons=("projection_lineage_unavailable",)
        )
    actual_select, actual_projections = actual_map
    group_columns = _select_group_columns(actual_sql, schema)
    closure = _functional_closure(set(group_columns or ()), schema)
    proofs: list[SemanticProof] = []
    for identity in shape.extra_fields:
        if identity in optional:
            proofs.append(
                SemanticProof(
                    status=SemanticProofStatus.PROVEN_EQUIVALENT,
                    reasons=("explicit_optional_projection",),
                )
            )
            continue
        expression = actual_projections.get(identity)
        aliases = _table_aliases(actual_select, schema)
        if isinstance(expression, (exp.Alias, exp.Paren)):
            expression = expression.this
        if isinstance(expression, exp.Column):
            column_identity = _column_identity(expression, aliases, schema)
            if column_identity is not None and column_identity in closure:
                proofs.append(
                    SemanticProof(
                        status=SemanticProofStatus.PROVEN_EQUIVALENT,
                        reasons=("extra_projection_functionally_dependent_on_group",),
                    )
                )
                continue
        proofs.append(
            SemanticProof(
                status=SemanticProofStatus.UNKNOWN, reasons=("extra_projection_safety_unproven",)
            )
        )
    return _combine_proofs(proofs)


def _comparison(
    *,
    equivalent: bool,
    assessable: bool,
    required_fields_present: bool,
    required_values_match: bool,
    row_grain_match: bool,
    identity_semantics_match: bool,
    row_set_match: bool,
    ordering_semantics_match: bool,
    actual_row_count: int,
    expected_row_count: int,
    shape: OutputShapeCompliance,
    failure_reason: str | None = None,
    proof_reasons: Sequence[str] = (),
) -> SemanticComparison:
    return SemanticComparison(
        equivalent=equivalent,
        assessable=assessable,
        required_fields_present=required_fields_present,
        required_values_match=required_values_match,
        row_grain_match=row_grain_match,
        identity_semantics_match=identity_semantics_match,
        row_set_match=row_set_match,
        ordering_semantics_match=ordering_semantics_match,
        actual_row_count=actual_row_count,
        expected_row_count=expected_row_count,
        optional_extra_fields=shape.extra_fields,
        presentation_differences=shape.presentation_differences,
        proof_reasons=tuple(dict.fromkeys(proof_reasons)),
        failure_reason=failure_reason,
        shape=shape,
    )


def _query_structure_matches(expected_sql: str | None, actual_sql: str | None) -> bool | None:
    """Compatibility helper for the retired pre-proof implementation only."""

    proof = _query_semantic_proof(
        expected_sql,
        actual_sql,
        schema=None,
        order_required=False,
        actual_literals_available=True,
    )
    if proof.status is SemanticProofStatus.PROVEN_EQUIVALENT:
        return True
    if proof.status is SemanticProofStatus.PROVEN_DIFFERENT:
        return False
    return None


def _legacy_compare_semantic_result(
    actual: QueryExecutionResult,
    expected: ExpectedQueryResult,
    *,
    policy: SemanticCasePolicy,
    actual_sql: str | None = None,
    expected_sql: str | None = None,
) -> SemanticComparison:
    """Compare business semantics while retaining a separate shape result.

    The function is conservative: missing rows, unsafe lineage, or an
    unparseable projection yield ``assessable=False`` rather than a guess.
    """

    expected_ids = projected_field_identities(expected_sql, expected.columns)
    actual_columns = tuple(column.name for column in actual.columns)
    actual_ids = projected_field_identities(actual_sql, actual_columns)
    if (
        expected_ids is None
        or actual_ids is None
        or len(set(expected_ids)) != len(expected_ids)
        or len(set(actual_ids)) != len(actual_ids)
    ):
        shape = OutputShapeCompliance(
            status=OutputShapeStatus.UNASSESSABLE,
            shape_compliant=False,
            exact_column_sequence=actual_columns == tuple(expected.columns),
            structural_shape_compliant=None,
            presentation_differences=("projection_lineage_unavailable",),
        )
        return SemanticComparison(
            equivalent=False,
            assessable=False,
            required_fields_present=False,
            required_values_match=False,
            row_grain_match=False,
            identity_semantics_match=False,
            row_set_match=False,
            ordering_semantics_match=False,
            actual_row_count=len(actual.rows),
            expected_row_count=len(expected.rows),
            failure_reason="unassessable: projection lineage is unavailable",
            shape=shape,
        )

    shape = _shape_compliance(
        expected.columns,
        actual_columns,
        expected_ids,
        actual_ids,
        policy,
    )
    required = policy.normalized_required_fields()
    expected_index = {identity: index for index, identity in enumerate(expected_ids)}
    actual_index = {identity: index for index, identity in enumerate(actual_ids)}
    required_present = required <= set(expected_index) and required <= set(actual_index)
    _, expected_distinct = _query_shape(expected_sql)
    actual_shape, actual_distinct = _query_shape(actual_sql)
    query_structure_match = _query_structure_matches(expected_sql, actual_sql)
    if query_structure_match is None:
        return SemanticComparison(
            equivalent=False,
            assessable=False,
            required_fields_present=required_present,
            required_values_match=False,
            row_grain_match=False,
            identity_semantics_match=False,
            row_set_match=False,
            ordering_semantics_match=False,
            actual_row_count=len(actual.rows),
            expected_row_count=len(expected.rows),
            optional_extra_fields=shape.extra_fields,
            presentation_differences=shape.presentation_differences,
            failure_reason="unassessable: query structure is unavailable",
            shape=shape,
        )
    row_grain_match = _actual_shape_matches_policy(actual_shape, policy) and query_structure_match
    if policy.row_grain is SemanticRowGrain.DETAIL:
        row_grain_match = row_grain_match and not actual_distinct
    identity_fields = policy.normalized_identity_fields()
    required_identity_fields = identity_fields & required
    identity_match = (
        required_identity_fields <= set(actual_index) if required_identity_fields else True
    )
    identity_match = identity_match and expected_distinct == actual_distinct

    if actual.state is not QueryLifecycleState.SUCCEEDED:
        shape = _mark_shape_unassessable(shape)
        return SemanticComparison(
            equivalent=False,
            assessable=True,
            required_fields_present=required_present,
            required_values_match=False,
            row_grain_match=row_grain_match,
            identity_semantics_match=identity_match,
            row_set_match=False,
            ordering_semantics_match=False,
            actual_row_count=len(actual.rows),
            expected_row_count=len(expected.rows),
            optional_extra_fields=shape.extra_fields,
            presentation_differences=shape.presentation_differences,
            failure_reason="execution did not succeed",
            shape=shape,
        )

    if not required_present:
        reason = "required semantic field is missing"
        return SemanticComparison(
            equivalent=False,
            assessable=True,
            required_fields_present=False,
            required_values_match=False,
            row_grain_match=row_grain_match,
            identity_semantics_match=identity_match,
            row_set_match=False,
            ordering_semantics_match=False,
            actual_row_count=len(actual.rows),
            expected_row_count=len(expected.rows),
            optional_extra_fields=shape.extra_fields,
            presentation_differences=shape.presentation_differences,
            failure_reason=reason,
            shape=shape,
        )

    if (
        actual.truncated != expected.expected_truncated
        or (actual.truncation_reason.value if actual.truncation_reason else None)
        != expected.truncation_reason
    ):
        return SemanticComparison(
            equivalent=False,
            assessable=True,
            required_fields_present=True,
            required_values_match=False,
            row_grain_match=row_grain_match,
            identity_semantics_match=identity_match,
            row_set_match=False,
            ordering_semantics_match=False,
            actual_row_count=len(actual.rows),
            expected_row_count=len(expected.rows),
            optional_extra_fields=shape.extra_fields,
            presentation_differences=shape.presentation_differences,
            failure_reason="truncation metadata differs",
            shape=shape,
        )

    if not actual.rows and not expected.rows:
        expected_empty_signature = _empty_query_signature(expected_sql)
        actual_empty_signature = _empty_query_signature(actual_sql)
        if expected_empty_signature is None or actual_empty_signature is None:
            return SemanticComparison(
                equivalent=False,
                assessable=False,
                required_fields_present=True,
                required_values_match=False,
                row_grain_match=row_grain_match,
                identity_semantics_match=identity_match,
                row_set_match=False,
                ordering_semantics_match=False,
                actual_row_count=0,
                expected_row_count=0,
                optional_extra_fields=shape.extra_fields,
                presentation_differences=shape.presentation_differences,
                failure_reason="unassessable: empty-result query lineage is unavailable",
                shape=shape,
            )
        empty_structure_match = expected_empty_signature == actual_empty_signature
        if not empty_structure_match:
            return SemanticComparison(
                equivalent=False,
                assessable=True,
                required_fields_present=True,
                required_values_match=False,
                row_grain_match=row_grain_match,
                identity_semantics_match=identity_match,
                row_set_match=False,
                ordering_semantics_match=False,
                actual_row_count=0,
                expected_row_count=0,
                optional_extra_fields=shape.extra_fields,
                presentation_differences=shape.presentation_differences,
                failure_reason="empty-result query semantics differ",
                shape=shape,
            )

    required_order = tuple(identity for identity in expected_ids if identity in required)
    expected_indices = tuple(expected_index[identity] for identity in required_order)
    actual_indices = tuple(actual_index[identity] for identity in required_order)
    numeric_indices = frozenset(expected.numeric_columns)
    row_count_match = len(actual.rows) == len(expected.rows)
    if not row_count_match:
        row_set_match = False
        ordering_match = False
    elif policy.semantic_order_required:
        row_set_match = all(
            _row_matches(
                actual_row,
                expected_row,
                actual_indices,
                expected_indices,
                numeric_indices,
            )
            for actual_row, expected_row in zip(actual.rows, expected.rows, strict=True)
        )
        ordering_match = row_set_match
    else:
        remaining = list(expected.rows)
        row_set_match = True
        for actual_row in actual.rows:
            match = next(
                (
                    index
                    for index, expected_row in enumerate(remaining)
                    if _row_matches(
                        actual_row,
                        expected_row,
                        actual_indices,
                        expected_indices,
                        numeric_indices,
                    )
                ),
                None,
            )
            if match is None:
                row_set_match = False
                break
            remaining.pop(match)
        row_set_match = row_set_match and not remaining
        ordering_match = True

    values_match = row_set_match
    extra_fields_allowed = not shape.extra_fields or (
        policy.allow_safe_extra_fields and query_structure_match
    )
    equivalent = all(
        (
            required_present,
            values_match,
            row_grain_match,
            identity_match,
            row_set_match,
            ordering_match,
            extra_fields_allowed,
        )
    )
    reason = None if equivalent else "semantic result differs"
    if not row_grain_match:
        reason = "row grain differs"
    elif not identity_match:
        reason = "stable identity semantics differ"
    elif not row_set_match:
        reason = "required row values or multiplicity differ"
    elif not ordering_match:
        reason = "semantic row ordering differs"
    elif not extra_fields_allowed:
        reason = "extra output fields are not allowed by policy"
    return SemanticComparison(
        equivalent=equivalent,
        assessable=True,
        required_fields_present=True,
        required_values_match=values_match,
        row_grain_match=row_grain_match,
        identity_semantics_match=identity_match,
        row_set_match=row_set_match,
        ordering_semantics_match=ordering_match,
        actual_row_count=len(actual.rows),
        expected_row_count=len(expected.rows),
        optional_extra_fields=shape.extra_fields,
        presentation_differences=shape.presentation_differences,
        failure_reason=reason,
        shape=shape,
    )


def _legacy_structural_artifact_assessment(
    *,
    record: dict[str, object],
    expected: ExpectedQueryResult,
    expected_sql: str,
    policy: SemanticCasePolicy,
) -> SemanticComparison:
    """Assess a safe provider artifact without inventing missing result rows."""

    actual_sql = record.get("redacted_sql")
    actual_sql = actual_sql if isinstance(actual_sql, str) else None
    expected_ids = projected_field_identities(expected_sql, expected.columns)
    actual_select = _select_expression(actual_sql) if actual_sql else None
    if actual_sql is None or expected_ids is None or actual_select is None:
        shape = OutputShapeCompliance(
            status=OutputShapeStatus.UNASSESSABLE,
            shape_compliant=False,
            exact_column_sequence=False,
            structural_shape_compliant=None,
            presentation_differences=("historical artifact omits materialized rows or lineage",),
        )
        return SemanticComparison(
            equivalent=False,
            assessable=False,
            required_fields_present=False,
            required_values_match=False,
            row_grain_match=False,
            identity_semantics_match=False,
            row_set_match=False,
            ordering_semantics_match=False,
            actual_row_count=int(record.get("row_count") or 0),
            expected_row_count=len(expected.rows),
            failure_reason="unassessable: historical artifact contains no materialized rows",
            shape=shape,
        )

    # Result columns are not persisted in v2 provider artifacts.  Use the
    # SELECT aliases/names from the redacted SQL for shape-only diagnostics.
    output_names = tuple(
        expression.alias_or_name or expression.name for expression in actual_select.expressions
    )
    actual_ids = projected_field_identities(actual_sql, output_names)
    if actual_ids is None:
        return _legacy_structural_artifact_assessment(
            record={**record, "redacted_sql": None},
            expected=expected,
            expected_sql=expected_sql,
            policy=policy,
        )
    shape = _shape_compliance(expected.columns, output_names, expected_ids, actual_ids, policy)
    required = policy.normalized_required_fields()
    required_present = required <= set(actual_ids)
    _, expected_distinct = _query_shape(expected_sql)
    actual_shape, actual_distinct = _query_shape(actual_sql)
    query_structure_match = _query_structure_matches(expected_sql, actual_sql)
    grain_match = (
        _actual_shape_matches_policy(actual_shape, policy) and query_structure_match is True
    )
    if policy.row_grain is SemanticRowGrain.DETAIL:
        grain_match = grain_match and not actual_distinct
    required_identity_fields = policy.normalized_identity_fields() & required
    identity_match = required_identity_fields <= set(actual_ids)
    identity_match = identity_match and expected_distinct == actual_distinct
    if record.get("result_status") != QueryLifecycleState.SUCCEEDED.value:
        shape = _mark_shape_unassessable(shape)
        return SemanticComparison(
            equivalent=False,
            assessable=False,
            required_fields_present=required_present,
            required_values_match=False,
            row_grain_match=grain_match,
            identity_semantics_match=identity_match,
            row_set_match=False,
            ordering_semantics_match=False,
            actual_row_count=int(record.get("row_count") or 0),
            expected_row_count=len(expected.rows),
            optional_extra_fields=shape.extra_fields,
            presentation_differences=shape.presentation_differences,
            failure_reason="unassessable: execution did not produce a result",
            shape=shape,
        )
    if query_structure_match is None:
        return SemanticComparison(
            equivalent=False,
            assessable=False,
            required_fields_present=required_present,
            required_values_match=False,
            row_grain_match=False,
            identity_semantics_match=False,
            row_set_match=False,
            ordering_semantics_match=False,
            actual_row_count=int(record.get("row_count") or 0),
            expected_row_count=len(expected.rows),
            optional_extra_fields=shape.extra_fields,
            presentation_differences=shape.presentation_differences,
            failure_reason="unassessable: query structure is unavailable",
            shape=shape,
        )
    if not required_present:
        return SemanticComparison(
            equivalent=False,
            assessable=True,
            required_fields_present=False,
            required_values_match=False,
            row_grain_match=grain_match,
            identity_semantics_match=identity_match,
            row_set_match=False,
            ordering_semantics_match=False,
            actual_row_count=int(record.get("row_count") or 0),
            expected_row_count=len(expected.rows),
            optional_extra_fields=shape.extra_fields,
            presentation_differences=shape.presentation_differences,
            failure_reason="required semantic projection is absent",
            shape=shape,
        )
    if not grain_match or not identity_match:
        return SemanticComparison(
            equivalent=False,
            assessable=True,
            required_fields_present=True,
            required_values_match=False,
            row_grain_match=grain_match,
            identity_semantics_match=identity_match,
            row_set_match=False,
            ordering_semantics_match=False,
            actual_row_count=int(record.get("row_count") or 0),
            expected_row_count=len(expected.rows),
            optional_extra_fields=shape.extra_fields,
            presentation_differences=shape.presentation_differences,
            failure_reason="structural row-grain or identity mismatch",
            shape=shape,
        )
    if int(record.get("row_count") or 0) != len(expected.rows):
        return SemanticComparison(
            equivalent=False,
            assessable=True,
            required_fields_present=True,
            required_values_match=False,
            row_grain_match=True,
            identity_semantics_match=True,
            row_set_match=False,
            ordering_semantics_match=False,
            actual_row_count=int(record.get("row_count") or 0),
            expected_row_count=len(expected.rows),
            optional_extra_fields=shape.extra_fields,
            presentation_differences=shape.presentation_differences,
            failure_reason="row count or multiplicity differs",
            shape=shape,
        )
    return SemanticComparison(
        equivalent=False,
        assessable=False,
        required_fields_present=True,
        required_values_match=False,
        row_grain_match=True,
        identity_semantics_match=True,
        row_set_match=False,
        ordering_semantics_match=False,
        actual_row_count=int(record.get("row_count") or 0),
        expected_row_count=len(expected.rows),
        optional_extra_fields=shape.extra_fields,
        presentation_differences=shape.presentation_differences,
        failure_reason="unassessable: historical artifact contains no materialized rows",
        shape=shape,
    )


# The implementations below are the public comparison paths.  The earlier
# helper-shaped code is retained only as local history while the proof-based
# paths are developed; these definitions intentionally shadow it so there is
# one conservative production-facing diagnostic behavior.
def compare_semantic_result(
    actual: QueryExecutionResult,
    expected: ExpectedQueryResult,
    *,
    policy: SemanticCasePolicy,
    actual_sql: str | None = None,
    expected_sql: str | None = None,
    schema: SemanticSchemaMetadata | None = None,
) -> SemanticComparison:
    """Compare result semantics only when every required dimension is proven."""

    if schema is not None:
        _ensure_schema_metadata_integrity(schema)

    expected_columns = tuple(expected.columns)
    actual_columns = tuple(column.name for column in actual.columns)
    expected_ids = projected_field_identities(expected_sql, expected_columns)
    actual_ids = projected_field_identities(actual_sql, actual_columns)
    if (
        expected_ids is None
        or actual_ids is None
        or len(set(expected_ids)) != len(expected_ids)
        or len(set(actual_ids)) != len(actual_ids)
    ):
        shape = OutputShapeCompliance(
            status=OutputShapeStatus.UNASSESSABLE,
            shape_compliant=False,
            exact_column_sequence=actual_columns == expected_columns,
            structural_shape_compliant=None,
            presentation_differences=("projection_lineage_unavailable",),
        )
        return _comparison(
            equivalent=False,
            assessable=False,
            required_fields_present=False,
            required_values_match=False,
            row_grain_match=False,
            identity_semantics_match=False,
            row_set_match=False,
            ordering_semantics_match=False,
            actual_row_count=len(actual.rows),
            expected_row_count=len(expected.rows),
            shape=shape,
            failure_reason="unassessable: projection lineage is unavailable",
            proof_reasons=("projection_lineage_unavailable",),
        )

    shape = _shape_compliance(expected_columns, actual_columns, expected_ids, actual_ids, policy)
    required = policy.normalized_required_fields()
    required_present = required <= set(expected_ids) and required <= set(actual_ids)
    if not required:
        return _comparison(
            equivalent=False,
            assessable=False,
            required_fields_present=required_present,
            required_values_match=False,
            row_grain_match=False,
            identity_semantics_match=False,
            row_set_match=False,
            ordering_semantics_match=False,
            actual_row_count=len(actual.rows),
            expected_row_count=len(expected.rows),
            shape=shape,
            failure_reason="unassessable: semantic policy has no required fields",
            proof_reasons=("insufficient_semantic_policy",),
        )

    _, expected_distinct = _query_shape(expected_sql)
    actual_shape, actual_distinct = _query_shape(actual_sql)
    group_proof = _group_proof(expected_sql, actual_sql, schema)
    query_proof = _query_semantic_proof(
        expected_sql,
        actual_sql,
        schema=schema,
        order_required=policy.semantic_order_required,
        actual_literals_available=True,
    )
    projection_proof = _projection_proof(
        expected_sql,
        actual_sql,
        required,
        schema,
        actual_literals_available=True,
    )
    extra_proof = _extra_projection_proof(expected_sql, actual_sql, shape, policy, schema)
    distinct_proof = _distinct_proof(expected_sql, actual_sql)
    identity_fields = policy.normalized_identity_fields()
    required_identity_fields = identity_fields & required
    identity_present = (
        required_identity_fields <= set(actual_ids) if required_identity_fields else True
    )
    identity_proof = SemanticProof(
        status=(
            SemanticProofStatus.PROVEN_EQUIVALENT
            if identity_present
            else SemanticProofStatus.PROVEN_DIFFERENT
        ),
        reasons=("identity_fields_present",) if identity_present else ("identity_field_missing",),
    )
    if expected_distinct != actual_distinct:
        identity_proof = SemanticProof(
            status=SemanticProofStatus.PROVEN_DIFFERENT,
            reasons=("distinct_semantics_differ",),
        )
    row_grain_match = (
        actual_shape is not None
        and _actual_shape_matches_policy(actual_shape, policy)
        and group_proof.status is SemanticProofStatus.PROVEN_EQUIVALENT
    )
    if policy.row_grain is SemanticRowGrain.DETAIL and actual_distinct:
        row_grain_match = False

    if actual.state is not QueryLifecycleState.SUCCEEDED:
        failure_shape = _mark_shape_unassessable(shape)
        return _comparison(
            equivalent=False,
            assessable=True,
            required_fields_present=required_present,
            required_values_match=False,
            row_grain_match=row_grain_match,
            identity_semantics_match=identity_proof.status is SemanticProofStatus.PROVEN_EQUIVALENT,
            row_set_match=False,
            ordering_semantics_match=False,
            actual_row_count=len(actual.rows),
            expected_row_count=len(expected.rows),
            shape=failure_shape,
            failure_reason="execution did not succeed",
            proof_reasons=query_proof.reasons,
        )

    if not required_present or projection_proof.status is SemanticProofStatus.PROVEN_DIFFERENT:
        return _comparison(
            equivalent=False,
            assessable=True,
            required_fields_present=required_present,
            required_values_match=False,
            row_grain_match=row_grain_match,
            identity_semantics_match=identity_proof.status is SemanticProofStatus.PROVEN_EQUIVALENT,
            row_set_match=False,
            ordering_semantics_match=False,
            actual_row_count=len(actual.rows),
            expected_row_count=len(expected.rows),
            shape=shape,
            failure_reason=(
                "required semantic field is missing"
                if not required_present
                else "required semantic projection differs"
            ),
            proof_reasons=query_proof.reasons + projection_proof.reasons,
        )

    if (
        actual.truncated != expected.expected_truncated
        or (actual.truncation_reason.value if actual.truncation_reason else None)
        != expected.truncation_reason
    ):
        return _comparison(
            equivalent=False,
            assessable=True,
            required_fields_present=True,
            required_values_match=False,
            row_grain_match=row_grain_match,
            identity_semantics_match=identity_proof.status is SemanticProofStatus.PROVEN_EQUIVALENT,
            row_set_match=False,
            ordering_semantics_match=False,
            actual_row_count=len(actual.rows),
            expected_row_count=len(expected.rows),
            shape=shape,
            failure_reason="truncation metadata differs",
            proof_reasons=("truncation_metadata_differs",),
        )

    ordering_proof = _order_limit_proof(
        expected_sql,
        actual_sql,
        schema,
        order_required=policy.semantic_order_required,
        actual_literals_available=True,
    )
    required_order = tuple(identity for identity in expected_ids if identity in required)
    expected_index = {identity: index for index, identity in enumerate(expected_ids)}
    actual_index = {identity: index for index, identity in enumerate(actual_ids)}
    expected_indices = tuple(expected_index[identity] for identity in required_order)
    actual_indices = tuple(actual_index[identity] for identity in required_order)
    numeric_indices = frozenset(expected.numeric_columns)
    row_set_match = False
    ordering_match = False
    if len(actual.rows) == len(expected.rows):
        if policy.semantic_order_required:
            row_set_match = all(
                _row_matches(
                    actual_row,
                    expected_row,
                    actual_indices,
                    expected_indices,
                    numeric_indices,
                )
                for actual_row, expected_row in zip(actual.rows, expected.rows, strict=True)
            )
            ordering_match = row_set_match
        else:
            remaining = list(expected.rows)
            row_set_match = True
            for actual_row in actual.rows:
                match = next(
                    (
                        index
                        for index, expected_row in enumerate(remaining)
                        if _row_matches(
                            actual_row,
                            expected_row,
                            actual_indices,
                            expected_indices,
                            numeric_indices,
                        )
                    ),
                    None,
                )
                if match is None:
                    row_set_match = False
                    break
                remaining.pop(match)
            row_set_match = row_set_match and not remaining
            ordering_match = True

    proof = _combine_proofs(
        (query_proof, projection_proof, extra_proof, distinct_proof, identity_proof, ordering_proof)
    )
    if proof.status is SemanticProofStatus.PROVEN_DIFFERENT:
        assessable = True
        equivalent = False
        failure_reason = (
            "empty-result query semantics differ"
            if not actual.rows
            and not expected.rows
            and query_proof.status is SemanticProofStatus.PROVEN_DIFFERENT
            else "proven semantic difference"
        )
    elif proof.status is SemanticProofStatus.UNKNOWN:
        assessable = False
        equivalent = False
        failure_reason = "unassessable: semantic proof is incomplete"
    else:
        assessable = True
        equivalent = row_set_match and ordering_match and len(actual.rows) == len(expected.rows)
        failure_reason = None if equivalent else "required row values or multiplicity differ"
    return _comparison(
        equivalent=equivalent,
        assessable=assessable,
        required_fields_present=True,
        required_values_match=row_set_match,
        row_grain_match=row_grain_match,
        identity_semantics_match=identity_proof.status is SemanticProofStatus.PROVEN_EQUIVALENT,
        row_set_match=row_set_match,
        ordering_semantics_match=ordering_match,
        actual_row_count=len(actual.rows),
        expected_row_count=len(expected.rows),
        shape=shape,
        failure_reason=failure_reason,
        proof_reasons=proof.reasons,
    )


def _structural_artifact_assessment(
    *,
    record: dict[str, object],
    expected: ExpectedQueryResult,
    expected_sql: str,
    policy: SemanticCasePolicy,
    schema: SemanticSchemaMetadata | None = None,
) -> SemanticComparison:
    """Assess only what a repository-safe artifact can prove independently."""

    if schema is not None:
        _ensure_schema_metadata_integrity(schema)

    actual_sql = record.get("redacted_sql")
    actual_sql = actual_sql if isinstance(actual_sql, str) else None
    expected_ids = projected_field_identities(expected_sql, expected.columns)
    actual_select = _select_expression(actual_sql) if actual_sql else None
    if actual_sql is None or expected_ids is None or actual_select is None:
        shape = OutputShapeCompliance(
            status=OutputShapeStatus.UNASSESSABLE,
            shape_compliant=False,
            exact_column_sequence=False,
            structural_shape_compliant=None,
            presentation_differences=("historical artifact omits materialized rows or lineage",),
        )
        return _comparison(
            equivalent=False,
            assessable=False,
            required_fields_present=False,
            required_values_match=False,
            row_grain_match=False,
            identity_semantics_match=False,
            row_set_match=False,
            ordering_semantics_match=False,
            actual_row_count=int(record.get("row_count") or 0),
            expected_row_count=len(expected.rows),
            shape=shape,
            failure_reason="unassessable: historical artifact contains no materialized rows",
            proof_reasons=("result_rows_unavailable",),
        )

    output_names = tuple(
        expression.alias_or_name or expression.name for expression in actual_select.expressions
    )
    actual_ids = projected_field_identities(actual_sql, output_names)
    if actual_ids is None:
        shape = OutputShapeCompliance(
            status=OutputShapeStatus.UNASSESSABLE,
            shape_compliant=False,
            exact_column_sequence=False,
            structural_shape_compliant=None,
            presentation_differences=("projection_lineage_unavailable",),
        )
        return _comparison(
            equivalent=False,
            assessable=False,
            required_fields_present=False,
            required_values_match=False,
            row_grain_match=False,
            identity_semantics_match=False,
            row_set_match=False,
            ordering_semantics_match=False,
            actual_row_count=int(record.get("row_count") or 0),
            expected_row_count=len(expected.rows),
            shape=shape,
            failure_reason="unassessable: projection lineage is unavailable",
            proof_reasons=("projection_lineage_unavailable",),
        )

    shape = _shape_compliance(expected.columns, output_names, expected_ids, actual_ids, policy)
    required = policy.normalized_required_fields()
    required_present = required <= set(actual_ids) and bool(required)
    if record.get("result_status") != QueryLifecycleState.SUCCEEDED.value:
        shape = _mark_shape_unassessable(shape)
        return _comparison(
            equivalent=False,
            assessable=False,
            required_fields_present=required_present,
            required_values_match=False,
            row_grain_match=False,
            identity_semantics_match=False,
            row_set_match=False,
            ordering_semantics_match=False,
            actual_row_count=int(record.get("row_count") or 0),
            expected_row_count=len(expected.rows),
            shape=shape,
            failure_reason="unassessable: execution did not produce a result",
            proof_reasons=("result_unavailable",),
        )
    if not required:
        return _comparison(
            equivalent=False,
            assessable=False,
            required_fields_present=required_present,
            required_values_match=False,
            row_grain_match=False,
            identity_semantics_match=False,
            row_set_match=False,
            ordering_semantics_match=False,
            actual_row_count=int(record.get("row_count") or 0),
            expected_row_count=len(expected.rows),
            shape=shape,
            failure_reason="unassessable: semantic policy has no required fields",
            proof_reasons=("insufficient_semantic_policy",),
        )

    _, expected_distinct = _query_shape(expected_sql)
    actual_shape, actual_distinct = _query_shape(actual_sql)
    query_proof = _query_semantic_proof(
        expected_sql,
        actual_sql,
        schema=schema,
        order_required=policy.semantic_order_required,
        actual_literals_available=False,
    )
    projection_proof = _projection_proof(
        expected_sql,
        actual_sql,
        required,
        schema,
        actual_literals_available=False,
    )
    extra_proof = _extra_projection_proof(expected_sql, actual_sql, shape, policy, schema)
    group_proof = _group_proof(expected_sql, actual_sql, schema)
    distinct_proof = _distinct_proof(expected_sql, actual_sql)
    identity_fields = policy.normalized_identity_fields()
    required_identity_fields = identity_fields & required
    identity_present = (
        required_identity_fields <= set(actual_ids) if required_identity_fields else True
    )
    identity_proof = SemanticProof(
        status=(
            SemanticProofStatus.PROVEN_EQUIVALENT
            if identity_present and expected_distinct == actual_distinct
            else SemanticProofStatus.PROVEN_DIFFERENT
        ),
        reasons=("identity_semantics_match",)
        if identity_present and expected_distinct == actual_distinct
        else ("identity_semantics_differ",),
    )
    row_grain_match = (
        actual_shape is not None
        and _actual_shape_matches_policy(actual_shape, policy)
        and group_proof.status is SemanticProofStatus.PROVEN_EQUIVALENT
    )
    if policy.row_grain is SemanticRowGrain.DETAIL and actual_distinct:
        row_grain_match = False
    if not row_grain_match:
        group_proof = _combine_proofs(
            (
                group_proof,
                SemanticProof(
                    status=SemanticProofStatus.PROVEN_DIFFERENT,
                    reasons=("row_grain_policy_mismatch",),
                ),
            )
        )
    if not required_present or projection_proof.status is SemanticProofStatus.PROVEN_DIFFERENT:
        return _comparison(
            equivalent=False,
            assessable=True,
            required_fields_present=required_present,
            required_values_match=False,
            row_grain_match=row_grain_match,
            identity_semantics_match=identity_proof.status is SemanticProofStatus.PROVEN_EQUIVALENT,
            row_set_match=False,
            ordering_semantics_match=False,
            actual_row_count=int(record.get("row_count") or 0),
            expected_row_count=len(expected.rows),
            shape=shape,
            failure_reason="required semantic projection is absent or differs",
            proof_reasons=query_proof.reasons + projection_proof.reasons,
        )
    if int(record.get("row_count") or 0) != len(expected.rows):
        return _comparison(
            equivalent=False,
            assessable=True,
            required_fields_present=True,
            required_values_match=False,
            row_grain_match=row_grain_match,
            identity_semantics_match=identity_proof.status is SemanticProofStatus.PROVEN_EQUIVALENT,
            row_set_match=False,
            ordering_semantics_match=False,
            actual_row_count=int(record.get("row_count") or 0),
            expected_row_count=len(expected.rows),
            shape=shape,
            failure_reason="row count or multiplicity differs",
            proof_reasons=("row_count_differs",),
        )
    proof = _combine_proofs(
        (query_proof, projection_proof, extra_proof, group_proof, distinct_proof, identity_proof)
    )
    if proof.status is SemanticProofStatus.PROVEN_DIFFERENT:
        return _comparison(
            equivalent=False,
            assessable=True,
            required_fields_present=True,
            required_values_match=False,
            row_grain_match=row_grain_match,
            identity_semantics_match=identity_proof.status is SemanticProofStatus.PROVEN_EQUIVALENT,
            row_set_match=False,
            ordering_semantics_match=False,
            actual_row_count=int(record.get("row_count") or 0),
            expected_row_count=len(expected.rows),
            shape=shape,
            failure_reason="proven semantic difference",
            proof_reasons=proof.reasons,
        )
    if proof.status is SemanticProofStatus.UNKNOWN:
        return _comparison(
            equivalent=False,
            assessable=False,
            required_fields_present=True,
            required_values_match=False,
            row_grain_match=row_grain_match,
            identity_semantics_match=identity_proof.status is SemanticProofStatus.PROVEN_EQUIVALENT,
            row_set_match=False,
            ordering_semantics_match=False,
            actual_row_count=int(record.get("row_count") or 0),
            expected_row_count=len(expected.rows),
            shape=shape,
            failure_reason="unassessable: historical artifact contains no materialized rows",
            proof_reasons=(*proof.reasons, "result_rows_unavailable"),
        )
    return _comparison(
        equivalent=True,
        assessable=True,
        required_fields_present=True,
        required_values_match=True,
        row_grain_match=True,
        identity_semantics_match=True,
        row_set_match=True,
        ordering_semantics_match=True,
        actual_row_count=int(record.get("row_count") or 0),
        expected_row_count=len(expected.rows),
        shape=shape,
        proof_reasons=(*proof.reasons, "structural_query_proof"),
    )


def evaluate_provider_artifact(
    dataset_path: Path,
    artifact_path: Path,
    *,
    policy_path: Path = DEFAULT_SEMANTIC_POLICY_PATH,
) -> SemanticRunReport:
    """Apply the diagnostic to a safe historical provider artifact.

    Historical v2 artifacts intentionally omit full result rows.  The report
    therefore uses formal passes and provable AST-level failures, and marks
    value-dependent cases unassessable instead of reconstructing oracle rows.
    """

    from knowledge_scope.evaluation.chatbi_evaluation_v2 import (
        load_chatbi_evaluation_dataset_v2,
    )

    dataset = load_chatbi_evaluation_dataset_v2(dataset_path)
    policy_artifact = load_versioned_semantic_policy(policy_path)
    _ensure_versioned_policy_integrity(policy_artifact)
    if policy_artifact.dataset_fingerprint != dataset.fingerprint:
        raise SemanticEvaluationError("semantic policy is bound to another frozen dataset")
    fixture_path = _REPOSITORY_ROOT / "tests/fixtures/chatbi_demo_v2.sql"
    actual_fixture_fingerprint = _sha256_file(fixture_path)
    if actual_fixture_fingerprint != policy_artifact.fixture_fingerprint:
        raise SemanticEvaluationError("semantic policy is bound to another fixture")
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SemanticEvaluationError(f"invalid provider artifact: {artifact_path}") from error
    if artifact.get("dataset_fingerprint") != dataset.fingerprint:
        raise SemanticEvaluationError("provider artifact dataset fingerprint mismatch")
    artifact_provenance = artifact.get("provenance")
    artifact_provenance = artifact_provenance if isinstance(artifact_provenance, dict) else {}
    if artifact.get("artifact_schema_version") == "a5.7h2-provider-run-v3":
        if not isinstance(policy_artifact, SemanticPolicyArtifactV3):
            raise SemanticEvaluationError(
                "a5.7h2 provider artifacts require the frozen g3 semantic policy"
            )
        if artifact_provenance.get("semantic_policy_version") != policy_artifact.schema_version:
            raise SemanticEvaluationError("provider artifact semantic policy version mismatch")
        if (
            artifact_provenance.get("semantic_policy_fingerprint")
            != policy_artifact.policy_fingerprint
        ):
            raise SemanticEvaluationError("provider artifact semantic policy fingerprint mismatch")
    artifact_fixture_version = artifact.get(
        "fixture_version", artifact_provenance.get("fixture_version")
    )
    if artifact_fixture_version != policy_artifact.fixture_version:
        raise SemanticEvaluationError("provider artifact fixture version mismatch")
    if artifact.get("fixture_fingerprint") != policy_artifact.fixture_fingerprint:
        raise SemanticEvaluationError("provider artifact fixture fingerprint mismatch")
    if artifact.get("fixture_schema_fingerprint") != policy_artifact.fixture_schema_fingerprint:
        raise SemanticEvaluationError("provider artifact fixture schema fingerprint mismatch")
    if artifact.get("split") != "dev":
        raise SemanticEvaluationError("semantic Run #6 analysis requires a DEV artifact")
    raw_records = artifact.get("records")
    if not isinstance(raw_records, list):
        raise SemanticEvaluationError("provider artifact records must be a list")
    records: dict[str, dict[str, object]] = {}
    for record in raw_records:
        if not isinstance(record, dict) or not isinstance(record.get("case_id"), str):
            raise SemanticEvaluationError("provider artifact contains an invalid case record")
        case_id = record["case_id"]
        if case_id in records:
            raise SemanticEvaluationError(f"provider artifact repeats case {case_id}")
        records[case_id] = record
    comparisons: dict[str, SemanticComparison] = {}
    positives = [case for case in dataset.cases if case.positive and case.split == "dev"]
    for case in positives:
        record = records.get(case.case_id)
        if record is None or case.expected_result is None or case.reference_sql is None:
            raise SemanticEvaluationError(f"provider artifact is missing case {case.case_id}")
        policy = policy_for_case(
            policy_artifact,
            case_id=case.case_id,
            question=case.question,
            expected=case.expected_result,
            expected_sql=case.reference_sql,
        )
        comparisons[case.case_id] = _structural_artifact_assessment(
            record=record,
            expected=case.expected_result,
            expected_sql=case.reference_sql,
            policy=policy,
            schema=policy_artifact.schema,
        )
    assessable = [comparison for comparison in comparisons.values() if comparison.assessable]
    shape_compliant = [
        comparison
        for comparison in comparisons.values()
        if comparison.shape.status is OutputShapeStatus.COMPLIANT
    ]
    shape_non_compliant = [
        comparison
        for comparison in comparisons.values()
        if comparison.shape.status is OutputShapeStatus.NON_COMPLIANT
    ]
    shape_unassessable = [
        comparison
        for comparison in comparisons.values()
        if comparison.shape.status is OutputShapeStatus.UNASSESSABLE
    ]
    presentation_only = [
        comparison
        for comparison in comparisons.values()
        if (
            comparison.assessable
            and comparison.equivalent is not False
            and comparison.shape.status is OutputShapeStatus.NON_COMPLIANT
        )
    ]
    semantic_failures = [
        comparison
        for comparison in comparisons.values()
        if comparison.assessable and not comparison.equivalent
    ]
    structurally_compatible_unassessable = [
        comparison
        for case_id, comparison in comparisons.items()
        if (
            not comparison.assessable
            and comparison.required_fields_present
            and records[case_id].get("result_status") == QueryLifecycleState.SUCCEEDED.value
        )
    ]
    result_unavailable_unassessable = [
        comparison
        for case_id, comparison in comparisons.items()
        if (
            not comparison.assessable
            and records[case_id].get("result_status") != QueryLifecycleState.SUCCEEDED.value
        )
    ]
    formal_count = sum(
        records[case.case_id].get("execution_equivalent") is True for case in positives
    )
    aggregate = SemanticRunAggregate(
        positive_case_count=len(positives),
        formal_execution_equivalent_count=formal_count,
        assessable_case_count=len(assessable),
        semantic_equivalent_count=sum(comparison.equivalent for comparison in assessable),
        shape_compliant_count=len(shape_compliant),
        shape_non_compliant_count=len(shape_non_compliant),
        shape_unassessable_count=len(shape_unassessable),
        presentation_only_failure_count=len(presentation_only),
        semantic_failure_count=len(semantic_failures),
        unassessable_count=sum(not comparison.assessable for comparison in comparisons.values()),
        structurally_compatible_unassessable_count=len(structurally_compatible_unassessable),
        result_unavailable_unassessable_count=len(result_unavailable_unassessable),
    )
    return SemanticRunReport(
        run_id=str(artifact.get("run_id", "unknown")),
        dataset_fingerprint=dataset.fingerprint,
        fixture_version=policy_artifact.fixture_version,
        fixture_fingerprint=policy_artifact.fixture_fingerprint,
        fixture_schema_fingerprint=policy_artifact.fixture_schema_fingerprint,
        policy_version=policy_artifact.schema_version,
        positive_case_count=len(positives),
        aggregate=aggregate,
        cases=comparisons,
    )


__all__ = [
    "DEFAULT_G3_SEMANTIC_POLICY_PATH",
    "DEFAULT_SEMANTIC_POLICY_PATH",
    "G3_DEV_POSITIVE_CASE_IDS",
    "SEMANTIC_POLICY_G3_VERSION",
    "SEMANTIC_POLICY_VERSION",
    "OutputShapeCompliance",
    "OutputShapeStatus",
    "SemanticCasePolicy",
    "SemanticCasePolicyV3",
    "SemanticCheckConstraint",
    "SemanticComparison",
    "SemanticEvaluationError",
    "SemanticField",
    "SemanticForeignKey",
    "SemanticPolicyArtifact",
    "SemanticPolicyArtifactV3",
    "SemanticPolicyAuthoringConfidence",
    "SemanticPolicyDuplicateKeyError",
    "SemanticProof",
    "SemanticProofStatus",
    "SemanticRelationMetadata",
    "SemanticRowGrain",
    "SemanticRunAggregate",
    "SemanticRunReport",
    "SemanticSchemaColumn",
    "SemanticSchemaMetadata",
    "build_default_semantic_policy",
    "compare_semantic_result",
    "evaluate_provider_artifact",
    "load_semantic_policy",
    "load_semantic_policy_v3",
    "load_versioned_semantic_policy",
    "policy_for_case",
    "projected_field_identities",
    "semantic_field_is_supported",
    "semantic_policy_v3_fingerprint",
    "semantic_schema_metadata_fingerprint",
]
