"""Bounded, evaluation-only semantic evidence for ChatBI runs.

This module deliberately has no dependency on the production Agent, prompt
builder, validator authorization path, or SQL executor.  It observes the
already validated SQL and bounded normalized result produced by an evaluation
runner and turns them into a deterministic, privacy-limited sidecar.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr, model_validator
from sqlglot import exp, parse_one
from sqlglot.errors import SqlglotError

from knowledge_scope.chatbi.agent import ChatBIResult
from knowledge_scope.chatbi.nl2sql_models import ValidatedSQL
from knowledge_scope.chatbi.schemas import QueryExecutionResult, QueryLifecycleState
from knowledge_scope.evaluation.schema_provenance import EvaluationValidatedSQL
from knowledge_scope.llm.schemas import LLMProviderInvocation

SEMANTIC_EVIDENCE_VERSION = "a5.7g2-semantic-evidence-v1"
SEMANTIC_EVIDENCE_CANONICALIZATION_VERSION = "a5.7g2-canonical-json-v1"
SEMANTIC_EVIDENCE_COLLECTOR_VERSION = "a5.7g2-collector-v1"
ROW_FINGERPRINT_VERSION = "a5.7g2-streaming-rows-v1"
SYNTHETIC_EXACT_MODE = "synthetic_exact_v1"
SENSITIVE_FUTURE_MODE = "sensitive_future"

MAX_EXPRESSION_DEPTH = 32
MAX_AST_NODES = 512
MAX_LIST_ENTRIES = 128
MAX_IDENTIFIER_BYTES = 512
MAX_RESULT_COLUMNS = 64
MAX_EXACT_ROWS = 256
MAX_EXACT_RESULT_BYTES = 64 * 1024
MAX_RECORD_BYTES = 128 * 1024
MAX_LITERAL_TEXT_BYTES = 4 * 1024


def _sql_parser_version() -> str:
    try:
        return f"sqlglot-{package_version('sqlglot')}"
    except PackageNotFoundError:
        return "sqlglot-unknown"


SQL_PARSER_VERSION = _sql_parser_version()

_SAFE_AGGREGATES = frozenset({"count", "sum", "avg", "min", "max"})
_SAFE_FUNCTIONS = frozenset(
    {
        "abs",
        "array_length",
        "btrim",
        "ceil",
        "ceiling",
        "char_length",
        "coalesce",
        "concat",
        "concat_ws",
        "date_part",
        "date_trunc",
        "floor",
        "greatest",
        "least",
        "length",
        "lower",
        "mod",
        "nullif",
        "power",
        "regexp_replace",
        "replace",
        "round",
        "rtrim",
        "split_part",
        "sqrt",
        "substring",
        "to_char",
        "trim",
        "upper",
    }
)
_BINARY_OPERATORS = {
    "Add": "+",
    "Sub": "-",
    "Mul": "*",
    "Div": "/",
    "Mod": "%",
    "EQ": "=",
    "NEQ": "<>",
    "GT": ">",
    "GTE": ">=",
    "LT": "<",
    "LTE": "<=",
    "Like": "LIKE",
    "ILike": "ILIKE",
    "DPipe": "||",
}
_PREDICATE_OPERATORS = frozenset(
    {
        "EQ",
        "NEQ",
        "GT",
        "GTE",
        "LT",
        "LTE",
        "Like",
        "ILike",
    }
)
_NUMERIC_LITERAL = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


class SemanticEvidenceError(ValueError):
    """Raised when evidence configuration or provenance is unsafe."""


class EvidenceStatus(StrEnum):
    AVAILABLE = "available"
    TRUNCATED = "truncated"
    UNAVAILABLE = "unavailable"


EvidenceCompleteness = Literal["complete", "partial", "unavailable"]
GenerationLifecycleStatus = Literal[
    "not_attempted",
    "provider_failure",
    "no_response",
    "parse_not_attempted",
    "parse_failed",
    "parsed",
]


class EvidenceMode(StrEnum):
    SYNTHETIC_EXACT_V1 = SYNTHETIC_EXACT_MODE
    SENSITIVE_FUTURE = SENSITIVE_FUTURE_MODE


class _EvidenceBase(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        hide_input_in_errors=True,
    )


class LiteralEvidence(_EvidenceBase):
    """A typed literal identity without the literal value."""

    value_type: Literal[
        "null",
        "boolean",
        "integer",
        "numeric",
        "string",
        "date",
        "time",
        "timestamp",
        "interval",
        "unknown",
    ]
    identity: StrictStr = Field(min_length=1, max_length=64)
    hash_mode: Literal["symbolic", "sha256-typed-v1", "unavailable"]


class ExpressionEvidence(_EvidenceBase):
    """Bounded tagged expression lineage; raw SQL is intentionally absent."""

    kind: Literal[
        "source_column",
        "literal",
        "aggregate",
        "derived",
        "function",
        "unknown",
    ]
    relation: StrictStr | None = Field(default=None, max_length=MAX_IDENTIFIER_BYTES)
    column: StrictStr | None = Field(default=None, max_length=MAX_IDENTIFIER_BYTES)
    function: StrictStr | None = Field(default=None, max_length=64)
    operator: StrictStr | None = Field(default=None, max_length=32)
    type_name: StrictStr | None = Field(default=None, max_length=64)
    distinct: StrictBool | None = None
    literal: LiteralEvidence | None = None
    children: tuple[ExpressionEvidence, ...] = Field(default=(), max_length=MAX_LIST_ENTRIES)
    reason: StrictStr | None = Field(default=None, max_length=96)


class ProjectionEvidence(_EvidenceBase):
    ordinal: StrictInt = Field(ge=0, le=MAX_LIST_ENTRIES - 1)
    kind: Literal["source_column", "aggregate", "derived", "literal", "unknown"]
    expression: ExpressionEvidence
    alias: StrictStr | None = Field(default=None, max_length=MAX_IDENTIFIER_BYTES)


class PredicateEvidence(_EvidenceBase):
    kind: Literal["and", "or", "not", "comparison", "is_null", "in", "between", "unknown"]
    operator: StrictStr | None = Field(default=None, max_length=32)
    expression: ExpressionEvidence | None = None
    values: tuple[ExpressionEvidence, ...] = Field(default=(), max_length=MAX_LIST_ENTRIES)
    children: tuple[PredicateEvidence, ...] = Field(default=(), max_length=MAX_LIST_ENTRIES)
    reason: StrictStr | None = Field(default=None, max_length=96)


ExpressionEvidence.model_rebuild()
PredicateEvidence.model_rebuild()


class RelationEvidence(_EvidenceBase):
    relation: StrictStr = Field(min_length=1, max_length=MAX_IDENTIFIER_BYTES)
    alias: StrictStr | None = Field(default=None, max_length=MAX_IDENTIFIER_BYTES)
    relation_kind: Literal["physical", "cte", "unknown"] = "physical"


class JoinKeyEvidence(_EvidenceBase):
    left: ExpressionEvidence
    right: ExpressionEvidence


class JoinEvidence(_EvidenceBase):
    join_type: Literal["inner", "left", "right", "full", "cross", "unknown"]
    left_relation: StrictStr | None = Field(default=None, max_length=MAX_IDENTIFIER_BYTES)
    right_relation: StrictStr | None = Field(default=None, max_length=MAX_IDENTIFIER_BYTES)
    keys: tuple[JoinKeyEvidence, ...] = Field(default=(), max_length=MAX_LIST_ENTRIES)
    predicate: PredicateEvidence | None = None


class OrderEvidence(_EvidenceBase):
    expression: ExpressionEvidence
    direction: Literal["asc", "desc"]
    nulls: Literal["first", "last", "unspecified"] = "unspecified"


class BoundEvidence(_EvidenceBase):
    kind: Literal["absent", "value", "all", "unavailable", "unsupported"]
    value: StrictInt | None = Field(default=None, ge=0, le=100_000)
    reason: StrictStr | None = Field(default=None, max_length=96)


class SetOperationEvidence(_EvidenceBase):
    operation: Literal["union", "intersect", "except"]
    all: StrictBool
    left_digest: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    right_digest: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")


class SemanticSQLEvidence(_EvidenceBase):
    status: EvidenceStatus
    completeness: EvidenceCompleteness = "complete"
    completeness_reasons: tuple[StrictStr, ...] = Field(default=(), max_length=MAX_LIST_ENTRIES)
    relations: tuple[RelationEvidence, ...] = Field(default=(), max_length=MAX_LIST_ENTRIES)
    projections: tuple[ProjectionEvidence, ...] = Field(default=(), max_length=MAX_LIST_ENTRIES)
    predicates: tuple[PredicateEvidence, ...] = Field(default=(), max_length=MAX_LIST_ENTRIES)
    joins: tuple[JoinEvidence, ...] = Field(default=(), max_length=MAX_LIST_ENTRIES)
    group_by: tuple[ExpressionEvidence, ...] = Field(default=(), max_length=MAX_LIST_ENTRIES)
    distinct: StrictBool = False
    distinct_mode: Literal["none", "distinct", "distinct_on"] = "none"
    distinct_on: tuple[ExpressionEvidence, ...] = Field(default=(), max_length=MAX_LIST_ENTRIES)
    grain_status: Literal["proven", "unknown"] = "unknown"
    order_by: tuple[OrderEvidence, ...] = Field(default=(), max_length=MAX_LIST_ENTRIES)
    limit: BoundEvidence
    offset: BoundEvidence
    fetch: BoundEvidence
    set_operations: tuple[SetOperationEvidence, ...] = Field(
        default=(), max_length=MAX_LIST_ENTRIES
    )
    unavailable_reasons: tuple[StrictStr, ...] = Field(default=(), max_length=MAX_LIST_ENTRIES)


class ResultColumnEvidence(_EvidenceBase):
    name: StrictStr = Field(min_length=1, max_length=MAX_IDENTIFIER_BYTES)
    data_type: StrictStr = Field(min_length=1, max_length=MAX_IDENTIFIER_BYTES)
    nullable: StrictBool
    ordinal: StrictInt = Field(ge=0, le=MAX_RESULT_COLUMNS - 1)


class ResultEvidence(_EvidenceBase):
    status: EvidenceStatus
    execution_status: StrictStr = Field(min_length=1, max_length=32)
    columns: tuple[ResultColumnEvidence, ...] = Field(default=(), max_length=MAX_RESULT_COLUMNS)
    row_count: StrictInt = Field(ge=0)
    truncated: StrictBool = False
    truncation_reason: StrictStr | None = Field(default=None, max_length=64)
    exact_rows: tuple[tuple[object, ...], ...] | None = None
    exact_rows_bytes: StrictInt | None = Field(default=None, ge=0, le=MAX_EXACT_RESULT_BYTES)
    row_fingerprint_version: Literal["a5.7g2-streaming-rows-v1"] = ROW_FINGERPRINT_VERSION
    row_multiset_sha256: StrictStr | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    row_order_sha256: StrictStr | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    row_order_semantics: Literal["ordered", "unordered", "unknown"] = "unknown"
    unavailable_reasons: tuple[StrictStr, ...] = Field(default=(), max_length=MAX_LIST_ENTRIES)

    @model_validator(mode="after")
    def validate_rows(self) -> ResultEvidence:
        if self.exact_rows is not None:
            if len(self.exact_rows) > MAX_EXACT_ROWS:
                raise ValueError("exact result evidence exceeds the row bound")
            width = len(self.columns)
            if any(len(row) != width for row in self.exact_rows):
                raise ValueError("exact result evidence row width is inconsistent")
            try:
                serialized = _canonical_json(self.exact_rows).encode("utf-8")
            except SemanticEvidenceError as error:
                raise ValueError("exact result evidence is not JSON-safe") from error
            if len(serialized) > MAX_EXACT_RESULT_BYTES:
                raise ValueError("exact result evidence exceeds the byte bound")
            if self.exact_rows_bytes != len(serialized):
                raise ValueError("exact result evidence byte count is inconsistent")
        return self


class GenerationAttemptEvidence(_EvidenceBase):
    logical_stage: Literal["generation", "repair_generation"]
    attempt_index: StrictInt = Field(ge=1)
    provider: StrictStr = Field(min_length=1, max_length=64)
    model: StrictStr = Field(min_length=1, max_length=255)
    outcome: Literal["success", "failure", "cancelled"]
    lifecycle_status: GenerationLifecycleStatus
    finish_reason: StrictStr | None = Field(default=None, max_length=64)
    response_parse_outcome: StrictStr | None = Field(default=None, max_length=64)
    output_token_budget: StrictInt | None = Field(default=None, ge=1)
    input_tokens: StrictInt | None = Field(default=None, ge=0)
    output_tokens: StrictInt | None = Field(default=None, ge=0)


class GenerationEvidence(_EvidenceBase):
    prompt_version: StrictStr = Field(min_length=1, max_length=64)
    reasoning_mode: StrictStr = Field(min_length=1, max_length=16)
    initial_parse_status: GenerationLifecycleStatus
    repair_attempted: StrictBool
    attempts: tuple[GenerationAttemptEvidence, ...] = Field(default=(), max_length=MAX_LIST_ENTRIES)
    attempts_truncated: StrictBool = False
    completeness: EvidenceCompleteness = "complete"
    completeness_reasons: tuple[StrictStr, ...] = Field(default=(), max_length=MAX_LIST_ENTRIES)


class SemanticEvidenceRecord(_EvidenceBase):
    """Versioned candidate evidence, never an authorization or execution token."""

    evidence_version: Literal["a5.7g2-semantic-evidence-v1"] = SEMANTIC_EVIDENCE_VERSION
    case_id: StrictStr = Field(min_length=1, max_length=128)
    split: Literal["dev", "test"]
    datasource_id: UUID
    dataset_fingerprint: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    fixture_fingerprint: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    schema_fingerprint: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    canonicalization_version: Literal["a5.7g2-canonical-json-v1"] = (
        SEMANTIC_EVIDENCE_CANONICALIZATION_VERSION
    )
    sql_dialect: Literal["postgres"] = "postgres"
    parser_version: StrictStr = Field(min_length=1, max_length=64)
    collector_version: Literal["a5.7g2-collector-v1"] = SEMANTIC_EVIDENCE_COLLECTOR_VERSION
    evidence_mode: Literal["synthetic_exact_v1", "sensitive_future"]
    generation: GenerationEvidence
    sql: SemanticSQLEvidence | None = None
    execution: ResultEvidence | None = None
    evidence_status: EvidenceStatus
    unavailable_reasons: tuple[StrictStr, ...] = Field(default=(), max_length=MAX_LIST_ENTRIES)
    evidence_digest: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_digest(self) -> SemanticEvidenceRecord:
        try:
            serialized = _canonical_json(
                self.model_dump(mode="json", exclude={"evidence_digest"})
            ).encode("utf-8")
        except SemanticEvidenceError as error:
            raise ValueError("semantic evidence is not JSON-safe") from error
        if len(serialized) > MAX_RECORD_BYTES:
            raise ValueError("semantic evidence exceeds the record byte bound")
        if (
            self.evidence_mode != SYNTHETIC_EXACT_MODE
            and self.execution is not None
            and self.execution.exact_rows is not None
        ):
            raise ValueError("exact result evidence is only allowed for synthetic_exact_v1")
        computed = _sha256(serialized)
        if self.evidence_digest != computed:
            raise ValueError("semantic evidence digest does not match canonical content")
        full_payload = self.model_dump(mode="json")
        if len(_canonical_json(full_payload).encode("utf-8")) > MAX_RECORD_BYTES:
            raise ValueError("semantic evidence exceeds the record byte bound")
        if self.evidence_mode == SENSITIVE_FUTURE_MODE:
            raise ValueError("sensitive future evidence mode is not implemented")
        return self

    def validate_provenance(
        self,
        *,
        dataset_fingerprint: str,
        fixture_fingerprint: str,
        schema_fingerprint: str,
    ) -> None:
        if (
            self.dataset_fingerprint != dataset_fingerprint
            or self.fixture_fingerprint != fixture_fingerprint
            or self.schema_fingerprint != schema_fingerprint
        ):
            raise SemanticEvidenceError("semantic evidence provenance does not match the run")


class _EvidenceBoundExceeded(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise SemanticEvidenceError("semantic evidence contains a non-JSON value") from error


def _sha256(value: str | bytes) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()


def evidence_digest_for(record: SemanticEvidenceRecord) -> str:
    """Return the deterministic digest excluding the digest field itself."""
    payload = record.model_dump(mode="json", exclude={"evidence_digest"})
    return _sha256(_canonical_json(payload))


def _safe_identifier(value: object) -> str:
    text = unicodedata.normalize("NFC", str(value))
    if not text or not text.strip() or len(text.encode("utf-8")) > MAX_IDENTIFIER_BYTES:
        raise _EvidenceBoundExceeded("identifier_bound")
    return text


def _safe_reason(value: str) -> str:
    return re.sub(r"[^a-z0-9_:-]", "_", value.lower())[:96] or "unknown"


def _literal_parts(node: exp.Expression) -> tuple[str, str, str]:
    if isinstance(node, exp.Null):
        return "null", "null", "symbolic"
    if isinstance(node, exp.Boolean):
        value_type, canonical = "boolean", str(node.this).lower()
    elif isinstance(node, exp.Literal):
        raw = str(node.this)
        if node.is_string:
            value_type, canonical = "string", raw
        elif _NUMERIC_LITERAL.fullmatch(raw):
            value_type = "integer" if re.fullmatch(r"[+-]?\d+", raw) else "numeric"
            canonical = raw
        else:
            value_type, canonical = "unknown", raw
    else:
        name = type(node).__name__.lower()
        value_type = (
            "date"
            if name == "date"
            else "timestamp"
            if name == "timestamp"
            else "time"
            if name == "time"
            else "interval"
            if name == "interval"
            else "unknown"
        )
        canonical = str(node.this) if getattr(node, "this", None) is not None else ""
    if len(canonical.encode("utf-8")) > MAX_LITERAL_TEXT_BYTES:
        raise _EvidenceBoundExceeded("literal_bound")
    if value_type == "unknown":
        return value_type, "", "unavailable"
    identity = _sha256(f"a5.7g2-literal-v1|{value_type}|{canonical}")
    return value_type, identity, "sha256-typed-v1"


class _ExpressionBuilder:
    def __init__(self, relation_aliases: Mapping[str, str] | None = None) -> None:
        self.nodes = 0
        self.relation_aliases = {
            unicodedata.normalize("NFKC", alias).casefold(): relation
            for alias, relation in (relation_aliases or {}).items()
        }

    def _enter(self, depth: int) -> None:
        if depth > MAX_EXPRESSION_DEPTH:
            raise _EvidenceBoundExceeded("expression_depth")
        self.nodes += 1
        if self.nodes > MAX_AST_NODES:
            raise _EvidenceBoundExceeded("semantic_ast_nodes")

    def build(self, node: exp.Expression | None, *, depth: int = 0) -> ExpressionEvidence:
        self._enter(depth)
        if node is None:
            return ExpressionEvidence(kind="unknown", reason="missing_expression")
        if isinstance(node, exp.Alias):
            return self.build(node.this, depth=depth + 1)
        if isinstance(node, exp.Column):
            if node.catalog or node.db:
                relation = ".".join(
                    _safe_identifier(part) for part in (node.catalog, node.db, node.table) if part
                )
            elif node.table:
                relation = self.relation_aliases.get(
                    unicodedata.normalize("NFKC", node.table).casefold(),
                    _safe_identifier(node.table),
                )
            else:
                relation = None
            return ExpressionEvidence(
                kind="source_column",
                relation=relation,
                column=_safe_identifier(node.name),
            )
        if isinstance(node, exp.Star):
            return ExpressionEvidence(kind="source_column", column="*")
        if isinstance(
            node,
            (exp.Null, exp.Boolean, exp.Literal, exp.Date, exp.Time, exp.Timestamp, exp.Interval),
        ):
            value_type, identity, hash_mode = _literal_parts(node)
            return ExpressionEvidence(
                kind="literal",
                literal=LiteralEvidence(
                    value_type=value_type,
                    identity=identity or "unavailable",
                    hash_mode=hash_mode,
                ),
            )
        if isinstance(node, exp.Cast):
            return ExpressionEvidence(
                kind="derived",
                function="cast",
                type_name=_safe_identifier(node.to.sql(dialect="postgres"))
                if node.to is not None
                else None,
                children=(self.build(node.this, depth=depth + 1),),
            )
        class_name = type(node).__name__
        if class_name in _BINARY_OPERATORS:
            return ExpressionEvidence(
                kind="derived",
                operator=_BINARY_OPERATORS[class_name],
                children=(
                    self.build(node.args.get("this"), depth=depth + 1),
                    self.build(node.args.get("expression"), depth=depth + 1),
                ),
            )
        if isinstance(node, exp.Neg):
            return ExpressionEvidence(
                kind="derived",
                operator="neg",
                children=(self.build(node.this, depth=depth + 1),),
            )
        function_name = node.sql_name().lower() if isinstance(node, exp.Func) else None
        if function_name in _SAFE_AGGREGATES:
            child = node.this
            distinct = isinstance(child, exp.Distinct)
            aggregate_items = child.expressions if distinct and child is not None else ()
            if len(aggregate_items) > MAX_LIST_ENTRIES:
                raise _EvidenceBoundExceeded("expression_entries")
            children = (
                tuple(self.build(item, depth=depth + 1) for item in aggregate_items)
                if distinct
                else (self.build(child, depth=depth + 1),)
            )
            return ExpressionEvidence(
                kind="aggregate",
                function=function_name,
                distinct=distinct,
                children=children,
            )
        if function_name in _SAFE_FUNCTIONS:
            args = []
            if node.this is not None:
                args.append(self.build(node.this, depth=depth + 1))
            if len(node.expressions) > MAX_LIST_ENTRIES:
                raise _EvidenceBoundExceeded("expression_entries")
            args.extend(self.build(item, depth=depth + 1) for item in node.expressions)
            if len(args) > MAX_LIST_ENTRIES:
                raise _EvidenceBoundExceeded("expression_entries")
            return ExpressionEvidence(kind="function", function=function_name, children=tuple(args))
        if isinstance(node, exp.Paren):
            return self.build(node.this, depth=depth + 1)
        return ExpressionEvidence(kind="unknown", reason=_safe_reason(class_name))


def _projection_kind(expression: ExpressionEvidence) -> str:
    if expression.kind in {"source_column", "aggregate", "literal", "unknown"}:
        return expression.kind
    return "derived"


def _relation_from_table(
    table: exp.Table, *, cte_names: frozenset[str] = frozenset()
) -> RelationEvidence:
    name = _safe_identifier(table.name)
    schema = _safe_identifier(table.db) if table.db else None
    relation = f"{schema}.{name}" if schema else name
    alias = table.alias or None
    return RelationEvidence(
        relation=relation,
        alias=_safe_identifier(alias) if alias else None,
        relation_kind=(
            "physical" if schema else "cte" if name.casefold() in cte_names else "unknown"
        ),
    )


def _predicate(
    node: exp.Expression | None,
    builder: _ExpressionBuilder,
    *,
    depth: int = 0,
) -> PredicateEvidence:
    if depth > MAX_EXPRESSION_DEPTH:
        raise _EvidenceBoundExceeded("expression_depth")
    if node is None:
        return PredicateEvidence(kind="unknown", reason="missing_predicate")
    class_name = type(node).__name__
    if isinstance(node, (exp.And, exp.Or)):
        return PredicateEvidence(
            kind="and" if isinstance(node, exp.And) else "or",
            children=(
                _predicate(node.this, builder, depth=depth + 1),
                _predicate(node.expression, builder, depth=depth + 1),
            ),
        )
    if isinstance(node, exp.Not):
        return PredicateEvidence(
            kind="not", children=(_predicate(node.this, builder, depth=depth + 1),)
        )
    if isinstance(node, exp.Is) and isinstance(node.expression, exp.Null):
        return PredicateEvidence(
            kind="is_null",
            operator="is_not_null" if isinstance(node.this, exp.Not) else "is_null",
            expression=builder.build(node.this),
        )
    if class_name in _PREDICATE_OPERATORS:
        return PredicateEvidence(
            kind="comparison",
            operator=_BINARY_OPERATORS[class_name],
            values=(
                builder.build(node.this),
                builder.build(node.expression),
            ),
        )
    if isinstance(node, exp.In):
        if len(node.expressions) > MAX_LIST_ENTRIES:
            raise _EvidenceBoundExceeded("predicate_entries")
        return PredicateEvidence(
            kind="in",
            expression=builder.build(node.this),
            values=tuple(builder.build(item) for item in node.expressions),
        )
    if isinstance(node, exp.Between):
        return PredicateEvidence(
            kind="between",
            expression=builder.build(node.this),
            values=(
                builder.build(node.args.get("low")),
                builder.build(node.args.get("high")),
            ),
        )
    return PredicateEvidence(kind="unknown", reason=_safe_reason(class_name))


def _bound(node: exp.Expression | None, builder: _ExpressionBuilder) -> BoundEvidence:
    if node is None:
        return BoundEvidence(kind="absent")
    options = node.args.get("limit_options")
    if options is not None and (
        bool(options.args.get("percent")) or bool(options.args.get("with_ties"))
    ):
        return BoundEvidence(kind="unsupported", reason="unsupported_bound_options")
    value = node.args.get("count") if isinstance(node, exp.Fetch) else node.args.get("expression")
    if isinstance(value, exp.Column) and value.name.upper() == "ALL":
        return BoundEvidence(kind="all")
    if isinstance(value, exp.Literal) and not value.is_string:
        try:
            number = int(str(value.this))
        except (TypeError, ValueError):
            return BoundEvidence(kind="unavailable", reason="non_integer_bound")
        if 0 <= number <= 100_000:
            return BoundEvidence(kind="value", value=number)
    if isinstance(value, exp.Null):
        return BoundEvidence(kind="unsupported", reason="null_bound")
    return BoundEvidence(kind="unavailable", reason="parameterized_or_complex_bound")


def _expression_has_unknown(expression: ExpressionEvidence) -> bool:
    return expression.kind == "unknown" or any(
        _expression_has_unknown(child) for child in expression.children
    )


def _predicate_has_unknown(predicate: PredicateEvidence) -> bool:
    return (
        predicate.kind == "unknown"
        or (predicate.expression is not None and _expression_has_unknown(predicate.expression))
        or any(_expression_has_unknown(value) for value in predicate.values)
        or any(_predicate_has_unknown(child) for child in predicate.children)
    )


def _bound_completeness_reason(bound: BoundEvidence) -> str | None:
    if bound.kind in {"unavailable", "unsupported"}:
        return "unsupported_clause_semantics"
    return None


def _unsupported_select_reasons(statement: exp.Select) -> set[str]:
    """Return safe reason codes for syntax we do not model completely."""
    reasons: set[str] = set()
    select_nodes = (statement, *statement.find_all(exp.Select))
    for select in select_nodes:
        with_clause = select.args.get("with") or select.args.get("with_")
        if isinstance(with_clause, exp.With) and with_clause.args.get("recursive"):
            reasons.add("unsupported_recursive_cte")
        if select is not statement and isinstance(with_clause, exp.With):
            reasons.add("unresolved_nested_cte_scope")
        for key in ("qualify", "into", "locks", "connect", "match", "settings"):
            if select.args.get(key) is not None:
                reasons.add("unsupported_statement_clause")
    for table in statement.find_all(exp.Table):
        if table.args.get("sample") is not None or table.args.get("pivots"):
            reasons.add("unsupported_statement_clause")
    return reasons


def _source_expressions(statement: exp.Select) -> tuple[exp.Expression, ...]:
    """Return every FROM/JOIN source in the select's own scope.

    The production validator accepts more source shapes than this collector
    currently models.  Keeping this dispatch explicit prevents a newly
    accepted source from disappearing while the parent is still reported as
    complete.
    """
    sources: list[exp.Expression] = []
    from_clause = statement.args.get("from") or statement.args.get("from_")
    if isinstance(from_clause, exp.From):
        if from_clause.this is not None:
            sources.append(from_clause.this)
        sources.extend(from_clause.expressions)
    for join in statement.args.get("joins") or ():
        if join.this is not None:
            sources.append(join.this)
    return tuple(sources)


def _source_completeness_reasons(statement: exp.Select) -> set[str]:
    """Return bounded reasons for source forms not fully represented here."""
    reasons: set[str] = set()
    for source in _source_expressions(statement):
        if isinstance(source, exp.Table):
            continue
        if isinstance(source, exp.Values):
            reasons.add("unsupported_values_source")
            continue
        if isinstance(source, exp.Lateral):
            reasons.add("unsupported_lateral_source")
            continue
        if isinstance(source, exp.Subquery):
            reasons.add("unresolved_derived_source_lineage")
            continue
        reasons.add("unsupported_source_type")
    return reasons


def _nested_set_completeness_reasons(statement: exp.Select) -> set[str]:
    """Prevent nested set trees from being mistaken for a complete SELECT."""
    if any(isinstance(node, exp.SetOperation) for node in statement.find_all(exp.SetOperation)):
        return {"unsupported_nested_set_operation"}
    return set()


def _order_evidence(
    order: exp.Order | None, builder: _ExpressionBuilder
) -> tuple[OrderEvidence, ...]:
    if order is not None and len(order.expressions) > MAX_LIST_ENTRIES:
        raise _EvidenceBoundExceeded("order_entries")
    return tuple(
        OrderEvidence(
            expression=builder.build(ordered.this),
            direction="desc" if ordered.args.get("desc") else "asc",
            nulls=(
                "first"
                if ordered.args.get("nulls_first")
                else "last"
                if ordered.args.get("nulls_first") is False
                else "unspecified"
            ),
        )
        for ordered in (order.expressions if order else ())
    )


def _completeness(
    reasons: Iterable[str],
) -> tuple[EvidenceCompleteness, tuple[str, ...]]:
    unique = tuple(sorted(set(reasons)))
    return ("partial" if unique else "complete", unique)


def _select_evidence(
    statement: exp.Select, *, _include_nested_completeness: bool = True
) -> SemanticSQLEvidence:
    with_clause = statement.args.get("with") or statement.args.get("with_")
    cte_names = (
        frozenset(
            unicodedata.normalize("NFKC", cte.alias_or_name).casefold()
            for cte in with_clause.expressions
            if isinstance(with_clause, exp.With) and cte.alias_or_name
        )
        if with_clause is not None
        else frozenset()
    )
    tables = tuple(statement.find_all(exp.Table))
    relations = tuple(_relation_from_table(table, cte_names=cte_names) for table in tables)
    if len(relations) > MAX_LIST_ENTRIES:
        raise _EvidenceBoundExceeded("relation_entries")
    unique_relations = tuple({relation.relation: relation for relation in relations}.values())
    relation_aliases: dict[str, str] = {}
    for table, relation in zip(tables, relations, strict=True):
        relation_aliases[table.name] = relation.relation
        if table.alias:
            relation_aliases[table.alias] = relation.relation
    builder = _ExpressionBuilder(relation_aliases)
    if len(statement.expressions) > MAX_LIST_ENTRIES:
        raise _EvidenceBoundExceeded("projection_entries")
    projections: list[ProjectionEvidence] = []
    for ordinal, expression in enumerate(statement.expressions):
        alias = expression.alias or None if isinstance(expression, exp.Alias) else None
        canonical = builder.build(expression)
        projections.append(
            ProjectionEvidence(
                ordinal=ordinal,
                kind=_projection_kind(canonical),
                expression=canonical,
                alias=_safe_identifier(alias) if alias else None,
            )
        )

    predicates: list[PredicateEvidence] = []
    for key in ("where", "having"):
        clause = statement.args.get(key)
        if clause is not None:
            predicates.append(_predicate(clause.this, builder))

    joins: list[JoinEvidence] = []
    previous_relation = unique_relations[0].relation if unique_relations else None
    for join in statement.args.get("joins") or []:
        right_relation = None
        if isinstance(join.this, exp.Table):
            right_relation = _relation_from_table(join.this).relation
        side = str(join.args.get("side") or "").lower()
        kind = str(join.args.get("kind") or "").lower()
        join_type = (
            side if side in {"left", "right", "full"} else "cross" if kind == "cross" else "inner"
        )
        keys: list[JoinKeyEvidence] = []
        on = join.args.get("on")
        for equality in on.find_all(exp.EQ) if on is not None else ():
            keys.append(
                JoinKeyEvidence(
                    left=builder.build(equality.this),
                    right=builder.build(equality.expression),
                )
            )
        joins.append(
            JoinEvidence(
                join_type=join_type
                if join_type in {"inner", "left", "right", "full", "cross"}
                else "unknown",
                left_relation=previous_relation,
                right_relation=right_relation,
                keys=tuple(keys[:MAX_LIST_ENTRIES]),
                predicate=_predicate(on, builder) if on is not None else None,
            )
        )
        previous_relation = right_relation or previous_relation
    if len(joins) > MAX_LIST_ENTRIES:
        raise _EvidenceBoundExceeded("join_entries")

    group = statement.args.get("group")
    if group is not None and len(group.expressions) > MAX_LIST_ENTRIES:
        raise _EvidenceBoundExceeded("group_entries")
    group_by = tuple(builder.build(item) for item in (group.expressions if group else ()))
    distinct_node = statement.args.get("distinct")
    distinct_mode: Literal["none", "distinct", "distinct_on"] = "none"
    distinct_on: tuple[ExpressionEvidence, ...] = ()
    if distinct_node is not None:
        distinct_mode = "distinct"
        on = distinct_node.args.get("on") if isinstance(distinct_node, exp.Distinct) else None
        if on is not None:
            distinct_mode = "distinct_on"
            if isinstance(on, exp.Tuple):
                distinct_items = tuple(on.expressions)
            elif isinstance(on, (list, tuple)):
                distinct_items = tuple(on)
            else:
                distinct_items = (on,)
            if len(distinct_items) > MAX_LIST_ENTRIES:
                raise _EvidenceBoundExceeded("distinct_on_entries")
            distinct_on = tuple(builder.build(item) for item in distinct_items)
    order_by = _order_evidence(statement.args.get("order"), builder)
    limit_node = statement.args.get("limit")
    is_fetch = isinstance(limit_node, exp.Fetch)
    limit = _bound(None if is_fetch else limit_node, builder)
    offset = _bound(statement.args.get("offset"), builder)
    fetch = _bound(limit_node if is_fetch else statement.args.get("fetch"), builder)
    completeness_reasons = _unsupported_select_reasons(statement)
    completeness_reasons.update(_source_completeness_reasons(statement))
    completeness_reasons.update(_nested_set_completeness_reasons(statement))
    if any(relation.relation_kind == "unknown" for relation in unique_relations):
        completeness_reasons.add("unknown_relation_scope")
    completeness_reasons.update(
        reason
        for reason in (
            _bound_completeness_reason(limit),
            _bound_completeness_reason(offset),
            _bound_completeness_reason(fetch),
        )
        if reason is not None
    )
    if any(_expression_has_unknown(item.expression) for item in projections):
        completeness_reasons.add("unsupported_expression")
    if any(_predicate_has_unknown(item) for item in predicates):
        completeness_reasons.add("unsupported_expression")
    if any(_expression_has_unknown(item) for item in group_by):
        completeness_reasons.add("unsupported_expression")
    if any(_expression_has_unknown(item.expression) for item in order_by):
        completeness_reasons.add("unsupported_expression")
    for join in joins:
        if any(
            _expression_has_unknown(key.left) or _expression_has_unknown(key.right)
            for key in join.keys
        ) or (join.predicate is not None and _predicate_has_unknown(join.predicate)):
            completeness_reasons.add("unsupported_expression")
    if any(_expression_has_unknown(item) for item in distinct_on):
        completeness_reasons.add("unsupported_distinct_on")
    if _include_nested_completeness:
        for nested in statement.find_all(exp.Select):
            try:
                nested_evidence = _select_evidence(nested, _include_nested_completeness=False)
            except _EvidenceBoundExceeded:
                completeness_reasons.add("nested_child_evidence_truncated")
                continue
            if nested_evidence.status is not EvidenceStatus.AVAILABLE:
                completeness_reasons.add("nested_child_evidence_unavailable")
            if nested_evidence.completeness != "complete":
                completeness_reasons.add(f"nested_child_sql_{nested_evidence.completeness}")
                completeness_reasons.update(
                    f"nested_{reason}" for reason in nested_evidence.completeness_reasons
                )
    completeness, completeness_codes = _completeness(completeness_reasons)
    return SemanticSQLEvidence(
        status=EvidenceStatus.AVAILABLE,
        completeness=completeness,
        completeness_reasons=completeness_codes,
        relations=tuple(sorted(unique_relations, key=lambda item: item.relation)),
        projections=tuple(projections),
        predicates=tuple(predicates),
        joins=tuple(joins),
        group_by=group_by,
        distinct=distinct_node is not None,
        distinct_mode=distinct_mode,
        distinct_on=distinct_on,
        grain_status="proven" if group_by or distinct_node is not None else "unknown",
        order_by=order_by,
        limit=limit,
        offset=offset,
        fetch=fetch,
    )


def _child_completeness_reasons(child: SemanticSQLEvidence) -> set[str]:
    reasons: set[str] = set()
    if child.status is not EvidenceStatus.AVAILABLE:
        reasons.add(f"child_evidence_{child.status.value}")
    if child.completeness != "complete":
        reasons.add(f"child_sql_{child.completeness}")
        reasons.update(child.completeness_reasons)
    return reasons


def _mark_partial(evidence: SemanticSQLEvidence, reason: str) -> SemanticSQLEvidence:
    reasons = tuple(sorted({*evidence.completeness_reasons, reason}))
    return evidence.model_copy(
        update={
            "completeness": "partial",
            "completeness_reasons": reasons,
        }
    )


def _query_evidence(statement: exp.Expression) -> SemanticSQLEvidence:
    if isinstance(statement, (exp.Union, exp.Intersect, exp.Except)):
        left = _query_evidence(statement.this)
        right = _query_evidence(statement.expression)
        operation = (
            "union"
            if isinstance(statement, exp.Union)
            else "intersect"
            if isinstance(statement, exp.Intersect)
            else "except"
        )
        left_digest = _sha256(_canonical_json(left.model_dump(mode="json")))
        right_digest = _sha256(_canonical_json(right.model_dump(mode="json")))
        tables = tuple(statement.find_all(exp.Table))
        relation_aliases = {
            table.alias or table.name: _relation_from_table(table).relation for table in tables
        }
        builder = _ExpressionBuilder(relation_aliases)
        outer_order = _order_evidence(statement.args.get("order"), builder)
        limit_node = statement.args.get("limit")
        is_fetch = isinstance(limit_node, exp.Fetch)
        outer_limit = _bound(None if is_fetch else limit_node, builder)
        outer_offset = _bound(statement.args.get("offset"), builder)
        outer_fetch = _bound(limit_node if is_fetch else statement.args.get("fetch"), builder)
        completeness_reasons = _child_completeness_reasons(left) | _child_completeness_reasons(
            right
        )
        root_with = statement.args.get("with") or statement.args.get("with_")
        if root_with is not None:
            completeness_reasons.add("unresolved_root_cte_scope")
        if statement.args.get("by_name"):
            completeness_reasons.add("unsupported_set_by_name")
        if any(statement.args.get(key) is not None for key in ("side", "kind", "on")):
            completeness_reasons.add("unsupported_set_modifier")
        if any(_expression_has_unknown(item.expression) for item in outer_order):
            completeness_reasons.add("unsupported_outer_ordering")
        for label, bound in (
            ("limit", outer_limit),
            ("offset", outer_offset),
            ("fetch", outer_fetch),
        ):
            if _bound_completeness_reason(bound) is not None:
                completeness_reasons.add(f"unsupported_outer_{label}")
        completeness, completeness_codes = _completeness(completeness_reasons)
        set_operations = (
            *left.set_operations,
            *right.set_operations,
            SetOperationEvidence(
                operation=operation,
                all=not bool(statement.args.get("distinct")),
                left_digest=left_digest,
                right_digest=right_digest,
            ),
        )
        if len(set_operations) > MAX_LIST_ENTRIES:
            raise _EvidenceBoundExceeded("set_operation_entries")
        return SemanticSQLEvidence(
            status=EvidenceStatus.AVAILABLE,
            completeness=completeness,
            completeness_reasons=completeness_codes,
            relations=tuple(
                sorted(
                    {item.relation: item for item in (*left.relations, *right.relations)}.values(),
                    key=lambda item: item.relation,
                )
            ),
            projections=left.projections,
            predicates=left.predicates,
            joins=left.joins,
            group_by=left.group_by,
            distinct=left.distinct,
            distinct_mode=left.distinct_mode,
            distinct_on=left.distinct_on,
            grain_status="unknown",
            order_by=outer_order,
            limit=outer_limit,
            offset=outer_offset,
            fetch=outer_fetch,
            set_operations=set_operations,
        )
    if isinstance(statement, exp.Select):
        return _select_evidence(statement)
    select = statement.find(exp.Select)
    if select is not None:
        evidence = _select_evidence(select)
        if any(isinstance(node, exp.SetOperation) for node in statement.find_all(exp.SetOperation)):
            evidence = _mark_partial(evidence, "unsupported_nested_set_operation")
        return evidence
    raise _EvidenceBoundExceeded("unsupported_statement_shape")


def _check_ast_node_bound(statement: exp.Expression) -> None:
    """Count the parsed tree iteratively before building semantic evidence."""
    pending: list[exp.Expression] = [statement]
    count = 0
    while pending:
        node = pending.pop()
        count += 1
        if count > MAX_AST_NODES:
            raise _EvidenceBoundExceeded("semantic_ast_nodes")
        for value in node.args.values():
            if isinstance(value, exp.Expression):
                pending.append(value)
            elif isinstance(value, (list, tuple)):
                pending.extend(item for item in value if isinstance(item, exp.Expression))


@dataclass(frozen=True)
class _CanonicalRows:
    ordered_sha256: str
    multiset_sha256: str
    exact_rows: tuple[tuple[object, ...], ...]
    exact_rows_bytes: int
    exact_rows_truncated_reason: str | None


def _canonical_rows(rows: Iterable[Sequence[object]]) -> _CanonicalRows:
    """Fingerprint rows in one pass while retaining only a bounded exact prefix."""
    ordered_hasher = hashlib.sha256()
    ordered_hasher.update(b"[")
    multiset_count = 0
    multiset_sum = 0
    multiset_xor = 0
    exact_rows: list[tuple[object, ...]] = []
    exact_bytes = 1  # opening ``[``; the closing bracket is added below
    exact_truncated_reason: str | None = None
    modulus = 1 << 256

    for row_index, row in enumerate(rows):
        row_values = tuple(row)
        row_json = _canonical_json(list(row_values))
        row_bytes = row_json.encode("utf-8")
        if row_index:
            ordered_hasher.update(b",")
        ordered_hasher.update(_canonical_json(row_json).encode("utf-8"))

        row_digest = hashlib.sha256(row_bytes).digest()
        row_value = int.from_bytes(row_digest, byteorder="big", signed=False)
        multiset_count += 1
        multiset_sum = (multiset_sum + row_value) % modulus
        multiset_xor ^= row_value

        if exact_truncated_reason is None:
            if row_index >= MAX_EXACT_ROWS:
                exact_truncated_reason = "exact_result_rows_bound"
            else:
                candidate_bytes = exact_bytes + (1 if exact_rows else 0) + len(row_bytes)
                if candidate_bytes + 1 > MAX_EXACT_RESULT_BYTES:
                    exact_truncated_reason = "exact_result_bytes_bound"
                else:
                    exact_rows.append(row_values)
                    exact_bytes = candidate_bytes

    ordered_hasher.update(b"]")
    multiset_payload = (
        f"{ROW_FINGERPRINT_VERSION}|{multiset_count}|{multiset_sum:064x}|{multiset_xor:064x}"
    )
    return _CanonicalRows(
        ordered_sha256=ordered_hasher.hexdigest(),
        multiset_sha256=_sha256(multiset_payload),
        exact_rows=tuple(exact_rows),
        exact_rows_bytes=exact_bytes + 1,
        exact_rows_truncated_reason=exact_truncated_reason,
    )


def _result_evidence(result: QueryExecutionResult | None, *, mode: str) -> ResultEvidence:
    if result is None:
        return ResultEvidence(
            status=EvidenceStatus.UNAVAILABLE,
            execution_status="unavailable",
            row_count=0,
            unavailable_reasons=("execution_result_unavailable",),
        )
    if len(result.columns) > MAX_RESULT_COLUMNS:
        return ResultEvidence(
            status=EvidenceStatus.TRUNCATED,
            execution_status=result.state.value,
            row_count=result.row_count,
            truncated=result.truncated,
            truncation_reason=result.truncation_reason.value if result.truncation_reason else None,
            unavailable_reasons=("result_columns_bound",),
        )
    if any(column.ordinal >= MAX_RESULT_COLUMNS for column in result.columns):
        return ResultEvidence(
            status=EvidenceStatus.TRUNCATED,
            execution_status=result.state.value,
            row_count=result.row_count,
            truncated=result.truncated,
            truncation_reason=result.truncation_reason.value if result.truncation_reason else None,
            unavailable_reasons=("result_column_ordinal_bound",),
        )
    columns = tuple(
        ResultColumnEvidence(
            name=_safe_identifier(column.name),
            data_type=_safe_identifier(column.data_type),
            nullable=column.nullable,
            ordinal=column.ordinal,
        )
        for column in result.columns
    )
    try:
        canonical = _canonical_rows(result.rows)
    except SemanticEvidenceError:
        return ResultEvidence(
            status=EvidenceStatus.UNAVAILABLE,
            execution_status=result.state.value,
            row_count=result.row_count,
            truncated=result.truncated,
            truncation_reason=result.truncation_reason.value if result.truncation_reason else None,
            unavailable_reasons=("result_json_unavailable",),
        )
    reasons: list[str] = []
    status = EvidenceStatus.TRUNCATED if result.truncated else EvidenceStatus.AVAILABLE
    exact_rows: tuple[tuple[object, ...], ...] | None = None
    exact_bytes: int | None = None
    exact_truncated_reason: str | None = None
    if result.state is QueryLifecycleState.SUCCEEDED and mode == SYNTHETIC_EXACT_MODE:
        exact_rows = canonical.exact_rows
        exact_bytes = canonical.exact_rows_bytes
        exact_truncated_reason = canonical.exact_rows_truncated_reason
        if exact_truncated_reason is not None:
            status = EvidenceStatus.TRUNCATED
            reasons.append(exact_truncated_reason)
    elif result.state is not QueryLifecycleState.SUCCEEDED:
        status = EvidenceStatus.UNAVAILABLE
        reasons.append("execution_not_successful")
    if result.truncated and result.truncation_reason is not None:
        reasons.append(result.truncation_reason.value)
    return ResultEvidence(
        status=status,
        execution_status=result.state.value,
        columns=columns,
        row_count=result.row_count,
        truncated=result.truncated or exact_truncated_reason is not None,
        truncation_reason=(
            result.truncation_reason.value if result.truncation_reason else exact_truncated_reason
        ),
        exact_rows=exact_rows,
        exact_rows_bytes=exact_bytes,
        row_fingerprint_version=ROW_FINGERPRINT_VERSION,
        row_multiset_sha256=canonical.multiset_sha256,
        row_order_sha256=canonical.ordered_sha256,
        row_order_semantics="unknown",
        unavailable_reasons=tuple(reasons),
    )


def _generation_evidence(
    *,
    prompt_version: str,
    reasoning_mode: str,
    result: ChatBIResult,
    invocations: Iterable[LLMProviderInvocation],
) -> GenerationEvidence:
    def lifecycle_status(invocation: LLMProviderInvocation) -> GenerationLifecycleStatus:
        if invocation.outcome == "cancelled":
            return "cancelled"
        if invocation.outcome == "failure":
            return "provider_failure" if invocation.llm_result_returned else "no_response"
        if invocation.response_parse_outcome == "parsed":
            return "parsed"
        if invocation.response_parse_outcome in {
            "structured_output_parse_error",
            "structured_output_schema_error",
        }:
            return "parse_failed"
        return "parse_not_attempted"

    generation_invocations: list[GenerationAttemptEvidence] = []
    initial_status: GenerationLifecycleStatus = "not_attempted"
    repair_attempted = False
    attempts_truncated = False
    for invocation in invocations:
        if invocation.logical_stage not in {"generation", "repair_generation"}:
            continue
        lifecycle = lifecycle_status(invocation)
        if invocation.logical_stage == "generation" and initial_status == "not_attempted":
            initial_status = lifecycle
        if invocation.logical_stage == "repair_generation":
            repair_attempted = True
        if len(generation_invocations) >= MAX_LIST_ENTRIES:
            attempts_truncated = True
            break
        generation_invocations.append(
            GenerationAttemptEvidence(
                logical_stage=invocation.logical_stage,
                attempt_index=invocation.attempt_index,
                provider=invocation.provider,
                model=invocation.model,
                outcome=invocation.outcome,
                lifecycle_status=lifecycle,
                finish_reason=invocation.finish_reason,
                response_parse_outcome=invocation.response_parse_outcome,
                output_token_budget=invocation.output_token_budget,
                input_tokens=invocation.input_tokens,
                output_tokens=invocation.output_tokens,
            )
        )
    completeness_reasons = ("attempt_metadata_truncated",) if attempts_truncated else ()
    return GenerationEvidence(
        prompt_version=prompt_version,
        reasoning_mode=reasoning_mode,
        initial_parse_status=initial_status,
        repair_attempted=repair_attempted,
        attempts=tuple(generation_invocations),
        attempts_truncated=attempts_truncated,
        completeness="partial" if attempts_truncated else "complete",
        completeness_reasons=completeness_reasons,
    )


def _generation_reasons(generation: GenerationEvidence) -> tuple[str, ...]:
    reasons: list[str] = list(generation.completeness_reasons)
    for attempt in generation.attempts:
        if attempt.lifecycle_status in {"provider_failure", "no_response", "cancelled"}:
            reasons.append(attempt.lifecycle_status)
        elif attempt.lifecycle_status == "parse_not_attempted":
            reasons.append("parse_not_attempted")
        elif attempt.lifecycle_status == "parse_failed":
            reasons.append("parse_failed")
    return tuple(dict.fromkeys(reasons))


def _compact_generation(generation: GenerationEvidence) -> GenerationEvidence:
    return generation.model_copy(update={"attempts": ()})


def _without_exact_rows(execution: ResultEvidence | None) -> ResultEvidence | None:
    if execution is None:
        return None
    return execution.model_copy(update={"exact_rows": None, "exact_rows_bytes": None})


def _compact_execution(execution: ResultEvidence | None) -> ResultEvidence | None:
    if execution is None:
        return None
    return execution.model_copy(
        update={
            "columns": (),
            "exact_rows": None,
            "exact_rows_bytes": None,
            "row_multiset_sha256": None,
            "row_order_sha256": None,
            "row_order_semantics": "unknown",
            "unavailable_reasons": tuple(
                dict.fromkeys((*execution.unavailable_reasons, "result_metadata_omitted"))
            ),
        }
    )


def _record_candidate(
    *,
    case_id: str,
    split: Literal["dev", "test"],
    datasource_id: UUID,
    dataset_fingerprint: str,
    fixture_fingerprint: str,
    schema_fingerprint: str,
    evidence_mode: Literal["synthetic_exact_v1", "sensitive_future"],
    generation: GenerationEvidence,
    sql: SemanticSQLEvidence | None,
    execution: ResultEvidence | None,
    status: EvidenceStatus,
    reasons: Iterable[str],
) -> SemanticEvidenceRecord | None:
    unique_reasons = tuple(dict.fromkeys(reasons))
    candidate = SemanticEvidenceRecord.model_construct(
        evidence_version=SEMANTIC_EVIDENCE_VERSION,
        case_id=case_id,
        split=split,
        datasource_id=datasource_id,
        dataset_fingerprint=dataset_fingerprint,
        fixture_fingerprint=fixture_fingerprint,
        schema_fingerprint=schema_fingerprint,
        evidence_mode=evidence_mode,
        parser_version=SQL_PARSER_VERSION,
        generation=generation,
        sql=sql,
        execution=execution,
        evidence_status=status,
        unavailable_reasons=unique_reasons,
        evidence_digest="0" * 64,
    )
    try:
        without_digest = candidate.model_dump(mode="json", exclude={"evidence_digest"})
        serialized = _canonical_json(without_digest).encode("utf-8")
        if len(serialized) > MAX_RECORD_BYTES:
            return None
        payload = dict(without_digest)
        payload["evidence_digest"] = _sha256(serialized)
        if len(_canonical_json(payload).encode("utf-8")) > MAX_RECORD_BYTES:
            return None
        return SemanticEvidenceRecord.model_validate(payload)
    except (SemanticEvidenceError, TypeError, ValueError):
        return None


def _bounded_record(
    *,
    case_id: str,
    split: Literal["dev", "test"],
    datasource_id: UUID,
    dataset_fingerprint: str,
    fixture_fingerprint: str,
    schema_fingerprint: str,
    evidence_mode: Literal["synthetic_exact_v1", "sensitive_future"],
    generation: GenerationEvidence,
    sql: SemanticSQLEvidence | None,
    execution: ResultEvidence | None,
    status: EvidenceStatus,
    reasons: Iterable[str],
) -> SemanticEvidenceRecord:
    base_reasons = tuple(dict.fromkeys(reasons))
    candidate = _record_candidate(
        case_id=case_id,
        split=split,
        datasource_id=datasource_id,
        dataset_fingerprint=dataset_fingerprint,
        fixture_fingerprint=fixture_fingerprint,
        schema_fingerprint=schema_fingerprint,
        evidence_mode=evidence_mode,
        generation=generation,
        sql=sql,
        execution=execution,
        status=status,
        reasons=base_reasons,
    )
    if candidate is not None:
        return candidate

    degraded_reasons = (*base_reasons, "evidence_record_bytes_bound")
    bounded_status = status if status is EvidenceStatus.UNAVAILABLE else EvidenceStatus.TRUNCATED
    candidate = _record_candidate(
        case_id=case_id,
        split=split,
        datasource_id=datasource_id,
        dataset_fingerprint=dataset_fingerprint,
        fixture_fingerprint=fixture_fingerprint,
        schema_fingerprint=schema_fingerprint,
        evidence_mode=evidence_mode,
        generation=generation,
        sql=sql,
        execution=_without_exact_rows(execution),
        status=bounded_status,
        reasons=(*degraded_reasons, "exact_rows_omitted"),
    )
    if candidate is not None:
        return candidate

    candidate = _record_candidate(
        case_id=case_id,
        split=split,
        datasource_id=datasource_id,
        dataset_fingerprint=dataset_fingerprint,
        fixture_fingerprint=fixture_fingerprint,
        schema_fingerprint=schema_fingerprint,
        evidence_mode=evidence_mode,
        generation=_compact_generation(generation),
        sql=sql,
        execution=_without_exact_rows(execution),
        status=bounded_status,
        reasons=(*degraded_reasons, "exact_rows_omitted", "generation_attempts_omitted"),
    )
    if candidate is not None:
        return candidate

    candidate = _record_candidate(
        case_id=case_id,
        split=split,
        datasource_id=datasource_id,
        dataset_fingerprint=dataset_fingerprint,
        fixture_fingerprint=fixture_fingerprint,
        schema_fingerprint=schema_fingerprint,
        evidence_mode=evidence_mode,
        generation=_compact_generation(generation),
        sql=sql,
        execution=_compact_execution(execution),
        status=bounded_status,
        reasons=(
            *degraded_reasons,
            "exact_rows_omitted",
            "generation_attempts_omitted",
            "result_metadata_omitted",
        ),
    )
    if candidate is not None:
        return candidate

    # The final form is intentionally static and contains no provider- or
    # result-sized fields. It remains far below MAX_RECORD_BYTES.
    candidate = _record_candidate(
        case_id=case_id,
        split=split,
        datasource_id=datasource_id,
        dataset_fingerprint=dataset_fingerprint,
        fixture_fingerprint=fixture_fingerprint,
        schema_fingerprint=schema_fingerprint,
        evidence_mode=evidence_mode,
        generation=_compact_generation(generation),
        sql=None,
        execution=None,
        status=bounded_status,
        reasons=(
            *degraded_reasons,
            "exact_rows_omitted",
            "generation_attempts_omitted",
            "result_metadata_omitted",
        ),
    )
    if candidate is None:
        raise SemanticEvidenceError("static semantic evidence fallback could not be built")
    return candidate


def build_unavailable_semantic_evidence(
    *,
    case_id: str,
    split: Literal["dev", "test"],
    datasource_id: UUID,
    dataset_fingerprint: str,
    fixture_fingerprint: str,
    schema_fingerprint: str,
    prompt_version: str,
    reasoning_mode: str,
    reason: str = "evidence_capture_failed",
) -> SemanticEvidenceRecord:
    """Build a bounded failure-only record when collection cannot complete."""
    generation = GenerationEvidence(
        prompt_version=prompt_version,
        reasoning_mode=reasoning_mode,
        initial_parse_status="not_attempted",
        repair_attempted=False,
        attempts=(),
    )
    return _bounded_record(
        case_id=case_id,
        split=split,
        datasource_id=datasource_id,
        dataset_fingerprint=dataset_fingerprint,
        fixture_fingerprint=fixture_fingerprint,
        schema_fingerprint=schema_fingerprint,
        evidence_mode=SYNTHETIC_EXACT_MODE,
        generation=generation,
        sql=None,
        execution=None,
        status=EvidenceStatus.UNAVAILABLE,
        reasons=(reason,),
    )


def capture_semantic_evidence(
    *,
    case_id: str,
    split: Literal["dev", "test"],
    datasource_id: UUID,
    dataset_fingerprint: str,
    fixture_fingerprint: str,
    schema_fingerprint: str,
    synthetic_fixture_fingerprint: str,
    validated_sql: ValidatedSQL | EvaluationValidatedSQL | None,
    execution_result: QueryExecutionResult | None,
    chatbi_result: ChatBIResult,
    provider_invocations: Iterable[LLMProviderInvocation],
    prompt_version: str,
    reasoning_mode: str,
    evidence_mode: Literal["synthetic_exact_v1", "sensitive_future"] = SYNTHETIC_EXACT_MODE,
) -> SemanticEvidenceRecord:
    """Capture evidence from authoritative in-memory evaluation state."""
    if evidence_mode != SYNTHETIC_EXACT_MODE:
        raise SemanticEvidenceError("only synthetic_exact_v1 is implemented")
    if fixture_fingerprint != synthetic_fixture_fingerprint:
        raise SemanticEvidenceError("synthetic exact evidence requires the frozen fixture")
    native_validated_sql: ValidatedSQL | None
    if isinstance(validated_sql, EvaluationValidatedSQL):
        native_validated_sql = validated_sql.validated_sql
        validated_schema_fingerprint = validated_sql.evaluation_schema_fingerprint
    else:
        native_validated_sql = validated_sql
        validated_schema_fingerprint = (
            validated_sql.schema_fingerprint if validated_sql is not None else None
        )
    if native_validated_sql is not None:
        if native_validated_sql.datasource_id != datasource_id:
            raise SemanticEvidenceError("validated SQL datasource does not match evidence")
        if validated_schema_fingerprint != schema_fingerprint:
            raise SemanticEvidenceError("validated SQL schema fingerprint does not match evidence")
    if chatbi_result.datasource_id != datasource_id:
        raise SemanticEvidenceError("ChatBI result datasource does not match evidence")
    if execution_result is not None and execution_result.datasource_id != datasource_id:
        raise SemanticEvidenceError("execution result datasource does not match evidence")
    generation = _generation_evidence(
        prompt_version=prompt_version,
        reasoning_mode=reasoning_mode,
        result=chatbi_result,
        invocations=provider_invocations,
    )
    execution = _result_evidence(execution_result, mode=evidence_mode)
    trace_events = {event.event for event in chatbi_result.trace}
    capture_reasons = list(_generation_reasons(generation))
    if validated_sql is None:
        capture_reasons.append("validated_sql_unavailable")
        if chatbi_result.redacted_sql is None:
            capture_reasons.append("sql_candidate_unavailable")
        if not {"validation_rejected", "validation_accepted"} & trace_events:
            capture_reasons.append("validation_not_reached")
        if execution_result is None and not {"sql_executed", "execution_failed"} & trace_events:
            capture_reasons.append("execution_not_reached")
        return _bounded_record(
            case_id=case_id,
            split=split,
            datasource_id=datasource_id,
            dataset_fingerprint=dataset_fingerprint,
            fixture_fingerprint=fixture_fingerprint,
            schema_fingerprint=schema_fingerprint,
            evidence_mode=evidence_mode,
            generation=generation,
            sql=None,
            execution=execution,
            status=EvidenceStatus.UNAVAILABLE,
            reasons=capture_reasons,
        )
    try:
        if native_validated_sql is None:
            raise SemanticEvidenceError("validated SQL is required for SQL evidence")
        statement = parse_one(native_validated_sql.normalized_sql, read="postgres")
        _check_ast_node_bound(statement)
        sql_evidence = _query_evidence(statement)
    except _EvidenceBoundExceeded as error:
        return _bounded_record(
            case_id=case_id,
            split=split,
            datasource_id=datasource_id,
            dataset_fingerprint=dataset_fingerprint,
            fixture_fingerprint=fixture_fingerprint,
            schema_fingerprint=schema_fingerprint,
            evidence_mode=evidence_mode,
            generation=generation,
            sql=None,
            execution=execution,
            status=EvidenceStatus.TRUNCATED,
            reasons=(*capture_reasons, error.reason),
        )
    except (SqlglotError, RecursionError, ValueError):
        return _bounded_record(
            case_id=case_id,
            split=split,
            datasource_id=datasource_id,
            dataset_fingerprint=dataset_fingerprint,
            fixture_fingerprint=fixture_fingerprint,
            schema_fingerprint=schema_fingerprint,
            evidence_mode=evidence_mode,
            generation=generation,
            sql=None,
            execution=execution,
            status=EvidenceStatus.UNAVAILABLE,
            reasons=(*capture_reasons, "validated_sql_parse_unavailable"),
        )
    if execution is not None:
        execution = execution.model_copy(
            update={"row_order_semantics": ("ordered" if sql_evidence.order_by else "unordered")}
        )
    if execution_result is None:
        capture_reasons.append("execution_not_reached")
    return _bounded_record(
        case_id=case_id,
        split=split,
        datasource_id=datasource_id,
        dataset_fingerprint=dataset_fingerprint,
        fixture_fingerprint=fixture_fingerprint,
        schema_fingerprint=schema_fingerprint,
        evidence_mode=evidence_mode,
        generation=generation,
        sql=sql_evidence,
        execution=execution,
        status=EvidenceStatus.AVAILABLE,
        reasons=capture_reasons,
    )


__all__ = [
    "MAX_AST_NODES",
    "MAX_EXACT_RESULT_BYTES",
    "MAX_EXACT_ROWS",
    "MAX_EXPRESSION_DEPTH",
    "MAX_RECORD_BYTES",
    "ROW_FINGERPRINT_VERSION",
    "SQL_PARSER_VERSION",
    "EvidenceMode",
    "EvidenceStatus",
    "SemanticEvidenceError",
    "SemanticEvidenceRecord",
    "build_unavailable_semantic_evidence",
    "capture_semantic_evidence",
    "evidence_digest_for",
]
