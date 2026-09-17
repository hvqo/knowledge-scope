"""Typed result-shape contracts for structured NL2SQL generation.

The contract describes the intended shape of a result.  It is deliberately
smaller than a SQL AST and is never an authorization mechanism; the existing
PostgreSQL safety validator remains the final SQL trust boundary.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Literal, NoReturn

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)
from sqlglot import exp, parse
from sqlglot.errors import SqlglotError

from .errors import ChatBIError, ChatBIErrorCategory
from .policy import QueryPolicy
from .schema_models import (
    SchemaObjectKind,
    SchemaRelation,
    SchemaSnapshot,
    SemanticSchemaContext,
)

RESULT_CONTRACT_VERSION: Final = "1.0"
RESULT_CONTRACT_MAX_COLUMNS: Final = 128
RESULT_CONTRACT_MAX_IDENTIFIER_LENGTH: Final = 512
RESULT_CONTRACT_MAX_EXPRESSION_LENGTH: Final = 4_000


def build_result_contract_prompt_example() -> dict[str, object]:
    """Return the canonical structural example embedded in the NL2SQL prompt.

    The schema/relation names are placeholders for structure only.  The
    generation prompt separately requires model output to use only the
    approved discovered schema.  Keeping this example next to the production
    result-contract models lets parser tests detect prompt/schema drift.
    """
    return {
        "result_contract": {
            "contract_version": RESULT_CONTRACT_VERSION,
            "row_grain": "grouped",
            "grain_keys": ["public.example.category"],
            "output_columns": [
                {
                    "kind": "source",
                    "source": "public.example.category",
                    "alias": "category",
                },
                {
                    "kind": "aggregate",
                    "function": "sum",
                    "source": "public.example.amount",
                    "alias": "total_amount",
                },
                {
                    "kind": "derived",
                    "expression": "amount * 1.0",
                    "source_columns": ["public.example.amount"],
                    "alias": "normalized_amount",
                },
            ],
            "group_by": ["public.example.category"],
            "order_by": [
                {"key": "public.example.category", "direction": "asc"},
                {"key": "total_amount", "direction": "desc"},
            ],
            "limit": None,
        },
        "sql": (
            "SELECT category, SUM(amount) AS total_amount, amount * 1.0 AS normalized_amount "
            "FROM public.example GROUP BY category ORDER BY category ASC, total_amount DESC"
        ),
    }


class ResultGrain(StrEnum):
    """The small set of generic row-grain descriptions supported by v1."""

    SCALAR = "scalar"
    DETAIL = "detail"
    GROUPED = "grouped"


class OutputColumnKind(StrEnum):
    """Semantic kind of one ordered output projection."""

    SOURCE = "source"
    AGGREGATE = "aggregate"
    DERIVED = "derived"


class AggregateFunction(StrEnum):
    """The aggregate functions that can be named by the compact contract."""

    COUNT = "count"
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"


class SortDirection(StrEnum):
    ASC = "asc"
    DESC = "desc"


class _ResultContractModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        hide_input_in_errors=True,
    )


def _normalize_text(value: str, field_name: str, *, max_length: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = unicodedata.normalize("NFC", value.strip())
    if not normalized:
        raise ValueError(f"{field_name} must contain non-whitespace characters")
    if len(normalized) > max_length:
        raise ValueError(f"{field_name} exceeds its length bound")
    if any(unicodedata.category(character).startswith("C") for character in normalized):
        raise ValueError(f"{field_name} must not contain control characters")
    return normalized


def _normalize_identifier(value: str, field_name: str = "identifier") -> str:
    return _normalize_text(
        value,
        field_name,
        max_length=RESULT_CONTRACT_MAX_IDENTIFIER_LENGTH,
    )


def _normalize_name_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raise ValueError(f"{field_name} must be a sequence of names")
    try:
        values = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(f"{field_name} must be a sequence of names") from error
    return tuple(_normalize_identifier(item, field_name) for item in values)


class ResultOutputColumn(_ResultContractModel):
    """One ordered source, aggregate, or derived result column.

    ``source`` and ``source_columns`` use canonical schema-qualified source
    references such as ``public.sales.amount``.  A derived expression remains
    a small semantic SQL expression string; it is not accepted as an AST or as
    an authorization input.
    """

    kind: OutputColumnKind
    source: StrictStr | None = Field(
        default=None,
        max_length=RESULT_CONTRACT_MAX_IDENTIFIER_LENGTH,
    )
    function: AggregateFunction | None = None
    expression: StrictStr | None = Field(
        default=None,
        max_length=RESULT_CONTRACT_MAX_EXPRESSION_LENGTH,
    )
    source_columns: tuple[StrictStr, ...] = Field(default=(), max_length=32)
    alias: StrictStr | None = Field(
        default=None,
        max_length=RESULT_CONTRACT_MAX_IDENTIFIER_LENGTH,
    )

    @field_validator("source", "alias")
    @classmethod
    def normalize_optional_identifiers(cls, value: str | None, info: object) -> str | None:
        if value is None:
            return None
        field_name = getattr(info, "field_name", "identifier")
        return _normalize_identifier(value, str(field_name))

    @field_validator("expression")
    @classmethod
    def normalize_expression(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _normalize_text(
            value,
            "expression",
            max_length=RESULT_CONTRACT_MAX_EXPRESSION_LENGTH,
        )

    @field_validator("source_columns", mode="before")
    @classmethod
    def normalize_sources(cls, value: object) -> tuple[str, ...]:
        return _normalize_name_tuple(value, "source_columns")

    @model_validator(mode="after")
    def validate_shape(self) -> ResultOutputColumn:
        if self.kind is OutputColumnKind.SOURCE:
            if self.source is None or self.function is not None or self.expression is not None:
                raise ValueError("source output columns require only source")
            if self.source_columns:
                raise ValueError("source output columns cannot declare source_columns")
        elif self.kind is OutputColumnKind.AGGREGATE:
            if self.function is None or self.expression is not None or self.source_columns:
                raise ValueError("aggregate output columns require function and optional source")
            if self.source is None and self.function is not AggregateFunction.COUNT:
                raise ValueError("only count may aggregate without a source column")
        elif self.kind is OutputColumnKind.DERIVED:
            if self.expression is None or self.source is not None or self.function is not None:
                raise ValueError("derived output columns require expression only")
        return self


class ResultOrderByTerm(_ResultContractModel):
    """One semantic output/source ordering term."""

    key: StrictStr = Field(min_length=1, max_length=RESULT_CONTRACT_MAX_IDENTIFIER_LENGTH)
    direction: SortDirection

    @field_validator("key")
    @classmethod
    def normalize_key(cls, value: str) -> str:
        return _normalize_identifier(value, "order_by key")

    @field_validator("direction", mode="before")
    @classmethod
    def normalize_direction(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value


class ResultContract(_ResultContractModel):
    """Compact semantic result contract paired with one generated SQL query."""

    contract_version: Literal["1.0"] = RESULT_CONTRACT_VERSION
    row_grain: ResultGrain
    grain_keys: tuple[StrictStr, ...] = Field(default=(), max_length=32)
    output_columns: tuple[ResultOutputColumn, ...] = Field(
        min_length=1,
        max_length=RESULT_CONTRACT_MAX_COLUMNS,
    )
    group_by: tuple[StrictStr, ...] = Field(default=(), max_length=32)
    order_by: tuple[ResultOrderByTerm, ...] = Field(default=(), max_length=32)
    limit: StrictInt | None = Field(default=None, ge=0, le=100_000)

    @field_validator("grain_keys", "group_by", mode="before")
    @classmethod
    def normalize_key_sequences(cls, value: object, info: object) -> tuple[str, ...]:
        field_name = getattr(info, "field_name", "keys")
        return _normalize_name_tuple(value, str(field_name))

    @model_validator(mode="after")
    def validate_structure(self) -> ResultContract:
        if self.contract_version != RESULT_CONTRACT_VERSION:
            raise ValueError("unsupported result contract version")
        if len(set(self.grain_keys)) != len(self.grain_keys):
            raise ValueError("grain_keys must be unique")
        if len(set(self.group_by)) != len(self.group_by):
            raise ValueError("group_by keys must be unique")
        if len({term.key for term in self.order_by}) != len(self.order_by):
            raise ValueError("order_by keys must be unique")

        output_signatures = {
            (
                column.kind,
                column.source,
                column.function,
                column.expression,
                column.source_columns,
            )
            for column in self.output_columns
        }
        if len(output_signatures) != len(self.output_columns):
            raise ValueError("output_columns contain duplicate structural entries")
        aliases = tuple(column.alias for column in self.output_columns if column.alias is not None)
        if len(set(aliases)) != len(aliases):
            raise ValueError("output column aliases must be unique")

        if self.row_grain is ResultGrain.SCALAR:
            if self.grain_keys or self.group_by:
                raise ValueError("scalar results cannot declare grain or grouping keys")
        elif self.row_grain is ResultGrain.DETAIL:
            if self.group_by:
                raise ValueError("detail results cannot declare group_by")
        elif self.row_grain is ResultGrain.GROUPED:
            if not self.grain_keys or not self.group_by:
                raise ValueError("grouped results require grain_keys and group_by")
            if not set(self.grain_keys) <= set(self.group_by):
                raise ValueError("group_by must retain every stable grain key")
        return self


@dataclass(frozen=True, slots=True)
class _ColumnReference:
    schema: str
    relation: str
    column: str

    @property
    def key(self) -> tuple[str, str, str]:
        return self.schema, self.relation, self.column


@dataclass(frozen=True, slots=True)
class _SourceBinding:
    relation_label: str | None
    columns: dict[str, _ColumnReference | None]


def _contract_error(message: str, *, inconsistent: bool = False) -> NoReturn:
    category = (
        ChatBIErrorCategory.RESULT_CONTRACT_INCONSISTENT
        if inconsistent
        else ChatBIErrorCategory.RESULT_CONTRACT_INVALID
    )
    raise ChatBIError(category, message)


def _split_reference(value: str) -> tuple[tuple[str, bool], ...]:
    """Split a dotted identifier while respecting PostgreSQL double quotes."""
    parts: list[tuple[str, bool]] = []
    current: list[str] = []
    quoted = False
    current_was_quoted = False
    index = 0
    while index < len(value):
        character = value[index]
        if quoted:
            if character == '"':
                if index + 1 < len(value) and value[index + 1] == '"':
                    current.append('"')
                    index += 2
                    continue
                quoted = False
            else:
                current.append(character)
        elif character == '"':
            if current:
                _contract_error("source references must use complete identifiers")
            quoted = True
            current_was_quoted = True
        elif character == ".":
            if not current:
                _contract_error("source references must contain non-empty identifier parts")
            parts.append(("".join(current), current_was_quoted))
            current = []
            current_was_quoted = False
        else:
            if current_was_quoted and not quoted:
                _contract_error("source references must use complete identifiers")
            current.append(character)
        index += 1
    if quoted or not current:
        _contract_error("source references contain an unterminated or empty identifier")
    parts.append(("".join(current), current_was_quoted))
    if len(parts) not in {2, 3}:
        _contract_error("source references must be relation.column or schema.relation.column")
    return tuple(parts)


def _identifier_matches(value: str, quoted: bool, actual: str) -> bool:
    return value == actual if quoted else value.lower() == actual


def _is_allowed_relation(relation: SchemaRelation, policy: QueryPolicy) -> bool:
    schema = relation.schema_name.lower()
    if schema in {"pg_catalog", "information_schema", "pg_temp", "pg_toast"} or schema.startswith(
        ("pg_temp_", "pg_toast_")
    ):
        return False
    if any(schema == denied.lower() for denied in policy.denied_schemas):
        return False
    return any(schema == allowed.lower() for allowed in policy.allowed_schemas)


def _included_relations(
    snapshot: SchemaSnapshot,
    context: SemanticSchemaContext,
    policy: QueryPolicy,
) -> tuple[SchemaRelation, ...]:
    included = set(context.included_relations)
    return tuple(
        relation
        for relation in snapshot.relations
        if f"{relation.schema_name}.{relation.name}" in included
        and _is_allowed_relation(relation, policy)
        and relation.kind is SchemaObjectKind.TABLE
    )


def _resolve_reference(
    value: str,
    *,
    snapshot: SchemaSnapshot,
    context: SemanticSchemaContext,
    policy: QueryPolicy,
) -> _ColumnReference:
    parts = _split_reference(value)
    if len(parts) == 3:
        schema_part, relation_part, column_part = parts
        candidates = tuple(
            relation
            for relation in _included_relations(snapshot, context, policy)
            if _identifier_matches(schema_part[0], schema_part[1], relation.schema_name)
            and _identifier_matches(relation_part[0], relation_part[1], relation.name)
        )
    else:
        relation_part, column_part = parts
        candidates = tuple(
            relation
            for relation in _included_relations(snapshot, context, policy)
            if _identifier_matches(relation_part[0], relation_part[1], relation.name)
        )
    if len(candidates) != 1:
        _contract_error("result contract references an unknown or ambiguous relation")
    relation = candidates[0]
    column_name = column_part[0]
    columns = tuple(
        column
        for column in relation.columns
        if _identifier_matches(column_name, column_part[1], column.name)
    )
    if len(columns) != 1:
        _contract_error("result contract references an unknown or ambiguous source column")
    return _ColumnReference(relation.schema_name, relation.name, columns[0].name)


def _contract_descriptor(
    column: ResultOutputColumn,
    *,
    snapshot: SchemaSnapshot,
    context: SemanticSchemaContext,
    policy: QueryPolicy,
) -> tuple[object, ...]:
    if column.kind is OutputColumnKind.SOURCE:
        reference = _resolve_reference(
            column.source or "",
            snapshot=snapshot,
            context=context,
            policy=policy,
        )
        return ("source", reference.key)
    if column.kind is OutputColumnKind.AGGREGATE:
        reference = (
            _resolve_reference(
                column.source,
                snapshot=snapshot,
                context=context,
                policy=policy,
            ).key
            if column.source is not None
            else None
        )
        return ("aggregate", column.function.value if column.function else None, reference)
    sources = tuple(
        _resolve_reference(
            source,
            snapshot=snapshot,
            context=context,
            policy=policy,
        ).key
        for source in column.source_columns
    )
    return ("derived", column.expression, sources)


def validate_result_contract(
    contract: object,
    *,
    schema_snapshot: SchemaSnapshot,
    semantic_context: SemanticSchemaContext,
    policy: QueryPolicy,
) -> ResultContract:
    """Validate contract structure and every referenced source object."""
    if not isinstance(contract, ResultContract):
        _contract_error("result contract is invalid")
    try:
        contract = ResultContract.model_validate(contract.model_dump())
    except Exception:
        _contract_error("result contract is invalid")
    if contract.limit is not None and contract.limit > policy.max_rows:
        _contract_error("result contract limit exceeds the query policy")

    for value in (*contract.grain_keys, *contract.group_by):
        _resolve_reference(
            value,
            snapshot=schema_snapshot,
            context=semantic_context,
            policy=policy,
        )
    for column in contract.output_columns:
        _contract_descriptor(
            column,
            snapshot=schema_snapshot,
            context=semantic_context,
            policy=policy,
        )
    aliases = {column.alias for column in contract.output_columns if column.alias is not None}
    for term in contract.order_by:
        if term.key not in aliases:
            _resolve_reference(
                term.key,
                snapshot=schema_snapshot,
                context=semantic_context,
                policy=policy,
            )
    return contract


def _node_identifier(node: object) -> tuple[str, bool] | None:
    if isinstance(node, exp.Identifier):
        return node.this, node.quoted
    if isinstance(node, str) and node:
        return node, False
    return None


def _binding_lookup(bindings: dict[str, _SourceBinding], value: str) -> _SourceBinding | None:
    return bindings.get(value.lower())


def _relation_for_table(
    table: exp.Table,
    *,
    snapshot: SchemaSnapshot,
    context: SemanticSchemaContext,
    policy: QueryPolicy,
) -> SchemaRelation:
    table_part = _node_identifier(table.this)
    if table_part is None:
        _contract_error("SQL table reference cannot be resolved", inconsistent=True)
    schema_node = table.args.get("db")
    schema_part = _node_identifier(schema_node) if schema_node is not None else None
    if table.args.get("catalog") is not None:
        _contract_error("cross-database SQL references are not supported", inconsistent=True)
    candidates = tuple(
        relation
        for relation in _included_relations(snapshot, context, policy)
        if _identifier_matches(table_part[0], table_part[1], relation.name)
        and (
            schema_part is None
            or _identifier_matches(schema_part[0], schema_part[1], relation.schema_name)
        )
    )
    if len(candidates) != 1:
        _contract_error("SQL table reference is unknown or ambiguous", inconsistent=True)
    return candidates[0]


def _binding_for_table(
    table: exp.Table,
    *,
    cte_bindings: dict[str, _SourceBinding],
    snapshot: SchemaSnapshot,
    context: SemanticSchemaContext,
    policy: QueryPolicy,
) -> tuple[str, _SourceBinding]:
    table_part = _node_identifier(table.this)
    if table_part is None:
        _contract_error("SQL table reference cannot be resolved", inconsistent=True)
    table_name = table_part[0]
    cte = _binding_lookup(cte_bindings, table_name) if table.args.get("db") is None else None
    if cte is not None:
        binding = cte
    else:
        relation = _relation_for_table(
            table,
            snapshot=snapshot,
            context=context,
            policy=policy,
        )
        binding = _SourceBinding(
            relation_label=f"{relation.schema_name}.{relation.name}",
            columns={
                column.name.lower(): _ColumnReference(
                    relation.schema_name,
                    relation.name,
                    column.name,
                )
                for column in relation.columns
            },
        )
    alias = table.alias
    visible_name = alias if alias else table_name
    return visible_name, binding


def _source_items(select: exp.Select) -> tuple[exp.Expression, ...]:
    from_clause = select.args.get("from_")
    items: list[exp.Expression] = []
    if from_clause is not None:
        if from_clause.this is not None:
            items.append(from_clause.this)
        items.extend(from_clause.expressions)
    for join in select.args.get("joins") or ():
        if join.this is not None:
            items.append(join.this)
    return tuple(items)


def _query_cte_bindings(
    query: exp.Expression,
    inherited: dict[str, _SourceBinding],
    *,
    snapshot: SchemaSnapshot,
    context: SemanticSchemaContext,
    policy: QueryPolicy,
) -> dict[str, _SourceBinding]:
    bindings = dict(inherited)
    with_clause = query.args.get("with_")
    if not isinstance(with_clause, exp.With):
        return bindings
    for cte in with_clause.expressions:
        alias = cte.args.get("alias")
        alias_part = _node_identifier(alias.this) if isinstance(alias, exp.TableAlias) else None
        if alias_part is None:
            _contract_error("CTE must have a valid alias", inconsistent=True)
        binding = _binding_for_query(
            cte.this,
            bindings,
            snapshot=snapshot,
            context=context,
            policy=policy,
        )
        bindings[alias_part[0].lower()] = binding
    return bindings


def _select_scope(
    select: exp.Select,
    inherited: dict[str, _SourceBinding],
    *,
    snapshot: SchemaSnapshot,
    context: SemanticSchemaContext,
    policy: QueryPolicy,
) -> dict[str, _SourceBinding]:
    ctes = _query_cte_bindings(
        select,
        inherited,
        snapshot=snapshot,
        context=context,
        policy=policy,
    )
    result: dict[str, _SourceBinding] = {}
    for source in _source_items(select):
        if isinstance(source, exp.Table):
            name, binding = _binding_for_table(
                source,
                cte_bindings=ctes,
                snapshot=snapshot,
                context=context,
                policy=policy,
            )
        elif isinstance(source, exp.Subquery):
            alias = source.alias
            if not alias:
                _contract_error("subqueries must have aliases", inconsistent=True)
            name, binding = (
                alias,
                _binding_for_query(
                    source.this,
                    ctes,
                    snapshot=snapshot,
                    context=context,
                    policy=policy,
                ),
            )
        else:
            _contract_error("SQL source is outside the result-contract subset", inconsistent=True)
        key = name.lower()
        if key in result:
            _contract_error("SQL source aliases are ambiguous", inconsistent=True)
        result[key] = binding
    return result


def _binding_for_query(
    query: exp.Expression,
    inherited: dict[str, _SourceBinding],
    *,
    snapshot: SchemaSnapshot,
    context: SemanticSchemaContext,
    policy: QueryPolicy,
) -> _SourceBinding:
    if isinstance(query, exp.SetOperation):
        return _binding_for_query(
            query.left,
            _query_cte_bindings(
                query,
                inherited,
                snapshot=snapshot,
                context=context,
                policy=policy,
            ),
            snapshot=snapshot,
            context=context,
            policy=policy,
        )
    if not isinstance(query, exp.Select):
        _contract_error("SQL subquery is outside the result-contract subset", inconsistent=True)
    scope = _select_scope(
        query,
        inherited,
        snapshot=snapshot,
        context=context,
        policy=policy,
    )
    columns: dict[str, _ColumnReference | None] = {}
    for index, expression in enumerate(query.expressions):
        base = expression.this if isinstance(expression, exp.Alias) else expression
        output_name = expression.alias_or_name or getattr(base, "name", "") or f"_col_{index}"
        reference = _canonical_sql_column(base, scope)
        columns[output_name.lower()] = reference
    return _SourceBinding(relation_label=None, columns=columns)


def _canonical_sql_column(
    column: object,
    scope: dict[str, _SourceBinding],
) -> _ColumnReference | None:
    if not isinstance(column, exp.Column):
        return None
    identifier = _node_identifier(column.this)
    if identifier is None:
        _contract_error("SQL column reference cannot be resolved", inconsistent=True)
    table_part = _node_identifier(column.args.get("table"))
    db_part = _node_identifier(column.args.get("db"))
    name = identifier[0]
    if table_part is not None:
        binding = _binding_lookup(scope, table_part[0])
        if binding is None:
            _contract_error("SQL column qualifier is not visible", inconsistent=True)
        if db_part is not None and binding.relation_label != f"{db_part[0]}.{table_part[0]}":
            _contract_error("SQL column qualifier does not match its source", inconsistent=True)
        reference = binding.columns.get(name.lower())
        if reference is None and name.lower() not in binding.columns:
            _contract_error("SQL column is not visible", inconsistent=True)
        if reference is None:
            _contract_error("SQL column does not resolve to a physical source", inconsistent=True)
        return reference
    matches = [
        reference
        for binding in scope.values()
        for key, reference in binding.columns.items()
        if key == name.lower() and reference is not None
    ]
    if len(matches) != 1:
        _contract_error("unqualified SQL column is unknown or ambiguous", inconsistent=True)
    return matches[0]


def _all_columns(
    expression: exp.Expression,
    scope: dict[str, _SourceBinding],
) -> tuple[_ColumnReference, ...]:
    references: list[_ColumnReference] = []
    for node in expression.walk():
        if isinstance(node, exp.Column):
            reference = _canonical_sql_column(node, scope)
            if reference is None:
                _contract_error("SQL expression contains an unresolved column", inconsistent=True)
            references.append(reference)
    return tuple(references)


def _expression_signature(
    expression: exp.Expression,
    scope: dict[str, _SourceBinding],
) -> tuple[object, ...]:
    if isinstance(expression, exp.Column):
        reference = _canonical_sql_column(expression, scope)
        if reference is None:
            _contract_error("SQL expression contains an unresolved column", inconsistent=True)
        return ("column", reference.key)
    if isinstance(expression, exp.Star):
        return ("star",)
    if isinstance(expression, exp.Literal):
        return ("literal", expression.is_string, expression.this)
    values: list[object] = []
    for key in sorted(expression.arg_types):
        if key in {"alias", "comments"}:
            continue
        value = expression.args.get(key)
        if isinstance(value, exp.Expression):
            values.append((key, _expression_signature(value, scope)))
        elif isinstance(value, list):
            values.append(
                (
                    key,
                    tuple(
                        _expression_signature(item, scope)
                        if isinstance(item, exp.Expression)
                        else item
                        for item in value
                    ),
                )
            )
        elif isinstance(value, (str, int, float, bool)) or value is None:
            values.append((key, value))
    return (type(expression).__name__, tuple(values))


def _aggregate_signature(
    expression: exp.Expression,
    scope: dict[str, _SourceBinding],
) -> tuple[object, ...] | None:
    aggregate_types = {
        exp.Count: AggregateFunction.COUNT.value,
        exp.Sum: AggregateFunction.SUM.value,
        exp.Avg: AggregateFunction.AVG.value,
        exp.Min: AggregateFunction.MIN.value,
        exp.Max: AggregateFunction.MAX.value,
    }
    function_name = next(
        (
            name
            for aggregate_type, name in aggregate_types.items()
            if isinstance(expression, aggregate_type)
        ),
        None,
    )
    if function_name is None or isinstance(expression, exp.Window):
        return None
    if expression.args.get("distinct"):
        _contract_error(
            "DISTINCT aggregates are outside the result-contract subset",
            inconsistent=True,
        )
    operand = expression.this
    if isinstance(operand, exp.Star):
        reference = None
    elif isinstance(operand, exp.Column):
        reference = _canonical_sql_column(operand, scope)
        if reference is None:
            _contract_error("aggregate source column cannot be resolved", inconsistent=True)
        reference = reference.key
    else:
        return None
    return ("aggregate", function_name, reference)


def _sql_projection_signature(
    expression: exp.Expression,
    scope: dict[str, _SourceBinding],
) -> tuple[object, ...]:
    base = expression.this if isinstance(expression, exp.Alias) else expression
    if isinstance(base, exp.Star):
        _contract_error("SELECT * cannot satisfy an exact result contract", inconsistent=True)
    if isinstance(base, exp.Column):
        reference = _canonical_sql_column(base, scope)
        if reference is None:
            _contract_error("SQL projection column cannot be resolved", inconsistent=True)
        return ("source", reference.key)
    aggregate = _aggregate_signature(base, scope)
    if aggregate is not None:
        return aggregate
    references = tuple(sorted({reference.key for reference in _all_columns(base, scope)}))
    return ("derived", _expression_signature(base, scope), references)


def _contract_projection_signatures(
    contract: ResultContract,
    *,
    snapshot: SchemaSnapshot,
    context: SemanticSchemaContext,
    policy: QueryPolicy,
) -> tuple[tuple[object, ...], ...]:
    signatures: list[tuple[object, ...]] = []
    for column in contract.output_columns:
        if column.kind is OutputColumnKind.SOURCE:
            reference = _resolve_reference(
                column.source or "",
                snapshot=snapshot,
                context=context,
                policy=policy,
            )
            signatures.append(("source", reference.key))
        elif column.kind is OutputColumnKind.AGGREGATE:
            reference = (
                _resolve_reference(
                    column.source,
                    snapshot=snapshot,
                    context=context,
                    policy=policy,
                ).key
                if column.source is not None
                else None
            )
            signatures.append(("aggregate", column.function.value, reference))
        else:
            sources = tuple(
                _resolve_reference(
                    source,
                    snapshot=snapshot,
                    context=context,
                    policy=policy,
                ).key
                for source in column.source_columns
            )
            signatures.append(("derived", column.expression, sources))
    return tuple(signatures)


def _order_descriptor(
    key: str,
    *,
    contract: ResultContract,
    contract_signatures: tuple[tuple[object, ...], ...],
    snapshot: SchemaSnapshot,
    context: SemanticSchemaContext,
    policy: QueryPolicy,
) -> tuple[object, ...]:
    aliases = {
        column.alias: signature
        for column, signature in zip(contract.output_columns, contract_signatures, strict=True)
        if column.alias is not None
    }
    if key in aliases:
        return aliases[key]
    return (
        "source",
        _resolve_reference(
            key,
            snapshot=snapshot,
            context=context,
            policy=policy,
        ).key,
    )


def _sql_order_descriptor(
    expression: exp.Expression,
    *,
    scope: dict[str, _SourceBinding],
    contract: ResultContract,
    contract_signatures: tuple[tuple[object, ...], ...],
) -> tuple[object, ...]:
    base = expression.this if isinstance(expression, exp.Ordered) else expression
    if isinstance(base, exp.Literal) and base.is_int:
        position = int(base.this) - 1
        if 0 <= position < len(contract_signatures):
            return contract_signatures[position]
        _contract_error("ORDER BY position is outside the result contract", inconsistent=True)
    if (
        isinstance(base, exp.Column)
        and not base.table
        and base.name
        in {column.alias for column in contract.output_columns if column.alias is not None}
    ):
        alias = base.name
        return next(
            signature
            for column, signature in zip(contract.output_columns, contract_signatures, strict=True)
            if column.alias == alias
        )
    if isinstance(base, exp.Column):
        reference = _canonical_sql_column(base, scope)
        if reference is None:
            _contract_error("ORDER BY source cannot be resolved", inconsistent=True)
        return ("source", reference.key)
    _contract_error("ORDER BY expression is outside the result-contract subset", inconsistent=True)


def _leaf_selects(statement: exp.Expression) -> tuple[exp.Select, ...]:
    if isinstance(statement, exp.Select):
        return (statement,)
    if isinstance(statement, exp.SetOperation):
        return _leaf_selects(statement.left) + _leaf_selects(statement.right)
    _contract_error("SQL statement cannot be checked against a result contract", inconsistent=True)


def _top_level_limit(statement: exp.Expression) -> int | None:
    limit = statement.args.get("limit")
    if limit is None:
        return None
    expression = limit.args.get("expression")
    if not isinstance(expression, exp.Literal) or not expression.is_int:
        _contract_error("SQL LIMIT must be an integer literal", inconsistent=True)
    try:
        value = int(expression.this)
    except (TypeError, ValueError):
        _contract_error("SQL LIMIT must be an integer literal", inconsistent=True)
    if value < 0:
        _contract_error("SQL LIMIT must be non-negative", inconsistent=True)
    return value


def validate_sql_result_contract(
    sql: str,
    contract: ResultContract,
    *,
    schema_snapshot: SchemaSnapshot,
    semantic_context: SemanticSchemaContext,
    policy: QueryPolicy,
) -> None:
    """Check SQL projection/grain/order/limit against a validated contract."""
    if not isinstance(sql, str) or not sql.strip():
        _contract_error("SQL is required for result-contract validation", inconsistent=True)
    try:
        statements = parse(sql, read="postgres")
    except (RecursionError, SqlglotError, ValueError):
        _contract_error("SQL could not be parsed for result-contract validation", inconsistent=True)
    if len(statements) != 1 or statements[0] is None:
        _contract_error("exactly one SQL statement is required", inconsistent=True)
    statement = statements[0]
    contract = validate_result_contract(
        contract,
        schema_snapshot=schema_snapshot,
        semantic_context=semantic_context,
        policy=policy,
    )
    contract_signatures = _contract_projection_signatures(
        contract,
        snapshot=schema_snapshot,
        context=semantic_context,
        policy=policy,
    )
    inherited: dict[str, _SourceBinding] = {}
    query_ctes = _query_cte_bindings(
        statement,
        inherited,
        snapshot=schema_snapshot,
        context=semantic_context,
        policy=policy,
    )
    leaves = _leaf_selects(statement)
    for select in leaves:
        scope = _select_scope(
            select,
            query_ctes,
            snapshot=schema_snapshot,
            context=semantic_context,
            policy=policy,
        )
        actual_signatures = tuple(
            _sql_projection_signature(expression, scope) for expression in select.expressions
        )
        # Contract-derived expressions are checked structurally, with source
        # columns resolved in the same SQL source scope as the actual query.
        rebuilt_expected: list[tuple[object, ...]] = []
        for column, expected in zip(contract.output_columns, contract_signatures, strict=True):
            if column.kind is not OutputColumnKind.DERIVED:
                rebuilt_expected.append(expected)
                continue
            try:
                expression_statements = parse(f"SELECT {column.expression}", read="postgres")
            except (RecursionError, SqlglotError, ValueError):
                _contract_error("derived result-contract expression is invalid", inconsistent=True)
            if len(expression_statements) != 1 or not isinstance(
                expression_statements[0], exp.Select
            ):
                _contract_error("derived result-contract expression is invalid", inconsistent=True)
            derived_expression = expression_statements[0].expressions[0]
            source_keys = tuple(
                sorted({reference.key for reference in _all_columns(derived_expression, scope)})
            )
            declared_source_keys = expected[2]
            if source_keys != declared_source_keys:
                _contract_error(
                    "derived SQL sources do not match the result contract",
                    inconsistent=True,
                )
            rebuilt_expected.append(
                ("derived", _expression_signature(derived_expression, scope), declared_source_keys)
            )
        if actual_signatures != tuple(rebuilt_expected):
            _contract_error(
                "SQL projection does not match the result contract",
                inconsistent=True,
            )

        group = select.args.get("group")
        actual_group = (
            tuple(
                _canonical_sql_column(item, scope).key  # type: ignore[union-attr]
                for item in group.expressions
                if isinstance(item, exp.Column)
            )
            if isinstance(group, exp.Group)
            else ()
        )
        if isinstance(group, exp.Group) and len(actual_group) != len(group.expressions):
            _contract_error("SQL GROUP BY contains an unsupported expression", inconsistent=True)
        expected_group = tuple(
            _resolve_reference(
                value,
                snapshot=schema_snapshot,
                context=semantic_context,
                policy=policy,
            ).key
            for value in contract.group_by
        )
        if set(actual_group) != set(expected_group):
            _contract_error("SQL grouping keys do not match the result contract", inconsistent=True)
        if contract.row_grain is ResultGrain.DETAIL and select.args.get("distinct") is not None:
            _contract_error("DISTINCT would destroy the contracted detail grain", inconsistent=True)
        if contract.row_grain is ResultGrain.GROUPED and not actual_group:
            _contract_error("grouped result SQL must contain GROUP BY", inconsistent=True)

    top_order = statement.args.get("order")
    actual_order = (
        tuple(
            (
                _sql_order_descriptor(
                    item,
                    scope=_select_scope(
                        leaves[0],
                        query_ctes,
                        snapshot=schema_snapshot,
                        context=semantic_context,
                        policy=policy,
                    ),
                    contract=contract,
                    contract_signatures=contract_signatures,
                ),
                SortDirection.DESC.value if item.args.get("desc") else SortDirection.ASC.value,
            )
            for item in top_order.expressions
        )
        if isinstance(top_order, exp.Order)
        else ()
    )
    expected_order = tuple(
        (
            _order_descriptor(
                term.key,
                contract=contract,
                contract_signatures=contract_signatures,
                snapshot=schema_snapshot,
                context=semantic_context,
                policy=policy,
            ),
            term.direction.value,
        )
        for term in contract.order_by
    )
    if actual_order != expected_order:
        _contract_error("SQL ordering does not match the result contract", inconsistent=True)

    actual_limit = _top_level_limit(statement)
    if contract.limit is None and actual_limit is not None:
        _contract_error("SQL contains an unexpected LIMIT", inconsistent=True)
    if contract.limit is not None and actual_limit != contract.limit:
        _contract_error("SQL LIMIT does not match the result contract", inconsistent=True)


__all__ = [
    "RESULT_CONTRACT_VERSION",
    "AggregateFunction",
    "OutputColumnKind",
    "ResultContract",
    "ResultGrain",
    "ResultOrderByTerm",
    "ResultOutputColumn",
    "SortDirection",
    "build_result_contract_prompt_example",
    "validate_result_contract",
    "validate_sql_result_contract",
]
