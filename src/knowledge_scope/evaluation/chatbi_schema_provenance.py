"""Authoritative PostgreSQL catalog provenance for ChatBI evaluation.

The provenance catalog is deliberately separate from the executable
``SchemaSnapshot``.  It observes the frozen benchmark contract, including
views and CHECK constraints, while the executable discovery policy may still
exclude those objects from prompts and SQL authorization.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import create_async_engine

from knowledge_scope.evaluation.chatbi_schema_fingerprint import (
    SchemaFingerprintError,
    canonical_schema_payload,
    schema_fingerprint,
)


class PostgreSQLSchemaProvenanceError(ValueError):
    """Raised when PostgreSQL metadata cannot form the frozen contract."""


@dataclass(frozen=True, slots=True)
class PostgreSQLSchemaProvenance:
    """Canonical catalog metadata and its shared evaluation fingerprint."""

    schema: str
    payload: Mapping[str, object]
    fingerprint: str

    @property
    def relation_labels(self) -> tuple[str, ...]:
        relations = self.payload.get("relations")
        if not isinstance(relations, list):
            raise PostgreSQLSchemaProvenanceError("canonical schema relations are invalid")
        labels: list[str] = []
        for relation in relations:
            if not isinstance(relation, Mapping):
                raise PostgreSQLSchemaProvenanceError("canonical schema relation is invalid")
            name = relation.get("name")
            if not isinstance(name, str):
                raise PostgreSQLSchemaProvenanceError("canonical schema relation name is invalid")
            labels.append(f"{self.schema}.{name}")
        return tuple(labels)


POSTGRES_SCHEMA_RELATIONS_QUERY = text(
    """
    SELECT
        c.relname AS relation_name,
        CASE c.relkind WHEN 'r' THEN 'table' WHEN 'v' THEN 'view' END AS relation_kind,
        a.attnum AS column_position,
        a.attname AS column_name,
        format_type(a.atttypid, a.atttypmod) AS data_type,
        a.attnotnull AS not_null
    FROM pg_catalog.pg_class AS c
    JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
    JOIN pg_catalog.pg_attribute AS a ON a.attrelid = c.oid
    WHERE n.nspname = :schema
      AND c.relkind IN ('r', 'v')
      AND a.attnum > 0
      AND NOT a.attisdropped
    ORDER BY c.relkind, c.relname, a.attnum
    """
)
POSTGRES_SCHEMA_CONSTRAINTS_QUERY = text(
    """
    SELECT
        c.relname AS relation_name,
        con.contype AS constraint_type,
        ARRAY(
            SELECT a.attname
            FROM unnest(con.conkey) WITH ORDINALITY AS key(attnum, position)
            JOIN pg_catalog.pg_attribute AS a
              ON a.attrelid = con.conrelid AND a.attnum = key.attnum
            ORDER BY key.position
        ) AS columns,
        rn.nspname AS referenced_schema,
        rc.relname AS referenced_relation,
        ARRAY(
            SELECT a.attname
            FROM unnest(con.confkey) WITH ORDINALITY AS key(attnum, position)
            JOIN pg_catalog.pg_attribute AS a
              ON a.attrelid = con.confrelid AND a.attnum = key.attnum
            ORDER BY key.position
        ) AS referenced_columns,
        CASE
            WHEN con.contype = 'c' THEN pg_catalog.pg_get_constraintdef(con.oid, true)
            ELSE NULL
        END AS check_definition
    FROM pg_catalog.pg_constraint AS con
    JOIN pg_catalog.pg_class AS c ON c.oid = con.conrelid
    JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
    LEFT JOIN pg_catalog.pg_class AS rc ON rc.oid = con.confrelid
    LEFT JOIN pg_catalog.pg_namespace AS rn ON rn.oid = rc.relnamespace
    WHERE n.nspname = :schema
      AND c.relkind = 'r'
      AND con.contype IN ('p', 'u', 'f', 'c')
    ORDER BY
        c.relname,
        con.contype,
        columns,
        referenced_schema,
        referenced_relation,
        referenced_columns,
        check_definition
    """
)


def _metadata_array(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    raise PostgreSQLSchemaProvenanceError("PostgreSQL schema metadata has an invalid array")


def _metadata_text(value: object, field_name: str) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str) or not value:
        raise PostgreSQLSchemaProvenanceError(
            f"PostgreSQL schema metadata field {field_name} is invalid"
        )
    return value


def canonical_postgres_schema_payload(
    relation_rows: Sequence[object],
    constraint_rows: Sequence[object],
    *,
    schema: str,
) -> dict[str, object]:
    """Convert catalog rows to the shared frozen canonical payload."""
    relation_map: dict[tuple[str, str], dict[str, object]] = {}
    for row in relation_rows:
        mapping = getattr(row, "_mapping", None)
        if not isinstance(mapping, Mapping):
            raise PostgreSQLSchemaProvenanceError("PostgreSQL relation row is invalid")
        relation_name = _metadata_text(mapping.get("relation_name"), "relation_name")
        relation_kind = _metadata_text(mapping.get("relation_kind"), "relation_kind")
        if relation_kind not in {"table", "view"}:
            raise PostgreSQLSchemaProvenanceError("PostgreSQL relation kind is invalid")
        key = (relation_kind, relation_name)
        relation = relation_map.setdefault(
            key,
            {"name": relation_name, "kind": relation_kind, "columns": [], "constraints": []},
        )
        columns = relation["columns"]
        if not isinstance(columns, list):
            raise PostgreSQLSchemaProvenanceError("PostgreSQL column list is invalid")
        position = mapping.get("column_position")
        if isinstance(position, bool) or not isinstance(position, int) or position < 1:
            raise PostgreSQLSchemaProvenanceError("PostgreSQL column position is invalid")
        not_null = mapping.get("not_null")
        if not isinstance(not_null, bool):
            raise PostgreSQLSchemaProvenanceError("PostgreSQL nullability metadata is invalid")
        columns.append(
            {
                "position": position,
                "name": _metadata_text(mapping.get("column_name"), "column_name"),
                "type": _metadata_text(mapping.get("data_type"), "data_type"),
                "nullable": not not_null,
            }
        )

    constraint_types = {"p": "primary_key", "u": "unique", "f": "foreign_key", "c": "check"}
    for row in constraint_rows:
        mapping = getattr(row, "_mapping", None)
        if not isinstance(mapping, Mapping):
            raise PostgreSQLSchemaProvenanceError("PostgreSQL constraint row is invalid")
        relation_name = _metadata_text(mapping.get("relation_name"), "relation_name")
        constraint_type = _metadata_text(mapping.get("constraint_type"), "constraint_type")
        canonical_type = constraint_types.get(constraint_type)
        if canonical_type is None:
            raise PostgreSQLSchemaProvenanceError("PostgreSQL constraint type is invalid")
        relation = relation_map.get(("table", relation_name))
        if relation is None:
            raise PostgreSQLSchemaProvenanceError(
                "PostgreSQL constraint references an undiscovered table"
            )
        constraints = relation["constraints"]
        if not isinstance(constraints, list):
            raise PostgreSQLSchemaProvenanceError("PostgreSQL constraint list is invalid")
        referenced_schema = mapping.get("referenced_schema")
        referenced_relation = mapping.get("referenced_relation")
        check_definition = mapping.get("check_definition")
        constraints.append(
            {
                "type": canonical_type,
                "columns": _metadata_array(mapping.get("columns")),
                "referenced_schema": (
                    _metadata_text(referenced_schema, "referenced_schema")
                    if referenced_schema is not None
                    else None
                ),
                "referenced_relation": (
                    _metadata_text(referenced_relation, "referenced_relation")
                    if referenced_relation is not None
                    else None
                ),
                "referenced_columns": _metadata_array(mapping.get("referenced_columns")),
                "check_definition": (
                    _metadata_text(check_definition, "check_definition")
                    if check_definition is not None
                    else None
                ),
            }
        )
    try:
        return canonical_schema_payload(schema=schema, relations=relation_map.values())
    except SchemaFingerprintError as error:
        raise PostgreSQLSchemaProvenanceError(
            "PostgreSQL schema metadata does not satisfy the canonical contract"
        ) from error


async def discover_postgres_schema_provenance(
    database_url: str,
    *,
    schema: str,
) -> PostgreSQLSchemaProvenance:
    """Read the authoritative catalog contract without changing executable policy."""
    try:
        parsed_url = make_url(database_url)
        if parsed_url.drivername == "postgresql":
            database_url = parsed_url.set(drivername="postgresql+asyncpg").render_as_string(
                hide_password=False
            )
    except Exception as error:
        raise PostgreSQLSchemaProvenanceError(
            "PostgreSQL schema provenance connection configuration is invalid"
        ) from error
    engine = create_async_engine(database_url)
    try:
        async with engine.begin() as connection:
            await connection.execute(text("SET TRANSACTION READ ONLY"))
            relations = await connection.execute(
                POSTGRES_SCHEMA_RELATIONS_QUERY,
                {"schema": schema},
            )
            constraints = await connection.execute(
                POSTGRES_SCHEMA_CONSTRAINTS_QUERY,
                {"schema": schema},
            )
            payload = canonical_postgres_schema_payload(
                relations.fetchall(),
                constraints.fetchall(),
                schema=schema,
            )
            return PostgreSQLSchemaProvenance(
                schema=schema,
                payload=payload,
                fingerprint=schema_fingerprint(payload),
            )
    except PostgreSQLSchemaProvenanceError:
        raise
    except (SQLAlchemyError, OSError) as error:
        raise PostgreSQLSchemaProvenanceError(
            "could not discover PostgreSQL schema provenance"
        ) from error
    finally:
        await engine.dispose()


__all__ = [
    "POSTGRES_SCHEMA_CONSTRAINTS_QUERY",
    "POSTGRES_SCHEMA_RELATIONS_QUERY",
    "PostgreSQLSchemaProvenance",
    "PostgreSQLSchemaProvenanceError",
    "canonical_postgres_schema_payload",
    "discover_postgres_schema_provenance",
]
