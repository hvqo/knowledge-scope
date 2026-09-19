"""Shared, deterministic schema provenance for ChatBI evaluation artifacts.

This module is intentionally evaluation-only.  Provider preflight and semantic
diagnostics both convert their source-specific schema metadata into this one
canonical representation before hashing it.  Keeping the representation and
hashing contract here prevents a sidecar digest from drifting away from the
authoritative catalog digest.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any


class SchemaFingerprintError(ValueError):
    """Raised when schema metadata cannot enter the canonical contract."""


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise SchemaFingerprintError(
            "schema metadata is not deterministically serializable"
        ) from error


def _text(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise SchemaFingerprintError(f"schema metadata field {key!r} must be a non-empty string")
    return value


def _optional_text(mapping: Mapping[str, Any], key: str) -> str | None:
    value = mapping.get(key)
    if value is not None and not isinstance(value, str):
        raise SchemaFingerprintError(f"schema metadata field {key!r} must be a string or null")
    return value


def _balanced_outer_parentheses(value: str) -> bool:
    """Return whether *value* is enclosed by one balanced parenthesis pair."""
    if len(value) < 2 or value[0] != "(" or value[-1] != ")":
        return False
    depth = 0
    for index, character in enumerate(value):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0 and index != len(value) - 1:
                return False
            if depth < 0:
                return False
    return depth == 0


def normalize_check_definition(value: str | None) -> str | None:
    """Normalize equivalent PostgreSQL CHECK display forms.

    PostgreSQL's ``pg_get_constraintdef`` output can vary in harmless ways
    across server versions, notably redundant parentheses around the complete
    predicate or a numeric literal before a type cast.  This helper only
    removes those formatting differences; it does not parse or reinterpret a
    predicate.  The frozen representation remains ``CHECK (...)``.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise SchemaFingerprintError("schema check definition must be a string or null")
    normalized = " ".join(unicodedata.normalize("NFC", value).split())
    if not normalized:
        raise SchemaFingerprintError("schema check definition must not be empty")
    match = re.fullmatch(r"(?i:check)\s*\((.*)\)", normalized)
    if match is None:
        return normalized
    body = match.group(1).strip()
    while _balanced_outer_parentheses(body):
        body = body[1:-1].strip()
    body = re.sub(
        r"\(\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\)::",
        r"\1::",
        body,
    )
    return f"CHECK ({body})"


def _sequence(mapping: Mapping[str, Any], key: str) -> list[Any]:
    value = mapping.get(key)
    if not isinstance(value, (list, tuple)):
        raise SchemaFingerprintError(f"schema metadata field {key!r} must be an array")
    return list(value)


def canonical_schema_payload(
    *,
    schema: str,
    relations: Iterable[Mapping[str, Any]],
) -> dict[str, object]:
    """Build the one canonical catalog payload used by all evaluators.

    The shape deliberately matches the existing provider preflight contract:
    relation kind/name, ordered column identity/type/nullability, and the full
    constraint tuple including CHECK definitions.  Sorting is limited to
    catalog-order-independent collections; column order and composite-key
    order remain authoritative.
    """

    if not isinstance(schema, str) or not schema:
        raise SchemaFingerprintError("schema identity must be a non-empty string")

    canonical_relations: list[dict[str, object]] = []
    for relation in relations:
        if not isinstance(relation, Mapping):
            raise SchemaFingerprintError("schema relation metadata must be an object")
        name = _text(relation, "name")
        kind = _text(relation, "kind")
        raw_columns = _sequence(relation, "columns")
        raw_constraints = _sequence(relation, "constraints")

        columns: list[dict[str, object]] = []
        for column in raw_columns:
            if not isinstance(column, Mapping):
                raise SchemaFingerprintError("schema column metadata must be an object")
            position = column.get("position")
            if isinstance(position, bool) or not isinstance(position, int) or position < 1:
                raise SchemaFingerprintError("schema column position must be a positive integer")
            nullable = column.get("nullable")
            if not isinstance(nullable, bool):
                raise SchemaFingerprintError("schema column nullable flag must be boolean")
            columns.append(
                {
                    "position": position,
                    "name": _text(column, "name"),
                    "type": _text(column, "type"),
                    "nullable": nullable,
                }
            )
        columns.sort(key=lambda item: int(item["position"]))

        constraints: list[dict[str, object]] = []
        for constraint in raw_constraints:
            if not isinstance(constraint, Mapping):
                raise SchemaFingerprintError("schema constraint metadata must be an object")
            raw_constraint_columns = _sequence(constraint, "columns")
            if any(not isinstance(column, str) or not column for column in raw_constraint_columns):
                raise SchemaFingerprintError("schema constraint columns must be non-empty strings")
            raw_referenced_columns = _sequence(constraint, "referenced_columns")
            if any(not isinstance(column, str) or not column for column in raw_referenced_columns):
                raise SchemaFingerprintError(
                    "schema referenced constraint columns must be non-empty strings"
                )
            constraints.append(
                {
                    "type": _text(constraint, "type"),
                    "columns": list(raw_constraint_columns),
                    "referenced_schema": _optional_text(constraint, "referenced_schema"),
                    "referenced_relation": _optional_text(constraint, "referenced_relation"),
                    "referenced_columns": list(raw_referenced_columns),
                    "check_definition": normalize_check_definition(
                        _optional_text(constraint, "check_definition")
                    ),
                }
            )
        # Preserve the provider's frozen ordering contract.  The old
        # provider sorted each canonical constraint by its own SHA-256 before
        # hashing the complete payload; retaining that ordering keeps the
        # authoritative historical fingerprint stable while moving the
        # serialization and hashing code into this shared module.
        constraints.sort(
            key=lambda constraint: hashlib.sha256(
                _canonical_json(constraint).encode("utf-8")
            ).hexdigest()
        )

        canonical_relations.append(
            {
                "name": name,
                "kind": kind,
                "columns": columns,
                "constraints": constraints,
            }
        )

    canonical_relations.sort(key=lambda item: (str(item["kind"]), str(item["name"])))
    return {"schema": schema, "relations": canonical_relations}


def schema_fingerprint(payload: Mapping[str, object]) -> str:
    """Hash an already canonical schema payload with the shared contract."""

    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


__all__ = [
    "SchemaFingerprintError",
    "canonical_schema_payload",
    "normalize_check_definition",
    "schema_fingerprint",
]
