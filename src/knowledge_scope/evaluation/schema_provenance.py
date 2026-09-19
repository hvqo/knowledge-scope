"""Explicit provenance bindings used by evaluation sidecars.

Production ``ValidatedSQL.schema_fingerprint`` remains the native
``SchemaSnapshot`` fingerprint. A provider evaluation may also carry a
verified catalog-contract fingerprint; the two values are intentionally
separate and the native validation result is never rewritten.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from knowledge_scope.chatbi.nl2sql_models import ValidatedSQL

_FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class EvaluationValidatedSQL:
    """A native validation result bound to a verified evaluation contract."""

    validated_sql: ValidatedSQL
    evaluation_schema_fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.validated_sql, ValidatedSQL):
            raise TypeError("evaluation binding requires a native ValidatedSQL result")
        if not _FINGERPRINT_PATTERN.fullmatch(self.evaluation_schema_fingerprint):
            raise ValueError("evaluation schema fingerprint must be a SHA-256 hex digest")

    @property
    def native_schema_fingerprint(self) -> str:
        """Return the fingerprint produced by the production validator."""
        return self.validated_sql.schema_fingerprint


__all__ = ["EvaluationValidatedSQL"]
