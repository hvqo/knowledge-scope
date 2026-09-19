"""Explicit, safe error categories for future ChatBI execution paths."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from knowledge_scope.llm.schemas import (
    StructuredOutputBoundaryDiagnostic,
    StructuredOutputBoundaryObservation,
)


class ChatBIErrorCategory(StrEnum):
    """Stable categories exposed to application-level error handling."""

    DATASOURCE_NOT_FOUND = "datasource_not_found"
    DATASOURCE_DISABLED = "datasource_disabled"
    UNSUPPORTED_DIALECT = "unsupported_dialect"
    CREDENTIAL_RESOLUTION_FAILED = "credential_resolution_failed"
    SCHEMA_DISCOVERY_FAILED = "schema_discovery_failed"
    INVALID_QUERY = "invalid_query"
    GENERATION_FAILED = "generation_failed"
    MALFORMED_MODEL_OUTPUT = "malformed_model_output"
    RESULT_CONTRACT_INVALID = "result_contract_invalid"
    RESULT_CONTRACT_INCONSISTENT = "result_contract_inconsistent"
    SQL_PARSE_ERROR = "sql_parse_error"
    UNSAFE_QUERY = "unsafe_query"
    POLICY_VIOLATION = "policy_violation"
    UNKNOWN_TABLE = "unknown_table"
    UNKNOWN_COLUMN = "unknown_column"
    DATASOURCE_UNAVAILABLE = "datasource_unavailable"
    EXECUTION_TIMEOUT = "execution_timeout"
    EXECUTION_FAILED = "execution_failed"
    RESULT_NORMALIZATION_FAILED = "result_normalization_failed"
    RESULT_SIZE_EXCEEDED = "result_size_exceeded"
    EXECUTION_CANCELLED = "execution_cancelled"
    ANALYSIS_FAILED = "analysis_failed"
    AGENT_LIMIT_EXCEEDED = "agent_limit_exceeded"
    ELIGIBILITY_UNAVAILABLE = "eligibility_unavailable"


class ChatBIError(RuntimeError):
    """An application error whose message is safe to expose to a caller."""

    def __init__(self, category: ChatBIErrorCategory, message: str) -> None:
        self.category = category
        self.safe_message = message
        super().__init__(message)


class StructuredOutputError(ChatBIError):
    """Controlled parse/schema failure for a completed structured LLM response."""

    def __init__(
        self,
        category: ChatBIErrorCategory,
        message: str,
        *,
        output_category: Literal["structured_output_parse_error", "structured_output_schema_error"],
        diagnostic: StructuredOutputBoundaryDiagnostic | None = None,
        boundary_observation: StructuredOutputBoundaryObservation | None = None,
    ) -> None:
        super().__init__(category, message)
        self.output_category = output_category
        self.diagnostic = diagnostic
        self.boundary_observation = boundary_observation
