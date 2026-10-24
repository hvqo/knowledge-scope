"""Planning, candidate, and tool-result contracts for the agent router."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from knowledge_scope.rag.schemas import RAGCitation
from knowledge_scope.table_analysis.models import TableOperation
from knowledge_scope.vision.schemas import VisionTaskType

AGENT_PROMPT_VERSION = "agent-rag-v1"
AGENT_TOOL_NAMES = ("analyze_image", "analyze_table")
AgentToolName = Literal["analyze_image", "analyze_table"]
AgentPlanAction = Literal["final", "call_tool"]
AgentToolStatus = Literal["completed", "failed", "skipped"]


class AgentError(RuntimeError):
    """Base class for controlled agent orchestration failures."""

    code: str = "agent_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class AgentPlanningError(AgentError):
    """The planner could not produce one valid, policy-conforming plan."""

    code = "agent_planning_failed"


class AgentToolError(AgentError):
    """A deterministic tool could not be executed."""

    code = "agent_tool_failed"


class AgentNotConfiguredError(AgentError):
    """Agent mode is not available in this deployment."""

    code = "agent_not_configured"


class AgentPlan(BaseModel):
    """One bounded planner decision.

    The planner either calls exactly one deterministic tool or finalizes the
    answer from the sources already collected.  Free-form code, SQL, and
    arbitrary expressions are not part of this contract.
    """

    model_config = ConfigDict(extra="forbid")

    action: AgentPlanAction
    reason: str | None = Field(default=None, max_length=200)
    tool: AgentToolName | None = None
    evidence_index: int | None = Field(default=None, ge=1, le=20)
    second_evidence_index: int | None = Field(default=None, ge=1, le=20)
    asset_index: int = Field(default=0, ge=0, le=255)
    vision_task: VisionTaskType | None = None
    question: str | None = Field(default=None, max_length=500)
    operation: TableOperation | None = None
    parameters: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_action_contract(self) -> AgentPlan:
        if self.action == "final":
            tool_fields = (
                self.tool,
                self.evidence_index,
                self.second_evidence_index,
                self.vision_task,
                self.operation,
                self.parameters,
            )
            if any(value is not None for value in tool_fields):
                raise ValueError("a final plan must not carry tool fields")
            return self

        if self.tool is None or self.evidence_index is None:
            raise ValueError("a tool plan requires tool and evidence_index")
        if self.tool == "analyze_image":
            if self.vision_task is None:
                raise ValueError("analyze_image requires a vision_task")
            if (
                self.operation is not None
                or self.parameters is not None
                or self.second_evidence_index is not None
            ):
                raise ValueError("analyze_image only accepts vision parameters")
        else:
            if self.operation is None:
                raise ValueError("analyze_table requires an operation")
            if self.vision_task is not None or self.question is not None:
                raise ValueError("analyze_table only accepts table parameters")
            if self.operation == "compare":
                if self.second_evidence_index is None:
                    raise ValueError("compare requires second_evidence_index")
            elif self.second_evidence_index is not None:
                raise ValueError("second_evidence_index is only valid for compare")
        return self


@dataclass(frozen=True, slots=True)
class AgentCandidate:
    """One retrieval candidate exposed to the planner and the final answer.

    The citation marker is assigned by the application, never by the model.
    """

    index: int
    marker: str
    citation: RAGCitation
    document_title: str | None = None
    table_columns: tuple[str, ...] = ()
    table_rows: tuple[str, ...] = ()
    table_issue: str | None = None

    @property
    def evidence_id(self) -> str | None:
        return self.citation.evidence_id

    @property
    def modality(self) -> str | None:
        return self.citation.modality

    @property
    def tool_capable(self) -> bool:
        return self.citation.candidate_kind == "evidence" and self.evidence_id is not None


@dataclass(frozen=True, slots=True)
class AgentToolObservation:
    """One executed tool call, kept bounded for planner and answer prompts."""

    tool: str
    summary: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentToolOutcome:
    """Normalized outcome of one deterministic tool execution."""

    tool: str
    status: AgentToolStatus
    summary: str
    payload: dict[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None
    # Application-assigned fact provenance of this tool result.
    fact_status: str | None = None
    numeric_claims: list[dict[str, Any]] = field(default_factory=list)


def _extract_json_object(content: str) -> dict[str, Any]:
    """Extract one JSON object from a model response without executing anything."""
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("planner output does not contain a JSON object")
    candidate = text[start : end + 1]
    decoded = json.loads(candidate)
    if not isinstance(decoded, dict):
        raise ValueError("planner output JSON is not an object")
    return decoded


def parse_agent_plan(content: str) -> AgentPlan:
    """Parse and validate one planner response; failures are controlled."""
    try:
        payload = _extract_json_object(content)
    except (ValueError, json.JSONDecodeError) as error:
        raise AgentPlanningError("规划结果不是可解析的 JSON") from error
    try:
        return AgentPlan.model_validate(payload)
    except ValidationError as error:
        raise AgentPlanningError("规划结果不符合工具调用约定") from error


__all__ = [
    "AGENT_PROMPT_VERSION",
    "AGENT_TOOL_NAMES",
    "AgentCandidate",
    "AgentError",
    "AgentNotConfiguredError",
    "AgentPlan",
    "AgentPlanningError",
    "AgentToolError",
    "AgentToolName",
    "AgentToolObservation",
    "AgentToolOutcome",
    "AgentToolStatus",
    "parse_agent_plan",
]
