# ruff: noqa: RUF001
"""Strict, reproducible scoring for the M5 multimodal agentic RAG evaluation.

The harness drives the real FastAPI agent endpoint (``POST /api/v1/rag/query``
with ``answer_mode="agent"``), captures the raw SSE events, and scores each
question with explicit checks instead of substring/any() shortcuts.

Scoring model (all metrics are reported with numerator and denominator):

* tool selection         - required tools executed, forbidden tools absent
* tool execution         - required tools actually completed (or produced the
                           expected controlled error)
* parameter validity     - operation, target document and vision task are taken
                           from the server-side ``agent_tool_trace`` parameters
* exact calculation      - deterministic table questions whose tool completed
                           and whose numbers match within tolerance
* final numeric accuracy - answer numbers for deterministic calculation tasks
* visual number accuracy - numbers read from an image table (needs_review, never
                           counted as an exact calculation)
* visual refusal         - unreadable visuals are answered with a controlled
                           refusal instead of invented numbers
* cited document cover.  - every required document appears in final citations
* citation structure     - citation markers/schema are valid
* evidence accessibility - every cited Evidence is still readable via the API
* required text          - answer contains the expected qualitative statements
* E2E success            - all applicable checks above pass for the question

True Retrieval Hit@K and citation *semantic* support are explicitly marked
UNVERIFIED: the public SSE contract does not expose retrieval stage Top-K
candidates, and a structural citation check cannot prove factual support.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from functools import reduce
from pathlib import Path
from typing import Any

import httpx

QUARTER_PATTERN = re.compile(r"\b[qQ][1-4]\b")
NUMBER_PATTERN = re.compile(r"(?<![A-Za-z0-9_.])-?\d+(?:,\d{3})*(?:\.\d+)?%?")
EVIDENCE_ID_PATTERN = re.compile(r"^evidence_v1_[0-9a-f]{64}$")
MARKER_PATTERN = re.compile(r"^C[1-9][0-9]*$")

MAX_ANSWER_CHARS = 200_000
UNVERIFIED_RETRIEVAL = "UNVERIFIED"
UNVERIFIED_SEMANTIC = "UNVERIFIED"


# ---------------------------------------------------------------------------
# Pure helpers (unit tested)
# ---------------------------------------------------------------------------
def normalize_text(value: str) -> str:
    """Normalize answer/dataset text for qualitative matching."""
    text = value.lower().replace("−", "-").replace("–", "-").replace("—", "-")
    text = text.replace("×", "x").replace("÷", "/").replace("（", "(").replace("）", ")")
    text = text.replace("，", ",").replace("。", ".").replace("：", ":")
    text = re.sub(r"\s+", "", text)
    return text


def strip_quarter_labels(text: str) -> str:
    return QUARTER_PATTERN.sub(" ", text)


def extract_numbers(text: str) -> list[Decimal]:
    """Extract decimal numbers from an answer (percent sign is stripped)."""
    cleaned = strip_quarter_labels(text)
    numbers: list[Decimal] = []
    for token in NUMBER_PATTERN.findall(cleaned):
        literal = token.replace(",", "").rstrip("%")
        try:
            numbers.append(Decimal(literal))
        except InvalidOperation:  # pragma: no cover - defensive
            continue
    return numbers


def _decimal(value: str | int | float) -> Decimal:
    return Decimal(str(value).replace(",", "").rstrip("%"))


def match_number_groups(
    numbers: Iterable[Decimal],
    groups: list[list[str]],
    *,
    tolerance: str | Decimal = "0",
) -> tuple[bool, list[list[str]]]:
    """Return (all groups matched, unmatched groups). Each group is an OR list."""
    tolerance_value = Decimal(str(tolerance))
    remaining = list(numbers)
    missing: list[list[str]] = []
    for group in groups:
        matched = False
        for candidate in group:
            expected = _decimal(candidate)
            for actual in remaining:
                if abs(actual - expected) <= tolerance_value:
                    remaining.remove(actual)
                    matched = True
                    break
            if matched:
                break
        if not matched:
            missing.append(group)
    return not missing, missing


def match_text_groups(normalized: str, groups: list[list[str]]) -> tuple[bool, list[list[str]]]:
    missing: list[list[str]] = []
    for group in groups:
        if not any(normalize_text(candidate) in normalized for candidate in group):
            missing.append(group)
    return not missing, missing


def _label_for_document(document_id: str | None, label_by_document: dict[str, str]) -> str | None:
    if not document_id:
        return None
    return label_by_document.get(document_id)


def _param_document_label(
    parameters: dict[str, Any] | None,
    label_by_document: dict[str, str],
) -> str | None:
    if not isinstance(parameters, dict):
        return None
    label = _label_for_document(parameters.get("document_id"), label_by_document)
    if label:
        return label
    table = parameters.get("table")
    if isinstance(table, dict):
        label = _label_for_document(table.get("document_id"), label_by_document)
        if label:
            return label
    return None


def _trace_entries(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [entry for entry in trace if isinstance(entry, dict) and entry.get("tool")]


def _analysis_tools(trace: list[dict[str, Any]]) -> list[str]:
    return [
        str(entry["tool"])
        for entry in _trace_entries(trace)
        if entry.get("tool") in {"analyze_image", "analyze_table"}
        and entry.get("status") != "skipped"
    ]


def _find_trace(trace: list[dict[str, Any]], tool: str) -> dict[str, Any] | None:
    for entry in _trace_entries(trace):
        if entry.get("tool") == tool:
            return entry
    return None


def _operation_matches(question: dict[str, Any], parameters: dict[str, Any] | None) -> bool:
    expected = question.get("expected_operations") or (
        [question["expected_operation"]] if question.get("expected_operation") else []
    )
    if not expected:
        return True
    if not isinstance(parameters, dict):
        return False
    actual = parameters.get("operation")
    return actual in expected


def _vision_task_matches(question: dict[str, Any], parameters: dict[str, Any] | None) -> bool:
    expected = question.get("vision_tasks") or (
        [question["vision_task"]] if question.get("vision_task") else []
    )
    if not expected:
        return True
    if not isinstance(parameters, dict):
        return False
    return parameters.get("vision_task") in expected


def _parameter_document_matches(
    question: dict[str, Any],
    entry: dict[str, Any],
    label_by_document: dict[str, str],
) -> tuple[bool, list[str]]:
    expected_docs = question.get("expected_parameter_docs") or []
    parameters = entry.get("parameters") if isinstance(entry, dict) else None
    primary = _param_document_label(parameters, label_by_document)
    compare = None
    if isinstance(parameters, dict):
        compare = _label_for_document(parameters.get("second_document_id"), label_by_document)
        if compare is None and isinstance(parameters.get("compare_table"), dict):
            compare = _label_for_document(
                parameters["compare_table"].get("document_id"), label_by_document
            )
    observed = {label for label in (primary, compare) if label}
    if not expected_docs:
        return True, sorted(observed)
    missing = [label for label in expected_docs if label not in observed]
    return not missing, missing


# ---------------------------------------------------------------------------
# Run capture
# ---------------------------------------------------------------------------
def _parse_sse(lines: Iterable[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    event_name: str | None = None
    data_lines: list[str] = []

    def flush() -> None:
        nonlocal event_name, data_lines
        if event_name is None:
            data_lines = []
            return
        raw_data = "\n".join(data_lines)
        try:
            payload = json.loads(raw_data)
        except json.JSONDecodeError:
            payload = {"_raw": raw_data[:500]}
        events.append({"event": event_name, "data": payload})
        event_name = None
        data_lines = []

    for raw in lines:
        line = raw.rstrip("\r")
        if not line:
            flush()
            continue
        if line.startswith("event:"):
            event_name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())
    flush()
    return events


def _run_query(
    client: httpx.Client,
    api_base: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    text_parts: list[str] = []
    citations: list[dict[str, Any]] = []
    complete: dict[str, Any] | None = None
    errors: list[dict[str, Any]] = []
    tool_statuses: list[dict[str, Any]] = []
    with client.stream("POST", f"{api_base}/api/v1/rag/query", json=payload) as response:
        response.raise_for_status()
        for event in _parse_sse(response.iter_lines()):
            name = event["event"]
            data = event["data"]
            if name == "answer_delta":
                chunk = data.get("text") if isinstance(data, dict) else None
                if isinstance(chunk, str):
                    text_parts.append(chunk)
                    if sum(len(part) for part in text_parts) > MAX_ANSWER_CHARS:
                        break
            elif name == "citations":
                items = data.get("items") if isinstance(data, dict) else None
                if isinstance(items, list):
                    citations.extend(item for item in items if isinstance(item, dict))
            elif name == "tool_status":
                if isinstance(data, dict):
                    tool_statuses.append(data)
            elif name == "error":
                if isinstance(data, dict):
                    errors.append(data)
            elif name == "complete":
                if isinstance(data, dict):
                    complete = data
    return {
        "answer": "".join(text_parts),
        "citations": citations,
        "complete": complete or {},
        "errors": errors,
        "tool_statuses": tool_statuses,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def _evidence_accessible(
    client: httpx.Client,
    api_base: str,
    knowledge_base_id: str,
    citation: dict[str, Any],
) -> bool | None:
    evidence_id = citation.get("evidence_id")
    document_id = citation.get("document_id")
    if not evidence_id or not document_id:
        return None
    url = (
        f"{api_base}/api/v1/knowledge-bases/{knowledge_base_id}/"
        f"documents/{document_id}/evidence/{evidence_id}"
    )
    try:
        response = client.get(url)
    except httpx.HTTPError:
        return False
    return response.status_code == 200


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
@dataclass
class QuestionResult:
    question: dict[str, Any]
    raw: dict[str, Any]
    checks: dict[str, Any] = field(default_factory=dict)
    failure_codes: list[str] = field(default_factory=list)
    accessibility: dict[str, bool | None] = field(default_factory=dict)

    def fail(self, code: str) -> None:
        if code not in self.failure_codes:
            self.failure_codes.append(code)


def _citation_structure_valid(
    citations: list[dict[str, Any]],
    knowledge_base_id: str,
) -> tuple[bool, list[str]]:
    problems: list[str] = []
    if not citations:
        return False, ["no_citations"]
    for index, citation in enumerate(citations, start=1):
        marker = citation.get("marker")
        if not isinstance(marker, str) or not MARKER_PATTERN.match(marker):
            problems.append(f"bad_marker:{index}")
        if citation.get("knowledge_base_id") != knowledge_base_id:
            problems.append(f"bad_kb:{index}")
        kind = citation.get("candidate_kind")
        if kind == "evidence":
            evidence_id = citation.get("evidence_id")
            if not isinstance(evidence_id, str) or not EVIDENCE_ID_PATTERN.match(evidence_id):
                problems.append(f"bad_evidence_id:{index}")
            if citation.get("modality") not in {"text", "image", "table", "formula"}:
                problems.append(f"bad_modality:{index}")
            representation_ids = citation.get("representation_ids")
            if not isinstance(representation_ids, list) or not representation_ids:
                problems.append(f"no_representations:{index}")
        elif kind == "chunk":
            if not citation.get("chunk_id"):
                problems.append(f"bad_chunk_id:{index}")
        else:
            problems.append(f"bad_candidate_kind:{index}")
    return not problems, problems


def score_question(
    question: dict[str, Any],
    raw: dict[str, Any],
    *,
    knowledge_base_id: str,
    label_by_document: dict[str, str],
) -> QuestionResult:
    result = QuestionResult(question=question, raw=raw)
    complete = raw["complete"]
    raw_trace = complete.get("agent_tool_trace")
    trace = raw_trace if isinstance(raw_trace, list) else []
    citations = raw["citations"]
    answer = raw["answer"]
    normalized = normalize_text(answer)
    numbers = extract_numbers(answer)

    tools = question.get("tools") or {}
    required = list(tools.get("required") or [])
    forbidden = set(tools.get("forbidden") or [])
    allowed = set(tools.get("allowed") or required)
    allow_none = bool(tools.get("allow_none"))

    executed = _analysis_tools(trace)
    extras = [tool for tool in executed if tool in forbidden]
    missing_required = [tool for tool in required if tool not in executed]
    if required:
        tool_selection = not missing_required and not extras
    else:
        unexpected = [tool for tool in executed if tool not in allowed]
        tool_selection = not extras and not unexpected
        if not allow_none and not executed:
            tool_selection = False
    result.checks["tool_selection"] = tool_selection
    if not tool_selection:
        result.fail("tool_selection")
        if missing_required:
            result.fail("missing_required_tool")
        if extras:
            result.fail("forbidden_tool")

    execution_ok = True
    expected_error = question.get("expected_tool_error")
    for tool in required:
        entry = _find_trace(trace, tool)
        if entry is None:
            execution_ok = False
            continue
        if expected_error:
            if entry.get("status") != "failed" or entry.get("error_code") != expected_error:
                execution_ok = False
        elif entry.get("status") != "completed":
            execution_ok = False
    result.checks["tool_execution"] = execution_ok
    result.checks["tool_executed_any"] = any(
        _find_trace(trace, tool) is not None for tool in required
    )
    if not execution_ok:
        result.fail("tool_execution")

    parameter_ok = True
    if required:
        for tool in required:
            entry = _find_trace(trace, tool)
            parameters = entry.get("parameters") if isinstance(entry, dict) else None
            if entry is None or not isinstance(parameters, dict):
                parameter_ok = False
                result.fail("parameter_unobservable")
                continue
            if tool == "analyze_table":
                if not _operation_matches(question, parameters):
                    parameter_ok = False
                    result.fail("parameter_operation")
                document_ok, missing = _parameter_document_matches(
                    question, entry, label_by_document
                )
                if not document_ok:
                    parameter_ok = False
                    result.fail("parameter_document")
                    if missing:
                        result.fail("parameter_missing_docs")
            elif tool == "analyze_image":
                if not _vision_task_matches(question, parameters):
                    parameter_ok = False
                    result.fail("parameter_vision_task")
                document_ok, missing = _parameter_document_matches(
                    question, entry, label_by_document
                )
                if not document_ok:
                    parameter_ok = False
                    result.fail("parameter_document")
                    if missing:
                        result.fail("parameter_missing_docs")
    result.checks["parameter_validity"] = parameter_ok

    expected_docs = list(question.get("required_docs") or [])
    cited_docs = {
        label_by_document.get(str(citation.get("document_id")))
        for citation in citations
        if citation.get("document_id")
    }
    cited_docs.discard(None)
    doc_coverage = all(label in cited_docs for label in expected_docs)
    result.checks["cited_document_coverage"] = doc_coverage
    if not doc_coverage:
        result.fail("document_coverage")

    if not citations and not question.get("citation_required"):
        citation_structure: bool | None = None
        structure_problems: list[str] = []
    else:
        citation_structure, structure_problems = _citation_structure_valid(
            citations, knowledge_base_id
        )
        if question.get("citation_required") and not citations:
            citation_structure = False
            structure_problems = structure_problems or ["no_citations"]
    result.checks["citation_structure"] = citation_structure
    if citation_structure is False:
        result.fail("citation_structure")
    result.checks["citation_structure_problems"] = structure_problems

    # Evidence accessibility is filled by the caller with a live HTTP client.

    numbers_spec = question.get("numbers") or {}
    number_groups = numbers_spec.get("groups") or []
    if number_groups:
        tolerance = numbers_spec.get("tolerance", "0")
        numbers_ok, missing_groups = match_number_groups(
            numbers, number_groups, tolerance=tolerance
        )
    else:
        numbers_ok, missing_groups = True, []
    if question.get("forbid_any_number") and numbers:
        numbers_ok = False
        result.fail("unexpected_number")
    result.checks["numbers"] = numbers_ok
    if not numbers_ok:
        result.fail("numbers_mismatch")
        result.checks["missing_number_groups"] = missing_groups

    text_groups = question.get("required_text_groups") or []
    if text_groups:
        text_ok, missing_text = match_text_groups(normalized, text_groups)
    else:
        text_ok, missing_text = True, []
    result.checks["required_text"] = text_ok
    if not text_ok:
        result.fail("required_text")
        result.checks["missing_text_groups"] = missing_text

    needs_review = any(
        "人工核对" in str(entry.get("summary") or "") or "needs_review" in str(entry.get("status"))
        for entry in _trace_entries(trace)
    )
    result.checks["needs_review_observed"] = needs_review
    if question.get("needs_review_expected") and not needs_review:
        result.fail("needs_review_missing")

    status = complete.get("status")
    refusal_text = bool(text_ok) if question.get("required_text_groups") else False
    if question.get("expect_insufficient"):
        insufficient_ok = status == "insufficient_evidence" or refusal_text
        result.checks["insufficient_answer"] = insufficient_ok
        if not insufficient_ok:
            result.fail("not_insufficient")

    if question.get("deterministic_calc") and not expected_error:
        exact_ok = (
            bool(required)
            and execution_ok
            and parameter_ok
            and numbers_ok
            and doc_coverage
            and bool(citation_structure)
            and status == "completed"
        )
        result.checks["exact_calculation"] = exact_ok
    else:
        result.checks["exact_calculation"] = None

    e2e = (
        not raw["errors"]
        and tool_selection
        and execution_ok
        and parameter_ok
        and numbers_ok
        and text_ok
        and doc_coverage
        and (citation_structure if citation_structure is not None else True)
        and status in {"completed", "insufficient_evidence"}
        and not any(question.get("needs_review_expected") and not needs_review for _ in [0])
    )
    if question.get("expect_insufficient"):
        e2e = e2e and (status == "insufficient_evidence" or refusal_text)
    result.checks["e2e"] = e2e
    if not e2e:
        result.fail("e2e")

    result.checks["status"] = status
    result.checks["answer_markers"] = sorted(
        marker for marker in re.findall(r"\[(C[1-9][0-9]*)\]", answer) if marker
    )
    result.checks["cited_documents"] = sorted(cited_docs)
    result.checks["tool_calls"] = complete.get("tool_calls")
    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _ratio(passed: int, total: int) -> dict[str, Any]:
    return {
        "passed": passed,
        "total": total,
        "value": round(passed / total, 4) if total else None,
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(percentile * len(ordered)) - 1))
    return round(ordered[index], 2)


def compute_metrics(results: list[QuestionResult]) -> dict[str, Any]:
    def rate(predicate) -> dict[str, Any]:
        subset = [result for result in results if predicate(result.question)]
        passed = sum(1 for result in subset if result.checks.get("e2e"))
        return {
            "questions": len(subset),
            "passed": passed,
            "value": round(passed / len(subset), 4) if subset else None,
        }

    total = len(results)
    tool_selection = _ratio(sum(1 for result in results if result.checks["tool_selection"]), total)
    tool_required = [
        result for result in results if (result.question.get("tools") or {}).get("required")
    ]
    tool_selected_questions = [
        result for result in tool_required if result.checks.get("tool_executed_any")
    ]
    tool_execution = _ratio(
        sum(1 for result in tool_selected_questions if result.checks["tool_execution"]),
        len(tool_selected_questions),
    )
    parameter = _ratio(
        sum(1 for result in tool_selected_questions if result.checks["parameter_validity"]),
        len(tool_selected_questions),
    )
    deterministic = [
        result
        for result in results
        if result.question.get("deterministic_calc")
        and not result.question.get("expected_tool_error")
    ]
    exact = _ratio(
        sum(1 for result in deterministic if result.checks.get("exact_calculation")),
        len(deterministic),
    )
    numeric_tasks = [
        result for result in deterministic if (result.question.get("numbers") or {}).get("groups")
    ]
    final_numeric = _ratio(
        sum(1 for result in numeric_tasks if result.checks["numbers"]),
        len(numeric_tasks),
    )
    executed_numeric_tasks = [
        result for result in numeric_tasks if result.checks.get("tool_execution")
    ]
    executed_numeric = _ratio(
        sum(1 for result in executed_numeric_tasks if result.checks["numbers"]),
        len(executed_numeric_tasks),
    )
    visual_number = [
        result
        for result in results
        if (result.question.get("numbers") or {}).get("kind") == "visual_number"
    ]
    visual_number_rate = _ratio(
        sum(
            1
            for result in visual_number
            if result.checks["numbers"] and result.checks["tool_execution"]
        ),
        len(visual_number),
    )
    visual_refusal = [
        result
        for result in results
        if (result.question.get("numbers") or {}).get("kind") == "visual_refusal"
        and result.question.get("category") == "image"
    ]
    visual_refusal_rate = _ratio(
        sum(
            1
            for result in visual_refusal
            if result.checks["required_text"]
            and result.checks["numbers"]
            and not result.raw["errors"]
        ),
        len(visual_refusal),
    )
    doc_required = [result for result in results if result.question.get("required_docs")]
    doc_coverage = _ratio(
        sum(1 for result in doc_required if result.checks["cited_document_coverage"]),
        len(doc_required),
    )
    citation_required = [result for result in results if result.question.get("citation_required")]
    citation_structure = _ratio(
        sum(1 for result in citation_required if result.checks["citation_structure"]),
        len(citation_required),
    )
    evidence_total = sum(len(result.accessibility) for result in results)
    evidence_accessible = sum(
        1
        for result in results
        for accessible in result.accessibility.values()
        if accessible is True
    )
    text_required = [result for result in results if result.question.get("required_text_groups")]
    text_accuracy = _ratio(
        sum(1 for result in text_required if result.checks["required_text"]),
        len(text_required),
    )
    e2e = _ratio(sum(1 for result in results if result.checks["e2e"]), total)
    latencies = [result.raw["latency_ms"] for result in results]
    failure_counts: dict[str, int] = {}
    for result in results:
        for code in result.failure_codes:
            failure_counts[code] = failure_counts.get(code, 0) + 1
    return {
        "total_questions": total,
        "tool_selection_accuracy": tool_selection,
        "tool_execution_success_rate": tool_execution,
        "parameter_validity_strict": parameter,
        "exact_calculation_success_rate": exact,
        "final_answer_numeric_accuracy": final_numeric,
        "executed_tool_numeric_accuracy": executed_numeric,
        "visual_number_accuracy": visual_number_rate,
        "visual_controlled_refusal_rate": visual_refusal_rate,
        "cited_document_coverage": doc_coverage,
        "citation_structure_validity": citation_structure,
        "evidence_accessibility_rate": _ratio(evidence_accessible, evidence_total),
        "required_text_accuracy": text_accuracy,
        "e2e_success_rate": e2e,
        "retrieval_hit_at_k": UNVERIFIED_RETRIEVAL,
        "citation_semantic_support": UNVERIFIED_SEMANTIC,
        "latency_ms": {
            "average": round(sum(latencies) / len(latencies), 2) if latencies else None,
            "median": _percentile(latencies, 0.5),
            "p95": _percentile(latencies, 0.95),
        },
        "failure_counts": failure_counts,
        "by_category": {
            category: rate(lambda question, category=category: question.get("category") == category)
            for category in sorted({str(result.question.get("category")) for result in results})
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _git_head(repo_root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rescore(
    source_path: Path,
    dataset: dict[str, Any],
    fixtures: dict[str, Any],
    args: argparse.Namespace,
) -> int:
    previous = _load_json(source_path)
    knowledge_base_id = fixtures["knowledge_base_id"]
    label_by_document = {doc_id: label for label, doc_id in fixtures["documents"].items()}
    questions_by_id = {question["id"]: question for question in dataset["questions"]}
    results: list[QuestionResult] = []
    for entry in previous.get("questions", []):
        question = questions_by_id.get(entry.get("id"))
        if question is None:
            continue
        raw = {
            "answer": entry.get("answer", ""),
            "citations": entry.get("citations", []),
            "complete": {
                **(entry.get("complete") or {}),
                "agent_tool_trace": entry.get("agent_tool_trace") or [],
            },
            "errors": entry.get("errors") or [],
            "tool_statuses": [],
            "latency_ms": entry.get("latency_ms", 0.0),
        }
        result = score_question(
            question,
            raw,
            knowledge_base_id=knowledge_base_id,
            label_by_document=label_by_document,
        )
        result.accessibility = {
            citation.get("evidence_id"): True
            for citation in raw["citations"]
            if citation.get("evidence_id")
        }
        results.append(result)

    metrics = compute_metrics(results)
    payload = dict(previous)
    payload["environment"] = {
        **(previous.get("environment") or {}),
        "rescored_at": datetime.now(UTC).isoformat(),
        "rescored_from": str(source_path),
    }
    payload["metrics"] = metrics
    payload["questions"] = [
        {
            "id": result.question["id"],
            "category": result.question.get("category"),
            "query": result.question["query"],
            "checks": result.checks,
            "failure_codes": result.failure_codes,
            "accessibility": result.accessibility,
            "answer": result.raw["answer"],
            "citations": result.raw["citations"],
            "agent_tool_trace": result.raw["complete"].get("agent_tool_trace"),
            "complete": result.raw["complete"],
            "errors": result.raw["errors"],
            "latency_ms": result.raw["latency_ms"],
        }
        for result in results
    ]
    output_path = Path(args.output)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "metrics": metrics,
                "environment": payload["environment"],
                "dataset": payload["dataset"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"rescored into {output_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", default="/tmp/m5-eval-fixtures.json")
    parser.add_argument("--dataset", default="scripts/m5_eval_dataset.json")
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--output", default="/tmp/m5-eval-strict.json")
    parser.add_argument(
        "--rescore",
        default="",
        help="re-score a previously saved strict result JSON with the current dataset/checks",
    )
    parser.add_argument("--only", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--repo-root", default=str(Path.cwd()))
    args = parser.parse_args()

    dataset = _load_json(Path(args.dataset))
    fixtures = _load_json(Path(args.fixtures))
    if args.rescore:
        return _rescore(Path(args.rescore), dataset, fixtures, args)
    questions: list[dict[str, Any]] = dataset["questions"]
    if args.only:
        wanted = {value.strip() for value in args.only.split(",") if value.strip()}
        questions = [question for question in questions if question.get("category") in wanted]
    if args.limit:
        questions = questions[: args.limit]

    knowledge_base_id = fixtures["knowledge_base_id"]
    label_by_document = {doc_id: label for label, doc_id in fixtures["documents"].items()}
    dataset_aliases = dataset.get("document_aliases") or {}
    missing_aliases = [
        alias
        for alias in reduce(
            lambda accumulated, question: accumulated.union(question.get("required_docs") or []),
            questions,
            set(),
        )
        if alias not in fixtures["documents"]
    ]
    if missing_aliases:
        print(f"WARNING: dataset references unavailable documents: {sorted(missing_aliases)}")

    results: list[QuestionResult] = []
    followup_history: list[dict[str, str]] = []
    started_at = datetime.now(UTC).isoformat()
    with httpx.Client(
        timeout=httpx.Timeout(connect=10.0, read=300.0, write=60.0, pool=10.0),
        trust_env=False,
    ) as client:
        for index, question in enumerate(questions, start=1):
            payload: dict[str, Any] = {
                "query": question["query"],
                "knowledge_base_id": knowledge_base_id,
                "retrieval_mode": "unified",
                "answer_mode": "agent",
            }
            if question.get("history"):
                payload["history"] = question["history"]
            elif question.get("category") == "followup" and followup_history:
                payload["history"] = list(followup_history[-4:])
            print(
                f"[{index}/{len(questions)}] {question['id']} ({question['category']}) ...",
                flush=True,
            )
            try:
                raw = _run_query(client, args.api_base, payload)
            except httpx.HTTPError as error:
                raw = {
                    "answer": "",
                    "citations": [],
                    "complete": {},
                    "errors": [{"category": "transport", "message": str(error)[:300]}],
                    "tool_statuses": [],
                    "latency_ms": 0.0,
                }
            result = score_question(
                question,
                raw,
                knowledge_base_id=knowledge_base_id,
                label_by_document=label_by_document,
            )
            # Fill accessibility with the live client.
            for citation in raw["citations"]:
                evidence_id = citation.get("evidence_id")
                if evidence_id:
                    result.accessibility[evidence_id] = _evidence_accessible(
                        client, args.api_base, knowledge_base_id, citation
                    )
            results.append(result)
            if question.get("category") == "followup":
                followup_history.append({"role": "user", "content": question["query"]})
                followup_history.append({"role": "assistant", "content": raw["answer"][:4000]})
            print(
                f"    e2e={result.checks.get('e2e')} failures={result.failure_codes} "
                f"latency={raw['latency_ms']}ms",
                flush=True,
            )

    metrics = compute_metrics(results)
    payload = {
        "schema": "m5-strict-evaluation/v1",
        "dataset": {
            "path": str(args.dataset),
            "name": dataset.get("name"),
            "version": dataset.get("version"),
            "document_aliases": dataset_aliases,
        },
        "fixtures": {
            "knowledge_base_id": knowledge_base_id,
            "documents": fixtures["documents"],
            "sources": fixtures.get("sources"),
            "published_at": fixtures.get("published_at"),
            "math_pdf_available": fixtures.get("math_pdf_available"),
        },
        "environment": {
            "git_head": _git_head(Path(args.repo_root).resolve()),
            "api_base": args.api_base,
            "started_at": started_at,
            "finished_at": datetime.now(UTC).isoformat(),
        },
        "metrics": metrics,
        "questions": [
            {
                "id": result.question["id"],
                "category": result.question.get("category"),
                "query": result.question["query"],
                "checks": result.checks,
                "failure_codes": result.failure_codes,
                "accessibility": result.accessibility,
                "answer": result.raw["answer"],
                "citations": [
                    {
                        "marker": citation.get("marker"),
                        "document_id": citation.get("document_id"),
                        "document_label": label_by_document.get(str(citation.get("document_id"))),
                        "candidate_kind": citation.get("candidate_kind"),
                        "evidence_id": citation.get("evidence_id"),
                        "modality": citation.get("modality"),
                        "page_start": citation.get("page_start"),
                        "page_end": citation.get("page_end"),
                        "source_block_ids": citation.get("source_block_ids"),
                        "snippet": (citation.get("snippet") or "")[:300],
                    }
                    for citation in result.raw["citations"]
                ],
                "agent_tool_trace": result.raw["complete"].get("agent_tool_trace"),
                "complete": {
                    key: result.raw["complete"].get(key)
                    for key in (
                        "status",
                        "prompt_version",
                        "provider",
                        "model",
                        "answer_mode",
                        "tool_calls",
                        "input_tokens",
                        "output_tokens",
                        "latency_ms",
                        "agent_degraded",
                        "agent_degraded_reason",
                    )
                },
                "errors": result.raw["errors"],
                "latency_ms": result.raw["latency_ms"],
            }
            for result in results
        ],
    }
    output_path = Path(args.output)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(
            {
                "metrics": metrics,
                "environment": payload["environment"],
                "dataset": payload["dataset"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"written {output_path} and {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
