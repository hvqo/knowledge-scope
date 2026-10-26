# ruff: noqa: RUF001
"""M9 evidence-sufficiency acceptance harness against a running API."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from pathlib import Path
from typing import Any

import httpx

NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _normalize_text(text: str) -> str:
    value = text.lower()
    for old, new in {
        "，": ",",
        "。": ".",
        "×": "x",
        "（": "(",
        "）": ")",
        " ": "",
        "\n": "",
        "\t": "",
    }.items():
        value = value.replace(old, new)
    return value


def _numbers_in(text: str) -> list[float]:
    values = []
    for match in NUMBER_RE.findall(text.replace(",", "")):
        try:
            values.append(float(match))
        except ValueError:
            continue
    return values


def _number_matches(expected: float, answer: str) -> bool:
    for value in _numbers_in(answer):
        scale = max(abs(expected), 1.0)
        if abs(value - expected) / scale <= 0.02:
            return True
    return False


def _stream_events(
    client: httpx.Client,
    base_url: str,
    query: str,
    kb_id: str,
    *,
    history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": query,
        "knowledge_base_id": kb_id,
        "answer_mode": "agent",
        "retrieval_mode": "unified",
    }
    if history:
        payload["history"] = history
    started = time.perf_counter()
    answer_parts: list[str] = []
    citations: list[dict[str, Any]] = []
    complete: dict[str, Any] | None = None
    tool_statuses: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    event_name: str | None = None
    with client.stream("POST", f"{base_url}/api/v1/rag/query", json=payload) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                event_name = None
                continue
            if line.startswith("event: "):
                event_name = line[len("event: ") :]
                continue
            if not line.startswith("data: "):
                continue
            data = json.loads(line[len("data: ") :])
            if event_name == "answer_delta":
                answer_parts.append(str(data.get("text", "")))
            elif event_name == "citations":
                citations = list(data.get("items", []))
            elif event_name == "tool_status":
                tool_statuses.append(data)
            elif event_name == "error":
                errors.append(data)
            elif event_name == "complete":
                complete = data
    latency_ms = (time.perf_counter() - started) * 1000
    return {
        "answer": "".join(answer_parts),
        "citations": citations,
        "complete": complete or {},
        "tool_statuses": tool_statuses,
        "errors": errors,
        "client_latency_ms": latency_ms,
    }


def _evaluate(
    question: dict[str, Any],
    result: dict[str, Any],
    document_ids: dict[str, str],
) -> dict[str, Any]:
    complete = result["complete"]
    trace = list(complete.get("agent_tool_trace") or [])
    tool_names = {"analyze_image", "analyze_table"}
    tools = [str(item.get("tool")) for item in trace if item.get("tool") in tool_names]
    unique_tools = sorted(set(tools))
    expected_tools = sorted(question.get("expected_tools") or [])
    tools_ok = set(expected_tools).issubset(set(unique_tools)) and (
        bool(expected_tools) or not unique_tools
    )
    cited_ids = {
        str(item.get("document_id")) for item in result["citations"] if item.get("document_id")
    }
    for source in complete.get("visual_sources") or []:
        if source.get("document_id"):
            cited_ids.add(str(source["document_id"]))
    expected_docs = {document_ids[alias] for alias in question.get("expected_docs") or []}
    docs_ok = (
        bool(expected_docs & cited_ids)
        if question.get("expected_docs_mode") == "any"
        else expected_docs.issubset(cited_ids)
    )
    numbers = [float(value) for value in question.get("expected_numbers") or []]
    number_checks = [_number_matches(value, result["answer"]) for value in numbers]
    numbers_ok = (
        (
            any(number_checks)
            if question.get("expected_numbers_mode") == "any"
            else all(number_checks)
        )
        if number_checks
        else True
    )
    phrase_checks = [
        _normalize_text(phrase) in _normalize_text(result["answer"])
        for phrase in question.get("expected_phrases") or []
    ]
    phrases_ok = (
        any(phrase_checks) if question.get("expected_phrases_mode") == "any" else all(phrase_checks)
    )
    visual_ok = (not question.get("expect_visual_source")) or bool(complete.get("visual_sources"))
    m9_available = complete.get("agent_evidence_satisfied") is not None
    sufficient_ok: bool | None = None
    supplementary_ok: bool | None = None
    if m9_available:
        sufficient_ok = bool(complete.get("agent_evidence_satisfied")) == bool(
            question.get("expected_sufficient")
        )
        supplementary_ok = bool(complete.get("agent_supplementary_retrieval")) == bool(
            question.get("expected_supplementary")
        )
    answer_ok = numbers_ok and phrases_ok
    e2e = answer_ok and tools_ok and docs_ok and visual_ok
    if m9_available:
        e2e = e2e and bool(sufficient_ok) and bool(supplementary_ok)
    return {
        "category": question.get("category"),
        "answer_ok": answer_ok,
        "numbers_ok": numbers_ok,
        "phrases_ok": phrases_ok,
        "tools": unique_tools,
        "tools_ok": tools_ok,
        "docs_ok": docs_ok,
        "cited_document_ids": sorted(cited_ids),
        "expected_document_ids": sorted(expected_docs),
        "visual_ok": visual_ok,
        "sufficient_ok": sufficient_ok,
        "supplementary_ok": supplementary_ok,
        "evidence_satisfied": complete.get("agent_evidence_satisfied"),
        "evidence_missing": complete.get("agent_evidence_missing"),
        "supplementary_retrieval": complete.get("agent_supplementary_retrieval"),
        "supplementary_query": complete.get("agent_supplementary_query"),
        "supplementary_added": complete.get("agent_supplementary_added"),
        "visual_auto_called": complete.get("agent_visual_auto_called"),
        "tool_calls": complete.get("tool_calls"),
        "citations": len(result["citations"]),
        "visual_sources": len(complete.get("visual_sources") or []),
        "latency_ms": complete.get("latency_ms"),
        "client_latency_ms": result["client_latency_ms"],
        "e2e": e2e,
        "answer": result["answer"],
        "answer_chars": len(result["answer"]),
        "tool_statuses": result["tool_statuses"],
        "errors": result["errors"],
    }


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--only", default="", help="Comma separated question ids")
    args = parser.parse_args()

    fixtures = json.loads(Path(args.fixtures).read_text())
    gold = json.loads(Path(args.gold).read_text())
    kb_id = fixtures["kb_id"]
    document_ids = {alias: item["id"] for alias, item in fixtures["documents"].items()}
    client = httpx.Client(trust_env=False, timeout=600)
    history: list[dict[str, str]] = []
    previous_group: str | None = None
    results: list[dict[str, Any]] = []
    selected = {item.strip() for item in args.only.split(",") if item.strip()}
    for question in gold["questions"]:
        if selected and question["id"] not in selected:
            continue
        group = question.get("followup_group")
        request_history = history if group and group == previous_group else None
        result = _stream_events(
            client,
            args.base_url,
            question["query"],
            kb_id,
            history=request_history,
        )
        evaluation = _evaluate(question, result, document_ids)
        evaluation.update({"id": question["id"], "query": question["query"]})
        results.append(evaluation)
        print(
            json.dumps(
                {
                    "id": question["id"],
                    "e2e": evaluation["e2e"],
                    "tools": evaluation["tools"],
                    "latency_ms": evaluation["latency_ms"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if group:
            history = [
                {"role": "user", "content": question["query"]},
                {"role": "assistant", "content": result["answer"]},
            ]
            previous_group = group
        else:
            history = []
            previous_group = None

    m9_results = [item for item in results if item["sufficient_ok"] is not None]
    latencies = [float(item["latency_ms"]) for item in results if item["latency_ms"] is not None]
    questions = len(results)

    def rate(items: list[dict[str, Any]], key: str) -> dict[str, Any]:
        passed = sum(1 for item in items if bool(item.get(key)))
        denominator = len(items)
        return {
            "numerator": passed,
            "denominator": denominator,
            "rate": passed / denominator if denominator else None,
        }

    summary = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "questions": questions,
        "mode": "m9" if m9_results else "baseline",
        "tool_selection_accuracy": rate(results, "tools_ok"),
        "evidence_sufficiency_accuracy": (
            rate(m9_results, "sufficient_ok") if m9_results else None
        ),
        "supplementary_trigger_accuracy": (
            rate(m9_results, "supplementary_ok") if m9_results else None
        ),
        "key_evidence_coverage": rate(results, "docs_ok"),
        "visual_tool_call_accuracy": rate(
            [item for item in results if item["visual_ok"] or item["tools"]], "visual_ok"
        ),
        "answer_accuracy": rate(results, "answer_ok"),
        "e2e_success": rate(results, "e2e"),
        "latency_ms": {
            "mean": statistics.fmean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "p95": _p95(latencies),
        },
        "failures": [
            {"id": item["id"], "query": item["query"], "tools": item["tools"]}
            for item in results
            if not item["e2e"]
        ],
        "details": results,
    }
    Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False)[:2000])


if __name__ == "__main__":
    main()
