# ruff: noqa: RUF001
"""Run the M5 multimodal agentic RAG evaluation against a live API.

The dataset lives in ``scripts/m5_eval_dataset.json``. Every question is sent to
``POST /api/v1/rag/query`` in agent mode and the SSE stream is graded against the
expected tool, numbers, documents, and citation contract.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import httpx

MARKER_PATTERN = re.compile(r"\[C(\d+)\]")


def normalize(text: str) -> str:
    text = text.replace(",", "").replace("，", "").replace("$", "")
    text = text.replace("\\", "").replace("×", "x").replace(" ", "")
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = text.replace("\n", "").replace("\t", "").replace("\u00a0", "")
    return text.lower()


def parse_sse(lines: list[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    event_name: str | None = None
    data_lines: list[str] = []
    for line in lines:
        if line.startswith("event:"):
            event_name = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())
        elif line == "":
            if event_name is not None and data_lines:
                payload = json.loads("\n".join(data_lines))
                events.append({"event": event_name, "data": payload})
            event_name, data_lines = None, []
    return events


async def ask(
    client: httpx.AsyncClient, base_url: str, question: dict[str, Any], kb: str
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": question["query"],
        "knowledge_base_id": kb,
        "answer_mode": "agent",
        "retrieval_mode": "unified",
    }
    if question.get("history"):
        payload["history"] = question["history"]
    started = time.perf_counter()
    lines: list[str] = []
    async with client.stream("POST", f"{base_url}/api/v1/rag/query", json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            lines.append(line)
    events = parse_sse(lines)
    return {"events": events, "elapsed_ms": (time.perf_counter() - started) * 1000}


def evaluate(
    question: dict[str, Any],
    run: dict[str, Any],
    evidence_status: dict[str, bool],
    doc_map: dict[str, str],
) -> dict[str, Any]:
    events = run["events"]
    answer = "".join(
        event["data"].get("text", "") for event in events if event["event"] == "answer_delta"
    )
    citations: list[dict[str, Any]] = []
    complete: dict[str, Any] = {}
    error: dict[str, Any] | None = None
    tool_statuses: list[dict[str, Any]] = []
    for event in events:
        if event["event"] == "citations":
            citations = list(event["data"].get("items") or [])
        elif event["event"] == "complete":
            complete = event["data"]
        elif event["event"] == "error":
            error = event["data"]
        elif event["event"] == "tool_status":
            tool_statuses.append(event["data"])

    normalized = normalize(answer)
    numbers = [normalize(value) for value in question.get("expected_numbers") or []]
    phrases = [normalize(value) for value in question.get("expected_phrases") or []]
    number_ok = True if not numbers else any(value in normalized for value in numbers)
    phrase_ok = True if not phrases else any(value in normalized for value in phrases)

    trace = list(complete.get("agent_tool_trace") or [])
    analyze_tools = [
        entry.get("tool") for entry in trace if str(entry.get("tool", "")).startswith("analyze_")
    ]
    expected_tools = question.get("expected_tools")
    if expected_tools is None:
        tool_ok = not analyze_tools
    else:
        allowed_tools = {tool for tool in expected_tools if tool != "none"}
        allow_none = "none" in expected_tools
        tool_ok = (allow_none and not analyze_tools) or any(
            tool in allowed_tools for tool in analyze_tools
        )
    param_ok = not any(entry.get("error_code") == "invalid_parameters" for entry in trace)

    markers = sorted({int(match) for match in MARKER_PATTERN.findall(answer)})
    citation_markers = sorted(
        {int(match) for c in citations for match in re.findall(r"C(\d+)", str(c.get("marker", "")))}
    )
    marker_ok = all(marker in citation_markers for marker in markers)
    citation_evidence_ok = all(
        evidence_status.get(f"{c['document_id']}:{c['evidence_id']}", False)
        for c in citations
        if c.get("evidence_id")
    )
    citation_count_ok = len(citations) >= 1 if question.get("citation_required") else True
    citation_ok = marker_ok and citation_evidence_ok and citation_count_ok

    expected_docs = {doc_map.get(label, label) for label in (question.get("expected_docs") or [])}
    citation_docs = {c.get("document_id") for c in citations}
    hit_ok = True if not expected_docs else bool(expected_docs & citation_docs)

    insufficient = bool(question.get("expect_insufficient"))
    refusal_tokens = ["没有", "未找到", "无法", "不足", "不存在", "未提供", "没有找到"]
    refusal_ok = (
        (
            complete.get("status") == "insufficient_evidence"
            or any(token in answer for token in refusal_tokens)
            or not citations
        )
        if insufficient
        else True
    )

    status_ok = complete.get("status") == "completed"
    e2e = all([number_ok, phrase_ok, tool_ok, param_ok, citation_ok, hit_ok, refusal_ok, status_ok])

    failures: list[str] = []
    if not status_ok:
        failures.append("status_error")
    if error:
        failures.append("error_event")
    if not number_ok:
        failures.append("number_mismatch")
    if not phrase_ok:
        failures.append("phrase_mismatch")
    if not tool_ok:
        failures.append(f"tool_mismatch:{analyze_tools}")
    if not param_ok:
        failures.append("invalid_parameters")
    if not citation_ok:
        failures.append("citation_invalid")
    if not hit_ok:
        failures.append("retrieval_miss")
    if not refusal_ok:
        failures.append("unsupported_claim")

    return {
        "id": question["id"],
        "category": question["category"],
        "query": question["query"],
        "answer": answer,
        "citations": len(citations),
        "markers": markers,
        "citation_docs": sorted(doc for doc in citation_docs if doc),
        "analyze_tools": analyze_tools,
        "tool_statuses": tool_statuses,
        "complete": {
            "status": complete.get("status"),
            "tool_calls": complete.get("tool_calls"),
            "agent_degraded": complete.get("agent_degraded"),
            "model": complete.get("model"),
            "latency_ms": complete.get("latency_ms"),
        },
        "checks": {
            "number": number_ok,
            "phrase": phrase_ok,
            "tool": tool_ok,
            "parameter": param_ok,
            "citation": citation_ok,
            "citation_evidence": citation_evidence_ok,
            "retrieval": hit_ok,
            "refusal": refusal_ok,
            "status": status_ok,
        },
        "e2e": e2e,
        "failures": failures,
        "latency_ms": complete.get("latency_ms") or run["elapsed_ms"],
        "expect_insufficient": insufficient,
    }


async def check_evidence(
    client: httpx.AsyncClient,
    base_url: str,
    kb: str,
    question: dict[str, Any],
    result: dict[str, Any],
) -> None:
    for citation in result.get("citation_objects", []):
        evidence_id = citation.get("evidence_id")
        document_id = citation.get("document_id")
        if not evidence_id or not document_id:
            continue
        response = await client.get(
            f"{base_url}/api/v1/knowledge-bases/{kb}/documents/{document_id}/evidence/{evidence_id}"
        )
        result["evidence_status"][f"{document_id}:{evidence_id}"] = response.status_code == 200


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--dataset", default="scripts/m5_eval_dataset.json")
    parser.add_argument("--output", default="/tmp/m5-eval-results.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--only", default="")
    args = parser.parse_args()

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    questions = dataset["questions"]
    if args.only:
        categories = {item.strip() for item in args.only.split(",") if item.strip()}
        questions = [item for item in questions if item["category"] in categories]
    if args.limit:
        questions = questions[: args.limit]

    async with httpx.AsyncClient(timeout=360.0, trust_env=False) as client:
        results = []
        for index, question in enumerate(questions, start=1):
            print(f"[{index}/{len(questions)}] {question['id']} {question['query']}", flush=True)
            run = await ask(client, args.base_url, question, dataset["knowledge_base_id"])
            citation_objects: list[dict[str, Any]] = []
            for event in run["events"]:
                if event["event"] == "citations":
                    citation_objects = list(event["data"].get("items") or [])
            evidence_status: dict[str, bool] = {}
            for citation in citation_objects:
                evidence_id = citation.get("evidence_id")
                document_id = citation.get("document_id")
                if not evidence_id or not document_id:
                    continue
                response = await client.get(
                    f"{args.base_url}/api/v1/knowledge-bases/{dataset['knowledge_base_id']}"
                    f"/documents/{document_id}/evidence/{evidence_id}"
                )
                evidence_status[f"{document_id}:{evidence_id}"] = response.status_code == 200
            result = evaluate(question, run, evidence_status, dataset["documents"])
            result["evidence_status"] = evidence_status
            results.append(result)
            summary_line = (
                f"    e2e={result['e2e']} failures={result['failures']} "
                f"latency={result['latency_ms']:.0f}ms"
            )
            print(summary_line, flush=True)

    def rate(values: list[bool]) -> float:
        return round(100.0 * sum(values) / len(values), 1) if values else 0.0

    latencies = [float(r["latency_ms"]) for r in results]
    summary = {
        "questions": len(results),
        "tool_selection_accuracy": rate([r["checks"]["tool"] for r in results]),
        "parameter_validity": rate([r["checks"]["parameter"] for r in results]),
        "numerical_accuracy": rate(
            [r["checks"]["number"] and r["checks"]["phrase"] for r in results]
        ),
        "citation_validity": rate([r["checks"]["citation"] for r in results]),
        "retrieval_hit_at_k": rate([r["checks"]["retrieval"] for r in results]),
        "e2e_success": rate([r["e2e"] for r in results]),
        "latency_avg_ms": round(statistics.mean(latencies), 1) if latencies else 0,
        "latency_median_ms": round(statistics.median(latencies), 1) if latencies else 0,
        "latency_p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 1)
        if latencies
        else 0,
        "failures": {},
    }
    for result in results:
        for failure in result["failures"]:
            summary["failures"][failure] = summary["failures"].get(failure, 0) + 1

    output = {"summary": summary, "results": results}
    Path(args.output).write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
