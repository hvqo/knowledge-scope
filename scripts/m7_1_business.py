"""Run real business acceptance questions against the running agent API."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8000/api/v1"


def _parse_sse(lines: list[str]) -> list[dict]:
    events = []
    name = None
    data: list[str] = []
    for line in lines:
        if line.startswith("event: "):
            name = line[7:].strip()
        elif line.startswith("data: "):
            data.append(line[6:])
        elif line == "" and name:
            events.append({"event": name, "data": json.loads("\n".join(data) or "{}")})
            name, data = None, []
    return events


def _run_query(client: httpx.Client, query: str, kb: str) -> dict:
    started = time.perf_counter()
    timestamps: dict[str, float] = {}
    events: list[dict] = []
    answer_parts: list[str] = []
    with client.stream(
        "POST",
        f"{BASE}/rag/query",
        json={
            "query": query,
            "knowledge_base_id": kb,
            "retrieval_mode": "unified",
            "answer_mode": "agent",
        },
    ) as response:
        response.raise_for_status()
        buffer: list[str] = []
        for line in response.iter_lines():
            buffer.append(line)
            if line == "":
                batch = _parse_sse(buffer)
                buffer = []
                for event in batch:
                    events.append(event)
                    if event["event"] == "answer_delta":
                        answer_parts.append(str(event["data"].get("text") or ""))
                    timestamps.setdefault(event["event"], time.perf_counter())
    complete = next(
        (event["data"] for event in reversed(events) if event["event"] == "complete"), {}
    )
    citations = next(
        (event["data"] for event in reversed(events) if event["event"] == "citations"), {}
    )
    answer = "".join(answer_parts)
    return {
        "started": started,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        "events": [event["event"] for event in events],
        "tool_statuses": [
            {"tool": event["data"].get("tool"), "status": event["data"].get("status")}
            for event in events
            if event["event"] == "tool_status"
        ],
        "answer": answer,
        "citations": citations.get("items") or [],
        "complete": complete,
        "timestamps": timestamps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixtures = json.loads(args.fixtures.read_text())
    kb = fixtures["knowledge_base_id"]
    queries = json.loads(args.gold.read_text())["business"]
    results = []
    with httpx.Client(trust_env=False, timeout=600) as client:
        for item in queries:
            run = _run_query(client, item["query"], kb)
            citation_evidence = {
                citation.get("evidence_id")
                for citation in run["citations"]
                if citation.get("evidence_id")
            }
            visual_evidence = {
                source.get("evidence_id")
                for source in run["complete"].get("visual_sources") or []
                if source.get("evidence_id")
            }
            gold = set(item["gold_evidence_ids"])
            used_image = any(status["tool"] == "analyze_image" for status in run["tool_statuses"])
            results.append(
                {
                    "id": item["id"],
                    "query": item["query"],
                    "kind": item.get("kind"),
                    "gold_evidence_ids": sorted(gold),
                    "elapsed_ms": run["elapsed_ms"],
                    "link_ms": round(
                        (run["timestamps"].get("complete", run["started"]) - run["started"]) * 1000,
                        1,
                    ),
                    "tool_statuses": run["tool_statuses"],
                    "used_analyze_image": used_image,
                    "citations_evidence": sorted(citation_evidence),
                    "visual_sources_evidence": sorted(visual_evidence),
                    "citation_hit": bool(gold & citation_evidence),
                    "visual_source_hit": bool(gold & visual_evidence),
                    "gold_in_candidates": bool(gold & citation_evidence or gold & visual_evidence),
                    "safety_gate": run["complete"].get("safety_gate"),
                    "fact_status": run["complete"].get("fact_status"),
                    "provider": run["complete"].get("provider"),
                    "model": run["complete"].get("model"),
                    "status": run["complete"].get("status"),
                    "retrieval_latency_ms": run["complete"].get("retrieval_latency_ms"),
                    "llm_latency_ms": run["complete"].get("llm_latency_ms"),
                    "answer_excerpt": run["answer"][:400],
                }
            )
            print(
                f"{item['id']}: status={run['complete'].get('status')} "
                f"image={used_image} gold_hit={results[-1]['gold_in_candidates']} "
                f"latency={run['elapsed_ms']}ms gate={run['complete'].get('safety_gate')}"
            )
    args.output.write_text(
        json.dumps({"fixtures": fixtures, "results": results}, ensure_ascii=False, indent=2)
    )
    total = len(results)
    passed = sum(1 for item in results if item["gold_in_candidates"])
    print(
        json.dumps(
            {"queries": total, "gold_hits": passed, "output": str(args.output)}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
