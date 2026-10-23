# ruff: noqa: RUF001
"""Prepare (and clean up) the isolated M5 evaluation fixtures.

The M5 evaluation needs a fresh knowledge base, four parsed documents, chunk
indexes, and a multimodal evidence index.  The UUIDs are intentionally not
stored in ``scripts/m5_eval_dataset.json``; they are written to a fixture file
that the harness reads with ``--fixtures``.

Workflow (GPU friendly: upload first, then stop the API while the CLI parses and
indexes, then restart the API for the evaluation run)::

    # 1) API running
    uv run python scripts/m5_make_eval_pdfs.py --output-dir /tmp/m5-eval-pdfs
    uv run python scripts/m5_prepare_eval_fixtures.py prepare \
        --pdf-dir /tmp/m5-eval-pdfs \
        --math-pdf "<A1.5 benchmark>/auto/..._origin.pdf" \
        --output /tmp/m5-eval-fixtures.json

    # 2) stop the API, then run the local MinerU/embedding pipeline
    uv run python scripts/m5_prepare_eval_fixtures.py index \
        --fixtures /tmp/m5-eval-fixtures.json

    # 3) restart the API and run the strict evaluation
    uv run python scripts/m5_agent_eval.py --fixtures /tmp/m5-eval-fixtures.json

    # 4) optional cleanup (API running)
    uv run python scripts/m5_prepare_eval_fixtures.py cleanup \
        --fixtures /tmp/m5-eval-fixtures.json

The math textbook PDF is not shipped with the repository; it comes from the
frozen A1.5 benchmark corpus.  Without ``--math-pdf`` the four math questions
cannot be reproduced and the harness will report them as unverified.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx

VISION_PDF = "m3-vision-acceptance.pdf"
TABLE_A_PDF = "m2-pdf-a.pdf"
TABLE_B_PDF = "m2-pdf-b.pdf"
MATH_PDF = "m5-math.pdf"

MATH_SOURCE_DEFAULT = (
    "/home/amber/PycharmProjects/KnowledgeScope/data/benchmarks/a1-5/raw/"
    "item-e71e010348534ba57c9d2e421439f197659bf1a524dffa17b5f51d7b4bf08ded/mineru/"
    "普通高中教科书·数学（A版）选择性必修_第三册_21-40/auto/"
    "普通高中教科书·数学（A版）选择性必修_第三册_21-40_origin.pdf"
)


def _src_root() -> Path:
    return Path(__file__).resolve().parent.parent / "src"


def _cli_environment(cli_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_src_root())
    env["NO_PROXY"] = "localhost,127.0.0.1"
    env["no_proxy"] = "localhost,127.0.0.1"
    env.setdefault(
        "KNOWLEDGE_SCOPE_MINERU_COMMAND",
        "/home/amber/.local/share/knowledgescope/mineru-3.4.5/bin/mineru",
    )
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env["KNOWLEDGE_SCOPE_DATA_DIR"] = str(cli_root / "data")
    return env


def _cli(cli_root: Path, arguments: list[str]) -> subprocess.CompletedProcess[str]:
    venv_cli = cli_root / ".venv" / "bin" / "knowledgescope"
    if not venv_cli.is_file():
        raise SystemExit(f"CLI not found: {venv_cli}")
    return subprocess.run(
        [str(venv_cli), *arguments],
        cwd=str(cli_root),
        env=_cli_environment(cli_root),
        text=True,
        capture_output=True,
        check=False,
    )


def _upload_documents(
    client: httpx.Client, api_base: str, repo_root: Path, args: argparse.Namespace
) -> dict[str, Any]:
    pdf_dir = Path(args.pdf_dir)
    candidates: list[tuple[str, Path, str]] = []
    vision_pdf = Path(args.vision_pdf) if args.vision_pdf else pdf_dir / VISION_PDF
    table_a = Path(args.table_a) if args.table_a else pdf_dir / TABLE_A_PDF
    table_b = Path(args.table_b) if args.table_b else pdf_dir / TABLE_B_PDF
    for label, path, kind in (
        ("V", vision_pdf, "generated"),
        ("A", table_a, "generated"),
        ("B", table_b, "generated"),
    ):
        if not path.is_file():
            raise SystemExit(f"missing evaluation PDF: {path} (run m5_make_eval_pdfs.py)")
        candidates.append((label, path, kind))

    math_pdf_arg = args.math_pdf or MATH_SOURCE_DEFAULT
    math_pdf = Path(math_pdf_arg)
    if math_pdf.is_file():
        candidates.append(("M", math_pdf, "external_corpus"))
    else:
        print(f"WARNING: math PDF not found ({math_pdf}); math questions will be unverified")

    response = client.post(
        f"{api_base}/api/v1/knowledge-bases",
        json={"name": args.kb_name, "description": "M5 strict evaluation fixtures"},
    )
    response.raise_for_status()
    kb_id = response.json()["id"]

    documents: dict[str, str] = {}
    sources: dict[str, dict[str, str]] = {}
    for label, path, kind in candidates:
        filename = MATH_PDF if label == "M" else path.name
        with path.open("rb") as handle:
            upload = client.post(
                f"{api_base}/api/v1/knowledge-bases/{kb_id}/documents",
                files={"file": (filename, handle, "application/pdf")},
            )
        upload.raise_for_status()
        document_id = upload.json()["id"]
        documents[label] = document_id
        sources[label] = {"path": str(path), "kind": kind, "filename": filename}
        print(f"uploaded {label}: {document_id} <- {path.name}")

    return {
        "knowledge_base_id": kb_id,
        "knowledge_base_name": args.kb_name,
        "pdf_dir": str(pdf_dir),
        "repo_root": str(repo_root),
        "documents": documents,
        "sources": sources,
        "math_pdf_available": "M" in documents,
        "math_source_expected": str(math_pdf_arg),
    }


def _index_document(cli_root: Path, kb_id: str, document_id: str) -> None:
    for arguments in (
        ["parse-document", document_id],
        ["chunk-document", document_id],
        ["qdrant", "index-document", document_id],
        [
            "multimodal-index",
            "build",
            "--knowledge-base-id",
            kb_id,
            "--canonical-path",
            f"data/parsing/{document_id}/canonical.json",
        ],
    ):
        print("RUN", " ".join(arguments), flush=True)
        result = _cli(cli_root, arguments)
        if result.returncode != 0:
            tail = (result.stderr or result.stdout).strip().splitlines()[-6:]
            command_text = " ".join(arguments)
            raise SystemExit(
                f"command failed ({result.returncode}): {command_text}\n" + "\n".join(tail)
            )


def _phase_prepare(repo_root: Path, args: argparse.Namespace) -> None:
    with httpx.Client(timeout=120.0, trust_env=False) as client:
        fixtures = _upload_documents(client, args.api_base, repo_root, args)
    Path(args.output).write_text(
        json.dumps(fixtures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"fixtures written to {args.output}")
    print("next: stop the API, then run the index phase")


def _phase_index(cli_root: Path, args: argparse.Namespace) -> None:
    fixtures = json.loads(Path(args.fixtures).read_text(encoding="utf-8"))
    kb_id = fixtures["knowledge_base_id"]
    for label, document_id in fixtures["documents"].items():
        print(f"=== indexing {label} {document_id}", flush=True)
        _index_document(cli_root, kb_id, document_id)
    fixtures["indexed"] = True
    Path(args.fixtures).write_text(
        json.dumps(fixtures, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("index phase complete")


def _phase_cleanup(cli_root: Path, args: argparse.Namespace) -> None:
    fixtures = json.loads(Path(args.fixtures).read_text(encoding="utf-8"))
    kb_id = fixtures["knowledge_base_id"]
    with httpx.Client(timeout=120.0, trust_env=False) as client:
        for document_id in fixtures["documents"].values():
            response = client.delete(
                f"{args.api_base}/api/v1/knowledge-bases/{kb_id}/documents/{document_id}"
            )
            print(f"delete document {document_id}: {response.status_code}")
        response = client.delete(f"{args.api_base}/api/v1/knowledge-bases/{kb_id}")
        print(f"delete knowledge base: {response.status_code}")
    data_dir = Path(_cli_environment(cli_root)["KNOWLEDGE_SCOPE_DATA_DIR"])
    for document_id in fixtures["documents"].values():
        for directory in ("parsing", "chunking", "evidence"):
            shutil.rmtree(data_dir / directory / document_id, ignore_errors=True)
    print("cleanup complete")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "index", "cleanup"))
    parser.add_argument("--repo-root", default=str(Path.cwd()))
    parser.add_argument(
        "--cli-root",
        default="/home/amber/PycharmProjects/KnowledgeScope",
        help="checkout whose .env/data holds the API storage (must match the running API)",
    )
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--fixtures", default="/tmp/m5-eval-fixtures.json")
    parser.add_argument("--output", default="/tmp/m5-eval-fixtures.json")
    parser.add_argument("--pdf-dir", default="/tmp/m5-eval-pdfs")
    parser.add_argument("--math-pdf", default="")
    parser.add_argument("--vision-pdf", default="")
    parser.add_argument("--table-a", default="")
    parser.add_argument("--table-b", default="")
    parser.add_argument("--kb-name", default="M5 严格评测知识库")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    cli_root = Path(args.cli_root).resolve()
    if args.phase == "prepare":
        _phase_prepare(repo_root, args)
    elif args.phase == "index":
        _phase_index(cli_root, args)
    else:
        _phase_cleanup(cli_root, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
