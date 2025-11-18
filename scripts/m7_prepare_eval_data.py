"""Build an isolated canonical/evidence tree for the M7 visual-retrieval experiments.

The historical ``data/parsing`` tree was never retained (the benchmark kept
canonical JSON plus MinerU raw outputs), so the visual index experiments run on
a throwaway data directory rebuilt from the frozen A1.5 benchmark artifacts.
Nothing here touches ``data/parsing``, ``data/evidence`` or the historical
knowledge bases.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from uuid import UUID

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from knowledge_scope.evidence.lifecycle import write_evidence_artifact  # noqa: E402
from knowledge_scope.evidence.service import build_evidence_document  # noqa: E402
from knowledge_scope.parsing.models import (  # noqa: E402
    CanonicalDocument,
    ImageBlock,
    TextBlock,
    TitleBlock,
)

DEFAULT_KB_ID = UUID("00000000-0000-4000-8000-000000000007")


def _copy_assets(
    canonical: CanonicalDocument,
    raw_dir: Path,
    parsing_dir: Path,
) -> tuple[int, int]:
    copied = 0
    missing = 0
    for page in canonical.pages:
        for block in page.blocks:
            if not isinstance(block, ImageBlock):
                continue
            source = raw_dir / block.asset_ref
            target = parsing_dir / "mineru" / block.asset_ref
            if not source.is_file():
                missing += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied += 1
    return copied, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recommended", default="/tmp/m7-audit/recommended_docs.json")
    parser.add_argument("--canonical-root", default="data/benchmarks/a1-5/canonical")
    parser.add_argument("--raw-root", default="data/benchmarks/a1-5/raw")
    parser.add_argument("--output", default="/tmp/m7-data")
    parser.add_argument("--knowledge-base-id", default=str(DEFAULT_KB_ID))
    parser.add_argument("--docs", nargs="*", default=None, help="docname substrings to keep")
    args = parser.parse_args()

    recommended = json.loads(Path(args.recommended).read_text(encoding="utf-8"))
    if args.docs:
        recommended = [
            item for item in recommended if any(part in item["docname"] for part in args.docs)
        ]
    if not recommended:
        raise SystemExit("no matching recommended documents")

    output = Path(args.output).resolve()
    parsing_root = output / "parsing"
    knowledge_base_id = UUID(args.knowledge_base_id)
    canonical_root = Path(args.canonical_root)
    raw_root = Path(args.raw_root)

    manifest: list[dict[str, object]] = []
    for item in recommended:
        document_id = UUID(item["doc_uuid"])
        canonical_path = canonical_root / f"{item['item_id']}.json"
        if not canonical_path.is_file():
            print(f"skip {item['docname']}: canonical missing", file=sys.stderr)
            continue
        canonical = CanonicalDocument.model_validate_json(
            canonical_path.read_text(encoding="utf-8")
        )
        if canonical.document_id != document_id:
            print(
                f"skip {item['docname']}: canonical id {canonical.document_id} != {document_id}",
                file=sys.stderr,
            )
            continue
        parsing_dir = parsing_root / str(document_id)
        if parsing_dir.exists():
            shutil.rmtree(parsing_dir)
        parsing_dir.mkdir(parents=True)
        (parsing_dir / "canonical.json").write_text(
            canonical.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        raw_dir = raw_root / item["item_id"] / "mineru"
        copied, missing = _copy_assets(canonical, raw_dir, parsing_dir)

        context_by_block_id: dict[str, str] = {}
        for page in canonical.pages:
            page_text = " ".join(
                block.text
                for block in page.blocks
                if isinstance(block, (TextBlock, TitleBlock)) and block.text
            )[:800]
            if not page_text:
                continue
            for block in page.blocks:
                if isinstance(block, ImageBlock):
                    context_by_block_id[block.block_id] = page_text
        artifact = build_evidence_document(
            canonical,
            knowledge_base_id,
            context_by_block_id=context_by_block_id,
        )
        write_evidence_artifact(output, artifact)
        image_evidence = [entry for entry in artifact.evidence if entry.modality == "image"]
        captioned = sum(
            1
            for entry in image_evidence
            if any(rep.representation_type == "caption" for rep in entry.representations)
        )
        manifest.append(
            {
                "document_id": str(document_id),
                "docname": item["docname"],
                "category": item.get("category"),
                "pages": len(canonical.pages),
                "image_blocks": sum(
                    1
                    for page in canonical.pages
                    for block in page.blocks
                    if isinstance(block, ImageBlock)
                ),
                "image_assets_copied": copied,
                "image_assets_missing": missing,
                "image_evidence": len(image_evidence),
                "image_evidence_with_caption": captioned,
            }
        )
        print(
            f"{item['docname']}: pages={len(canonical.pages)} images={copied} "
            f"missing={missing} evidence={len(image_evidence)} captioned={captioned}"
        )

    output.mkdir(parents=True, exist_ok=True)
    manifest_doc = {
        "knowledge_base_id": str(knowledge_base_id),
        "output": str(output),
        "documents": manifest,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    total_images = sum(int(entry["image_assets_copied"]) for entry in manifest)
    print(f"prepared {len(manifest)} documents, {total_images} images under {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
