"""Build a labeled montage of acceptance-KB images for manual gold query authoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import UUID

from PIL import Image, ImageDraw

from knowledge_scope.evidence.lifecycle import EvidenceArtifactError, load_evidence_artifact

CELL = (360, 300)
COLS = 3
ROWS_PER_SHEET = 4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()
    fixtures = json.loads(args.fixtures.read_text())
    per_doc = max(1, args.limit // len(fixtures["documents"]))
    entries = []
    for doc in fixtures["documents"]:
        try:
            artifact = load_evidence_artifact(args.data_dir, UUID(doc["id"]))
        except EvidenceArtifactError:
            continue
        picked = 0
        for evidence in artifact.evidence:
            if evidence.modality != "image":
                continue
            asset_refs = evidence.lineage.asset_refs
            if not asset_refs:
                continue
            entries.append(
                {
                    "document_name": doc["name"],
                    "document_id": doc["id"],
                    "evidence_id": evidence.evidence_id,
                    "asset_ref": asset_refs[0],
                    "asset_index": 0,
                    "page": evidence.lineage.page_start,
                    "caption": None,
                }
            )
            picked += 1
            if picked >= per_doc:
                break
    entries = entries[: args.limit]
    for entry in entries:
        path = args.data_dir / "parsing" / entry["document_id"] / "mineru" / entry["asset_ref"]
        entry["source_path"] = str(path)
        entry["exists"] = path.is_file()
    sheets = []
    for start in range(0, len(entries), COLS * ROWS_PER_SHEET):
        chunk = entries[start : start + COLS * ROWS_PER_SHEET]
        sheet = Image.new("RGB", (CELL[0] * COLS, CELL[1] * ROWS_PER_SHEET), "white")
        draw = ImageDraw.Draw(sheet)
        for offset, entry in enumerate(chunk):
            row, col = divmod(offset, COLS)
            x, y = col * CELL[0], row * CELL[1]
            label = f"#{start + offset + 1} {entry['document_name']} p{entry['page']}"
            draw.text((x + 6, y + 4), label, fill="black")
            path = Path(entry["source_path"])
            if path.is_file():
                try:
                    image = Image.open(path).convert("RGB")
                    image.thumbnail((CELL[0] - 12, CELL[1] - 24))
                    sheet.paste(image, (x + 6, y + 20))
                except Exception as error:
                    draw.text((x + 6, y + 30), f"decode failed: {error}", fill="red")
            else:
                draw.text((x + 6, y + 30), "missing asset", fill="red")
        out = args.out.with_name(f"{args.out.stem}-{len(sheets) + 1}{args.out.suffix}")
        sheet.save(out)
        sheets.append(str(out))
    mapping = args.out.with_name(f"{args.out.stem}-map.json")
    mapping.write_text(
        json.dumps({"entries": entries, "sheets": sheets}, ensure_ascii=False, indent=2)
    )
    print(json.dumps({"entries": len(entries), "sheets": sheets, "map": str(mapping)}))


if __name__ == "__main__":
    main()
