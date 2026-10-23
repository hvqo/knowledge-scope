# ruff: noqa: RUF001
"""Generate the synthetic PDF fixtures used by the M5 agentic RAG evaluation.

The vision PDF embeds the three committed PNG fixtures under
``tests/fixtures/m5`` so the evaluation input is reproducible without an
external corpus.  The two GDP table PDFs are drawn with reportlab.

The real math textbook PDF is NOT generated here: it is extracted from the
frozen A1.5 benchmark corpus and must be passed explicitly to
``scripts/m5_prepare_eval_fixtures.py --math-pdf``.  Without it, the four
math-related questions cannot be reproduced (see the M5 evaluation report).

Usage::

    uv run python scripts/m5_make_eval_pdfs.py --output-dir /tmp/m5-eval-pdfs
"""

from __future__ import annotations

import argparse
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas

FIXTURE_ROOT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "m5"
FONT_NAME = "STSong-Light"

GDP_ROWS_A = [
    ("越南", "4300", "4760"),
    ("泰国", "5000", "5200"),
    ("文莱", "0", "50"),
]
GDP_ROWS_B = [
    ("越南", "4500", "4700"),
    ("泰国", "4900", "5300"),
    ("文莱", "10", "40"),
]


def _draw_table_pdf(path: Path, caption: str, rows: list[tuple[str, str, str]]) -> None:
    c = canvas.Canvas(str(path), pagesize=A4)
    _width, height = A4
    c.setFont(FONT_NAME, 16)
    c.drawString(60, height - 100, caption)

    x0 = 60
    y0 = height - 170
    col_widths = (150, 190, 190)
    row_height = 44
    headers = ("国家", "2023年GDP", "2024年GDP")

    def draw_row(values: tuple[str, ...], row_index: int, bold: bool) -> None:
        c.setFont(FONT_NAME, 13 if bold else 12)
        x = x0
        y_top = y0 - row_index * row_height
        for value, col_width in zip(values, col_widths, strict=True):
            c.rect(x, y_top - row_height, col_width, row_height)
            c.drawString(x + 10, y_top - row_height + 14, value)
            x += col_width

    draw_row(headers, 0, bold=True)
    for index, row in enumerate(rows, start=1):
        draw_row(row, index, bold=False)
    c.showPage()
    c.save()


def _draw_vision_pdf(path: Path) -> None:
    c = canvas.Canvas(str(path), pagesize=A4)
    _width, height = A4

    def page(image_name: str, caption: str, note: str) -> None:
        image_path = FIXTURE_ROOT / image_name
        if not image_path.is_file():
            raise FileNotFoundError(f"missing fixture image: {image_path}")
        c.drawImage(
            str(image_path),
            60,
            height - 460,
            width=475,
            height=300,
            preserveAspectRatio=True,
            anchor="c",
        )
        c.setFont(FONT_NAME, 14)
        c.drawString(60, height - 500, caption)
        c.setFont(FONT_NAME, 11)
        c.drawString(60, height - 522, note)
        c.showPage()

    page("chart.png", "图 1 2023 年季度销售额趋势", "该图表展示了四个季度的销售额变化。")
    page("table.png", "表 1 各国 GDP 数据", "数据单位为亿美元，包含 2023 年和 2024 年。")
    page("flow.png", "图 3 文档处理流程示意", "该流程图展示从输入 PDF 到输出 Evidence 的流程。")
    c.save()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/tmp/m5-eval-pdfs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdfmetrics.registerFont(UnicodeCIDFont(FONT_NAME))

    vision_pdf = output_dir / "m3-vision-acceptance.pdf"
    table_a = output_dir / "m2-pdf-a.pdf"
    table_b = output_dir / "m2-pdf-b.pdf"
    _draw_vision_pdf(vision_pdf)
    _draw_table_pdf(table_a, "表 1 各国 GDP 单位：亿美元", GDP_ROWS_A)
    _draw_table_pdf(table_b, "表 2 各国 GDP 预测值 单位：亿美元", GDP_ROWS_B)
    for path in (vision_pdf, table_a, table_b):
        print(f"created {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
