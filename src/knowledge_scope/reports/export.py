"""Deterministic, server-side report exports from persisted report state."""

# Chinese labels are intentionally part of the export format.
# ruff: noqa: RUF001

from __future__ import annotations

import io
import json
import re
import zipfile
from collections.abc import Iterable
from typing import Literal
from xml.etree import ElementTree as ET

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfgen import canvas

from .generation import stable_source_marker
from .models import REPORT_SECTION_CONTENT_MAX_LENGTH
from .schemas import ReportDetailResponse, ReportSourceReferenceResponse

_WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_CONTENT_TYPES_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
_XML_NS = "http://www.w3.org/XML/1998/namespace"
_SAFE_FILENAME = re.compile(r"[^\w\-\u4e00-\u9fff]+", re.UNICODE)


class ReportExportError(Exception):
    """Raised when the selected server-side export backend is unavailable."""

    def __init__(self, category: Literal["aggregate_bounds", "backend", "pdf_font_unavailable"]):
        super().__init__(category)
        self.category = category


REPORT_EXPORT_MAX_SECTIONS = 100
REPORT_EXPORT_MAX_SOURCE_REFERENCES = 256
REPORT_EXPORT_MAX_BODY_CHARS = REPORT_SECTION_CONTENT_MAX_LENGTH * 10
REPORT_EXPORT_MAX_CHATBI_CELLS = 50_000


def safe_report_filename(title: str, extension: str) -> str:
    """Return a bounded filename that cannot introduce a filesystem path."""
    normalized = _SAFE_FILENAME.sub("-", title.strip()).strip(".-_")[:80] or "report"
    return f"{normalized}.{extension}"


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _clean_text(str(value))


def _clean_text(value: str) -> str:
    """Remove XML/PDF control characters while preserving normal Unicode text."""
    return "".join(
        character for character in value if character in "\t\n\r" or ord(character) >= 32
    )


def _paragraph(root: ET.Element, text: str, *, style: str | None = None) -> None:
    paragraph = ET.SubElement(root, f"{{{_WORD_NS}}}p")
    if style is not None:
        properties = ET.SubElement(paragraph, f"{{{_WORD_NS}}}pPr")
        style_element = ET.SubElement(properties, f"{{{_WORD_NS}}}pStyle")
        style_element.set(f"{{{_WORD_NS}}}val", style)
    run = ET.SubElement(paragraph, f"{{{_WORD_NS}}}r")
    text_element = ET.SubElement(run, f"{{{_WORD_NS}}}t")
    text_element.set(f"{{{_XML_NS}}}space", "preserve")
    text_element.text = _clean_text(text)


def _table(root: ET.Element, source: ReportSourceReferenceResponse) -> None:
    if not source.columns:
        return
    table = ET.SubElement(root, f"{{{_WORD_NS}}}tbl")
    rows: list[list[str]] = [
        [column.name for column in source.columns],
        *[[_safe_text(cell) for cell in row] for row in source.rows],
    ]
    for values in rows[:101]:
        row = ET.SubElement(table, f"{{{_WORD_NS}}}tr")
        for value in values[:50]:
            cell = ET.SubElement(row, f"{{{_WORD_NS}}}tc")
            _paragraph(cell, value)


def _validate_export_bounds(report: ReportDetailResponse) -> None:
    """Reject oversized reports before either exporter starts rendering."""
    if len(report.sections) > REPORT_EXPORT_MAX_SECTIONS:
        raise ReportExportError("aggregate_bounds")

    source_count = 0
    body_chars = len(_safe_text(report.title))
    chatbi_cells = 0
    for section in report.sections:
        source_count += len(section.sources)
        body_chars += len(_safe_text(section.title)) + len(_safe_text(section.content))
        if source_count > REPORT_EXPORT_MAX_SOURCE_REFERENCES:
            raise ReportExportError("aggregate_bounds")
        if body_chars > REPORT_EXPORT_MAX_BODY_CHARS:
            raise ReportExportError("aggregate_bounds")
        for source in section.sources:
            body_chars += len(_safe_text(source.title))
            body_chars += len(_safe_text(source.snippet))
            body_chars += len(_safe_text(source.question))
            body_chars += len(_safe_text(source.answer))
            body_chars += sum(len(_safe_text(column.name)) for column in source.columns)
            if source.kind == "chatbi_result":
                chatbi_cells += len(source.columns) * len(source.rows)
                if chatbi_cells > REPORT_EXPORT_MAX_CHATBI_CELLS:
                    raise ReportExportError("aggregate_bounds")
            body_chars += sum(sum(len(_safe_text(cell)) for cell in row) for row in source.rows)
            if body_chars > REPORT_EXPORT_MAX_BODY_CHARS:
                raise ReportExportError("aggregate_bounds")


def _source_lines(source: ReportSourceReferenceResponse, marker: str) -> Iterable[str]:
    pages = ""
    if source.page_start is not None:
        pages = f"，第 {source.page_start}–{source.page_end or source.page_start} 页"
    yield f"[{marker}] {source.title}{pages}"
    if source.kind == "rag_evidence" and source.snippet:
        yield source.snippet
    if source.kind == "chatbi_result":
        if source.question:
            yield f"问题：{source.question}"
        if source.answer:
            yield f"回答：{source.answer}"


def build_docx(report: ReportDetailResponse) -> bytes:
    """Build a minimal OOXML document without writing a temporary file."""
    _validate_export_bounds(report)
    ET.register_namespace("w", _WORD_NS)
    ET.register_namespace(
        "r", "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    )
    root = ET.Element(f"{{{_WORD_NS}}}document")
    body = ET.SubElement(root, f"{{{_WORD_NS}}}body")
    _paragraph(body, report.title, style="Title")
    source_index = {
        source.id: stable_source_marker(source.id)
        for source in [source for section in report.sections for source in section.sources]
    }
    for section in report.sections:
        _paragraph(body, section.title, style="Heading1")
        for line in section.content.splitlines() or [""]:
            _paragraph(body, line)
        for source in section.sources:
            lines = list(_source_lines(source, source_index[source.id]))
            _paragraph(body, lines[0], style="Heading2")
            for line in lines[1:]:
                _paragraph(body, line)
            _table(body, source)
    _paragraph(body, "参考来源", style="Heading1")
    for source in [source for section in report.sections for source in section.sources]:
        _paragraph(body, next(_source_lines(source, source_index[source.id])))
    section_properties = ET.SubElement(body, f"{{{_WORD_NS}}}sectPr")
    page_size = ET.SubElement(section_properties, f"{{{_WORD_NS}}}pgSz")
    page_size.set(f"{{{_WORD_NS}}}w", "11906")
    page_size.set(f"{{{_WORD_NS}}}h", "16838")

    document_xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<Types xmlns="{_CONTENT_TYPES_NS}">'
        '<Default Extension="rels" '
        'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    ).encode()
    root_relationships = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<Relationships xmlns="{_REL_NS}">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/'
        'officeDocument" Target="word/document.xml"/>'
        "</Relationships>"
    ).encode()
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in (
            ("[Content_Types].xml", content_types),
            ("_rels/.rels", root_relationships),
            ("word/document.xml", document_xml),
        ):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, data)
    return result.getvalue()


def _wrap_pdf_line(value: str, max_chars: int = 46) -> list[str]:
    lines: list[str] = []
    for paragraph in value.splitlines() or [""]:
        if not paragraph:
            lines.append("")
            continue
        lines.extend(
            paragraph[index : index + max_chars] for index in range(0, len(paragraph), max_chars)
        )
    return lines


def _report_lines(report: ReportDetailResponse) -> Iterable[str]:
    yield report.title
    for section in report.sections:
        yield section.title
        yield from section.content.splitlines() or [""]
        for source in section.sources:
            yield from _source_lines(source, stable_source_marker(source.id))
            if source.kind == "chatbi_result" and source.columns:
                yield " | ".join(column.name for column in source.columns[:20])
            for row in source.rows[:20]:
                yield " | ".join(_safe_text(cell) for cell in row[:20])
    yield "参考来源"
    for source in [source for section in report.sections for source in section.sources]:
        yield next(_source_lines(source, stable_source_marker(source.id)))


def build_pdf(report: ReportDetailResponse) -> bytes:
    """Build a deterministic PDF using the standard Chinese CID font."""
    _validate_export_bounds(report)
    try:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    except Exception as error:  # pragma: no cover - depends on reportlab install.
        raise ReportExportError("pdf_font_unavailable") from error
    output = io.BytesIO()
    document = canvas.Canvas(output, pagesize=A4, pageCompression=1, invariant=1)
    document.setTitle(report.title)
    document.setAuthor("KnowledgeScope")
    _width, height = A4
    x, y = 48, height - 56
    document.setFont("STSong-Light", 16)
    for raw_line in _report_lines(report):
        for line in _wrap_pdf_line(raw_line):
            if y < 52:
                document.showPage()
                document.setFont("STSong-Light", 10)
                y = height - 48
            document.drawString(x, y, _clean_text(line))
            y -= 16 if raw_line == report.title else 13
            if raw_line == report.title:
                document.setFont("STSong-Light", 10)
    document.save()
    return output.getvalue()


__all__ = [
    "REPORT_EXPORT_MAX_BODY_CHARS",
    "REPORT_EXPORT_MAX_CHATBI_CELLS",
    "REPORT_EXPORT_MAX_SECTIONS",
    "REPORT_EXPORT_MAX_SOURCE_REFERENCES",
    "ReportExportError",
    "build_docx",
    "build_pdf",
    "safe_report_filename",
]
