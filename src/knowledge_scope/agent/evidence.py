# ruff: noqa: RUF001
"""Lightweight, inspectable evidence requirements and sufficiency checks.

The agent does not ask a model whether retrieval is sufficient.  Instead this
module extracts a small structured requirement from the question (entities,
years, metric, required modality, comparison intent) and checks the returned
candidates against it.  The check is deliberately conservative: it only proves
that a candidate mentions the required item, never that the evidence is
semantically sufficient.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from knowledge_scope.agent.schemas import AgentCandidate

_COUNTRIES: tuple[str, ...] = (
    "越南",
    "泰国",
    "中国",
    "美国",
    "日本",
    "印度",
    "德国",
    "法国",
    "英国",
    "俄罗斯",
    "巴西",
    "加拿大",
    "澳大利亚",
    "韩国",
    "新加坡",
    "马来西亚",
    "印度尼西亚",
    "菲律宾",
    "文莱",
    "柬埔寨",
    "老挝",
    "缅甸",
    "墨西哥",
    "意大利",
    "西班牙",
    "荷兰",
    "沙特阿拉伯",
    "南非",
    "埃及",
    "土耳其",
    "阿根廷",
)

_METRICS: dict[str, str] = {
    "gdp": "GDP",
    "国内生产总值": "GDP",
    "人均gdp": "人均GDP",
    "人口": "人口",
    "增长率": "增长率",
    "增长速度": "增长率",
    "销售额": "销售额",
    "销量": "销量",
    "收入": "收入",
    "产量": "产量",
    "出口": "出口",
    "进口": "进口",
    "碳排放": "碳排放",
    "二氧化碳": "碳排放",
    "温度": "温度",
    "降水": "降水",
    "准确率": "准确率",
    "识别率": "识别率",
    "精度": "精度",
    "召回率": "召回率",
}

_IMAGE_SOFT = re.compile(
    r"图|图表|图片|示意图|流程图|结构图|曲线|柱状|折线|饼图|散点|地图|照片|示意图|趋势图"
)
_IMAGE_FEATURE = re.compile(
    r"趋势|走势|峰值|低谷|变化|波动|上升|下降|增长趋势|最高|最低|关系|结构|步骤|流程|"
    r"分布|颜色|位置|形状|对比图|示意图|曲线|柱状|折线|饼图|散点"
)
_TABLE_INTENT = re.compile(r"表|表格|数据|统计|数值|指标|对比|比较|增长|汇总|合计|平均")
_COMPARISON = re.compile(r"对比|比较|相比|差异|分别|各自|vs|VS|和|与|、")
_NUMBER_WORDS = re.compile(r"\d+(?:\.\d+)?")
_TITLE_ONLY = re.compile(r"标题|题目|叫什么|哪一页|第几页|来源|出处|页码|作者|是什么图")


@dataclass(frozen=True, slots=True)
class EvidenceRequirement:
    """One small evidence checklist derived from the user question."""

    query: str
    objects: tuple[str, ...] = ()
    years: tuple[int, ...] = ()
    metrics: tuple[str, ...] = ()
    needs_image: bool = False
    needs_image_feature: bool = False
    needs_table: bool = False
    comparison: bool = False
    range_years: tuple[int, int] | None = None
    relation_terms: tuple[str, ...] = ()
    scanned_table: bool = False

    @property
    def empty(self) -> bool:
        return not (
            self.objects
            or self.years
            or self.metrics
            or self.needs_image
            or self.needs_table
            or self.relation_terms
        )


@dataclass(frozen=True, slots=True)
class EvidenceAssessment:
    """Candidate coverage against one requirement checklist."""

    satisfied: bool
    missing: tuple[str, ...] = ()
    object_coverage: dict[str, tuple[int, ...]] = field(default_factory=dict)
    relation_coverage: dict[str, tuple[int, ...]] = field(default_factory=dict)
    year_coverage: tuple[int, ...] = ()
    metric_coverage: tuple[str, ...] = ()
    image_indices: tuple[int, ...] = ()
    table_indices: tuple[int, ...] = ()
    conflicts: tuple[str, ...] = ()

    def coverage_summary(self) -> dict[str, int]:
        return {
            "objects": len(self.object_coverage),
            "years": len(self.year_coverage),
            "metrics": len(self.metric_coverage),
            "images": len(self.image_indices),
            "tables": len(self.table_indices),
            "missing": len(self.missing),
            "relations": len(self.relation_coverage),
        }


def visual_feature_requested(query: str) -> bool:
    """Return True when the question asks for native visual features."""
    if not _IMAGE_FEATURE.search(query):
        return False
    return not (
        _TITLE_ONLY.search(query) and not re.search(r"趋势|峰值|走势|变化|关系|结构|流程", query)
    )


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def _candidate_text(candidate: AgentCandidate) -> str:
    citation = candidate.citation
    parts = [
        candidate.document_title or "",
        citation.snippet or "",
        " ".join(citation.section_path),
        " ".join(candidate.table_columns),
        " ".join(candidate.table_rows),
    ]
    return _normalize(" ".join(part for part in parts if part))


def extract_requirement(query: str) -> EvidenceRequirement:
    """Extract a bounded evidence requirement from the question text."""
    text = query.strip()
    lowered = text.lower()
    objects = tuple(country for country in _COUNTRIES if country in text)
    years = tuple(sorted({int(match) for match in re.findall(r"(?:19|20)\d{2}", text)}))
    if len(years) > 4:
        years = years[:4]
    range_years: tuple[int, int] | None = None
    range_match = re.search(r"((?:19|20)\d{2})\s*[-—–至到]\s*((?:19|20)\d{2})", text)
    if range_match:
        start, end = int(range_match.group(1)), int(range_match.group(2))
        if start <= end:
            range_years = (start, end)
    metrics: list[str] = []
    for token, normalized in _METRICS.items():
        if token in lowered and normalized not in metrics:
            metrics.append(normalized)
    needs_image = bool(_IMAGE_SOFT.search(text))
    needs_image_feature = needs_image and bool(_IMAGE_FEATURE.search(text))
    scanned_table = bool(
        re.search(r"图片\s*表格|图像\s*表格|扫描\s*(?:表|件|图片)|扫描表|图片中", text)
    )
    if scanned_table:
        needs_image = True
        needs_image_feature = True
    needs_table = bool(_TABLE_INTENT.search(text)) or bool(metrics and len(objects) >= 2)
    comparison = len(objects) >= 2 or (bool(_COMPARISON.search(text)) and len(objects) >= 2)
    relation_terms = _relation_terms(text) if "关系" in text else ()
    return EvidenceRequirement(
        query=query,
        objects=objects,
        years=years,
        metrics=tuple(metrics),
        needs_image=needs_image,
        needs_image_feature=needs_image_feature,
        needs_table=needs_table,
        comparison=comparison,
        range_years=range_years,
        relation_terms=relation_terms,
        scanned_table=scanned_table,
    )


_RELATION_STOPWORDS = (
    "请解释",
    "解释",
    "请说明",
    "说明",
    "统计关系",
    "关系",
    "之间",
    "的",
    "请分析",
    "分析",
    "请说明",
)

_RELATION_SPLIT = re.compile(r"与|和|、|,|，|对比|比较|相比|差异|关系")


def _relation_terms(text: str) -> tuple[str, ...]:
    """Extract the concepts whose relationship the question asks about."""
    raw = text.strip()
    for word in _RELATION_STOPWORDS:
        raw = raw.replace(word, " ")
    lowered = raw.lower()
    for token in _METRICS:
        lowered = lowered.replace(token, " ")
    terms: list[str] = []
    for part in _RELATION_SPLIT.split(lowered):
        cleaned = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", part)
        cleaned = cleaned.strip("的了和与及统计指标数据")
        if len(cleaned) < 2 or cleaned in {"gdp"}:
            continue
        if any(country in cleaned for country in _COUNTRIES):
            continue
        if cleaned not in terms:
            terms.append(cleaned)
    return tuple(terms[:2])


def _object_covered(candidate: AgentCandidate, object_name: str) -> bool:
    if object_name in (candidate.table_rows or ()):
        return True
    return object_name in _candidate_text(candidate)


def _candidate_bound(candidate: AgentCandidate, requirement: EvidenceRequirement) -> bool:
    """Return True when one candidate binds the object to the year and metric."""
    text = _candidate_text(candidate)
    object_hits = [obj for obj in requirement.objects if _object_covered(candidate, obj)]
    year_hits = [year for year in requirement.years if str(year) in text]
    metric_hits = [metric for metric in requirement.metrics if _metric_in_text(metric, text)]
    if requirement.relation_terms and not any(term in text for term in requirement.relation_terms):
        return False
    if object_hits:
        if requirement.years and not year_hits:
            return False
        return not (requirement.metrics and not metric_hits)
    if requirement.years and requirement.metrics:
        return bool(year_hits and metric_hits)
    return False


def assess_evidence(
    requirement: EvidenceRequirement,
    candidates: Sequence[AgentCandidate],
) -> EvidenceAssessment:
    """Check whether candidates bind every required item inside one evidence item."""
    if requirement.empty:
        return EvidenceAssessment(satisfied=True)
    if not candidates:
        return EvidenceAssessment(satisfied=False, missing=("没有任何候选证据",))
    image_indices = tuple(
        candidate.index
        for candidate in candidates
        if candidate.modality == "image" and candidate.tool_capable
    )
    table_indices = tuple(
        candidate.index
        for candidate in candidates
        if candidate.modality == "table" and candidate.tool_capable
    )
    object_coverage: dict[str, tuple[int, ...]] = {}
    missing: list[str] = []
    for object_name in requirement.objects:
        indices = tuple(
            candidate.index
            for candidate in candidates
            if _object_covered(candidate, object_name) and _candidate_bound(candidate, requirement)
        )
        object_coverage[object_name] = indices
        if not indices:
            bound_hint = ""
            if requirement.years or requirement.metrics:
                parts = []
                if requirement.years:
                    parts.append("年份 " + "/".join(str(year) for year in requirement.years))
                if requirement.metrics:
                    parts.append("指标 " + "/".join(requirement.metrics))
                bound_hint = "，且未能绑定" + "、".join(parts)
            missing.append(f"{object_name} 缺少对应证据{bound_hint}")
    relation_coverage: dict[str, tuple[int, ...]] = {}
    for term in requirement.relation_terms:
        indices = tuple(
            candidate.index
            for candidate in candidates
            if _normalize(term) in _candidate_text(candidate)
        )
        relation_coverage[term] = indices
        if not indices:
            missing.append(f"关系对象「{term}」缺少直接证据")
    bound_candidates = [
        candidate for candidate in candidates if _candidate_bound(candidate, requirement)
    ]
    year_coverage = tuple(
        year
        for year in requirement.years
        if any(str(year) in _candidate_text(candidate) for candidate in bound_candidates)
    )
    if requirement.years and any(
        str(year) not in " ".join(_candidate_text(candidate) for candidate in candidates)
        for year in requirement.years
    ):
        missing.append("时间信息缺失：" + "、".join(str(year) for year in requirement.years))
    metric_coverage = tuple(
        metric
        for metric in requirement.metrics
        if any(
            _metric_in_text(metric, _candidate_text(candidate)) for candidate in bound_candidates
        )
    )
    for metric in requirement.metrics:
        if metric not in metric_coverage:
            missing.append(f"指标 {metric} 缺少对应证据")
    if requirement.needs_image_feature and not image_indices:
        missing.append("缺少可分析的图片证据")
    image_driven = bool(
        requirement.needs_image_feature
        and image_indices
        and not requirement.objects
        and not requirement.needs_table
    )
    if image_driven or (requirement.scanned_table and image_indices):
        # A question that asks to read a visual feature (or a scanned table)
        # can be answered by the image itself; metric/year words do not need
        # to appear in the text around the image.
        missing = [
            item
            for item in missing
            if "指标" not in item and "年份" not in item and "图片" not in item
        ]
    if requirement.needs_image_feature and image_indices and not requirement.scanned_table:
        if requirement.relation_terms:
            if not any(
                any(
                    _normalize(term) in _candidate_text(candidate)
                    for term in requirement.relation_terms
                )
                for candidate in candidates
                if candidate.modality == "image"
            ):
                missing.append("缺少与关系对象直接相关的图片证据")
        elif requirement.objects and not any(
            _object_covered(candidate, object_name)
            for candidate in candidates
            if candidate.modality == "image"
            for object_name in requirement.objects
        ):
            missing.append("缺少与对象绑定的图片证据")
    if requirement.needs_table and not table_indices and requirement.comparison:
        missing.append("跨对象比较缺少结构化表格证据")
    if (
        requirement.comparison
        and len(requirement.objects) >= 2
        and sum(1 for indices in object_coverage.values() if indices) < 2
        and not any(len(indices) >= 2 for indices in object_coverage.values())
    ):
        missing.append("比较对象未在同一或互补证据中同时覆盖")
    conflicts = _unit_conflicts(candidates) if requirement.comparison else ()
    if conflicts:
        missing.extend(conflicts)
    return EvidenceAssessment(
        satisfied=not missing,
        missing=tuple(dict.fromkeys(missing)),
        object_coverage=object_coverage,
        relation_coverage=relation_coverage,
        year_coverage=year_coverage,
        metric_coverage=metric_coverage,
        image_indices=image_indices,
        table_indices=table_indices,
        conflicts=conflicts,
    )


def _metric_in_text(metric: str, text: str) -> bool:
    if metric.lower() in text:
        return True
    aliases = {
        "GDP": ("gdp", "国内生产总值"),
        "人均GDP": ("人均gdp", "人均国内生产总值"),
        "增长率": ("增长", "增速"),
        "碳排放": ("碳排放", "二氧化碳"),
    }
    return any(alias in text for alias in aliases.get(metric, ()))


_UNIT_PATTERN = re.compile(
    r"亿美元|万元|亿元|万美元|亿吨|万吨|万吨标准煤|摄氏度|℃|％|%|人|美元|吨|公里|千米"
)


def _unit_conflicts(candidates: Sequence[AgentCandidate]) -> tuple[str, ...]:
    units: set[str] = set()
    for candidate in candidates:
        if candidate.modality != "table":
            continue
        text = _candidate_text(candidate)
        units.update(_UNIT_PATTERN.findall(text))
    if len(units) >= 2:
        return ("候选表格使用了不同单位：" + "、".join(sorted(units)),)
    return ()


def build_supplementary_query(
    requirement: EvidenceRequirement,
    assessment: EvidenceAssessment,
) -> str:
    """Build one focused query for the single supplementary retrieval pass."""
    parts: list[str] = []
    for object_name in requirement.objects:
        if not assessment.object_coverage.get(object_name):
            parts.append(object_name)
    for year in requirement.years:
        if year not in assessment.year_coverage:
            parts.append(str(year))
    if requirement.range_years is not None:
        parts.extend(
            str(year)
            for year in requirement.range_years
            if str(year) not in _candidate_year_text(assessment)
        )
    for metric in requirement.metrics:
        if metric not in assessment.metric_coverage:
            parts.append(metric)
    for term in requirement.relation_terms:
        if not assessment.relation_coverage.get(term):
            parts.append(term)
    if requirement.needs_image_feature:
        parts.append("图表 趋势")
    if requirement.needs_table and not assessment.table_indices:
        parts.append("统计表")
    if not parts:
        parts.extend(requirement.objects[:2] or list(requirement.metrics[:1]))
    query = " ".join(dict.fromkeys(part for part in parts if part)).strip()
    if not query:
        query = requirement.query
    suffix = " 补充检索"
    if query == requirement.query:
        query = f"{query}{suffix}"
    return query[:160]


def _candidate_year_text(assessment: EvidenceAssessment) -> set[str]:
    return {str(year) for year in assessment.year_coverage}


__all__ = [
    "EvidenceAssessment",
    "EvidenceRequirement",
    "assess_evidence",
    "build_supplementary_query",
    "extract_requirement",
    "visual_feature_requested",
]
