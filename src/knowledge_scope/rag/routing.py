# Chinese question patterns and product copy intentionally use Chinese punctuation.
# ruff: noqa: RUF001, RUF003

"""Deterministic query routing for the retrieval-augmented question flow.

Not every question is a question about the corpus.  Greetings, thanks, and
questions about the assistant itself were previously pushed through the whole
retrieval pipeline, which cost seconds and produced answers grounded in
irrelevant passages.  This module decides those turns up front so they can be
answered directly, while every content question keeps the caller's retrieval
mode.
"""

from __future__ import annotations

import re
from typing import Final, Literal

from .schemas import RAGRequestedRetrievalMode

RAGRoute = Literal["direct", "dense", "unified"]
DirectIntent = Literal["greeting", "thanks", "farewell", "identity", "capability"]

# A direct turn must be short; anything longer is treated as a content question.
RAG_DIRECT_QUERY_MAX_CHARACTERS: Final = 30
# ``auto`` only downgrades to the vector-only path for genuinely small lookups.
RAG_SIMPLE_QUERY_MAX_CHARACTERS: Final = 24

# Cues that one retrieval branch is not enough: comparisons, multi-hop
# relations, synthesis, and aggregation needs.  Everything else stays a lookup.
_COMPLEXITY_MARKERS: Final[tuple[str, ...]] = (
    "对比",
    "比较",
    "区别",
    "差异",
    "异同",
    "关系",
    "联系",
    "为什么",
    "原因",
    "影响",
    "作用",
    "流程",
    "步骤",
    "过程",
    "总结",
    "概括",
    "梳理",
    "归纳",
    "汇总",
    "统计",
    "有哪些",
    "哪些",
    "分别",
    "各自",
    "以及",
    "同时",
    "全文",
    "整篇",
    "整个",
    "所有",
    "时间线",
    "演变",
    "发展",
    "优缺点",
    "评价",
    "列举",
    "表格",
    "图表",
    "图片",
    "数据",
    "跨文档",
    "compare",
    "difference",
    "relationship",
    "summarise",
    "summarize",
    "overview",
    "list",
    "steps",
    "process",
    "across",
)
_CLAUSE_SEPARATORS: Final = "，,;；、"
# Trailing particles and question words do not make a lookup lookups complex.
_TRAILING_PARTICLES: Final = "呢吗吧啊呀哦哈嘛"
_TRAILING_FILLER_PATTERN: Final = re.compile(
    r"(?:请问|谢谢|多谢|麻烦了|一下|一下下|的话)*[?？！!。．\s]*$"
)
# A single-fact question word: the user wants one value, not a synthesis.
_FACT_QUESTION_MARKERS: Final[tuple[str, ...]] = (
    "是多少",
    "多少",
    "多少个",
    "几种",
    "几个",
    "多久",
    "多长时间",
    "什么时候",
    "何时",
    "哪一年",
    "几年",
    "哪里",
    "在哪",
    "哪个部门",
    "谁负责",
    "谁",
    "是否",
    "能不能",
    "多大",
    "多高",
    "多长",
    "几米",
    "几度",
    "what is",
    "how many",
    "how long",
    "when",
    "where",
    "who",
    "which",
)
# Longer lookups are still routed to the fast path when they are a single
# factual question phrased politely, for example
# "请问一下这份资料里提到的曝气池溶解氧标准是多少呢".
RAG_FACT_QUERY_MAX_CHARACTERS: Final = 40
# Content words are the query terms a usable context should actually mention.
_COVERAGE_STOP_WORDS: Final[tuple[str, ...]] = (
    "为什么",
    "是什么",
    "什么是",
    "怎么样",
    "怎么办",
    "怎么",
    "怎样",
    "如何",
    "哪些",
    "哪个",
    "多少",
    "什么",
    "这份",
    "这份资料",
    "该资料",
    "资料中",
    "文档中",
    "里面",
    "中的",
    # Generic document references: every context mentions them, so they say
    # nothing about whether the retrieved material is the right material.
    "资料",
    "文档",
    "报告",
    "文件",
    "内容",
    "部分",
    "情况",
    "方面",
    "关于",
    "根据",
    "有关",
    "以及",
    "还有",
    "资料里",
    "文档里",
    "报告里",
    "文中",
    "提到",
    "说到",
    "该文档",
    "该报告",
    "请说明",
    "请介绍",
    "请",
    "说明",
    "介绍",
    "告诉",
    "一下",
    "里",
    "我们",
    "你们",
    "他们",
    "可以",
    "需要",
    "是否",
    "的话",
    "的",
    "了",
    "是",
    "在",
    "和",
    "与",
    "及",
    "或",
    "对",
    "中",
    "上",
    "下",
    "有",
    "呢",
    "吗",
    "吧",
    "啊",
    "呀",
)
_COVERAGE_PARTS_PATTERN: Final = re.compile(r"[^\w\u4e00-\u9fff]+")


# Filler that may precede the real question ("你好，请问你能做什么？").
_LEADING_FILLER_PATTERN: Final = re.compile(
    r"^(?:你好|您好|哈喽|哈罗|嗨|喂|在吗|在么|hello|hi|hey|"
    r"请问一下|请问|麻烦问一下|麻烦|劳驾|想问一下|想问|我想问)"
    r"[，,。.!！?？~～、\s]*"
)
_PUNCTUATION: Final = "，,。.!！?？~～、;；:："

_DIRECT_PATTERNS: Final[tuple[tuple[DirectIntent, re.Pattern[str]], ...]] = tuple(
    (intent, re.compile(pattern))
    for intent, pattern in (
        (
            "greeting",
            r"^(?:你好|您好|哈喽|哈罗|嗨|喂|在吗|在么|早上好|上午好|中午好|下午好|晚上好|"
            r"goodmorning|goodafternoon|goodevening|hello|hi|hey|yo)$",
        ),
        (
            "thanks",
            r"^(?:谢谢|谢谢你|多谢|感谢|非常感谢|太感谢了|辛苦了|thanks|thankyou|thx|"
            r"muchappreciated)$",
        ),
        ("farewell", r"^(?:再见|拜拜|先这样|下次再聊|bye|goodbye|seeyou)$"),
        (
            "identity",
            r"^(?:你是谁|您是谁|你是什么|你叫什么|你叫什么名字|你是干什么的|你是做什么的|"
            r"你是什么模型|你用的什么模型|你是什么ai|你背后是什么模型|"
            r"whoareyou|whatareyou)$",
        ),
        (
            "capability",
            r"^(?:你能做什么|您能做什么|你能干什么|你可以做什么|你可以干什么|你会做什么|"
            r"你会干什么|你能帮我做什么|你可以帮我做什么|你能帮我干什么|"
            r"你有什么功能|你有哪些功能|你有什么能力|你有哪些能力|"
            r"怎么用|怎么使用|如何使用|使用说明|怎么提问|如何提问|"
            r"有哪些功能|功能有哪些|能做什么|你会什么|"
            r"whatcanyoudo|howtouse|howdoesitwork|whatdoyoudo)$",
        ),
    )
)

# Words that mean the turn is about the material after all, so it must be
# retrieved even when the phrasing looks like a capability question.
_CONTENT_CUES: Final[tuple[str, ...]] = (
    "资料",
    "文档",
    "文件",
    "课文",
    "章节",
    "段落",
    "原文",
    "表格",
    "图片",
    "图表",
    "数据",
    "指标",
    "页",
)


def normalize_direct_query(query: str) -> str:
    """Lowercase, strip punctuation and filler, so matching stays predictable."""

    normalized = re.sub(r"\s+", "", query.strip().lower()).strip(_PUNCTUATION)
    # Greetings and politeness can stack ("你好，请问…"), and a bare "你好" must
    # survive as itself, so stripping stops as soon as nothing is left.
    current = normalized
    while True:
        stripped = _LEADING_FILLER_PATTERN.sub("", current).strip(_PUNCTUATION)
        if not stripped or stripped == current:
            return current
        current = stripped


def classify_direct_intent(query: str) -> DirectIntent | None:
    """Return the small-talk intent of one question, or ``None`` for content."""

    normalized = normalize_direct_query(query)
    if not normalized or len(normalized) > RAG_DIRECT_QUERY_MAX_CHARACTERS:
        return None
    intent = next(
        (candidate for candidate, pattern in _DIRECT_PATTERNS if pattern.match(normalized)),
        None,
    )
    if intent is None:
        return None
    if intent in {"capability", "identity"} and any(cue in normalized for cue in _CONTENT_CUES):
        return None
    return intent


def direct_answer(intent: DirectIntent) -> str:
    """Product-level answer for a turn that does not need the corpus."""

    if intent == "greeting":
        return (
            "你好，我是知识库资料问答助手。直接问我资料里的内容就行，"
            "例如某份文件的关键结论、流程或数据。"
        )
    if intent == "thanks":
        return "不客气。需要继续查资料时随时问我。"
    if intent == "farewell":
        return "好的，需要时再叫我。"
    if intent == "identity":
        return (
            "我是 KnowledgeScope 的资料问答助手，只依据当前知识库中的资料回答，"
            "并在回答里标注来源引用。"
        )
    return (
        "我是知识库资料问答助手，可以：按问题检索当前知识库并给出带来源的回答；"
        "梳理文档里的结论、流程和数据；对比不同资料的说法；在追问中延续上下文。"
        "回答只依据检索到的资料，资料不足时会直接说明，不会凭空补充。"
    )


def strip_query_filler(query: str) -> str:
    """Return the question without greetings, trailing particles and clause noise."""

    normalized = normalize_direct_query(query)
    stripped = normalized.strip(_TRAILING_PARTICLES)
    return _TRAILING_FILLER_PATTERN.sub("", stripped).strip(_TRAILING_PARTICLES) or stripped


def content_terms(query: str) -> tuple[str, ...]:
    """Return the query's content terms, longest first.

    Stop words and question words are dropped, and what remains is split at the
    particles that normally join a subject to its predicate, so
    "曝气池的溶解氧控制在多少" yields ``曝气池`` and ``溶解氧控制``.
    """

    stripped = strip_query_filler(query)
    for stop_word in _COVERAGE_STOP_WORDS:
        stripped = stripped.replace(stop_word, " ")
    terms: list[str] = []
    for part in _COVERAGE_PARTS_PATTERN.split(stripped):
        if len(part) >= 2 or (part.isascii() and len(part) >= 2):
            terms.append(part)
    # Question order matters: the first term is the subject the context must
    # mention, so the tuple keeps the order the question used.
    return tuple(dict.fromkeys(terms))


def _is_covered(term: str, haystack: str) -> bool:
    """Return whether one content term appears in the text.

    The first two characters are accepted for longer terms, so compound words
    and light paraphrasing ("养鱼池增氧" against "增加养鱼池水中的含氧量") do
    not look like missing material.
    """

    return term in haystack or (len(term) >= 3 and term[:2] in haystack)


def term_coverage(query: str, text: str) -> float:
    """Return how much of the question the retrieved text actually covers.

    A question without content terms is always covered.
    """

    terms = content_terms(query)
    if not terms:
        return 1.0
    haystack = " ".join(text.split()).lower()
    return sum(1 for term in terms if _is_covered(term, haystack)) / len(terms)


def covers_subject(query: str, text: str) -> bool:
    """Return whether the text mentions the question's leading content term.

    In a question like "曝气池的溶解氧控制在多少" the subject comes first and
    carries the material being asked about, so a context that never mentions it
    is about something else, however well it matches the rest of the wording.
    """

    terms = content_terms(query)
    if not terms:
        return True
    return _is_covered(terms[0], " ".join(text.split()).lower())


def is_simple_query(query: str) -> bool:
    """Return whether the question is a single lookup the vector path can serve.

    The rule is deliberately narrow: short, single-clause questions without
    comparison, synthesis, aggregation, or multi-document cues.  Politely
    phrased single-fact questions are accepted up to a longer bound, because a
    user asking for one value does not need every retrieval branch.
    """

    normalized = strip_query_filler(query)
    if not normalized:
        return False
    if any(separator in normalized for separator in _CLAUSE_SEPARATORS):
        return False
    if any(marker in normalized for marker in _COMPLEXITY_MARKERS):
        return False
    if len(normalized) <= RAG_SIMPLE_QUERY_MAX_CHARACTERS:
        return True
    if len(normalized) > RAG_FACT_QUERY_MAX_CHARACTERS:
        return False
    return any(marker in normalized for marker in _FACT_QUESTION_MARKERS)


def resolve_route(
    query: str,
    requested_mode: RAGRequestedRetrievalMode,
) -> tuple[RAGRoute, DirectIntent | None]:
    """Resolve the route for one question.

    ``auto`` keeps small lookups on the fast vector path and sends everything
    else, including every follow-up with context needs, to unified retrieval.
    """

    intent = classify_direct_intent(query)
    if intent is not None:
        return "direct", intent
    if requested_mode == "auto":
        return ("dense" if is_simple_query(query) else "unified"), None
    return requested_mode, None


__all__ = [
    "RAG_DIRECT_QUERY_MAX_CHARACTERS",
    "RAG_FACT_QUERY_MAX_CHARACTERS",
    "RAG_SIMPLE_QUERY_MAX_CHARACTERS",
    "DirectIntent",
    "RAGRoute",
    "classify_direct_intent",
    "content_terms",
    "covers_subject",
    "direct_answer",
    "is_simple_query",
    "normalize_direct_query",
    "resolve_route",
    "strip_query_filler",
    "term_coverage",
]
