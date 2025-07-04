# Chinese questions intentionally keep their full-width punctuation.
# ruff: noqa: RUF001

from __future__ import annotations

import pytest

from knowledge_scope.rag.routing import (
    RAG_DIRECT_QUERY_MAX_CHARACTERS,
    classify_direct_intent,
    content_terms,
    direct_answer,
    is_simple_query,
    normalize_direct_query,
    resolve_route,
    strip_query_filler,
    term_coverage,
)


@pytest.mark.parametrize(
    ("query", "intent"),
    [
        ("你好", "greeting"),
        ("您好！", "greeting"),
        ("Hello", "greeting"),
        ("  hi  ", "greeting"),
        ("早上好", "greeting"),
        ("谢谢", "thanks"),
        ("多谢！", "thanks"),
        ("感谢你的帮助", None),
        ("再见", "farewell"),
        ("你是谁", "identity"),
        ("你是什么模型？", "identity"),
        ("你能做什么？", "capability"),
        ("你好，你能做什么？", "capability"),
        ("你好，请问你能做什么？", "capability"),
        ("请问你能做什么", "capability"),
        ("麻烦问一下你能做什么", "capability"),
        ("请问你是谁", "identity"),
        ("你能帮我做什么", "capability"),
        ("你有哪些功能", "capability"),
        ("怎么用", "capability"),
        ("What can you do?", "capability"),
    ],
)
def test_small_talk_is_classified(query: str, intent: str | None) -> None:
    assert classify_direct_intent(query) == intent


@pytest.mark.parametrize(
    "query",
    [
        "这份资料的核心结论是什么？",
        "文档里有哪些功能？",
        "你能总结一下课文吗",
        "课文里提到了哪些人物？",
        "你好，请介绍一下质量管理流程",
        "表格里的数据说明了什么？",
        "第 12 页讲了什么？",
        "谢谢，另外文档里提到的指标是什么？",
        "为什么质量成本会上升？",
        "请问这份资料讲了什么？",
        "麻烦问一下，课文里的人物有哪些？",
        "",
        "   ",
    ],
)
def test_content_questions_are_never_direct(query: str) -> None:
    assert classify_direct_intent(query) is None


def test_overlong_questions_are_treated_as_content() -> None:
    padded = f"你能做什么{'呢' * RAG_DIRECT_QUERY_MAX_CHARACTERS}"

    assert len(padded) > RAG_DIRECT_QUERY_MAX_CHARACTERS
    assert classify_direct_intent(padded) is None


def test_route_keeps_the_requested_retrieval_mode_for_content() -> None:
    assert resolve_route("你能做什么", "unified") == ("direct", "capability")
    assert resolve_route("这份资料讲了什么？", "unified") == ("unified", None)
    assert resolve_route("这份资料讲了什么？", "dense") == ("dense", None)


def test_direct_answers_describe_implemented_behaviour_only() -> None:
    capability = direct_answer("capability")

    assert "检索" in capability
    assert "来源" in capability
    assert "资料不足" in capability
    assert direct_answer("greeting").startswith("你好")
    assert direct_answer("thanks") == "不客气。需要继续查资料时随时问我。"
    assert "KnowledgeScope" in direct_answer("identity")


def test_normalization_strips_filler_and_punctuation() -> None:
    assert normalize_direct_query("  你好， 你能做什么？ ") == "你能做什么"
    assert normalize_direct_query("Hi!") == "hi"
    assert normalize_direct_query("。。。") == ""


def test_politely_phrased_single_fact_questions_stay_fast() -> None:
    assert is_simple_query("请问一下这份资料里提到的曝气池溶解氧标准是多少呢") is True
    assert strip_query_filler("请问一下这份资料里提到的曝气池溶解氧标准是多少呢") == (
        "这份资料里提到的曝气池溶解氧标准是多少"
    )
    assert is_simple_query("水土保持方案里的监测频次是多少？") is True


def test_long_questions_without_a_fact_word_keep_the_full_path() -> None:
    assert (
        is_simple_query("请把这份水保方案里关于边坡防护和排水沟设计的全部内容都介绍一下可以吗")
        is False
    )
    assert is_simple_query("这份资料里讲了哪些和施工组织有关的内容呢") is False


def test_content_terms_keep_subjects_and_drop_question_words() -> None:
    # The subject comes first, exactly as the question ordered it.
    assert content_terms("曝气池的溶解氧控制在多少") == ("曝气池", "溶解氧控制")
    assert content_terms("请问一下这份资料里提到的养鱼池增氧的具体措施是什么呢") == (
        "养鱼池增氧",
        "具体措施",
    )
    assert content_terms("请问这份资料说明了什么？") == ()


def test_term_coverage_separates_supported_from_missing_subjects() -> None:
    query = "曝气池的溶解氧控制在多少"

    assert term_coverage(query, "曝气池溶解氧控制范围为 2-4 mg/L") == 1.0
    assert term_coverage(query, "养鱼池增氧与气体溶解度实验") == 0.5
    assert term_coverage("这份资料说明了什么？", "任何内容") == 1.0
