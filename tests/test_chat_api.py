# Chinese question text intentionally keeps its full-width punctuation.
# ruff: noqa: RUF001

from __future__ import annotations

from uuid import uuid4

import pytest
from httpx import AsyncClient


async def create_knowledge_base(
    client: AsyncClient,
    *,
    name: str = "质量管理规范",
) -> dict[str, object]:
    response = await client.post(
        "/api/v1/knowledge-bases",
        json={"name": name, "description": None},
    )
    assert response.status_code == 201
    return response.json()


def _chunk_citation(marker: str = "C1", *, knowledge_base_id: str, document_id: str) -> dict:
    return {
        "marker": marker,
        "document_id": document_id,
        "knowledge_base_id": knowledge_base_id,
        "candidate_kind": "chunk",
        "chunk_id": "chunk-1",
        "evidence_id": None,
        "modality": None,
        "representation_ids": [],
        "asset_refs": [],
        "page_start": 1,
        "page_end": 2,
        "source_block_ids": ["block-1"],
        "section_path": ["第一章"],
        "section_title": "第一章",
        "snippet": "来源片段内容",
        "snippet_kind": "source",
        "final_rank": None,
        "final_reranker_score": None,
        "branch_provenance": [],
    }


async def _create_conversation(
    client: AsyncClient,
    *,
    knowledge_base_id: str | None = None,
    title: str | None = None,
) -> dict:
    payload: dict[str, object] = {}
    if knowledge_base_id is not None:
        payload["knowledge_base_id"] = knowledge_base_id
    if title is not None:
        payload["title"] = title
    response = await client.post("/api/v1/chat/conversations", json=payload)
    assert response.status_code == 201
    return response.json()


@pytest.mark.anyio
async def test_conversation_crud_and_message_history(client: AsyncClient) -> None:
    knowledge_base = await create_knowledge_base(client)
    knowledge_base_id = knowledge_base["id"]
    document_id = str(uuid4())

    created = await _create_conversation(client, knowledge_base_id=knowledge_base_id)

    assert created["title"] == "新对话"
    assert created["knowledge_base_id"] == knowledge_base_id
    assert created["message_count"] == 0

    listed = await client.get("/api/v1/chat/conversations")
    assert listed.status_code == 200
    assert listed.json()["total"] == 1
    assert listed.json()["items"][0]["id"] == created["id"]

    question = await client.post(
        f"/api/v1/chat/conversations/{created['id']}/messages",
        json={"role": "user", "content": "  产品有哪些质量要求？  "},
    )
    assert question.status_code == 201
    assert question.json()["position"] == 0
    assert question.json()["role"] == "user"

    answer = await client.post(
        f"/api/v1/chat/conversations/{created['id']}/messages",
        json={
            "role": "assistant",
            "content": "需要满足三点要求 [C1]",
            "citations": [
                _chunk_citation(knowledge_base_id=knowledge_base_id, document_id=document_id)
            ],
        },
    )
    assert answer.status_code == 201
    assert answer.json()["position"] == 1
    assert answer.json()["citations"][0]["marker"] == "C1"

    detail = await client.get(f"/api/v1/chat/conversations/{created['id']}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["conversation"]["title"] == "产品有哪些质量要求？"
    assert body["conversation"]["message_count"] == 2
    assert [message["role"] for message in body["messages"]] == ["user", "assistant"]
    assert [message["position"] for message in body["messages"]] == [0, 1]
    assert body["messages"][1]["citations"][0]["snippet"] == "来源片段内容"

    renamed = await client.patch(
        f"/api/v1/chat/conversations/{created['id']}",
        json={"title": "  质量管理  问答  "},
    )
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "质量管理 问答"
    assert renamed.json()["message_count"] == 2

    deleted = await client.delete(f"/api/v1/chat/conversations/{created['id']}")
    assert deleted.status_code == 204
    assert (await client.get(f"/api/v1/chat/conversations/{created['id']}")).status_code == 404
    assert (await client.get("/api/v1/chat/conversations")).json()["total"] == 0


@pytest.mark.anyio
async def test_conversation_list_orders_by_latest_activity(client: AsyncClient) -> None:
    first = await _create_conversation(client)
    second = await _create_conversation(client)

    response = await client.post(
        f"/api/v1/chat/conversations/{first['id']}/messages",
        json={"role": "user", "content": "最新提问"},
    )
    assert response.status_code == 201

    listed = (await client.get("/api/v1/chat/conversations")).json()

    assert listed["total"] == 2
    assert listed["items"][0]["id"] == first["id"]
    assert listed["items"][1]["id"] == second["id"]
    assert listed["items"][0]["message_count"] == 1


@pytest.mark.anyio
async def test_conversation_title_keeps_the_first_question(client: AsyncClient) -> None:
    conversation = await _create_conversation(client)
    long_question = "请说明" + "质量管理体系的适用范围" * 5

    await client.post(
        f"/api/v1/chat/conversations/{conversation['id']}/messages",
        json={"role": "user", "content": long_question},
    )
    await client.post(
        f"/api/v1/chat/conversations/{conversation['id']}/messages",
        json={"role": "user", "content": "第二个问题"},
    )

    detail = (await client.get(f"/api/v1/chat/conversations/{conversation['id']}")).json()

    assert detail["conversation"]["title"].startswith("请说明质量管理体系")
    assert detail["conversation"]["title"].endswith("…")
    assert detail["conversation"]["title"] != "第二个问题"


@pytest.mark.anyio
async def test_conversation_can_be_bound_and_rebound_to_a_knowledge_base(
    client: AsyncClient,
) -> None:
    first = await create_knowledge_base(client, name="第一库")
    second = await create_knowledge_base(client, name="第二库")
    conversation = await _create_conversation(client)

    bound = await client.patch(
        f"/api/v1/chat/conversations/{conversation['id']}",
        json={"knowledge_base_id": first["id"]},
    )
    assert bound.status_code == 200
    assert bound.json()["knowledge_base_id"] == first["id"]

    rebound = await client.patch(
        f"/api/v1/chat/conversations/{conversation['id']}",
        json={"knowledge_base_id": second["id"]},
    )
    assert rebound.json()["knowledge_base_id"] == second["id"]

    missing = await client.patch(
        f"/api/v1/chat/conversations/{conversation['id']}",
        json={"knowledge_base_id": str(uuid4())},
    )
    assert missing.status_code == 404


@pytest.mark.anyio
async def test_conversations_are_removed_with_their_knowledge_base(client: AsyncClient) -> None:
    knowledge_base = await create_knowledge_base(client)
    conversation = await _create_conversation(client, knowledge_base_id=knowledge_base["id"])

    deleted = await client.delete(f"/api/v1/knowledge-bases/{knowledge_base['id']}")

    assert deleted.status_code == 204
    assert (await client.get(f"/api/v1/chat/conversations/{conversation['id']}")).status_code == 404


@pytest.mark.anyio
async def test_conversation_requests_are_validated(client: AsyncClient) -> None:
    conversation = await _create_conversation(client)
    conversation_id = conversation["id"]

    blank_title = await client.post("/api/v1/chat/conversations", json={"title": "   "})
    assert blank_title.status_code == 422

    empty_patch = await client.patch(f"/api/v1/chat/conversations/{conversation_id}", json={})
    assert empty_patch.status_code == 422

    null_title = await client.patch(
        f"/api/v1/chat/conversations/{conversation_id}",
        json={"title": None},
    )
    assert null_title.status_code == 422

    unknown_knowledge_base = await client.post(
        "/api/v1/chat/conversations",
        json={"knowledge_base_id": str(uuid4())},
    )
    assert unknown_knowledge_base.status_code == 404

    blank_content = await client.post(
        f"/api/v1/chat/conversations/{conversation_id}/messages",
        json={"role": "assistant", "content": "   "},
    )
    assert blank_content.status_code == 422

    user_citations = await client.post(
        f"/api/v1/chat/conversations/{conversation_id}/messages",
        json={
            "role": "user",
            "content": "问题",
            "citations": [
                _chunk_citation(
                    knowledge_base_id=str(uuid4()),
                    document_id=str(uuid4()),
                )
            ],
        },
    )
    assert user_citations.status_code == 422

    duplicate_markers = await client.post(
        f"/api/v1/chat/conversations/{conversation_id}/messages",
        json={
            "role": "assistant",
            "content": "回答",
            "citations": [
                _chunk_citation(knowledge_base_id=str(uuid4()), document_id=str(uuid4())),
                _chunk_citation(knowledge_base_id=str(uuid4()), document_id=str(uuid4())),
            ],
        },
    )
    assert duplicate_markers.status_code == 422

    unknown_role = await client.post(
        f"/api/v1/chat/conversations/{conversation_id}/messages",
        json={"role": "system", "content": "内容"},
    )
    assert unknown_role.status_code == 422

    assert (await client.get(f"/api/v1/chat/conversations/{uuid4()}")).status_code == 404
    renamed_missing = await client.patch(
        f"/api/v1/chat/conversations/{uuid4()}",
        json={"title": "重命名"},
    )
    assert renamed_missing.status_code == 404
    assert (await client.delete(f"/api/v1/chat/conversations/{uuid4()}")).status_code == 404
    assert (
        await client.post(
            f"/api/v1/chat/conversations/{uuid4()}/messages",
            json={"role": "user", "content": "问题"},
        )
    ).status_code == 404


@pytest.mark.anyio
async def test_failed_answers_are_persisted_with_their_status(client: AsyncClient) -> None:
    conversation = await _create_conversation(client)

    failed = await client.post(
        f"/api/v1/chat/conversations/{conversation['id']}/messages",
        json={"role": "assistant", "content": "回答生成未完成，请稍后重试。", "status": "error"},
    )

    assert failed.status_code == 201
    assert failed.json()["status"] == "error"
    assert (await client.get(f"/api/v1/chat/conversations/{conversation['id']}")).json()[
        "messages"
    ][0]["status"] == "error"
