# Chinese titles intentionally keep their full-width punctuation.
# ruff: noqa: RUF001

from __future__ import annotations

import json
from typing import Any

import pytest
from httpx import AsyncClient

from conftest import _test_client


class _FakeGateway:
    """Deterministic gateway returning a fixed extraction payload."""

    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.requests: list[dict[str, Any]] = []

    async def complete(self, request: object) -> object:
        self.requests.append(request)
        return _GenerationResult(self.payload)


class _GenerationResult:
    def __init__(self, text: str) -> None:
        self.text = text


async def _create_knowledge_base(client: AsyncClient, name: str = "质量管理规范") -> str:
    payload = {"name": name, "description": None}
    response = await client.post("/api/v1/knowledge-bases", json=payload)
    assert response.status_code == 201
    return str(response.json()["id"])


async def _create_conversation(
    client: AsyncClient,
    *,
    knowledge_base_id: str,
    question: str,
    answer: str,
) -> str:
    conversation = (
        await client.post(
            "/api/v1/chat/conversations",
            json={"knowledge_base_id": knowledge_base_id},
        )
    ).json()
    conversation_id = str(conversation["id"])
    for role, content in (("user", question), ("assistant", answer)):
        response = await client.post(
            f"/api/v1/chat/conversations/{conversation_id}/messages",
            json={"role": role, "content": content},
        )
        assert response.status_code == 201
    return conversation_id


@pytest.mark.anyio
async def test_project_groups_conversations_and_survives_project_deletion(
    client: AsyncClient,
) -> None:
    created = await client.post(
        "/api/v1/chat/projects",
        json={"title": "中考化学复习", "description": "围绕真题与实验"},
    )
    assert created.status_code == 201
    project = created.json()

    conversation = (
        await client.post(
            "/api/v1/chat/conversations",
            json={"title": "第一次提问"},
        )
    ).json()
    moved = await client.patch(
        f"/api/v1/chat/conversations/{conversation['id']}",
        json={"project_id": project["id"]},
    )
    assert moved.status_code == 200
    assert moved.json()["project_id"] == project["id"]

    projects = (await client.get("/api/v1/chat/projects")).json()
    assert projects["total"] == 1
    assert projects["items"][0]["conversation_count"] == 1

    listed = (await client.get("/api/v1/chat/conversations")).json()
    assert listed["items"][0]["project_id"] == project["id"]

    # Deleting the project keeps the conversation, just ungrouped.
    removed = await client.delete(f"/api/v1/chat/projects/{project['id']}")
    assert removed.status_code == 204
    still_there = (await client.get("/api/v1/chat/conversations")).json()
    assert still_there["total"] == 1
    assert still_there["items"][0]["project_id"] is None
    assert (await client.get("/api/v1/chat/projects")).json()["total"] == 0


@pytest.mark.anyio
async def test_project_can_be_renamed_and_validates_input(client: AsyncClient) -> None:
    project = (await client.post("/api/v1/chat/projects", json={"title": "临时名称"})).json()

    updated = await client.patch(
        f"/api/v1/chat/projects/{project['id']}",
        json={"title": "中考化学复习", "description": "沉淀与溶解"},
    )
    assert updated.status_code == 200
    assert updated.json()["title"] == "中考化学复习"
    assert updated.json()["description"] == "沉淀与溶解"

    empty_patch = await client.patch(f"/api/v1/chat/projects/{project['id']}", json={})
    assert empty_patch.status_code == 422
    blank_title = await client.patch(
        f"/api/v1/chat/projects/{project['id']}",
        json={"title": "   "},
    )
    assert blank_title.status_code == 422
    missing = await client.patch(
        "/api/v1/chat/conversations/00000000-0000-0000-0000-000000000000",
        json={"project_id": project["id"]},
    )
    assert missing.status_code == 404


@pytest.mark.anyio
async def test_memories_are_scoped_to_one_knowledge_base(client: AsyncClient) -> None:
    chemistry = await _create_knowledge_base(client, "初中化学")
    physics = await _create_knowledge_base(client, "初中物理")

    manual = await client.post(
        "/api/v1/chat/memories",
        json={"knowledge_base_id": chemistry, "content": "用户在准备中考化学复习"},
    )
    assert manual.status_code == 201
    assert manual.json()["knowledge_base_id"] == chemistry
    assert manual.json()["source_conversation_id"] is None

    chemistry_view = (
        await client.get("/api/v1/chat/memories", params={"knowledge_base_id": chemistry})
    ).json()
    physics_view = (
        await client.get("/api/v1/chat/memories", params={"knowledge_base_id": physics})
    ).json()
    assert chemistry_view["total"] == 1
    assert physics_view["total"] == 0

    forgotten = await client.delete(f"/api/v1/chat/memories/{manual.json()['id']}")
    assert forgotten.status_code == 204
    assert (
        await client.get("/api/v1/chat/memories", params={"knowledge_base_id": chemistry})
    ).json()["total"] == 0


@pytest.mark.anyio
async def test_extraction_distills_durable_facts_from_one_conversation(
    postgres_test_database: str,
    postgres_test_engine: Any,
    test_data_dir: Any,
) -> None:
    gateway = _FakeGateway(
        json.dumps(
            ["用户在准备中考化学复习", "用户在准备中考化学复习", "偏好要点式回答"],
            ensure_ascii=False,
        )
    )
    async with _test_client(
        postgres_test_database,
        postgres_test_engine,
        test_data_dir,
        llm_gateway=gateway,
    ) as client:
        knowledge_base_id = await _create_knowledge_base(client, "初中化学")
        conversation_id = await _create_conversation(
            client,
            knowledge_base_id=knowledge_base_id,
            question="我想复习化学，从哪里开始？",
            answer="先从物质的组成与结构入手 [C1]。",
        )
        before = (
            await client.get(
                "/api/v1/chat/memories",
                params={"knowledge_base_id": knowledge_base_id},
            )
        ).json()
        assert before["total"] == 0

        extracted = await client.post(
            "/api/v1/chat/memories/extract",
            params={"conversation_id": conversation_id},
        )
        assert extracted.status_code == 200
        memories = extracted.json()
        # The duplicate is dropped; the restatement of nothing new is kept only
        # once, and each memory points back at its conversation.
        assert [item["content"] for item in memories] == [
            "用户在准备中考化学复习",
            "偏好要点式回答",
        ]
        assert memories[0]["source_conversation_id"] == conversation_id

        stored = (
            await client.get(
                "/api/v1/chat/memories",
                params={"knowledge_base_id": knowledge_base_id},
            )
        ).json()
        assert stored["total"] == 2

        # A second pass finds nothing new worth remembering.
        again = await client.post(
            "/api/v1/chat/memories/extract",
            params={"conversation_id": conversation_id},
        )
        assert again.status_code == 200
        assert again.json() == []


@pytest.mark.anyio
async def test_extraction_needs_a_knowledge_base(client: AsyncClient) -> None:
    conversation = (
        await client.post("/api/v1/chat/conversations", json={"title": "无知识库"})
    ).json()

    response = await client.post(
        "/api/v1/chat/memories/extract",
        params={"conversation_id": conversation["id"]},
    )

    assert response.status_code == 200
    assert response.json() == []
