"""Explicit request and response models for the persistent chat workspace."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from knowledge_scope.chat.models import (
    CHAT_CONVERSATION_TITLE_MAX_LENGTH,
    CHAT_MEMORY_CONTENT_MAX_LENGTH,
    CHAT_MESSAGE_CITATIONS_MAX_ITEMS,
    CHAT_MESSAGE_CONTENT_MAX_LENGTH,
    CHAT_PROJECT_DESCRIPTION_MAX_LENGTH,
    CHAT_PROJECT_TITLE_MAX_LENGTH,
)
from knowledge_scope.rag.schemas import RAGCitation

ChatMessageRole = Literal["user", "assistant"]
ChatMessageStatus = Literal["complete", "error"]


def normalize_chat_title(value: str) -> str:
    """Collapse whitespace so a stored title stays a single readable line."""

    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError("title must contain at least one non-whitespace character")
    return normalized


class ChatConversationCreate(BaseModel):
    """Validated input for creating one conversation."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, max_length=CHAT_CONVERSATION_TITLE_MAX_LENGTH)
    knowledge_base_id: UUID | None = None
    project_id: UUID | None = None

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str | None) -> str | None:
        return normalize_chat_title(value) if value is not None else None


class ChatConversationUpdate(BaseModel):
    """Validated partial input for renaming a conversation or rebinding it."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, max_length=CHAT_CONVERSATION_TITLE_MAX_LENGTH)
    knowledge_base_id: UUID | None = None
    project_id: UUID | None = None

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str | None) -> str | None:
        return normalize_chat_title(value) if value is not None else None

    @model_validator(mode="after")
    def validate_patch(self) -> Self:
        if not self.model_fields_set:
            raise ValueError("at least one field must be provided")
        if "title" in self.model_fields_set and self.title is None:
            raise ValueError("title cannot be null")
        return self


class ChatMessageCreate(BaseModel):
    """Validated input for appending one question or answer."""

    model_config = ConfigDict(extra="forbid")

    role: ChatMessageRole
    content: str = Field(min_length=1, max_length=CHAT_MESSAGE_CONTENT_MAX_LENGTH)
    status: ChatMessageStatus = "complete"
    citations: list[RAGCitation] = Field(
        default_factory=list,
        max_length=CHAT_MESSAGE_CITATIONS_MAX_ITEMS,
    )

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must contain non-whitespace characters")
        return value

    @model_validator(mode="after")
    def validate_role_payload(self) -> Self:
        if self.role == "user" and self.citations:
            raise ValueError("user messages cannot carry citations")
        if self.role == "user" and self.status != "complete":
            raise ValueError("user messages are always complete")
        markers = [item.marker for item in self.citations]
        if len(set(markers)) != len(markers):
            raise ValueError("citation markers must be unique")
        return self


class ChatMessageResponse(BaseModel):
    """One persisted message in reading order."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    role: ChatMessageRole
    status: ChatMessageStatus
    content: str
    citations: list[RAGCitation]
    position: int
    created_at: datetime


class ChatConversationResponse(BaseModel):
    """Public representation of one conversation."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    knowledge_base_id: UUID | None
    project_id: UUID | None
    message_count: int
    created_at: datetime
    updated_at: datetime


class ChatConversationListResponse(BaseModel):
    """Bounded list response for conversations."""

    items: list[ChatConversationResponse]
    total: int
    limit: int
    offset: int


class ChatConversationDetailResponse(BaseModel):
    """One conversation together with every persisted message."""

    conversation: ChatConversationResponse
    messages: list[ChatMessageResponse]


class ChatProjectCreate(BaseModel):
    """Validated input for creating one project."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=CHAT_PROJECT_TITLE_MAX_LENGTH)
    description: str | None = Field(
        default=None,
        max_length=CHAT_PROJECT_DESCRIPTION_MAX_LENGTH,
    )

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        return normalize_chat_title(value)

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class ChatProjectUpdate(BaseModel):
    """Validated partial input for renaming or redescribing a project."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, max_length=CHAT_PROJECT_TITLE_MAX_LENGTH)
    description: str | None = Field(
        default=None,
        max_length=CHAT_PROJECT_DESCRIPTION_MAX_LENGTH,
    )

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str | None) -> str | None:
        return normalize_chat_title(value) if value is not None else None

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @model_validator(mode="after")
    def validate_patch(self) -> Self:
        if not self.model_fields_set:
            raise ValueError("at least one field must be provided")
        if "title" in self.model_fields_set and self.title is None:
            raise ValueError("title cannot be null")
        return self


class ChatProjectResponse(BaseModel):
    """Public representation of one project."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    description: str | None
    conversation_count: int
    created_at: datetime
    updated_at: datetime


class ChatProjectListResponse(BaseModel):
    """Bounded list response for projects."""

    items: list[ChatProjectResponse]
    total: int
    limit: int
    offset: int


class ChatMemoryCreate(BaseModel):
    """Validated input for manually adding one memory."""

    model_config = ConfigDict(extra="forbid")

    knowledge_base_id: UUID
    content: str = Field(min_length=1, max_length=CHAT_MEMORY_CONTENT_MAX_LENGTH)

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("content must contain non-whitespace characters")
        return normalized


class ChatMemoryResponse(BaseModel):
    """Public representation of one long-term memory."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    knowledge_base_id: UUID
    content: str
    source_conversation_id: UUID | None
    created_at: datetime


class ChatMemoryListResponse(BaseModel):
    """Bounded list response for memories of one knowledge base."""

    items: list[ChatMemoryResponse]
    total: int
    limit: int
    offset: int


__all__ = [
    "ChatConversationCreate",
    "ChatConversationDetailResponse",
    "ChatConversationListResponse",
    "ChatConversationResponse",
    "ChatConversationUpdate",
    "ChatMemoryCreate",
    "ChatMemoryListResponse",
    "ChatMemoryResponse",
    "ChatMessageCreate",
    "ChatMessageResponse",
    "ChatMessageRole",
    "ChatMessageStatus",
    "ChatProjectCreate",
    "ChatProjectListResponse",
    "ChatProjectResponse",
    "ChatProjectUpdate",
    "normalize_chat_title",
]
