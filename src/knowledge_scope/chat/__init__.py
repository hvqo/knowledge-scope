"""Persistent chat workspace for knowledge-base question answering."""

from .models import (
    CHAT_CONVERSATION_DEFAULT_TITLE,
    CHAT_CONVERSATION_TITLE_MAX_LENGTH,
    CHAT_MESSAGE_CITATIONS_MAX_ITEMS,
    CHAT_MESSAGE_CONTENT_MAX_LENGTH,
    CHAT_MESSAGE_ROLES,
    CHAT_MESSAGE_STATUSES,
    ChatConversation,
    ChatMessage,
)

__all__ = [
    "CHAT_CONVERSATION_DEFAULT_TITLE",
    "CHAT_CONVERSATION_TITLE_MAX_LENGTH",
    "CHAT_MESSAGE_CITATIONS_MAX_ITEMS",
    "CHAT_MESSAGE_CONTENT_MAX_LENGTH",
    "CHAT_MESSAGE_ROLES",
    "CHAT_MESSAGE_STATUSES",
    "ChatConversation",
    "ChatMessage",
]
