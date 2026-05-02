# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.messages import trim_messages

from aiq_agent.common import parse_data_sources


def trim_message_history(messages: list[BaseMessage], max_tokens: int) -> list[BaseMessage]:
    """Trim messages to a maximum number of tokens."""
    return trim_messages(
        messages=[m.model_dump() for m in messages],
        max_tokens=max_tokens,
        strategy="last",
        token_counter=len,
        start_on="human",
        include_system=True,
    )


def _normalize_enum_value(value: Any) -> str | None:
    """Extract string value from enum or return as-is if already a string."""
    if value is None:
        return None
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def _is_text_type(type_value: Any) -> bool:
    """Check if type value represents 'text', handling both strings and enums."""
    normalized = _normalize_enum_value(type_value)
    return normalized is not None and normalized.lower() == "text"


def _is_user_role(role_value: Any) -> bool:
    """Check if role value represents 'user', handling both strings and enums."""
    normalized = _normalize_enum_value(role_value)
    return normalized is not None and normalized.lower() == "user"


def _extract_text_from_message(message: Any) -> str | None:
    if message is None:
        return None
    if isinstance(message, str):
        return message
    if hasattr(message, "content"):
        content_value = getattr(message, "content")
        if isinstance(content_value, str):
            return content_value
        if isinstance(content_value, list):
            parts = []
            for item in content_value:
                if hasattr(item, "type") and _is_text_type(getattr(item, "type")):
                    text_value = getattr(item, "text", None)
                    if text_value:
                        parts.append(str(text_value))
                elif isinstance(item, dict) and _is_text_type(item.get("type")):
                    text_value = item.get("text")
                    if text_value:
                        parts.append(str(text_value))
            if parts:
                return "\n".join(parts).strip()
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and _is_text_type(item.get("type")):
                    text = item.get("text")
                    if text:
                        parts.append(str(text))
            if parts:
                return "\n".join(parts).strip()
        if isinstance(content, str):
            return content
        text_value = message.get("text")
        if isinstance(text_value, str):
            return text_value
    return None


def _parse_available_documents(raw: Any) -> list[dict[str, Any]] | None:
    """Parse uploaded document metadata from a client payload."""
    if not isinstance(raw, list):
        return None

    documents: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        file_name = item.get("file_name") or item.get("fileName")
        if not isinstance(file_name, str) or not file_name.strip():
            continue
        summary = item.get("summary")
        documents.append(
            {
                "file_name": file_name.strip(),
                "summary": summary if isinstance(summary, str) else None,
            }
        )
    return documents or None


def _extract_query_from_text(text: str) -> tuple[str, list[str] | None, str | None, list[dict[str, Any]] | None]:
    if not text:
        return ("", None, None, None)
    trimmed = text.strip()
    if trimmed.startswith("{") and trimmed.endswith("}"):
        try:
            payload = json.loads(trimmed)
        except json.JSONDecodeError:
            return (text, None, None, None)
        if isinstance(payload, dict):
            data_sources = parse_data_sources(payload.get("data_sources"))
            collection_name = payload.get("collection_name")
            if not isinstance(collection_name, str) or not collection_name.strip():
                collection_name = None
            else:
                collection_name = collection_name.strip()
            available_documents = _parse_available_documents(payload.get("available_documents"))
            query_text = payload.get("query") or payload.get("text")
            if isinstance(query_text, str) and query_text.strip():
                return (query_text.strip(), data_sources, collection_name, available_documents)
    return (text, None, None, None)


def _extract_query_context(payload: Any) -> tuple[str, list[str] | None, str | None, list[dict[str, Any]] | None]:
    """Extract query text, data sources, and document metadata from various payload formats.

    Returns:
        Tuple of (query_text, data_sources, collection_name, available_documents).
        - data_sources is None if not specified, meaning use all configured tools
        - data_sources is a list if explicitly specified (use only those)
    """
    if isinstance(payload, dict):
        content = payload.get("content", {}) if isinstance(payload.get("content"), dict) else {}
        data_sources = parse_data_sources(payload.get("data_sources")) or parse_data_sources(
            content.get("data_sources")
        )
        collection_name = payload.get("collection_name") or content.get("collection_name")
        if not isinstance(collection_name, str) or not collection_name.strip():
            collection_name = None
        else:
            collection_name = collection_name.strip()
        available_documents = _parse_available_documents(
            payload.get("available_documents") or content.get("available_documents")
        )
        messages = content.get("messages", [])
        query_text = None
        if isinstance(messages, list) and messages:
            for msg in reversed(messages):
                if isinstance(msg, dict) and _is_user_role(msg.get("role")):
                    query_text = _extract_text_from_message(msg)
                    if query_text:
                        break
            if not query_text:
                query_text = _extract_text_from_message(messages[-1])
        if not query_text:
            query_text = _extract_text_from_message(payload.get("message")) or _extract_text_from_message(
                payload.get("text")
            )
        if query_text:
            inline_query, inline_sources, inline_collection, inline_documents = _extract_query_from_text(query_text)
            query_text = inline_query
            data_sources = data_sources or inline_sources
            collection_name = collection_name or inline_collection
            available_documents = available_documents or inline_documents
        return (query_text or "", data_sources, collection_name, available_documents)

    messages = getattr(payload, "messages", None)
    if isinstance(messages, list):
        data_sources = parse_data_sources(getattr(payload, "data_sources", None))
        collection_name = getattr(payload, "collection_name", None)
        if not isinstance(collection_name, str) or not collection_name.strip():
            collection_name = None
        else:
            collection_name = collection_name.strip()
        available_documents = _parse_available_documents(getattr(payload, "available_documents", None))
        query_text = None
        for msg in reversed(messages):
            if _is_user_role(getattr(msg, "role", None)):
                query_text = _extract_text_from_message(msg)
                if query_text:
                    break
        if not query_text and messages:
            query_text = _extract_text_from_message(messages[-1])
        if query_text:
            inline_query, inline_sources, inline_collection, inline_documents = _extract_query_from_text(query_text)
            query_text = inline_query
            data_sources = data_sources or inline_sources
            collection_name = collection_name or inline_collection
            available_documents = available_documents or inline_documents
        return (query_text or "", data_sources, collection_name, available_documents)

    query_text = str(payload)
    inline_query, inline_sources, inline_collection, inline_documents = _extract_query_from_text(query_text)
    return (inline_query, inline_sources, inline_collection, inline_documents)


def _extract_query_and_sources(payload: Any) -> tuple[str, list[str] | None]:
    """Backward-compatible query/data-source extractor."""
    query_text, data_sources, _, _ = _extract_query_context(payload)
    return (query_text, data_sources)
