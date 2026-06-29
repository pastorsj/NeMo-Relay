# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic vendor-protocol fakes for memory adapter tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class FakeVendorError(Exception):
    def __init__(self, status: int) -> None:
        super().__init__("fake vendor response body")
        self.status = status


@dataclass
class FakeMem0Item:
    id: str
    memory: str
    user_id: str
    metadata: dict[str, Any]


class FakeMem0Client:
    """In-memory fake matching the Mem0 2.0.8 async shapes used by Relay."""

    def __init__(self) -> None:
        self.items: list[FakeMem0Item] = []
        self.add_calls: list[dict[str, object]] = []
        self.search_calls: list[dict[str, object]] = []
        self.next_error: int | None = None

    async def add(
        self,
        messages: list[dict[str, str]],
        *,
        user_id: str,
        metadata: dict[str, Any],
        infer: bool,
    ) -> object:
        self._maybe_fail()
        self.add_calls.append({"messages": messages, "user_id": user_id, "metadata": metadata, "infer": infer})
        item = FakeMem0Item(f"mem0-{len(self.items) + 1}", messages[0]["content"], user_id, dict(metadata))
        self.items.append(item)
        return {"results": [{"id": item.id, "memory": item.memory, "event": "ADD"}]}

    async def search(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any],
        threshold: float,
        explain: bool,
    ) -> object:
        self._maybe_fail()
        self.search_calls.append(
            {
                "query": query,
                "top_k": top_k,
                "filters": dict(filters),
                "threshold": threshold,
                "explain": explain,
            }
        )
        query_tokens = set(query.lower().split())
        results = []
        for item in self.items:
            if filters.get("user_id") != item.user_id:
                continue
            if any(key != "user_id" and item.metadata.get(key) != value for key, value in filters.items()):
                continue
            memory_tokens = set(item.memory.lower().split())
            score = len(query_tokens & memory_tokens) / max(1, len(query_tokens))
            if score < threshold or score == 0:
                continue
            results.append(
                {
                    "id": item.id,
                    "memory": item.memory,
                    "score": score,
                    "metadata": dict(item.metadata),
                    "score_details": {"semantic": score},
                }
            )
        results.sort(key=lambda item: (-float(item["score"]), str(item["id"])))
        return {"results": results[:top_k]}

    def _maybe_fail(self) -> None:
        if self.next_error is not None:
            status, self.next_error = self.next_error, None
            raise FakeVendorError(status)


@dataclass
class FakeHindsightItem:
    id: str
    bank_id: str
    content: str
    timestamp: object
    context: str | None
    document_id: str
    metadata: dict[str, str]
    tags: list[str]


class FakeHindsightClient:
    """In-memory fake matching Hindsight 0.8.3 shapes used by Relay."""

    def __init__(self) -> None:
        self.items: list[FakeHindsightItem] = []
        self.retain_calls: list[dict[str, object]] = []
        self.recall_calls: list[dict[str, object]] = []
        self.reflect_calls: list[dict[str, object]] = []
        self.next_error: int | None = None

    async def aretain(
        self,
        bank_id: str,
        content: str,
        *,
        timestamp: object,
        context: str | None,
        document_id: str | None,
        metadata: dict[str, str] | None,
        tags: list[str] | None,
        update_mode: str | None,
        retain_async: bool,
    ) -> object:
        self._maybe_fail()
        if document_id is None:
            raise ValueError("fake requires document_id")
        call = {
            "bank_id": bank_id,
            "content": content,
            "timestamp": timestamp,
            "context": context,
            "document_id": document_id,
            "metadata": metadata,
            "tags": tags,
            "update_mode": update_mode,
            "retain_async": retain_async,
        }
        self.retain_calls.append(call)
        self.items = [item for item in self.items if not (item.bank_id == bank_id and item.document_id == document_id)]
        self.items.append(
            FakeHindsightItem(
                id=f"fact-{len(self.items) + 1}",
                bank_id=bank_id,
                content=f"extracted: {content}",
                timestamp=timestamp,
                context=context,
                document_id=document_id,
                metadata=dict(metadata or {}),
                tags=list(tags or []),
            )
        )
        return {"success": True, "bank_id": bank_id, "items_count": 1, "async": retain_async}

    async def arecall(
        self,
        bank_id: str,
        query: str,
        *,
        max_tokens: int,
        budget: str,
        trace: bool,
        include_source_facts: bool,
        tags: list[str] | None,
        tags_match: str,
        prefer_observations: bool,
    ) -> object:
        self._maybe_fail()
        self.recall_calls.append(
            {
                "bank_id": bank_id,
                "query": query,
                "max_tokens": max_tokens,
                "budget": budget,
                "trace": trace,
                "include_source_facts": include_source_facts,
                "tags": tags,
                "tags_match": tags_match,
                "prefer_observations": prefer_observations,
            }
        )
        query_tokens = set(query.lower().split())
        results = []
        for item in self.items:
            if item.bank_id != bank_id:
                continue
            if tags and not all(tag in item.tags for tag in tags):
                continue
            content_tokens = set(item.content.lower().split())
            score = len(query_tokens & content_tokens) / max(1, len(query_tokens))
            if score == 0:
                continue
            results.append(
                {
                    "id": item.id,
                    "text": item.content,
                    "type": "world",
                    "document_id": item.document_id,
                    "metadata": dict(item.metadata),
                    "tags": list(item.tags),
                    "scores": {"final": score, "semantic": score},
                }
            )
        results.sort(key=lambda item: (-float(item["scores"]["final"]), str(item["id"])))
        return {"results": results}

    async def areflect(
        self,
        bank_id: str,
        query: str,
        *,
        budget: str,
        context: str | None,
        max_tokens: int | None,
        tags: list[str] | None,
        tags_match: str,
        include_facts: bool,
        include_tool_calls: bool,
        include_tool_call_output: bool,
    ) -> object:
        self._maybe_fail()
        self.reflect_calls.append(
            {
                "bank_id": bank_id,
                "query": query,
                "budget": budget,
                "context": context,
                "max_tokens": max_tokens,
                "tags": tags,
                "tags_match": tags_match,
                "include_facts": include_facts,
                "include_tool_calls": include_tool_calls,
                "include_tool_call_output": include_tool_call_output,
            }
        )
        memories = [
            {"id": item.id, "text": item.content, "type": "world"}
            for item in self.items
            if item.bank_id == bank_id and (not tags or all(tag in item.tags for tag in tags))
        ]
        return {
            "text": f"Reflection: {query}",
            "based_on": {"memories": memories if include_facts else None, "mental_models": [], "directives": []},
        }

    def _maybe_fail(self) -> None:
        if self.next_error is not None:
            status, self.next_error = self.next_error, None
            raise FakeVendorError(status)
