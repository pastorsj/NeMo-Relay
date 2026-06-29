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
