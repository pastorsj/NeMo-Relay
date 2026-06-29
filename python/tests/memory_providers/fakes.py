# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic vendor-protocol fakes for memory adapter tests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from nemo_relay.memory_providers.cognee import (
    CogneeDataItem,
    CogneeDataSnapshot,
    CogneeRecallHit,
    CogneeRememberSnapshot,
)
from nemo_relay.memory_providers.graphiti import (
    GraphitiEdgeSnapshot,
    GraphitiEpisodeEvidence,
    GraphitiEpisodeSnapshot,
)


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
    """In-memory fake matching the Mem0 2.0.10 async shapes used by Relay."""

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


class FakeGraphitiClient:
    """Deterministic temporal-graph fake matching Relay's thin wrapper."""

    def __init__(self) -> None:
        self.items: dict[tuple[str, str], GraphitiEpisodeEvidence] = {}
        self.add_calls: list[dict[str, object]] = []
        self.search_calls: list[dict[str, object]] = []
        self.next_error: int | None = None
        self.duplicate_hits = False

    async def add_episode(
        self,
        *,
        name: str,
        body: str,
        source_description: str,
        reference_time: datetime,
        group_id: str,
        uuid: str,
    ) -> GraphitiEpisodeEvidence:
        self._maybe_fail()
        self.add_calls.append(
            {
                "name": name,
                "body": body,
                "source_description": source_description,
                "reference_time": reference_time,
                "group_id": group_id,
                "uuid": uuid,
            }
        )
        created_at = datetime.now(timezone.utc)
        edge = GraphitiEdgeSnapshot(
            uuid=f"edge-{uuid}",
            name="RELATES_TO",
            fact=f"Extracted fact: {body}",
            episodes=(uuid,),
            source_node_uuid=f"source-{uuid}",
            target_node_uuid=f"target-{uuid}",
            created_at=created_at,
            valid_at=reference_time,
            reference_time=reference_time,
            attributes={"extractor": "fake-graphiti"},
        )
        episode = GraphitiEpisodeSnapshot(
            uuid=uuid,
            name=name,
            group_id=group_id,
            content=body,
            created_at=created_at,
            valid_at=reference_time,
            entity_edge_ids=(edge.uuid,),
        )
        evidence = GraphitiEpisodeEvidence(episode, (edge,))
        self.items[(group_id, uuid)] = evidence
        return evidence

    async def search_episodes(
        self,
        query: str,
        *,
        group_id: str,
        num_results: int,
    ) -> tuple[GraphitiEpisodeEvidence, ...]:
        self._maybe_fail()
        self.search_calls.append({"query": query, "group_id": group_id, "num_results": num_results})
        query_tokens = set(query.lower().split())
        ranked: list[tuple[float, GraphitiEpisodeEvidence]] = []
        for (item_group, _), evidence in self.items.items():
            if item_group != group_id:
                continue
            fact_tokens = set(" ".join(edge.fact for edge in evidence.edges).lower().split())
            score = len(query_tokens & fact_tokens) / max(1, len(query_tokens))
            if score > 0:
                ranked.append((score, evidence))
        ranked.sort(key=lambda item: (-item[0], item[1].episode.uuid))
        output = [replace(evidence, fact_rank=rank) for rank, (_, evidence) in enumerate(ranked, 1)]
        if self.duplicate_hits and output:
            output.insert(1, output[0])
        return tuple(output[:num_results])

    def replace_edge(self, episode_id: str, **changes: object) -> None:
        for key, evidence in self.items.items():
            if evidence.episode.uuid == episode_id:
                edge = replace(evidence.edges[0], **changes)
                self.items[key] = replace(evidence, edges=(edge,))
                return
        raise KeyError(episode_id)

    def _maybe_fail(self) -> None:
        if self.next_error is not None:
            status, self.next_error = self.next_error, None
            raise FakeVendorError(status)


class FakeCogneeClient:
    """Deterministic fake for Cognee's memory-oriented API."""

    def __init__(self) -> None:
        self.items: dict[tuple[str, UUID], CogneeDataItem] = {}
        self.remember_calls: list[dict[str, object]] = []
        self.recall_calls: list[dict[str, object]] = []
        self.list_data_calls: list[dict[str, object]] = []
        self.forget_calls: list[dict[str, object]] = []
        self.improve_calls: list[dict[str, object]] = []
        self.next_error: int | None = None
        self.duplicate_chunks = False
        self.remember_status = "completed"

    async def remember(
        self,
        item: CogneeDataItem,
        *,
        dataset_name: str,
        run_in_background: bool,
        self_improvement: bool,
    ) -> CogneeRememberSnapshot:
        self._maybe_fail()
        self.remember_calls.append(
            {
                "item": item,
                "dataset_name": dataset_name,
                "run_in_background": run_in_background,
                "self_improvement": self_improvement,
            }
        )
        if self.remember_status == "completed":
            self.items[(dataset_name, item.data_id)] = item
        return CogneeRememberSnapshot(self.remember_status, item.data_id)

    async def recall(
        self,
        query_text: str,
        *,
        query_type: str,
        datasets: list[str],
        top_k: int,
        auto_route: bool,
        scope: str,
        include_references: bool,
    ) -> tuple[CogneeRecallHit, ...]:
        self._maybe_fail()
        self.recall_calls.append(
            {
                "query_text": query_text,
                "query_type": query_type,
                "datasets": list(datasets),
                "top_k": top_k,
                "auto_route": auto_route,
                "scope": scope,
                "include_references": include_references,
            }
        )
        query_tokens = set(query_text.lower().split())
        ranked: list[tuple[float, CogneeDataItem]] = []
        for (dataset_name, _), item in self.items.items():
            if dataset_name not in datasets:
                continue
            item_tokens = set(item.data.lower().split())
            score = len(query_tokens & item_tokens) / max(1, len(query_tokens))
            if score > 0:
                ranked.append((score, item))
        ranked.sort(key=lambda pair: (-pair[0], str(pair[1].data_id)))
        output: list[CogneeRecallHit] = []
        for score, item in ranked:
            output.append(CogneeRecallHit(item.data_id, item.data, f"chunk-{item.data_id}-0", score))
            if self.duplicate_chunks:
                output.append(CogneeRecallHit(item.data_id, item.data, f"chunk-{item.data_id}-1", score / 2))
        return tuple(output[:top_k])

    async def list_data(
        self,
        dataset_name: str,
        data_ids: Sequence[UUID],
    ) -> tuple[CogneeDataSnapshot, ...]:
        self._maybe_fail()
        self.list_data_calls.append({"dataset_name": dataset_name, "data_ids": tuple(data_ids)})
        return tuple(
            CogneeDataSnapshot(data_id, item.external_metadata)
            for (item_dataset, data_id), item in self.items.items()
            if item_dataset == dataset_name and data_id in data_ids
        )

    async def forget(self, *, data_id: UUID, dataset: str) -> bool:
        self._maybe_fail()
        self.forget_calls.append({"data_id": data_id, "dataset": dataset})
        return self.items.pop((dataset, data_id), None) is not None

    async def improve(self, dataset: str, *, run_in_background: bool) -> object:
        self._maybe_fail()
        self.improve_calls.append({"dataset": dataset, "run_in_background": run_in_background})
        return {"status": "completed"}

    def _maybe_fail(self) -> None:
        if self.next_error is not None:
            status, self.next_error = self.next_error, None
            raise FakeVendorError(status)
