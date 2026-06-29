# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused Graphiti OSS temporal adapter tests."""

import builtins
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from types import ModuleType
from typing import cast

import pytest
from tests.memory_providers.fakes import FakeGraphitiClient

from nemo_relay import JsonObject
from nemo_relay.memory import (
    MemoryContent,
    MemoryErrorCode,
    MemoryFilter,
    MemoryNamespace,
    MemoryProvenance,
    MemoryProviderError,
    MemoryRequestContext,
    MemorySearchRequest,
    MemorySearchScope,
    MemoryStoreRequest,
)
from nemo_relay.memory_providers.graphiti import GraphitiMemoryProvider, graphiti_group

NOW = datetime(2026, 6, 29, 12, tzinfo=timezone.utc)


def request(*, subject: str = "subject", session: str = "session-a") -> MemoryStoreRequest:
    return MemoryStoreRequest(
        context=MemoryRequestContext("store-1"),
        namespace=MemoryNamespace("tenant", subject, session, "agent-a"),
        content=MemoryContent.text_content("prefers solarized mode"),
        event_timestamp=NOW,
        provenance=MemoryProvenance("conversation", ("turn-1",)),
        metadata={"kind": "preference"},
        idempotency_key="turn-1",
    )


def search_request(*, subject: str = "subject", kind: str = "preference") -> MemorySearchRequest:
    return MemorySearchRequest(
        context=MemoryRequestContext("search-1"),
        namespace=MemoryNamespace("tenant", subject, "session-b", "agent-a"),
        query="solarized mode",
        scope=MemorySearchScope.AGENT,
        filter=MemoryFilter(metadata={"kind": kind}),
        limit=5,
    )


async def test_store_keeps_envelope_out_of_extraction_and_search_preserves_graph_evidence():
    client = FakeGraphitiClient()
    provider = GraphitiMemoryProvider(client)

    stored = await provider.store(request())
    searched = await provider.search(search_request())

    call = client.add_calls[0]
    assert call["source_description"] == "NeMo Relay canonical memory episode"
    assert "nemo-relay-record" not in str(call["source_description"])
    assert "tenant" not in str(call["group_id"])
    assert call["group_id"] == graphiti_group(request().namespace)
    assert len(searched.matches) == 1
    assert searched.matches[0].record == stored.record
    assert searched.matches[0].score == 1.0
    metadata = searched.matches[0].record.provider_metadata
    assert metadata["score_semantics"] == "reciprocal_rank"
    graph_edges = metadata["graph_edges"]
    assert isinstance(graph_edges, list) and isinstance(graph_edges[0], dict)
    edge = cast(JsonObject, graph_edges[0])
    assert edge["episodes"] == [stored.record.id]
    assert edge["valid_at"] == "2026-06-29T12:00:00Z"
    assert edge["reference_time"] == "2026-06-29T12:00:00Z"
    assert edge["source_node_uuid"] != edge["target_node_uuid"]


async def test_search_exposes_current_temporal_snapshot_without_changing_source_record():
    client = FakeGraphitiClient()
    provider = GraphitiMemoryProvider(client)
    stored = await provider.store(request())
    client.replace_edge(stored.record.id, invalid_at=NOW + timedelta(hours=1), expired_at=NOW + timedelta(hours=1))

    searched = await provider.search(search_request())

    recalled = searched.matches[0].record
    assert recalled.id == stored.record.id
    assert recalled.content == stored.record.content
    assert recalled.provenance == stored.record.provenance
    graph_edges = recalled.provider_metadata["graph_edges"]
    assert isinstance(graph_edges, list) and isinstance(graph_edges[0], dict)
    edge = cast(JsonObject, graph_edges[0])
    assert edge["invalid_at"] == "2026-06-29T13:00:00Z"
    assert edge["expired_at"] == "2026-06-29T13:00:00Z"


async def test_subject_partition_filters_and_duplicate_fact_hits_do_not_broaden_results():
    client = FakeGraphitiClient()
    client.duplicate_hits = True
    provider = GraphitiMemoryProvider(client)
    stored = await provider.store(request(subject="subject-a"))

    found = await provider.search(search_request(subject="subject-a"))
    other = await provider.search(search_request(subject="subject-b"))
    rejected = await provider.search(search_request(subject="subject-a", kind="fact"))

    assert [match.record.id for match in found.matches] == [stored.record.id]
    assert not other.matches
    assert not rejected.matches


async def test_vendor_failures_are_typed_and_redacted():
    client = FakeGraphitiClient()
    client.next_error = 503
    provider = GraphitiMemoryProvider(client)

    with pytest.raises(MemoryProviderError) as failure:
        await provider.store(request())

    assert failure.value.error.code is MemoryErrorCode.PROVIDER_UNAVAILABLE
    assert failure.value.error.retryable
    assert "response body" not in str(failure.value)


def test_graphiti_client_and_connection_settings_are_mutually_exclusive():
    with pytest.raises(ValueError, match="either"):
        GraphitiMemoryProvider(FakeGraphitiClient(), uri="bolt://localhost")


def test_missing_graphiti_extra_has_actionable_error(monkeypatch: pytest.MonkeyPatch):
    original_import = builtins.__import__

    def blocked_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> ModuleType:
        if name == "graphiti_core":
            raise ImportError("blocked for test")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(ImportError, match=r"nemo-relay\[graphiti\]"):
        GraphitiMemoryProvider(uri="bolt://localhost")
