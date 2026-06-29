# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused Cognee remember/recall/forget/improve adapter tests."""

import builtins
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from types import ModuleType

import pytest
from tests.memory_providers.fakes import FakeCogneeClient

from nemo_relay.memory import (
    MemoryContent,
    MemoryDeleteRequest,
    MemoryErrorCode,
    MemoryFilter,
    MemoryMaintenanceAction,
    MemoryMaintenanceRequest,
    MemoryMaintenanceWindow,
    MemoryNamespace,
    MemoryProvenance,
    MemoryProviderError,
    MemoryRequestContext,
    MemorySearchRequest,
    MemorySearchScope,
    MemoryStoreDisposition,
    MemoryStoreRequest,
)
from nemo_relay.memory_providers.cognee import CogneeDataItem, CogneeMemoryProvider, cognee_dataset

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


async def test_store_uses_permanent_remember_and_search_reconstructs_source_record():
    client = FakeCogneeClient()
    provider = CogneeMemoryProvider(client)

    stored = await provider.store(request())
    searched = await provider.search(search_request())

    call = client.remember_calls[0]
    assert call["run_in_background"] is False
    assert call["self_improvement"] is False
    assert call["dataset_name"] == cognee_dataset(request().namespace)
    assert "tenant" not in str(call["dataset_name"])
    item = call["item"]
    assert isinstance(item, CogneeDataItem)
    assert item.label == stored.record.id
    assert len(searched.matches) == 1
    assert searched.matches[0].record == stored.record
    assert searched.matches[0].score == 1.0
    assert searched.matches[0].record.provider_metadata["score_semantics"] == "reciprocal_rank"
    recall = client.recall_calls[0]
    assert recall["query_type"] == "CHUNKS"
    assert recall["auto_route"] is False
    assert recall["scope"] == "graph"


async def test_subject_filters_and_duplicate_chunks_do_not_broaden_results():
    client = FakeCogneeClient()
    client.duplicate_chunks = True
    provider = CogneeMemoryProvider(client)
    stored = await provider.store(request(subject="subject-a"))

    found = await provider.search(search_request(subject="subject-a"))
    other = await provider.search(search_request(subject="subject-b"))
    rejected = await provider.search(search_request(subject="subject-a", kind="fact"))

    assert [match.record.id for match in found.matches] == [stored.record.id]
    assert not other.matches
    assert not rejected.matches


async def test_forget_removes_record_and_invalidates_idempotent_replay():
    client = FakeCogneeClient()
    provider = CogneeMemoryProvider(client)
    store_request = request()
    stored = await provider.store(store_request)

    deleted = await provider.delete(
        MemoryDeleteRequest(MemoryRequestContext("delete-1"), store_request.namespace, stored.record.id)
    )
    absent = await provider.search(search_request())
    recreated = await provider.store(store_request)

    assert deleted.id == stored.record.id and deleted.deleted
    assert not absent.matches
    assert recreated.disposition is MemoryStoreDisposition.CREATED
    assert len(client.remember_calls) == 2


async def test_forget_is_scoped_to_the_requested_subject():
    client = FakeCogneeClient()
    provider = CogneeMemoryProvider(client)
    stored = await provider.store(request(subject="subject-a"))

    result = await provider.delete(
        MemoryDeleteRequest(
            MemoryRequestContext("delete-other"),
            MemoryNamespace("tenant", "subject-b"),
            stored.record.id,
        )
    )

    assert not result.deleted
    assert client.forget_calls == []


async def test_improve_maps_only_to_unbounded_consolidate():
    client = FakeCogneeClient()
    provider = CogneeMemoryProvider(client)
    namespace = request().namespace
    context = MemoryRequestContext("improve-1")

    result = await provider.maintain(
        MemoryMaintenanceRequest(
            context,
            namespace,
            MemoryMaintenanceAction.CONSOLIDATE,
            parameters={"checkpoint_id": "checkpoint-1", "query": "consolidate"},
        )
    )

    assert result.records == ()
    assert client.improve_calls == [{"dataset": cognee_dataset(namespace), "run_in_background": False}]

    with pytest.raises(MemoryProviderError) as wrong_action:
        await provider.maintain(MemoryMaintenanceRequest(context, namespace, MemoryMaintenanceAction.REFLECT))
    assert wrong_action.value.error.code is MemoryErrorCode.UNSUPPORTED

    with pytest.raises(MemoryProviderError) as bounded:
        await provider.maintain(
            MemoryMaintenanceRequest(
                context,
                namespace,
                MemoryMaintenanceAction.CONSOLIDATE,
                window=MemoryMaintenanceWindow("checkpoint", "query", 5),
            )
        )
    assert bounded.value.error.code is MemoryErrorCode.UNSUPPORTED


async def test_vendor_failures_and_incomplete_remember_are_typed():
    client = FakeCogneeClient()
    client.next_error = 503
    provider = CogneeMemoryProvider(client)

    with pytest.raises(MemoryProviderError) as unavailable:
        await provider.store(request())
    assert unavailable.value.error.code is MemoryErrorCode.PROVIDER_UNAVAILABLE
    assert "response body" not in str(unavailable.value)

    client.remember_status = "errored"
    with pytest.raises(MemoryProviderError, match="did not complete") as incomplete:
        await provider.store(request())
    assert incomplete.value.error.code is MemoryErrorCode.PROVIDER_UNAVAILABLE


def test_missing_cognee_extra_has_actionable_error(monkeypatch: pytest.MonkeyPatch):
    original_import = builtins.__import__

    def blocked_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> ModuleType:
        if name == "cognee":
            raise ImportError("blocked for test")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(ImportError, match=r"nemo-relay\[cognee\]"):
        CogneeMemoryProvider()
