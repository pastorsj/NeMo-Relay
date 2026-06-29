# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused Hindsight 0.8 adapter and Reflect tests."""

from datetime import datetime, timezone

import pytest
from tests.memory_providers.fakes import FakeHindsightClient

from nemo_relay.memory import (
    MemoryContent,
    MemoryErrorCode,
    MemoryMaintenanceAction,
    MemoryMaintenanceRequest,
    MemoryMaintenanceWindow,
    MemoryNamespace,
    MemoryProvenance,
    MemoryProviderError,
    MemoryRequestContext,
    MemorySearchRequest,
    MemorySearchScope,
    MemoryStoreRequest,
)
from nemo_relay.memory_providers._common import AGENT_TAG_PREFIX, vendor_partition
from nemo_relay.memory_providers.hindsight import HindsightMemoryProvider


def store_request(*, subject: str = "subject", session: str = "session-a") -> MemoryStoreRequest:
    return MemoryStoreRequest(
        context=MemoryRequestContext("store-1"),
        namespace=MemoryNamespace("tenant", subject, session, "agent-a"),
        content=MemoryContent.text_content("prefers solarized mode"),
        event_timestamp=datetime(2026, 6, 29, 12, tzinfo=timezone.utc),
        provenance=MemoryProvenance("conversation", ("turn-1",)),
        metadata={"kind": "preference"},
        idempotency_key="turn-1",
    )


async def test_store_waits_for_retain_and_recall_reconstructs_original_record():
    client = FakeHindsightClient()
    provider = HindsightMemoryProvider(client)
    stored = await provider.store(store_request())

    retain = client.retain_calls[0]
    assert retain["bank_id"] == vendor_partition(store_request().namespace)
    assert retain["retain_async"] is False
    assert retain["update_mode"] == "replace"
    assert AGENT_TAG_PREFIX in str(retain["tags"])
    assert client.items[0].content.startswith("extracted:")

    recalled = await provider.search(
        MemorySearchRequest(
            context=MemoryRequestContext("search-1"),
            namespace=MemoryNamespace("tenant", "subject", "session-b", "agent-a"),
            query="solarized mode",
            scope=MemorySearchScope.AGENT,
            limit=5,
        )
    )

    assert len(recalled.matches) == 1
    assert recalled.matches[0].record == stored.record
    assert recalled.matches[0].score > 0
    assert client.recall_calls[0]["tags_match"] == "all_strict"


async def test_bank_and_scope_tags_prevent_identity_broadening():
    client = FakeHindsightClient()
    provider = HindsightMemoryProvider(client)
    await provider.store(store_request(subject="subject-a"))

    other_subject = await provider.search(
        MemorySearchRequest(
            context=MemoryRequestContext("other-subject"),
            namespace=MemoryNamespace("tenant", "subject-b"),
            query="solarized",
            limit=5,
        )
    )
    other_agent = await provider.search(
        MemorySearchRequest(
            context=MemoryRequestContext("other-agent"),
            namespace=MemoryNamespace("tenant", "subject-a", agent_id="agent-b"),
            query="solarized",
            scope=MemorySearchScope.AGENT,
            limit=5,
        )
    )

    assert not other_subject.matches
    assert not other_agent.matches
    assert client.recall_calls[0]["tags"] is None
    assert client.recall_calls[0]["tags_match"] == "any"


async def test_multiple_extracted_facts_for_one_document_return_one_relay_record():
    client = FakeHindsightClient()
    provider = HindsightMemoryProvider(client)
    stored = await provider.store(store_request())
    duplicate = client.items[0]
    client.items.append(
        type(duplicate)(
            id="fact-duplicate",
            bank_id=duplicate.bank_id,
            content="another extracted solarized fact",
            timestamp=duplicate.timestamp,
            context=duplicate.context,
            document_id=duplicate.document_id,
            metadata=duplicate.metadata,
            tags=duplicate.tags,
        )
    )

    result = await provider.search(
        MemorySearchRequest(
            context=MemoryRequestContext("dedupe"),
            namespace=store_request().namespace,
            query="solarized",
            limit=5,
        )
    )

    assert [match.record.id for match in result.matches] == [stored.record.id]


async def test_reflect_commits_derived_record_with_mapped_evidence_parent():
    client = FakeHindsightClient()
    provider = HindsightMemoryProvider(client)
    source = await provider.store(store_request())
    maintenance = MemoryMaintenanceRequest(
        context=MemoryRequestContext("reflect-1"),
        namespace=store_request().namespace,
        action=MemoryMaintenanceAction.REFLECT,
        parameters={
            "budget": "low",
            "checkpoint_id": "checkpoint-1",
            "query": "What solarized preference is remembered?",
        },
    )

    result = await provider.maintain(maintenance)

    assert len(result.records) == 1
    derived = result.records[0]
    assert derived.content.text == "Reflection: What solarized preference is remembered?"
    assert derived.provenance.source == "maintenance"
    assert derived.provenance.parent_memory_ids == (source.record.id,)
    assert any(source_id.startswith("hindsight:fact-") for source_id in derived.provenance.source_ids)
    assert derived.metadata["relay_derived"] is True
    assert client.reflect_calls[0]["include_facts"] is True
    assert client.retain_calls[-1]["retain_async"] is False


async def test_consolidate_is_not_claimed_by_reflect_capability():
    provider = HindsightMemoryProvider(FakeHindsightClient())
    request = MemoryMaintenanceRequest(
        context=MemoryRequestContext("consolidate"),
        namespace=MemoryNamespace("tenant", "subject"),
        action=MemoryMaintenanceAction.CONSOLIDATE,
    )

    with pytest.raises(MemoryProviderError) as failure:
        await provider.maintain(request)

    assert failure.value.error.code is MemoryErrorCode.UNSUPPORTED


async def test_reflect_rejects_bounded_window_it_cannot_enforce():
    provider = HindsightMemoryProvider(FakeHindsightClient())
    request = MemoryMaintenanceRequest(
        context=MemoryRequestContext("bounded-window"),
        namespace=MemoryNamespace("tenant", "subject"),
        action=MemoryMaintenanceAction.REFLECT,
        window=MemoryMaintenanceWindow(
            checkpoint_id="checkpoint",
            query="Reflect",
            limit=5,
        ),
    )

    with pytest.raises(MemoryProviderError) as failure:
        await provider.maintain(request)

    assert failure.value.error.code is MemoryErrorCode.UNSUPPORTED


async def test_reflect_rejects_unknown_scope_parameter_as_typed_invalid_request():
    provider = HindsightMemoryProvider(FakeHindsightClient())
    request = MemoryMaintenanceRequest(
        context=MemoryRequestContext("invalid-scope"),
        namespace=MemoryNamespace("tenant", "subject"),
        action=MemoryMaintenanceAction.REFLECT,
        parameters={"query": "Reflect", "scope": "organization"},
    )

    with pytest.raises(MemoryProviderError) as failure:
        await provider.maintain(request)

    assert failure.value.error.code is MemoryErrorCode.INVALID_REQUEST


async def test_hindsight_vendor_failure_is_typed():
    client = FakeHindsightClient()
    client.next_error = 429
    provider = HindsightMemoryProvider(client)

    with pytest.raises(MemoryProviderError) as failure:
        await provider.store(store_request())

    assert failure.value.error.code is MemoryErrorCode.PROVIDER_UNAVAILABLE
    assert failure.value.error.retryable


def test_default_client_requires_base_url_before_importing_sdk():
    with pytest.raises(ValueError, match="base_url"):
        HindsightMemoryProvider()
