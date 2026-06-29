# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for dependency-free provider adapter helpers."""

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

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
    MemoryStoreDisposition,
    MemoryStoreRequest,
    MemoryStoreResult,
)
from nemo_relay.memory_providers._common import (
    RECORD_METADATA_KEY,
    IdempotencyLedger,
    await_vendor,
    content_search_text,
    prepare_record,
    reconstruct_record,
    record_matches,
    scope_tags,
    vendor_partition,
)

NOW = datetime(2026, 6, 29, 12, tzinfo=timezone.utc)


def store_request(*, content: str = "prefers solarized", key: str | None = "key-1") -> MemoryStoreRequest:
    return MemoryStoreRequest(
        context=MemoryRequestContext("store-1"),
        namespace=MemoryNamespace("tenant:a", "subject:b", "session", "agent"),
        content=MemoryContent.text_content(content),
        event_timestamp=NOW,
        provenance=MemoryProvenance("conversation", ("turn-1",)),
        metadata={"kind": "preference", "nested": {"safe": True}},
        idempotency_key=key,
    )


def test_partition_is_opaque_stable_and_delimiter_safe():
    first = MemoryNamespace("tenant:alpha", "subject:beta")
    confusable = MemoryNamespace("tenant", "alpha:subject:beta")

    assert vendor_partition(first) == vendor_partition(first)
    assert vendor_partition(first) != vendor_partition(confusable)
    assert "tenant:alpha" not in vendor_partition(first)
    assert "subject:beta" not in vendor_partition(first)
    assert scope_tags(MemoryNamespace("tenant", "subject", "session-secret", "agent-secret")) == scope_tags(
        MemoryNamespace("tenant", "subject", "session-secret", "agent-secret")
    )
    assert "secret" not in "".join(scope_tags(MemoryNamespace("tenant", "subject", "session-secret", "agent-secret")))


def test_record_envelope_round_trip_preserves_relay_fields_and_separates_vendor_data():
    request = store_request()
    record, reserved = prepare_record("fake", request, ingested_at=NOW + timedelta(seconds=1))

    rebuilt = reconstruct_record(
        "fake",
        "search-1",
        {**reserved, "user_attempt": "ignored"},
        provider_metadata={"vendor_id": "native-7", "score_source": "semantic"},
    )

    assert rebuilt.id == record.id
    assert rebuilt.namespace == request.namespace
    assert rebuilt.content == request.content
    assert rebuilt.event_timestamp == request.event_timestamp
    assert rebuilt.provenance == request.provenance
    assert rebuilt.metadata == request.metadata
    assert rebuilt.provider_metadata == {"vendor_id": "native-7", "score_source": "semantic"}
    assert RECORD_METADATA_KEY in reserved


def test_missing_or_cross_provider_envelope_is_typed_failure():
    with pytest.raises(MemoryProviderError) as missing:
        reconstruct_record("fake", "search", {})
    assert missing.value.error.code is MemoryErrorCode.INTERNAL

    _, reserved = prepare_record("other", store_request(), ingested_at=NOW)
    with pytest.raises(MemoryProviderError, match="another adapter"):
        reconstruct_record("fake", "search", reserved)


def test_content_projection_and_exact_local_filters():
    request = store_request()
    record, _ = prepare_record("fake", request, ingested_at=NOW + timedelta(seconds=1))
    search = MemorySearchRequest(
        context=MemoryRequestContext("search-1"),
        namespace=MemoryNamespace("tenant:a", "subject:b", "other-session", "agent"),
        query="solarized",
        scope=MemorySearchScope.AGENT,
        filter=MemoryFilter(
            metadata={"kind": "preference"},
            event_after=NOW - timedelta(seconds=1),
            ingested_before=NOW + timedelta(seconds=2),
        ),
        limit=5,
    )

    assert record_matches(record, search)
    assert not record_matches(
        record,
        MemorySearchRequest(
            context=search.context,
            namespace=search.namespace,
            query=search.query,
            scope=search.scope,
            filter=MemoryFilter(metadata={"kind": "fact"}),
            limit=search.limit,
        ),
    )
    assert content_search_text(MemoryContent.json_content({"b": 2, "a": 1})) == '{"a":1,"b":2}'
    assert content_search_text(MemoryContent.reference_content("opaque", "preview")) == "preview"


async def test_idempotency_ledger_replays_conflicts_and_does_not_serialize_other_keys():
    ledger = IdempotencyLedger(capacity=2)
    request = store_request()
    calls = 0

    async def mutation() -> MemoryStoreResult:
        nonlocal calls
        calls += 1
        record, _ = prepare_record("fake", request, ingested_at=NOW)
        return MemoryStoreResult(record, MemoryStoreDisposition.CREATED)

    created = await ledger.run(request, mutation)
    replayed = await ledger.run(request, mutation)

    assert created.disposition is MemoryStoreDisposition.CREATED
    assert replayed.disposition is MemoryStoreDisposition.EXISTING
    assert replayed.record == created.record
    assert calls == 1

    with pytest.raises(MemoryProviderError) as conflict:
        await ledger.run(store_request(content="different"), mutation)
    assert conflict.value.error.code is MemoryErrorCode.CONFLICT


async def test_await_vendor_preserves_cancellation_and_maps_deadlines_and_status():
    async def cancelled() -> None:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await await_vendor(cancelled(), provider="fake", operation_id="cancel", deadline=None)

    async def slow() -> None:
        await asyncio.sleep(1)

    with pytest.raises(MemoryProviderError) as timeout:
        await await_vendor(
            slow(),
            provider="fake",
            operation_id="timeout",
            deadline=datetime.now(timezone.utc) + timedelta(milliseconds=1),
        )
    assert timeout.value.error.code is MemoryErrorCode.DEADLINE_EXCEEDED
    assert timeout.value.error.retryable

    class ServiceFailure(Exception):
        status = 503

    async def failed() -> None:
        raise ServiceFailure("secret response body")

    with pytest.raises(MemoryProviderError) as unavailable:
        await await_vendor(failed(), provider="fake", operation_id="failed", deadline=None)
    assert unavailable.value.error.code is MemoryErrorCode.PROVIDER_UNAVAILABLE
    assert "secret" not in str(unavailable.value)
