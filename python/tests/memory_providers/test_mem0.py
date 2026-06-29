# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused Mem0 OSS adapter conversion tests."""

import builtins
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from types import ModuleType
from typing import cast

import pytest
from tests.memory_providers.fakes import FakeMem0Client

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
from nemo_relay.memory_providers._common import AGENT_TAG_PREFIX, vendor_partition
from nemo_relay.memory_providers.mem0 import Mem0MemoryProvider


def request(*, subject: str = "subject", session: str = "session-a") -> MemoryStoreRequest:
    return MemoryStoreRequest(
        context=MemoryRequestContext("store-1"),
        namespace=MemoryNamespace("tenant", subject, session, "agent-a"),
        content=MemoryContent.text_content("prefers solarized mode"),
        event_timestamp=datetime(2026, 6, 29, 12, tzinfo=timezone.utc),
        provenance=MemoryProvenance("conversation", ("turn-1",)),
        metadata={"kind": "preference", "nested": {"not": "flattened"}},
        idempotency_key="turn-1",
    )


async def test_store_uses_raw_add_and_search_reconstructs_record():
    client = FakeMem0Client()
    provider = Mem0MemoryProvider(client)
    stored = await provider.store(request())

    assert client.add_calls[0]["infer"] is False
    assert client.add_calls[0]["user_id"] == vendor_partition(request().namespace)
    assert "tenant" not in str(client.add_calls[0]["user_id"])

    result = await provider.search(
        MemorySearchRequest(
            context=MemoryRequestContext("search-1"),
            namespace=MemoryNamespace("tenant", "subject", "session-b", "agent-a"),
            query="solarized mode",
            scope=MemorySearchScope.AGENT,
            filter=MemoryFilter(metadata={"kind": "preference"}),
            limit=5,
        )
    )

    assert len(result.matches) == 1
    assert result.matches[0].record == stored.record
    assert result.matches[0].score > 0
    search_call = client.search_calls[0]
    filters = cast(dict[str, object], search_call["filters"])
    assert "user_id" in filters
    assert not any(key in search_call for key in ("user_id", "agent_id", "run_id"))
    assert AGENT_TAG_PREFIX in str(filters)


async def test_subject_partition_and_metadata_filter_do_not_broaden():
    client = FakeMem0Client()
    provider = Mem0MemoryProvider(client)
    await provider.store(request(subject="subject-a"))

    other = await provider.search(
        MemorySearchRequest(
            context=MemoryRequestContext("other"),
            namespace=MemoryNamespace("tenant", "subject-b"),
            query="solarized",
            limit=5,
        )
    )
    rejected = await provider.search(
        MemorySearchRequest(
            context=MemoryRequestContext("filter"),
            namespace=MemoryNamespace("tenant", "subject-a"),
            query="solarized",
            filter=MemoryFilter(metadata={"kind": "fact"}),
            limit=5,
        )
    )

    assert not other.matches
    assert not rejected.matches


async def test_vendor_failures_are_typed_and_redacted():
    client = FakeMem0Client()
    client.next_error = 503
    provider = Mem0MemoryProvider(client)

    with pytest.raises(MemoryProviderError) as failure:
        await provider.store(request())

    assert failure.value.error.code is MemoryErrorCode.PROVIDER_UNAVAILABLE
    assert failure.value.error.retryable
    assert "response body" not in str(failure.value)


def test_client_and_config_are_mutually_exclusive():
    with pytest.raises(ValueError, match="either"):
        Mem0MemoryProvider(FakeMem0Client(), config={})


def test_missing_mem0_extra_has_actionable_error(monkeypatch: pytest.MonkeyPatch):
    original_import = builtins.__import__

    def blocked_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> ModuleType:
        if name == "mem0":
            raise ImportError("blocked for test")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(ImportError, match=r"nemo-relay\[mem0\]"):
        Mem0MemoryProvider()
