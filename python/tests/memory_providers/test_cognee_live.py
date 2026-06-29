# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicitly opt-in Cognee live lifecycle smoke test."""

import os
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from nemo_relay.memory import (
    MemoryContent,
    MemoryDeleteRequest,
    MemoryNamespace,
    MemoryProvenance,
    MemoryRequestContext,
    MemorySearchRequest,
    MemoryStoreRequest,
)
from nemo_relay.memory_providers.cognee import CogneeMemoryProvider


@pytest.mark.live_memory
async def test_cognee_live_remember_recall_forget_round_trip():
    if os.getenv("NEMO_RELAY_RUN_COGNEE_LIVE") != "1":
        pytest.skip("set NEMO_RELAY_RUN_COGNEE_LIVE=1 to run the Cognee live test")
    pytest.importorskip("cognee")
    run_id = uuid4().hex
    provider = CogneeMemoryProvider()
    namespace = MemoryNamespace(f"relay-live-{run_id}", "subject")
    stored = await provider.store(
        MemoryStoreRequest(
            context=MemoryRequestContext(f"{run_id}-store"),
            namespace=namespace,
            content=MemoryContent.text_content(f"relay live sentinel {run_id}"),
            event_timestamp=datetime.now(timezone.utc),
            provenance=MemoryProvenance("live_test"),
            idempotency_key=f"{run_id}-key",
        )
    )
    searched = await provider.search(
        MemorySearchRequest(
            context=MemoryRequestContext(f"{run_id}-search"),
            namespace=namespace,
            query=f"sentinel {run_id}",
            limit=5,
        )
    )
    deleted = await provider.delete(
        MemoryDeleteRequest(
            context=MemoryRequestContext(f"{run_id}-delete"),
            namespace=namespace,
            id=stored.record.id,
        )
    )

    assert any(match.record.id == stored.record.id for match in searched.matches)
    assert deleted.deleted
