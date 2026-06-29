# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicitly opt-in Hindsight service smoke test."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from nemo_relay.memory import (
    MemoryContent,
    MemoryNamespace,
    MemoryProvenance,
    MemoryRequestContext,
    MemorySearchRequest,
    MemoryStoreRequest,
)
from nemo_relay.memory_providers.hindsight import HindsightMemoryProvider


@pytest.mark.live_memory
async def test_hindsight_live_store_search_round_trip():
    base_url = os.getenv("NEMO_RELAY_HINDSIGHT_URL")
    if not base_url:
        pytest.skip("set NEMO_RELAY_HINDSIGHT_URL to run the Hindsight live test")
    pytest.importorskip("hindsight_client")
    run_id = uuid4().hex
    provider = HindsightMemoryProvider(
        base_url=base_url,
        api_key=os.getenv("NEMO_RELAY_HINDSIGHT_API_KEY"),
    )
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

    assert any(match.record.id == stored.record.id for match in searched.matches)
