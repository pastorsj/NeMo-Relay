# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicitly opt-in Graphiti OSS live smoke test."""

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
from nemo_relay.memory_providers.graphiti import GraphitiMemoryProvider


@pytest.mark.live_memory
async def test_graphiti_live_store_search_round_trip():
    if os.getenv("NEMO_RELAY_RUN_GRAPHITI_LIVE") != "1":
        pytest.skip("set NEMO_RELAY_RUN_GRAPHITI_LIVE=1 to run the Graphiti live test")
    uri = os.getenv("NEMO_RELAY_GRAPHITI_URI")
    if not uri:
        pytest.skip("set NEMO_RELAY_GRAPHITI_URI and model credentials for the Graphiti live test")
    pytest.importorskip("graphiti_core")
    run_id = uuid4().hex
    provider = GraphitiMemoryProvider(
        uri=uri,
        user=os.getenv("NEMO_RELAY_GRAPHITI_USER"),
        password=os.getenv("NEMO_RELAY_GRAPHITI_PASSWORD"),
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
