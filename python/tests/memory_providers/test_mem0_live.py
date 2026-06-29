# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicitly opt-in Mem0 OSS live smoke test."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
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
from nemo_relay.memory_providers.mem0 import Mem0MemoryProvider


@pytest.mark.live_memory
async def test_mem0_live_store_search_round_trip():
    if os.getenv("NEMO_RELAY_RUN_MEM0_LIVE") != "1":
        pytest.skip("set NEMO_RELAY_RUN_MEM0_LIVE=1 to run the Mem0 live test")
    config_path = os.getenv("NEMO_RELAY_MEM0_LIVE_CONFIG")
    if not config_path:
        pytest.skip("set NEMO_RELAY_MEM0_LIVE_CONFIG to a Mem0 JSON config file")
    pytest.importorskip("mem0")
    config = json.loads(Path(config_path).read_text())
    run_id = uuid4().hex
    provider = Mem0MemoryProvider(config=config)
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
