# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional Python adapters for provider-neutral memory backends.

Vendor SDKs are imported only when their concrete adapter constructs a default
client. Importing this package does not require any provider extra.
"""

from nemo_relay.memory_providers.cognee import CogneeAsyncClient, CogneeMemoryProvider
from nemo_relay.memory_providers.conformance import (
    ConformanceCase,
    ConformanceReport,
    run_provider_conformance,
)
from nemo_relay.memory_providers.graphiti import GraphitiAsyncClient, GraphitiMemoryProvider
from nemo_relay.memory_providers.hindsight import HindsightAsyncClient, HindsightMemoryProvider
from nemo_relay.memory_providers.mem0 import Mem0AsyncClient, Mem0MemoryProvider

__all__ = [
    "CogneeAsyncClient",
    "CogneeMemoryProvider",
    "ConformanceCase",
    "ConformanceReport",
    "GraphitiAsyncClient",
    "GraphitiMemoryProvider",
    "HindsightAsyncClient",
    "HindsightMemoryProvider",
    "Mem0AsyncClient",
    "Mem0MemoryProvider",
    "run_provider_conformance",
]
