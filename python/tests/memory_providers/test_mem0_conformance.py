# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Supported provider conformance for the deterministic Mem0 profile."""

from tests.memory_providers.fakes import FakeMem0Client

from nemo_relay.memory_providers.conformance import run_provider_conformance
from nemo_relay.memory_providers.mem0 import Mem0MemoryProvider


async def test_mem0_fake_passes_supported_provider_conformance():
    report = await run_provider_conformance(Mem0MemoryProvider(FakeMem0Client()), "mem0-fake")

    assert report.passed, [(case.name, case.message) for case in report.failures]
