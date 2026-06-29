# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Supported provider conformance for the deterministic Hindsight profile."""

from tests.memory_providers.fakes import FakeHindsightClient

from nemo_relay.memory_providers.conformance import run_provider_conformance
from nemo_relay.memory_providers.hindsight import HindsightMemoryProvider


async def test_hindsight_fake_passes_supported_provider_conformance():
    report = await run_provider_conformance(HindsightMemoryProvider(FakeHindsightClient()), "hindsight-fake")

    assert report.passed, [(case.name, case.message) for case in report.failures]
