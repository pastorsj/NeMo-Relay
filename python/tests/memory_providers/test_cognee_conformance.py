# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Supported-conformance coverage for the Cognee adapter."""

from tests.memory_providers.fakes import FakeCogneeClient

from nemo_relay.memory import MemoryMaintenanceAction
from nemo_relay.memory_providers.cognee import CogneeMemoryProvider
from nemo_relay.memory_providers.conformance import run_provider_conformance


async def test_cognee_fake_passes_supported_conformance():
    report = await run_provider_conformance(
        CogneeMemoryProvider(FakeCogneeClient()),
        "cognee-fake",
        maintenance_action=MemoryMaintenanceAction.CONSOLIDATE,
    )

    assert report.passed, [(case.name, case.message) for case in report.failures]
