# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Supported-conformance coverage for the Graphiti OSS adapter."""

from tests.memory_providers.fakes import FakeGraphitiClient

from nemo_relay.memory_providers.conformance import run_provider_conformance
from nemo_relay.memory_providers.graphiti import GraphitiMemoryProvider


async def test_graphiti_fake_passes_supported_conformance():
    report = await run_provider_conformance(GraphitiMemoryProvider(FakeGraphitiClient()), "graphiti-fake")

    assert report.passed, [(case.name, case.message) for case in report.failures]
