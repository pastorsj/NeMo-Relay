# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packaging boundary tests for optional memory vendor SDKs."""

from __future__ import annotations

import subprocess
import sys
from importlib.metadata import requires


def test_importing_adapter_package_does_not_import_vendor_sdks():
    code = """
import sys
import nemo_relay.memory_providers
assert 'mem0' not in sys.modules
assert 'hindsight_client' not in sys.modules
assert 'graphiti_core' not in sys.modules
assert 'cognee' not in sys.modules
"""

    subprocess.run([sys.executable, "-c", code], check=True)


def test_vendor_requirements_are_extra_gated():
    requirements = requires("nemo-relay") or []
    vendor_requirements = [
        item for item in requirements if item.startswith(("cognee", "graphiti-core", "hindsight-client", "mem0ai"))
    ]

    assert len(vendor_requirements) == 4
    assert all("extra ==" in item for item in vendor_requirements)
