<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Relay Memory

`nemo-relay-memory` defines the provider-neutral asynchronous memory interface
for NeMo Relay. Providers implement required `search` and `store` operations,
advertise optional operations through capabilities, and use the canonical DTOs
from `nemo-relay-types`.

`MemoryRuntime` validates required requests and enforces their absolute
deadlines. It does not retry operations. Cancelling the runtime future drops the
provider future, so provider implementations must finish preparation before a
single cancellation-safe commit point.

The deterministic in-memory provider, reusable provider conformance harness,
automatic recall, and plugin activation are introduced in subsequent changes.
