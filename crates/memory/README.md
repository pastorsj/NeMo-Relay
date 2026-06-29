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

The network-free reference provider is useful for examples and deterministic
tests:

```rust,no_run
use nemo_relay_memory::memory::{MemoryNamespace, MemoryRequestContext, MemorySearchRequest};
use nemo_relay_memory::{InMemoryProvider, MemoryRuntime};

# async fn example() -> Result<(), nemo_relay_memory::memory::MemoryOperationError> {
let runtime = MemoryRuntime::new(InMemoryProvider::new());
let request = MemorySearchRequest::new(
    MemoryRequestContext::new("search-1")?,
    MemoryNamespace::new("tenant-demo", "subject-alex")?,
    "preferred editor theme",
)?;
let result = runtime.search(request).await?;
assert!(result.matches.is_empty());
# Ok(())
# }
```

Adapter test suites can call `run_provider_conformance` with an isolated
provider and a unique run ID. The returned named cases make failures usable in
local tests and CI without hiding provider errors behind harness panics.

This crate currently exposes direct provider behavior only. Automatic recall,
write-back, and plugin activation land in a later phase and are not enabled by
constructing `MemoryRuntime`.
