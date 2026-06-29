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

## Automatic reference mode

Enable the `relay` feature to install automatic memory around ordinary
non-streaming managed LLM calls. The component uses the same provider contract;
the built-in plugin and primary-language wrappers initially construct only the
dependency-free `InMemoryProvider`.

```bash
cargo add nemo-relay-memory --features relay
```

```rust,ignore
use std::sync::Arc;

use nemo_relay::api::llm::{LlmCallExecuteParams, LlmRequest, llm_call_execute};
use nemo_relay::api::runtime::LlmExecutionNextFn;
use nemo_relay::codec::openai_chat::OpenAIChatCodec;
use nemo_relay_memory::memory::MemoryNamespace;
use nemo_relay_memory::{AutomaticMemoryConfig, InMemoryProvider, MemoryComponent};
use serde_json::json;

# async fn automatic() -> Result<(), Box<dyn std::error::Error>> {
let component = MemoryComponent::new(
    InMemoryProvider::new(),
    AutomaticMemoryConfig {
        namespace: Some(MemoryNamespace::new("tenant-demo", "subject-alex")?),
        ..AutomaticMemoryConfig::default()
    },
)?;
let _installation = component.install_global("memory", 0)?;
let callback: LlmExecutionNextFn = Arc::new(|request| {
    Box::pin(async move {
        // `request` already contains any selected memory block.
        Ok(json!({
            "id": "response-1",
            "model": "demo-model",
            "choices": [{"message": {"role": "assistant", "content": "hello"}}]
        }))
    })
});

llm_call_execute(
    LlmCallExecuteParams::builder()
        .name("demo-provider")
        .request(LlmRequest {
            headers: serde_json::Map::new(),
            content: json!({
                "model": "demo-model",
                "messages": [{"role": "user", "content": "What do I prefer?"}]
            }),
        })
        .func(callback)
        .codec(Arc::new(OpenAIChatCodec))
        .response_codec(Arc::new(OpenAIChatCodec))
        .build(),
)
.await?;
# Ok(())
# }
```

Automatic mode resolves `metadata.memory.namespace` from the call, then the
nearest visible scope, then the component's static fallback. Tenant and subject
are required. `metadata.memory.enabled=false` opts a call or visible scope out.
The component searches with a deadline, deterministically deduplicates and
budgets results, prepends a versioned untrusted block to the last user message,
escapes memory tag delimiters inside untrusted records, and stores the original
user/assistant projection after a successful callback. It never writes the
injected block back as new memory.

Retrieval and storage have independent fail-open or fail-closed policies.
Identity defaults to fail-closed. Evidence uses categorized `memory` marks with
schema `nemo.relay.memory.operation/0.1`; the supported `references` mode emits
IDs, scores, hashes, lengths, latency, and policy without raw query, memory, or
answer text. Retrieved, injected, stored, and failed are separate facts. Relay
does not infer that an injected item caused the answer.

Automatic mode currently rejects streaming managed calls. Python and Node
expose native in-memory reference components, but arbitrary language-defined
`MemoryProvider` implementations are not yet bridged into automatic native
execution.

## Adaptive Prompt Cache Coordination

Automatic memory and the Adaptive Cache Governor (ACG) share a narrow seam.
The versioned leading memory envelope becomes a private `Memory` block in ACG
Prompt IR, and the learned stable prefix stops before it. Provider translation
still operates on the original request bytes.

ACG can emit hash-only `MemoryCacheFacts` beside provider cache-read and
cache-write token counts. The optional facts contain the envelope version,
current and previous short hashes, change status, Prompt IR sequence index, and
stable-prefix relationship. They never contain recalled text. A changed hash
and a cache miss are correlated observations; they do not prove that memory
caused the miss or that the model used the recalled context.

The memory runtime owns tenant and subject identity, semantic records, and
provider persistence. ACG owns prompt-prefix observations, provider cache
placement, and cache telemetry. The two systems do not share storage,
invalidation, retention, or exactly-once semantics.

## Background write-back

Inline write-back remains the default. Set `write_delivery` to `background` to
admit completed-turn storage to a bounded process-local queue and return without
waiting for the provider mutation:

```rust,ignore
use std::time::Duration;

use nemo_relay_memory::{
    AutomaticMemoryConfig, InMemoryProvider, MemoryComponent, MemoryWorkQueueConfig,
    WriteDelivery,
};

# async fn background() -> Result<(), Box<dyn std::error::Error>> {
let component = MemoryComponent::new(
    InMemoryProvider::new(),
    AutomaticMemoryConfig {
        write_delivery: WriteDelivery::Background,
        background_queue: MemoryWorkQueueConfig {
            capacity: 32,
            max_attempts: 3,
            ..MemoryWorkQueueConfig::default()
        },
        ..AutomaticMemoryConfig::default()
    },
)?;
let mut installation = component.install_global("memory", 0)?;

// Run ordinary managed LLM calls. Recall still completes before each callback,
// while successful-turn stores execute on the queue.

component.flush_background(Duration::from_secs(5)).await?;
installation.close()?;
component.shutdown_background(Duration::from_secs(5)).await?;
# Ok(())
# }
```

The queue has one consumer and bounded pending capacity. Backpressure is either
immediate rejection or a bounded capacity wait. Retryable typed failures retry
with a stable job ID and provider idempotency key; every attempt receives a
fresh deadline. `background_snapshot`, `background_job`, `flush_background`,
`drain_background`, and `shutdown_background` make tests and process handoff
deterministic. Python exposes the equivalent `background_status`,
`background_job_status`, `flush`, `drain`, and `shutdown` methods. Node uses the
same names in camel case where applicable.

In background mode, `storage_policy` applies to queue admission. A fail-closed
rejection can fail lifecycle completion. A provider failure after admission is
recorded as job state and evidence, but cannot retroactively fail a model
response that has already returned. Storage marks distinguish `queued`,
`running`, `retrying`, `stored`, `rejected`, `failed`, and `cancelled` without
including raw content or namespace identity.

The local queue is at-least-once and process-local. It does not persist jobs
across crashes or provide distributed exactly-once coordination. Terminal job
history is bounded; provider idempotency still protects a replay after an old
status entry is pruned.

## Maintenance and dreaming

`MemoryMaintainer` is separate from `MemoryProvider`: the provider owns storage
operations, while the maintainer owns derivation policy. A
`MemoryMaintenanceWindow` declares a checkpoint, optional predecessor, query,
scope, filters, and a source limit. The deterministic `ReferenceMaintainer`
searches that window and writes one versioned JSON artifact with
`source = "maintenance"`, the checkpoint in `source_ids`, and every source
record in `parent_memory_ids`. It is a source-preserving reference transform,
not an LLM summary.

Store and maintenance requests can use the same `MemoryWorkQueue`. A custom
maintainer can instead call a provider-native reflect operation or a separate
model. Heavy maintainers can run inside a `nemo-relay-worker` plugin: the worker
owns its provider client and scheduler, and its `WorkerPlugin::shutdown` hook
drains owned work when the host sends the existing authenticated `Shutdown`
RPC. ATOF events remain evidence and must not be treated as a durable command
queue.
