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

Automatic mode currently rejects streaming managed calls. Phase 3 awaits
write-back before returning; bounded queues and background maintenance are
separate work. Python and Node expose native in-memory reference components,
but arbitrary language-defined `MemoryProvider` implementations are not yet
bridged into automatic native execution.
