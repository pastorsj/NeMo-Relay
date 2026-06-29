<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

[![License](https://img.shields.io/github/license/NVIDIA/NeMo-Relay)](https://github.com/NVIDIA/NeMo-Relay/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/NVIDIA/NeMo-Relay/)
[![Release](https://img.shields.io/github/v/release/NVIDIA/NeMo-Relay?color=green)](https://github.com/NVIDIA/NeMo-Relay/releases)
[![Codecov](https://codecov.io/gh/NVIDIA/NeMo-Relay/branch/main/graph/badge.svg)](https://app.codecov.io/gh/NVIDIA/NeMo-Relay)
[![PyPI](https://img.shields.io/pypi/v/nemo-relay?color=4B8BBE&logo=pypi)](https://pypi.org/project/nemo-relay/)
[![npm node](https://img.shields.io/npm/v/nemo-relay-node?label=nemo-relay-node&color=CC3534&logo=npm)](https://www.npmjs.com/package/nemo-relay-node)
[![npm wasm](https://img.shields.io/npm/v/nemo-relay-wasm?label=nemo-relay-wasm&color=CC3534&logo=npm)](https://www.npmjs.com/package/nemo-relay-wasm)
[![Crates.io](https://img.shields.io/crates/v/nemo-relay?label=nemo-relay&color=B7410E&logo=rust)](https://crates.io/crates/nemo-relay)
[![Crates.io](https://img.shields.io/crates/v/nemo-relay-adaptive?label=nemo-relay-adaptive&color=B7410E&logo=rust)](https://crates.io/crates/nemo-relay-adaptive)
[![Crates.io](https://img.shields.io/crates/v/nemo-relay-cli?label=nemo-relay-cli&color=B7410E&logo=rust)](https://crates.io/crates/nemo-relay-cli)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/NVIDIA/NeMo-Relay)

# NeMo Relay

`nemo-relay-adaptive` is the Rust companion crate for adaptive NeMo Relay
runtime behavior. Use it with `nemo-relay` when an agent runtime should learn
from observed executions, inject runtime hints, or persist adaptive state.

Adaptive behavior is installed through the same plugin system used by the core
runtime, so applications can enable it without changing their orchestration
framework.

## Why Use It?

- ⚙️ **Install adaptive behavior through plugins**: Enable adaptive runtime
  components through the same configuration path as other NeMo Relay plugins.
- 📈 **Learn from observed executions**: Derive runtime hints from scope, tool,
  and LLM events without replacing the application framework.
- 💾 **Choose local or shared state**: Use in-memory state for local runs or the
  optional Redis backend for shared persistence.
- 🧩 **Keep adaptive behavior reusable**: Package telemetry, hint injection,
  tool parallelism, and cache-governor behavior behind stable component
  settings.

## What You Get

- ✅ **`AdaptiveConfig`**: A canonical config contract for the top-level
  `adaptive` plugin component.
- ✅ **Built-in component settings**: Typed config helpers for telemetry,
  adaptive hints, tool parallelism, and the Adaptive Cache Governor.
- ✅ **State backends**: In-memory state by default and Redis-backed state behind
  the `redis-backend` feature.
- ✅ **Learning primitives**: Runtime helpers and learners built on NeMo Relay
  events.
- ✅ **Adaptive Cache Governor (ACG) module surface**: The canonical
  `nemo_relay_adaptive::acg` module for PromptIR, provider plugins, stability
  analysis, and cache telemetry normalization.

## Memory-Aware Cache Planning

When automatic memory prepends a valid versioned Relay envelope, ACG classifies
the envelope as a private `Memory` block in Prompt IR. It does not change the
provider request. The learned stable prefix ends before that block, so existing
Anthropic and OpenAI translation keeps volatile recalled context outside the
planned reusable prefix.

Cache request facts and telemetry can include `MemoryCacheFacts`: the envelope
version, current and previous 12-hex SHA-256 prefixes, change status, Prompt IR
sequence index, and whether the block is outside the stable prefix. These facts
never contain recalled text. They correlate a memory revision with provider
cache-read and cache-write tokens, but they do not prove that the revision
caused a cache outcome or affected an answer.

Memory and adaptive state remain separate. The memory runtime owns semantic
identity and provider persistence. ACG owns prompt-prefix observations,
provider translation, and cache telemetry. They share Prompt IR and telemetry,
not a database, invalidation API, or retention policy.

For current provider behavior, refer to the
[OpenAI prompt caching guide](https://platform.openai.com/docs/guides/prompt-caching)
and the
[Anthropic prompt caching guide](https://platform.claude.com/docs/en/build-with-claude/prompt-caching).

## Installation

Install the published crate alongside the core runtime:

```bash
cargo add nemo-relay nemo-relay-adaptive
```

Enable Redis-backed state only when the application needs shared persistence:

```bash
cargo add nemo-relay-adaptive --features redis-backend
```

For local source development:

```bash
cargo build -p nemo-relay-adaptive
cargo test -p nemo-relay-adaptive
```

## Getting Started

Create a default adaptive config and select the in-memory backend:

```rust
use nemo_relay_adaptive::{AdaptiveConfig, BackendSpec, StateConfig};

let config = AdaptiveConfig {
    state: Some(StateConfig {
        backend: BackendSpec::in_memory(),
    }),
    ..Default::default()
};
```

Register the adaptive plugin component before validating or initializing plugin
configuration that includes an `adaptive` component:

```rust
nemo_relay_adaptive::plugin_component::register_adaptive_component()?;
```

## Feature Flags

- `redis-backend`: Enables the Redis-backed storage implementation.

Builds without `redis-backend` still support the in-memory backend and the rest
of the adaptive pipeline.

## Documentation

[NeMo Relay documentation](https://docs.nvidia.com/nemo/relay)
