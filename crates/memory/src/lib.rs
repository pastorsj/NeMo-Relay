// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider-neutral memory interfaces and reference implementations for NeMo Relay.
//!
//! This crate owns provider execution behavior while [`nemo_relay_types::memory`]
//! owns the serializable wire contract. The direct [`MemoryRuntime`] validates
//! required requests, enforces absolute deadlines, and never retries implicitly.
//! Dropping an operation future is the cancellation mechanism; providers must not
//! publish partial mutations before a cancellation-safe commit point.

pub mod provider;
pub mod runtime;

pub use nemo_relay_types::memory;
pub use provider::{MemoryProvider, MemoryProviderResult};
pub use runtime::MemoryRuntime;
