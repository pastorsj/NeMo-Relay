// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![doc = include_str!("../README.md")]

//! Provider-neutral memory interfaces and reference implementations for NeMo Relay.
//!
//! This crate owns provider execution behavior while [`nemo_relay_types::memory`]
//! owns the serializable wire contract. The direct [`MemoryRuntime`] validates
//! required requests, enforces absolute deadlines, and never retries implicitly.
//! Dropping an operation future is the cancellation mechanism; providers must not
//! publish partial mutations before a cancellation-safe commit point.

#[cfg(feature = "relay")]
pub mod automatic;
pub mod conformance;
#[cfg(feature = "relay")]
pub mod evidence;
pub mod in_memory;
#[cfg(feature = "relay")]
pub mod plugin;
pub mod provider;
pub mod runtime;

#[cfg(feature = "relay")]
pub use automatic::{
    AutomaticMemoryConfig, EvidenceMode, FailurePolicy, MemoryComponent, MemoryInstallation,
    WriteProjection,
};
pub use conformance::{ConformanceCase, ConformanceReport, run_provider_conformance};
pub use in_memory::InMemoryProvider;
pub use nemo_relay_types::memory;
#[cfg(feature = "relay")]
pub use plugin::{MEMORY_PLUGIN_KIND, register_memory_component};
pub use provider::{MemoryProvider, MemoryProviderResult};
pub use runtime::MemoryRuntime;
