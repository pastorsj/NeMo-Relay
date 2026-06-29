// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native reference automatic-memory wrapper for Node.js.

use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use napi_derive::napi;
use nemo_relay_memory::{
    AutomaticMemoryConfig, InMemoryProvider, MemoryComponent, MemoryInstallation,
};

use crate::types::ScopeHandle;

/// Native owner for one dependency-free in-memory automatic-memory component.
#[napi(js_name = "NativeInMemoryAutomaticMemory")]
pub struct NativeInMemoryAutomaticMemory {
    component: MemoryComponent,
    installation: Mutex<Option<MemoryInstallation>>,
}

#[napi]
impl NativeInMemoryAutomaticMemory {
    /// Create an isolated reference component from canonical snake-case config.
    #[napi(constructor)]
    pub fn new(config: Option<serde_json::Value>) -> napi::Result<Self> {
        let config = match config {
            Some(config) => {
                serde_json::from_value::<AutomaticMemoryConfig>(config).map_err(|error| {
                    napi::Error::from_reason(format!("invalid automatic memory config: {error}"))
                })?
            }
            None => AutomaticMemoryConfig::default(),
        };
        let component = MemoryComponent::new(InMemoryProvider::new(), config).map_err(|error| {
            napi::Error::from_reason(format!("invalid automatic memory config: {error}"))
        })?;
        Ok(Self {
            component,
            installation: Mutex::new(None),
        })
    }

    /// Install globally or on one active scope.
    #[napi]
    pub fn install(
        &self,
        name: Option<String>,
        priority: Option<i32>,
        scope: Option<&ScopeHandle>,
    ) -> napi::Result<()> {
        let mut installation = self.lock_installation()?;
        if installation.is_some() {
            return Err(napi::Error::from_reason(
                "automatic memory is already installed",
            ));
        }
        let name = name.unwrap_or_else(|| "automatic_memory".to_string());
        let installed = match scope {
            Some(scope) => {
                self.component
                    .install_scope(&scope.inner, name, priority.unwrap_or_default())
            }
            None => self
                .component
                .install_global(name, priority.unwrap_or_default()),
        }
        .map_err(|error| napi::Error::from_reason(error.to_string()))?;
        *installation = Some(installed);
        Ok(())
    }

    /// Deregister this installation once.
    #[napi]
    pub fn close(&self) -> napi::Result<bool> {
        let mut installation = self.lock_installation()?;
        let Some(mut installed) = installation.take() else {
            return Ok(false);
        };
        installed
            .close()
            .map_err(|error| napi::Error::from_reason(error.to_string()))
    }

    /// Number of prepared calls still awaiting completion.
    #[napi(getter, js_name = "activeTurns")]
    pub fn active_turns(&self) -> u32 {
        u32::try_from(self.component.active_turns()).unwrap_or(u32::MAX)
    }

    /// Aggregate background queue state, or null for inline delivery.
    #[napi(getter, js_name = "backgroundStatus")]
    pub fn background_status(&self) -> napi::Result<Option<serde_json::Value>> {
        self.component
            .background_snapshot()
            .map(serde_json::to_value)
            .transpose()
            .map_err(|error| napi::Error::from_reason(error.to_string()))
    }

    /// Retained background state for one job, or null when unavailable.
    #[napi(js_name = "backgroundJobStatus")]
    pub fn background_job_status(&self, job_id: String) -> napi::Result<Option<serde_json::Value>> {
        self.component
            .background_job(&job_id)
            .map(serde_json::to_value)
            .transpose()
            .map_err(|error| napi::Error::from_reason(error.to_string()))
    }

    /// Wait for work accepted before this call.
    #[napi(js_name = "flushBackground")]
    pub async fn flush_background(&self, timeout_millis: u32) -> napi::Result<bool> {
        self.component
            .flush_background(checked_timeout(timeout_millis)?)
            .await
            .map_err(memory_runtime_error)
    }

    /// Stop admission and drain all accepted work.
    #[napi(js_name = "drainBackground")]
    pub async fn drain_background(&self, timeout_millis: u32) -> napi::Result<bool> {
        self.component
            .drain_background(checked_timeout(timeout_millis)?)
            .await
            .map_err(memory_runtime_error)
    }

    /// Stop, drain, and join the background worker.
    #[napi(js_name = "shutdownBackground")]
    pub async fn shutdown_background(&self, timeout_millis: u32) -> napi::Result<bool> {
        self.component
            .shutdown_background(checked_timeout(timeout_millis)?)
            .await
            .map_err(memory_runtime_error)
    }
}

impl NativeInMemoryAutomaticMemory {
    fn lock_installation(&self) -> napi::Result<MutexGuard<'_, Option<MemoryInstallation>>> {
        self.installation.lock().map_err(|_| {
            napi::Error::from_reason("automatic memory installation lock was poisoned")
        })
    }
}

fn checked_timeout(timeout_millis: u32) -> napi::Result<Duration> {
    if timeout_millis == 0 {
        return Err(napi::Error::from_reason(
            "timeoutMillis must be positive".to_string(),
        ));
    }
    Ok(Duration::from_millis(u64::from(timeout_millis)))
}

fn memory_runtime_error(error: nemo_relay_memory::memory::MemoryOperationError) -> napi::Error {
    napi::Error::from_reason(format!(
        "memory background operation failed ({:?}): {}",
        error.code, error.message
    ))
}
