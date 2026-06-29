// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native reference automatic-memory wrapper for Python.

use std::sync::Mutex;
use std::time::Duration;

use nemo_relay_memory::{
    AutomaticMemoryConfig, InMemoryProvider, MemoryComponent, MemoryInstallation,
};
use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::convert::{json_to_py, py_to_json};
use crate::py_types::PyScopeHandle;

/// Native owner for one dependency-free in-memory automatic-memory component.
#[pyclass(name = "_NativeInMemoryAutomaticMemory")]
pub struct PyInMemoryAutomaticMemory {
    component: MemoryComponent,
    installation: Mutex<Option<MemoryInstallation>>,
}

#[pymethods]
impl PyInMemoryAutomaticMemory {
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(config: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let config = match config.filter(|value| !value.is_none()) {
            Some(config) => serde_json::from_value::<AutomaticMemoryConfig>(py_to_json(config)?)
                .map_err(|error| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "invalid automatic memory config: {error}"
                    ))
                })?,
            None => AutomaticMemoryConfig::default(),
        };
        let component = MemoryComponent::new(InMemoryProvider::new(), config).map_err(|error| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "invalid automatic memory config: {error}"
            ))
        })?;
        Ok(Self {
            component,
            installation: Mutex::new(None),
        })
    }

    /// Install the component globally or on one active scope.
    #[pyo3(signature = (name="automatic_memory", priority=0, scope=None))]
    fn install(&self, name: &str, priority: i32, scope: Option<&PyScopeHandle>) -> PyResult<()> {
        let mut installation = self.lock_installation()?;
        if installation.is_some() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "automatic memory is already installed",
            ));
        }
        let installed = match scope {
            Some(scope) => self.component.install_scope(&scope.inner, name, priority),
            None => self.component.install_global(name, priority),
        }
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
        *installation = Some(installed);
        Ok(())
    }

    /// Deregister this installation once.
    fn close(&self) -> PyResult<bool> {
        let mut installation = self.lock_installation()?;
        let Some(mut installed) = installation.take() else {
            return Ok(false);
        };
        installed
            .close()
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))
    }

    /// Number of prepared calls still awaiting completion.
    #[getter]
    fn active_turns(&self) -> usize {
        self.component.active_turns()
    }

    /// Aggregate background queue state, or ``None`` for inline delivery.
    #[getter]
    fn background_status(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let value = serde_json::to_value(self.component.background_snapshot())
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
        json_to_py(py, &value)
    }

    /// Retained background state for one job, or ``None`` when unavailable.
    fn background_job_status(&self, py: Python<'_>, job_id: &str) -> PyResult<Py<PyAny>> {
        let value = serde_json::to_value(self.component.background_job(job_id))
            .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
        json_to_py(py, &value)
    }

    /// Wait for work accepted before this call.
    #[pyo3(signature = (timeout_millis=5_000))]
    fn flush_background<'py>(
        &self,
        py: Python<'py>,
        timeout_millis: u64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let component = self.component.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            component
                .flush_background(checked_timeout(timeout_millis)?)
                .await
                .map_err(memory_runtime_error)
        })
    }

    /// Stop admission and drain all accepted work.
    #[pyo3(signature = (timeout_millis=5_000))]
    fn drain_background<'py>(
        &self,
        py: Python<'py>,
        timeout_millis: u64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let component = self.component.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            component
                .drain_background(checked_timeout(timeout_millis)?)
                .await
                .map_err(memory_runtime_error)
        })
    }

    /// Stop, drain, and join the background worker.
    #[pyo3(signature = (timeout_millis=5_000))]
    fn shutdown_background<'py>(
        &self,
        py: Python<'py>,
        timeout_millis: u64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let component = self.component.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            component
                .shutdown_background(checked_timeout(timeout_millis)?)
                .await
                .map_err(memory_runtime_error)
        })
    }
}

impl PyInMemoryAutomaticMemory {
    fn lock_installation(&self) -> PyResult<std::sync::MutexGuard<'_, Option<MemoryInstallation>>> {
        self.installation.lock().map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err(
                "automatic memory installation lock was poisoned",
            )
        })
    }
}

fn checked_timeout(timeout_millis: u64) -> PyResult<Duration> {
    if timeout_millis == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "timeout_millis must be positive",
        ));
    }
    Ok(Duration::from_millis(timeout_millis))
}

fn memory_runtime_error(error: nemo_relay_memory::memory::MemoryOperationError) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!(
        "memory background operation failed ({:?}): {}",
        error.code, error.message
    ))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyInMemoryAutomaticMemory>()?;
    Ok(())
}
