// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native reference automatic-memory wrapper for Python.

use std::sync::Mutex;

use nemo_relay_memory::{
    AutomaticMemoryConfig, InMemoryProvider, MemoryComponent, MemoryInstallation,
};
use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::convert::py_to_json;
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

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyInMemoryAutomaticMemory>()?;
    Ok(())
}
