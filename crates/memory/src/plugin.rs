// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Built-in automatic memory plugin component.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use nemo_relay::plugin::{
    ConfigDiagnostic, DiagnosticLevel, Plugin, PluginError, PluginRegistrationContext,
    Result as PluginResult, register_plugin,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value as Json};

use crate::InMemoryProvider;
use crate::automatic::{AutomaticMemoryConfig, MemoryComponent};

/// Plugin kind for automatic memory configuration.
pub const MEMORY_PLUGIN_KIND: &str = "memory";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
struct MemoryPluginConfig {
    provider: String,
    priority: i32,
    #[serde(flatten)]
    automatic: AutomaticMemoryConfig,
}

impl Default for MemoryPluginConfig {
    fn default() -> Self {
        Self {
            provider: "in_memory".to_string(),
            priority: 0,
            automatic: AutomaticMemoryConfig::default(),
        }
    }
}

struct MemoryPlugin;

impl Plugin for MemoryPlugin {
    fn plugin_kind(&self) -> &str {
        MEMORY_PLUGIN_KIND
    }

    fn allows_multiple_components(&self) -> bool {
        false
    }

    fn validate(&self, plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        match parse_config(plugin_config) {
            Ok(config) if config.provider != "in_memory" => vec![diagnostic(format!(
                "provider must be 'in_memory' in the reference automatic component, got {:?}",
                config.provider
            ))],
            Ok(config) => config
                .automatic
                .validate()
                .err()
                .map(|error| vec![diagnostic(error.to_string())])
                .unwrap_or_default(),
            Err(error) => vec![diagnostic(error.to_string())],
        }
    }

    fn register<'a>(
        &'a self,
        plugin_config: &Map<String, Json>,
        ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = PluginResult<()>> + Send + 'a>> {
        let plugin_config = plugin_config.clone();
        Box::pin(async move {
            let config = parse_config(&plugin_config)?;
            if config.provider != "in_memory" {
                return Err(PluginError::InvalidConfig(format!(
                    "memory provider {:?} is not supported by the reference component",
                    config.provider
                )));
            }
            config
                .automatic
                .validate()
                .map_err(|error| PluginError::InvalidConfig(error.to_string()))?;
            let component = MemoryComponent::new(InMemoryProvider::new(), config.automatic)
                .map_err(|error| PluginError::InvalidConfig(error.to_string()))?;
            ctx.register_llm_lifecycle_hook(
                "automatic",
                config.priority,
                component.lifecycle_hook(),
            )
        })
    }
}

/// Register the built-in automatic memory plugin kind idempotently.
pub fn register_memory_component() -> PluginResult<()> {
    match register_plugin(Arc::new(MemoryPlugin)) {
        Ok(()) => Ok(()),
        Err(PluginError::RegistrationFailed(message)) if message.contains("already registered") => {
            Ok(())
        }
        Err(error) => Err(error),
    }
}

fn parse_config(config: &Map<String, Json>) -> PluginResult<MemoryPluginConfig> {
    serde_json::from_value(Json::Object(config.clone()))
        .map_err(|error| PluginError::InvalidConfig(format!("invalid memory config: {error}")))
}

fn diagnostic(message: String) -> ConfigDiagnostic {
    ConfigDiagnostic {
        level: DiagnosticLevel::Error,
        code: "memory.invalid_config".to_string(),
        component: Some(MEMORY_PLUGIN_KIND.to_string()),
        field: None,
        message,
    }
}
