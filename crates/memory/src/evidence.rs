// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! ATOF memory operation evidence helpers.

use std::collections::BTreeMap;

use nemo_relay::api::event::{CategoryProfile, DataSchema, EventCategory};
use nemo_relay::api::runtime::LlmLifecycleContext;
use nemo_relay::api::scope::{EmitMarkEventParams, event};
use nemo_relay::error::Result;
use nemo_relay::json::Json;
use nemo_relay_types::memory::{MemoryContent, MemoryMatch, MemoryNamespace, MemoryOperationError};
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::automatic::{EvidenceMode, FailurePolicy};

/// Schema name used by automatic memory operation marks.
pub const MEMORY_OPERATION_SCHEMA: &str = "nemo.relay.memory.operation";
/// Initial automatic memory operation schema version.
pub const MEMORY_OPERATION_SCHEMA_VERSION: &str = "0.1";

pub(crate) fn namespace_reference(namespace: &MemoryNamespace) -> String {
    hash_bytes(
        &serde_json::to_vec(namespace)
            .expect("serializing a validated memory namespace cannot fail"),
    )
}

pub(crate) fn item_evidence(memory_match: &MemoryMatch, _mode: EvidenceMode) -> Json {
    let rendered = render_content(&memory_match.record.content);
    let item = serde_json::Map::from_iter([
        ("id".to_string(), json!(memory_match.record.id)),
        ("provider".to_string(), json!(memory_match.record.provider)),
        ("rank".to_string(), json!(memory_match.rank)),
        ("score".to_string(), json!(memory_match.score)),
        (
            "content_hash".to_string(),
            json!(hash_bytes(rendered.as_bytes())),
        ),
        (
            "content_length".to_string(),
            json!(rendered.chars().count()),
        ),
    ]);
    Json::Object(item)
}

pub(crate) fn error_evidence(error: &MemoryOperationError) -> Json {
    json!({
        "code": error.code,
        "retryable": error.retryable,
        "provider": error.provider,
        "operation_id": error.operation_id,
    })
}

pub(crate) fn emit_operation(
    context: &LlmLifecycleContext,
    subtype: &str,
    provider: &str,
    data: Json,
) -> Result<()> {
    let name = format!("memory.{subtype}");
    event(
        EmitMarkEventParams::builder()
            .name(&name)
            .llm_parent(&context.handle)
            .data(data)
            .data_schema(
                DataSchema::builder()
                    .name(MEMORY_OPERATION_SCHEMA)
                    .version(MEMORY_OPERATION_SCHEMA_VERSION)
                    .build(),
            )
            .category(EventCategory::memory())
            .category_profile(
                CategoryProfile::builder()
                    .subtype(subtype)
                    .extra(BTreeMap::from([("provider".to_string(), json!(provider))]))
                    .build(),
            )
            .build(),
    )
}

pub(crate) fn policy_name(policy: FailurePolicy) -> &'static str {
    match policy {
        FailurePolicy::FailOpen => "fail_open",
        FailurePolicy::FailClosed => "fail_closed",
    }
}

pub(crate) fn render_content(content: &MemoryContent) -> String {
    match content {
        MemoryContent::Text { text } => text.clone(),
        MemoryContent::Json { value } => value.to_string(),
        MemoryContent::Reference { reference, preview } => {
            preview.clone().unwrap_or_else(|| reference.clone())
        }
    }
}

fn hash_bytes(value: &[u8]) -> String {
    let digest = Sha256::digest(value);
    let encoded = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    format!("sha256:{encoded}")
}
