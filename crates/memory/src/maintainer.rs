// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider-neutral memory derivation policy.

use std::collections::{BTreeMap, BTreeSet};

use async_trait::async_trait;
use nemo_relay_types::memory::{
    MemoryContent, MemoryMaintenanceAction, MemoryMaintenanceRequest, MemoryMaintenanceResult,
    MemoryOperationError, MemoryProvenance, MemorySearchRequest, MemoryStoreRequest,
};
use serde_json::json;

use crate::MemoryRuntime;

/// Artifact format emitted by [`ReferenceMaintainer`].
pub const REFERENCE_ARTIFACT_VERSION: &str = "0.1";

/// Result returned by a memory maintainer.
pub type MemoryMaintainerResult<T> = Result<T, MemoryOperationError>;

/// Object-safe policy for deriving memory from a declared source window.
///
/// A maintainer is deliberately separate from [`crate::MemoryProvider`]. A
/// maintainer may compose required search/store operations, delegate to a
/// provider-native maintenance capability, or invoke a separate model. It must
/// preserve the request's tenant and subject partition.
#[async_trait]
pub trait MemoryMaintainer: Send + Sync {
    /// Stable maintainer identifier used in provenance and idempotency keys.
    fn name(&self) -> &str;

    /// Execute one validated maintenance request against a memory runtime.
    async fn maintain(
        &self,
        runtime: &MemoryRuntime,
        request: MemoryMaintenanceRequest,
    ) -> MemoryMaintainerResult<MemoryMaintenanceResult>;
}

/// Deterministic maintainer for tests, examples, and zero-service operation.
///
/// The reference transform preserves source contents in a versioned JSON
/// artifact. It is not an LLM summary. Records previously derived by a
/// maintainer are excluded so replay cannot recursively consolidate its own
/// output.
#[derive(Debug, Clone, Copy, Default)]
pub struct ReferenceMaintainer;

#[async_trait]
impl MemoryMaintainer for ReferenceMaintainer {
    fn name(&self) -> &str {
        "reference"
    }

    async fn maintain(
        &self,
        runtime: &MemoryRuntime,
        request: MemoryMaintenanceRequest,
    ) -> MemoryMaintainerResult<MemoryMaintenanceResult> {
        request.validate()?;
        let window = request.window.as_ref().ok_or_else(|| {
            MemoryOperationError::invalid_request("reference maintenance requires a source window")
                .with_operation_id(&request.context.operation_id)
                .with_provider(self.name())
        })?;
        let search = runtime
            .search(MemorySearchRequest {
                context: child_context(&request, "search"),
                namespace: request.namespace.clone(),
                query: window.query.clone(),
                scope: window.scope,
                filter: window.filter.clone(),
                limit: window.limit,
            })
            .await?;

        let mut seen = BTreeSet::new();
        let sources = search
            .matches
            .into_iter()
            .filter(|memory_match| memory_match.record.provenance.source != "maintenance")
            .filter(|memory_match| seen.insert(memory_match.record.id.clone()))
            .collect::<Vec<_>>();
        if sources.is_empty() {
            return Ok(MemoryMaintenanceResult {
                job_id: None,
                records: vec![],
                partial_errors: search.partial_errors,
            });
        }

        let parent_memory_ids = sources
            .iter()
            .map(|memory_match| memory_match.record.id.clone())
            .collect::<Vec<_>>();
        let event_timestamp = sources
            .iter()
            .map(|memory_match| memory_match.record.event_timestamp)
            .max()
            .expect("nonempty source window has an event timestamp");
        let action = action_name(request.action);
        let content = MemoryContent::Json {
            value: json!({
                "kind": "derived_memory",
                "artifact_version": REFERENCE_ARTIFACT_VERSION,
                "maintainer": self.name(),
                "action": action,
                "checkpoint_id": window.checkpoint_id,
                "sources": sources.iter().map(|memory_match| json!({
                    "memory_id": memory_match.record.id,
                    "content": memory_match.record.content.clone(),
                })).collect::<Vec<_>>(),
            }),
        };
        let previous_checkpoint = window
            .previous_checkpoint_id
            .as_ref()
            .map_or(serde_json::Value::Null, |value| json!(value));
        let result = runtime
            .store(MemoryStoreRequest {
                context: child_context(&request, "store"),
                namespace: request.namespace.clone(),
                content,
                event_timestamp,
                provenance: MemoryProvenance {
                    source: "maintenance".to_string(),
                    source_ids: vec![window.checkpoint_id.clone()],
                    parent_memory_ids,
                    metadata: BTreeMap::from([
                        ("action".to_string(), json!(action)),
                        (
                            "artifact_version".to_string(),
                            json!(REFERENCE_ARTIFACT_VERSION),
                        ),
                        ("maintainer".to_string(), json!(self.name())),
                        ("previous_checkpoint_id".to_string(), previous_checkpoint),
                    ]),
                },
                metadata: BTreeMap::from([
                    ("relay_derived".to_string(), json!(true)),
                    (
                        "artifact_version".to_string(),
                        json!(REFERENCE_ARTIFACT_VERSION),
                    ),
                ]),
                idempotency_key: Some(format!(
                    "maintenance:{}:{action}:{}:{}",
                    self.name(),
                    window.checkpoint_id,
                    REFERENCE_ARTIFACT_VERSION
                )),
            })
            .await?;

        Ok(MemoryMaintenanceResult {
            job_id: None,
            records: vec![result.record],
            partial_errors: search.partial_errors,
        })
    }
}

fn child_context(
    request: &MemoryMaintenanceRequest,
    stage: &str,
) -> nemo_relay_types::memory::MemoryRequestContext {
    nemo_relay_types::memory::MemoryRequestContext {
        operation_id: format!("{}:{stage}", request.context.operation_id),
        deadline: request.context.deadline,
    }
}

const fn action_name(action: MemoryMaintenanceAction) -> &'static str {
    match action {
        MemoryMaintenanceAction::Reflect => "reflect",
        MemoryMaintenanceAction::Consolidate => "consolidate",
    }
}
