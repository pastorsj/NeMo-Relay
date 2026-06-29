// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic maintenance contract and reference behavior tests.

use std::collections::BTreeMap;

use async_trait::async_trait;
use nemo_relay_memory::memory::{
    MemoryContent, MemoryErrorCode, MemoryFilter, MemoryMaintenanceAction,
    MemoryMaintenanceRequest, MemoryMaintenanceWindow, MemoryNamespace, MemoryOperationError,
    MemoryProvenance, MemoryRequestContext, MemorySearchRequest, MemorySearchResult,
    MemorySearchScope, MemoryStoreRequest, MemoryStoreResult,
};
use nemo_relay_memory::{
    InMemoryProvider, MemoryMaintainer, MemoryProvider, MemoryProviderResult, MemoryRuntime,
    REFERENCE_ARTIFACT_VERSION, ReferenceMaintainer,
};
use serde_json::json;

fn context(operation_id: &str) -> MemoryRequestContext {
    MemoryRequestContext::new(operation_id).expect("valid operation context")
}

fn namespace(subject: &str, session: &str) -> MemoryNamespace {
    MemoryNamespace {
        tenant_id: "tenant-a".to_string(),
        subject_id: subject.to_string(),
        session_id: Some(session.to_string()),
        agent_id: Some("agent-a".to_string()),
    }
}

async fn seed(
    runtime: &MemoryRuntime,
    operation_id: &str,
    namespace: MemoryNamespace,
    text: &str,
) -> MemoryStoreResult {
    runtime
        .store(MemoryStoreRequest {
            context: context(operation_id),
            namespace,
            content: MemoryContent::Text {
                text: text.to_string(),
            },
            event_timestamp: serde_json::from_str("\"2026-06-29T10:00:00Z\"")
                .expect("valid timestamp"),
            provenance: MemoryProvenance {
                source: "user".to_string(),
                source_ids: vec![operation_id.to_string()],
                parent_memory_ids: vec![],
                metadata: BTreeMap::new(),
            },
            metadata: BTreeMap::new(),
            idempotency_key: Some(format!("seed:{operation_id}")),
        })
        .await
        .expect("seed store succeeds")
}

fn maintenance_request(
    operation_id: &str,
    namespace: MemoryNamespace,
    query: &str,
) -> MemoryMaintenanceRequest {
    MemoryMaintenanceRequest {
        context: context(operation_id),
        namespace,
        action: MemoryMaintenanceAction::Consolidate,
        window: Some(MemoryMaintenanceWindow {
            checkpoint_id: "checkpoint-0002".to_string(),
            previous_checkpoint_id: Some("checkpoint-0001".to_string()),
            query: query.to_string(),
            scope: MemorySearchScope::Subject,
            filter: MemoryFilter::default(),
            limit: 10,
        }),
        parameters: BTreeMap::new(),
    }
}

#[tokio::test]
async fn reference_maintainer_preserves_partition_parent_provenance_and_replay() {
    let provider = InMemoryProvider::new();
    let runtime = MemoryRuntime::new(provider);
    let first = seed(
        &runtime,
        "source-1",
        namespace("subject-a", "session-a"),
        "editor preference is solarized dark",
    )
    .await;
    let second = seed(
        &runtime,
        "source-2",
        namespace("subject-a", "session-b"),
        "editor preference includes a larger font",
    )
    .await;
    seed(
        &runtime,
        "source-other-subject",
        namespace("subject-b", "session-c"),
        "editor preference is a light theme",
    )
    .await;
    let request = maintenance_request(
        "maintain-1",
        namespace("subject-a", "session-derived"),
        "editor preference",
    );

    let result = ReferenceMaintainer
        .maintain(&runtime, request.clone())
        .await
        .expect("maintenance succeeds");
    assert_eq!(result.records.len(), 1);
    assert!(result.partial_errors.is_empty());
    let derived = &result.records[0];
    assert_eq!(derived.namespace.tenant_id, "tenant-a");
    assert_eq!(derived.namespace.subject_id, "subject-a");
    assert_eq!(
        derived.namespace.session_id.as_deref(),
        Some("session-derived")
    );
    assert_eq!(derived.provenance.source, "maintenance");
    assert_eq!(derived.provenance.source_ids, ["checkpoint-0002"]);
    assert_eq!(
        derived.provenance.parent_memory_ids,
        [first.record.id.clone(), second.record.id.clone()]
    );
    assert_eq!(
        derived.provenance.metadata["artifact_version"],
        json!(REFERENCE_ARTIFACT_VERSION)
    );
    assert_eq!(derived.metadata["relay_derived"], json!(true));
    assert_eq!(
        derived.content,
        MemoryContent::Json {
            value: json!({
                "kind": "derived_memory",
                "artifact_version": REFERENCE_ARTIFACT_VERSION,
                "maintainer": "reference",
                "action": "consolidate",
                "checkpoint_id": "checkpoint-0002",
                "sources": [
                    {"memory_id": first.record.id, "content": first.record.content},
                    {"memory_id": second.record.id, "content": second.record.content},
                ],
            }),
        }
    );

    let replay = ReferenceMaintainer
        .maintain(&runtime, request)
        .await
        .expect("maintenance replay succeeds");
    assert_eq!(replay.records, result.records);

    let stored = runtime
        .search(MemorySearchRequest {
            context: context("verify-derived"),
            namespace: namespace("subject-a", "session-derived"),
            query: "derived memory solarized".to_string(),
            scope: MemorySearchScope::Exact,
            filter: MemoryFilter::default(),
            limit: 10,
        })
        .await
        .expect("verification search succeeds");
    assert_eq!(
        stored
            .matches
            .iter()
            .filter(|item| item.record.provenance.source == "maintenance")
            .count(),
        1
    );
}

#[tokio::test]
async fn empty_window_and_missing_window_do_not_fabricate_records() {
    let runtime = MemoryRuntime::new(InMemoryProvider::new());
    let empty = ReferenceMaintainer
        .maintain(
            &runtime,
            maintenance_request(
                "maintain-empty",
                namespace("subject-a", "session-a"),
                "nothing matches",
            ),
        )
        .await
        .expect("an empty source window is a valid outcome");
    assert!(empty.records.is_empty());

    let mut missing = maintenance_request(
        "maintain-missing-window",
        namespace("subject-a", "session-a"),
        "unused",
    );
    missing.window = None;
    let error = ReferenceMaintainer
        .maintain(&runtime, missing)
        .await
        .expect_err("reference maintenance requires a window");
    assert_eq!(error.code, MemoryErrorCode::InvalidRequest);
    assert_eq!(error.provider.as_deref(), Some("reference"));
}

struct FailingSearchProvider;

#[async_trait]
impl MemoryProvider for FailingSearchProvider {
    fn name(&self) -> &str {
        "failing_search"
    }

    async fn search(
        &self,
        request: MemorySearchRequest,
    ) -> MemoryProviderResult<MemorySearchResult> {
        Err(MemoryOperationError::new(
            MemoryErrorCode::ProviderUnavailable,
            "source index unavailable",
            true,
        )
        .with_operation_id(request.context.operation_id)
        .with_provider(self.name()))
    }

    async fn store(&self, _request: MemoryStoreRequest) -> MemoryProviderResult<MemoryStoreResult> {
        unreachable!("maintenance must not store after a failed source search")
    }
}

#[tokio::test]
async fn source_search_failure_is_preserved_without_a_partial_artifact() {
    let runtime = MemoryRuntime::new(FailingSearchProvider);
    let error = ReferenceMaintainer
        .maintain(
            &runtime,
            maintenance_request(
                "maintain-failure",
                namespace("subject-a", "session-a"),
                "editor preference",
            ),
        )
        .await
        .expect_err("source failure must propagate");

    assert_eq!(error.code, MemoryErrorCode::ProviderUnavailable);
    assert!(error.retryable);
    assert_eq!(error.provider.as_deref(), Some("failing_search"));
    assert_eq!(
        error.operation_id.as_deref(),
        Some("maintain-failure:search")
    );
}
