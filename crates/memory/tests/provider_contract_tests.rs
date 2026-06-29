// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Contract tests for provider object safety, defaults, and runtime deadlines.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use async_trait::async_trait;
use nemo_relay_memory::memory::{
    MemoryBatchStoreRequest, MemoryCapabilities, MemoryContent, MemoryDeleteRequest,
    MemoryErrorCode, MemoryFeedbackKind, MemoryFeedbackRequest, MemoryFilter, MemoryHealthRequest,
    MemoryMaintenanceAction, MemoryMaintenanceRequest, MemoryMatch, MemoryNamespace,
    MemoryOperationError, MemoryProvenance, MemoryRequestContext, MemorySearchRequest,
    MemorySearchResult, MemorySearchScope, MemoryStoreRequest, MemoryStoreResult,
    MemoryUpdateRequest,
};
use nemo_relay_memory::{MemoryProvider, MemoryProviderResult, MemoryRuntime};

#[derive(Clone)]
struct ProbeProvider {
    calls: Arc<AtomicUsize>,
    dropped: Arc<AtomicBool>,
    response: MemorySearchResult,
    block_search: bool,
}

#[async_trait]
impl MemoryProvider for ProbeProvider {
    fn name(&self) -> &str {
        "probe"
    }

    async fn search(
        &self,
        _request: MemorySearchRequest,
    ) -> MemoryProviderResult<MemorySearchResult> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.block_search {
            let _guard = DropFlag(self.dropped.clone());
            std::future::pending().await
        } else {
            Ok(self.response.clone())
        }
    }

    async fn store(&self, _request: MemoryStoreRequest) -> MemoryProviderResult<MemoryStoreResult> {
        Err(MemoryOperationError::new(
            MemoryErrorCode::Internal,
            "probe store",
            false,
        ))
    }
}

struct DropFlag(Arc<AtomicBool>);

impl Drop for DropFlag {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

fn context(operation_id: &str) -> MemoryRequestContext {
    MemoryRequestContext::new(operation_id).expect("valid context")
}

fn namespace() -> MemoryNamespace {
    MemoryNamespace::new("tenant-test", "subject-test").expect("valid namespace")
}

fn search_request(operation_id: &str) -> MemorySearchRequest {
    MemorySearchRequest {
        context: context(operation_id),
        namespace: namespace(),
        query: "preferred editor theme".to_string(),
        scope: MemorySearchScope::Subject,
        filter: MemoryFilter::default(),
        limit: 10,
    }
}

fn unsupported_provider() -> Arc<dyn MemoryProvider> {
    Arc::new(ProbeProvider {
        calls: Arc::new(AtomicUsize::new(0)),
        dropped: Arc::new(AtomicBool::new(false)),
        response: MemorySearchResult::default(),
        block_search: false,
    })
}

#[tokio::test]
async fn trait_object_executes_required_operations_once_and_preserves_partial_results() {
    let partial_error = MemoryOperationError::new(
        MemoryErrorCode::ProviderUnavailable,
        "one shard unavailable",
        true,
    );
    let response = MemorySearchResult {
        matches: Vec::<MemoryMatch>::new(),
        partial_errors: vec![partial_error],
    };
    let calls = Arc::new(AtomicUsize::new(0));
    let provider: Arc<dyn MemoryProvider> = Arc::new(ProbeProvider {
        calls: calls.clone(),
        dropped: Arc::new(AtomicBool::new(false)),
        response: response.clone(),
        block_search: false,
    });
    let runtime = MemoryRuntime::from_arc(provider);

    let actual = runtime.search(search_request("search-once")).await.unwrap();

    assert_eq!(actual, response);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(runtime.provider_name(), "probe");
}

#[tokio::test]
async fn false_capabilities_return_typed_unsupported_errors() {
    let provider = unsupported_provider();
    assert_eq!(provider.capabilities(), MemoryCapabilities::default());
    let namespace = namespace();

    let update = provider
        .update(MemoryUpdateRequest {
            context: context("update"),
            namespace: namespace.clone(),
            id: "memory-1".to_string(),
            content: Some(MemoryContent::Text {
                text: "changed".to_string(),
            }),
            metadata: None,
        })
        .await;
    let delete = provider
        .delete(MemoryDeleteRequest {
            context: context("delete"),
            namespace: namespace.clone(),
            id: "memory-1".to_string(),
        })
        .await;
    let batch = provider
        .batch_store(MemoryBatchStoreRequest {
            context: context("batch"),
            requests: vec![],
        })
        .await;
    let maintenance = provider
        .maintain(MemoryMaintenanceRequest {
            context: context("maintenance"),
            namespace: namespace.clone(),
            action: MemoryMaintenanceAction::Reflect,
            window: None,
            parameters: BTreeMap::new(),
        })
        .await;
    let feedback = provider
        .feedback(MemoryFeedbackRequest {
            context: context("feedback"),
            namespace,
            id: "memory-1".to_string(),
            kind: MemoryFeedbackKind::Relevant,
            confidence: Some(1.0),
            metadata: BTreeMap::new(),
        })
        .await;
    let health = provider
        .health(MemoryHealthRequest {
            context: context("health"),
        })
        .await;

    for result in [
        update.map(|_| ()),
        delete.map(|_| ()),
        batch.map(|_| ()),
        maintenance.map(|_| ()),
        feedback.map(|_| ()),
        health.map(|_| ()),
    ] {
        let error = result.expect_err("unsupported operation must fail");
        assert_eq!(error.code, MemoryErrorCode::Unsupported);
        assert_eq!(error.provider.as_deref(), Some("probe"));
    }
}

#[tokio::test(start_paused = true)]
async fn deadline_drops_blocking_provider_future_without_retry() {
    let calls = Arc::new(AtomicUsize::new(0));
    let dropped = Arc::new(AtomicBool::new(false));
    let runtime = MemoryRuntime::new(ProbeProvider {
        calls: calls.clone(),
        dropped: dropped.clone(),
        response: MemorySearchResult::default(),
        block_search: true,
    });
    let mut request = search_request("blocked-search");
    request.context.deadline =
        Some(serde_json::from_str("\"2999-01-01T00:00:00Z\"").expect("valid future deadline"));

    let operation = tokio::spawn(async move { runtime.search(request).await });
    tokio::task::yield_now().await;
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    tokio::time::advance(std::time::Duration::from_secs(40_000_000_000)).await;
    let error = operation
        .await
        .expect("task must join")
        .expect_err("deadline must expire");

    assert_eq!(error.code, MemoryErrorCode::DeadlineExceeded);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(dropped.load(Ordering::SeqCst));
}

#[tokio::test]
async fn already_expired_deadline_never_calls_provider() {
    let calls = Arc::new(AtomicUsize::new(0));
    let runtime = MemoryRuntime::new(ProbeProvider {
        calls: calls.clone(),
        dropped: Arc::new(AtomicBool::new(false)),
        response: MemorySearchResult::default(),
        block_search: false,
    });
    let mut request = search_request("expired-search");
    request.context.deadline =
        Some(serde_json::from_str("\"2000-01-01T00:00:00Z\"").expect("valid past deadline"));

    let error = runtime
        .search(request)
        .await
        .expect_err("deadline is expired");

    assert_eq!(error.code, MemoryErrorCode::DeadlineExceeded);
    assert_eq!(error.provider.as_deref(), Some("probe"));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[allow(dead_code)]
fn _store_request_example() -> MemoryStoreRequest {
    MemoryStoreRequest {
        context: context("store"),
        namespace: namespace(),
        content: MemoryContent::Text {
            text: "example".to_string(),
        },
        event_timestamp: serde_json::from_str("\"2026-01-01T00:00:00Z\"").expect("valid timestamp"),
        provenance: MemoryProvenance {
            source: "test".to_string(),
            ..MemoryProvenance::default()
        },
        metadata: BTreeMap::new(),
        idempotency_key: None,
    }
}
