// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Tests for the reusable provider conformance report.

use std::sync::Arc;

use async_trait::async_trait;
use nemo_relay_memory::memory::{
    MemoryErrorCode, MemoryOperationError, MemorySearchRequest, MemorySearchResult,
    MemoryStoreRequest, MemoryStoreResult,
};
use nemo_relay_memory::{
    InMemoryProvider, MemoryProvider, MemoryProviderResult, run_provider_conformance,
};

#[tokio::test]
async fn in_memory_provider_passes_every_conformance_case() {
    let report = run_provider_conformance(Arc::new(InMemoryProvider::new()), "in-memory-1").await;

    assert!(
        report.passed(),
        "conformance failures: {:?}",
        report.failures().collect::<Vec<_>>()
    );
    assert_eq!(report.provider, "in_memory");
    assert_eq!(report.run_id, "in-memory-1");
    assert_eq!(report.cases.len(), 9);
    assert!(report.cases.iter().all(|case| case.message.is_none()));
}

struct FailingProvider;

#[async_trait]
impl MemoryProvider for FailingProvider {
    fn name(&self) -> &str {
        "failing"
    }

    async fn search(
        &self,
        _request: MemorySearchRequest,
    ) -> MemoryProviderResult<MemorySearchResult> {
        Err(failure())
    }

    async fn store(&self, _request: MemoryStoreRequest) -> MemoryProviderResult<MemoryStoreResult> {
        Err(failure())
    }
}

#[tokio::test]
async fn harness_reports_provider_failures_as_named_data() {
    let report = run_provider_conformance(Arc::new(FailingProvider), "failure-1").await;

    assert!(!report.passed());
    assert!(report.cases.iter().any(|case| {
        case.name == "required_store"
            && !case.passed
            && case.message.as_deref() == Some("synthetic provider failure")
    }));
    assert!(
        report
            .cases
            .iter()
            .any(|case| case.name == "deadline_cancellation" && case.passed)
    );
}

#[tokio::test]
async fn empty_run_id_is_a_structured_failure() {
    let report = run_provider_conformance(Arc::new(InMemoryProvider::new()), "  ").await;

    assert_eq!(report.cases.len(), 1);
    assert_eq!(report.cases[0].name, "valid_run_id");
    assert!(!report.cases[0].passed);
}

fn failure() -> MemoryOperationError {
    MemoryOperationError::new(
        MemoryErrorCode::ProviderUnavailable,
        "synthetic provider failure",
        true,
    )
}
