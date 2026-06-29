// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Asynchronous provider contract for memory backends.

use async_trait::async_trait;
use nemo_relay_types::memory::{
    MemoryBatchStoreRequest, MemoryBatchStoreResult, MemoryCapabilities, MemoryDeleteRequest,
    MemoryDeleteResult, MemoryFeedbackRequest, MemoryFeedbackResult, MemoryHealthRequest,
    MemoryHealthResult, MemoryMaintenanceRequest, MemoryMaintenanceResult, MemoryOperationError,
    MemorySearchRequest, MemorySearchResult, MemoryStoreRequest, MemoryStoreResult,
    MemoryUpdateRequest,
};

/// Result returned by memory provider operations.
pub type MemoryProviderResult<T> = Result<T, MemoryOperationError>;

/// Object-safe asynchronous memory provider.
///
/// Implementations must provide only [`MemoryProvider::search`] and
/// [`MemoryProvider::store`]. Optional operations advertise support through
/// [`MemoryProvider::capabilities`] and otherwise return a stable `unsupported`
/// error from their default implementation.
///
/// Provider futures are cancelled by being dropped. Mutating implementations
/// must therefore prepare work before a single cancellation-safe commit point
/// and avoid awaiting after mutation begins.
#[async_trait]
pub trait MemoryProvider: Send + Sync {
    /// Return the stable provider identifier recorded on results and errors.
    fn name(&self) -> &str;

    /// Return optional operations supported by this provider.
    fn capabilities(&self) -> MemoryCapabilities {
        MemoryCapabilities::default()
    }

    /// Search memories within the request's tenant and subject partition.
    async fn search(
        &self,
        request: MemorySearchRequest,
    ) -> MemoryProviderResult<MemorySearchResult>;

    /// Store one memory using the request's origin namespace and provenance.
    async fn store(&self, request: MemoryStoreRequest) -> MemoryProviderResult<MemoryStoreResult>;

    /// Update an existing record when the provider advertises update support.
    async fn update(
        &self,
        request: MemoryUpdateRequest,
    ) -> MemoryProviderResult<MemoryStoreResult> {
        Err(unsupported(
            self.name(),
            "update",
            &request.context.operation_id,
        ))
    }

    /// Delete an existing record when the provider advertises delete support.
    async fn delete(
        &self,
        request: MemoryDeleteRequest,
    ) -> MemoryProviderResult<MemoryDeleteResult> {
        Err(unsupported(
            self.name(),
            "delete",
            &request.context.operation_id,
        ))
    }

    /// Store multiple records when the provider advertises batch support.
    async fn batch_store(
        &self,
        request: MemoryBatchStoreRequest,
    ) -> MemoryProviderResult<MemoryBatchStoreResult> {
        Err(unsupported(
            self.name(),
            "batch_store",
            &request.context.operation_id,
        ))
    }

    /// Reflect or consolidate when the provider advertises maintenance support.
    async fn maintain(
        &self,
        request: MemoryMaintenanceRequest,
    ) -> MemoryProviderResult<MemoryMaintenanceResult> {
        Err(unsupported(
            self.name(),
            "maintenance",
            &request.context.operation_id,
        ))
    }

    /// Accept relevance or attribution feedback when advertised.
    async fn feedback(
        &self,
        request: MemoryFeedbackRequest,
    ) -> MemoryProviderResult<MemoryFeedbackResult> {
        Err(unsupported(
            self.name(),
            "feedback",
            &request.context.operation_id,
        ))
    }

    /// Report explicit provider health when advertised.
    async fn health(
        &self,
        request: MemoryHealthRequest,
    ) -> MemoryProviderResult<MemoryHealthResult> {
        Err(unsupported(
            self.name(),
            "health",
            &request.context.operation_id,
        ))
    }
}

fn unsupported(provider: &str, capability: &str, operation_id: &str) -> MemoryOperationError {
    MemoryOperationError::unsupported(capability)
        .with_operation_id(operation_id)
        .with_provider(provider)
}
