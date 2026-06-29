// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Direct provider facade with validation and deadline enforcement.

use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use nemo_relay_types::memory::{
    MemoryBatchStoreRequest, MemoryBatchStoreResult, MemoryCapabilities, MemoryDeleteRequest,
    MemoryDeleteResult, MemoryFeedbackRequest, MemoryFeedbackResult, MemoryHealthRequest,
    MemoryHealthResult, MemoryMaintenanceRequest, MemoryMaintenanceResult, MemoryOperationError,
    MemoryRequestContext, MemorySearchRequest, MemorySearchResult, MemoryStoreRequest,
    MemoryStoreResult, MemoryUpdateRequest,
};

use crate::provider::{MemoryProvider, MemoryProviderResult};

/// Direct facade around one memory provider.
///
/// The facade validates required search and store requests, enforces the
/// absolute deadline carried by every operation context, and returns provider
/// results without retrying or rewriting them.
#[derive(Clone)]
pub struct MemoryRuntime {
    provider: Arc<dyn MemoryProvider>,
}

impl MemoryRuntime {
    /// Create a runtime from a concrete provider.
    pub fn new<P>(provider: P) -> Self
    where
        P: MemoryProvider + 'static,
    {
        Self::from_arc(Arc::new(provider))
    }

    /// Create a runtime from a shared provider trait object.
    pub fn from_arc(provider: Arc<dyn MemoryProvider>) -> Self {
        Self { provider }
    }

    /// Return the shared provider used by this runtime.
    pub fn provider(&self) -> &Arc<dyn MemoryProvider> {
        &self.provider
    }

    /// Return the provider's stable identifier.
    pub fn provider_name(&self) -> &str {
        self.provider.name()
    }

    /// Return optional operations advertised by the provider.
    pub fn capabilities(&self) -> MemoryCapabilities {
        self.provider.capabilities()
    }

    /// Validate and execute a memory search once.
    pub async fn search(
        &self,
        request: MemorySearchRequest,
    ) -> MemoryProviderResult<MemorySearchResult> {
        request.validate()?;
        let context = request.context.clone();
        self.execute(&context, self.provider.search(request)).await
    }

    /// Validate and execute a memory store once.
    pub async fn store(
        &self,
        request: MemoryStoreRequest,
    ) -> MemoryProviderResult<MemoryStoreResult> {
        request.validate()?;
        let context = request.context.clone();
        self.execute(&context, self.provider.store(request)).await
    }

    /// Execute an advertised update once, subject to its context deadline.
    pub async fn update(
        &self,
        request: MemoryUpdateRequest,
    ) -> MemoryProviderResult<MemoryStoreResult> {
        let context = request.context.clone();
        context.validate()?;
        request.namespace.validate()?;
        self.execute(&context, self.provider.update(request)).await
    }

    /// Execute an advertised delete once, subject to its context deadline.
    pub async fn delete(
        &self,
        request: MemoryDeleteRequest,
    ) -> MemoryProviderResult<MemoryDeleteResult> {
        let context = request.context.clone();
        context.validate()?;
        request.namespace.validate()?;
        self.execute(&context, self.provider.delete(request)).await
    }

    /// Execute an advertised batch store once, subject to its context deadline.
    pub async fn batch_store(
        &self,
        request: MemoryBatchStoreRequest,
    ) -> MemoryProviderResult<MemoryBatchStoreResult> {
        let context = request.context.clone();
        context.validate()?;
        for item in &request.requests {
            item.validate()?;
        }
        self.execute(&context, self.provider.batch_store(request))
            .await
    }

    /// Execute advertised maintenance once, subject to its context deadline.
    pub async fn maintain(
        &self,
        request: MemoryMaintenanceRequest,
    ) -> MemoryProviderResult<MemoryMaintenanceResult> {
        request.validate()?;
        let context = request.context.clone();
        self.execute(&context, self.provider.maintain(request))
            .await
    }

    /// Submit advertised feedback once, subject to its context deadline.
    pub async fn feedback(
        &self,
        request: MemoryFeedbackRequest,
    ) -> MemoryProviderResult<MemoryFeedbackResult> {
        let context = request.context.clone();
        context.validate()?;
        request.namespace.validate()?;
        self.execute(&context, self.provider.feedback(request))
            .await
    }

    /// Query advertised provider health once, subject to its context deadline.
    pub async fn health(
        &self,
        request: MemoryHealthRequest,
    ) -> MemoryProviderResult<MemoryHealthResult> {
        let context = request.context.clone();
        context.validate()?;
        self.execute(&context, self.provider.health(request)).await
    }

    async fn execute<T, F>(
        &self,
        context: &MemoryRequestContext,
        operation: F,
    ) -> MemoryProviderResult<T>
    where
        F: Future<Output = MemoryProviderResult<T>>,
    {
        let Some(remaining) = remaining_time(context, self.provider.name())? else {
            return operation.await;
        };

        match tokio::time::timeout(remaining, operation).await {
            Ok(result) => result,
            Err(_) => Err(deadline_error(context, self.provider.name())),
        }
    }
}

fn remaining_time(
    context: &MemoryRequestContext,
    provider: &str,
) -> Result<Option<Duration>, MemoryOperationError> {
    let Some(deadline) = context.deadline.as_ref() else {
        return Ok(None);
    };
    let now_millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let deadline_millis = i128::from(deadline.timestamp_millis());
    let remaining_millis = deadline_millis - i128::try_from(now_millis).unwrap_or(i128::MAX);
    if remaining_millis <= 0 {
        return Err(deadline_error(context, provider));
    }
    let remaining_millis = u64::try_from(remaining_millis).unwrap_or(u64::MAX);
    Ok(Some(Duration::from_millis(remaining_millis)))
}

fn deadline_error(context: &MemoryRequestContext, provider: &str) -> MemoryOperationError {
    MemoryOperationError::deadline_exceeded("memory operation deadline exceeded")
        .with_operation_id(&context.operation_id)
        .with_provider(provider)
}
