// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider-neutral memory operation data types.
//!
//! These types define the serializable contract shared by memory providers,
//! language bindings, and worker protocols. They contain no provider runtime,
//! storage implementation, or process-global state.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt::{self, Display, Formatter};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::Json;

/// Default maximum number of matches requested by convenience constructors.
pub const DEFAULT_SEARCH_LIMIT: usize = 10;

/// Largest result limit accepted by the shared contract.
pub const MAX_SEARCH_LIMIT: usize = 1_000;

/// Stable machine-readable memory operation error code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum MemoryErrorCode {
    /// A request violates the provider-neutral contract.
    InvalidRequest,
    /// The provider does not implement the requested optional capability.
    Unsupported,
    /// The operation did not complete before its deadline.
    DeadlineExceeded,
    /// A caller or runtime cancelled the operation.
    Cancelled,
    /// The request conflicts with existing state, such as an idempotency key.
    Conflict,
    /// The provider or its backing service is unavailable.
    ProviderUnavailable,
    /// The provider failed for an implementation-specific reason.
    Internal,
}

/// Serializable provider or contract failure.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct MemoryOperationError {
    /// Stable error classification.
    pub code: MemoryErrorCode,
    /// Human-readable diagnostic message.
    pub message: String,
    /// Whether a caller may safely retry according to the provider.
    #[serde(default)]
    pub retryable: bool,
    /// Correlation identifier from the operation context, when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operation_id: Option<String>,
    /// Provider that produced the failure, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// Provider-neutral structured diagnostic fields.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub details: BTreeMap<String, Json>,
}

impl MemoryOperationError {
    /// Create an invalid-request error.
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self::new(MemoryErrorCode::InvalidRequest, message, false)
    }

    /// Create an unsupported-capability error.
    pub fn unsupported(capability: impl Into<String>) -> Self {
        let capability = capability.into();
        Self::new(
            MemoryErrorCode::Unsupported,
            format!("memory provider does not support {capability}"),
            false,
        )
    }

    /// Create a deadline-exceeded error.
    pub fn deadline_exceeded(message: impl Into<String>) -> Self {
        Self::new(MemoryErrorCode::DeadlineExceeded, message, true)
    }

    /// Create an idempotency or mutation conflict.
    pub fn conflict(message: impl Into<String>) -> Self {
        Self::new(MemoryErrorCode::Conflict, message, false)
    }

    /// Attach an operation correlation identifier.
    pub fn with_operation_id(mut self, operation_id: impl Into<String>) -> Self {
        self.operation_id = Some(operation_id.into());
        self
    }

    /// Attach the provider that produced the error.
    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = Some(provider.into());
        self
    }

    /// Create an error with an explicit code and retry policy.
    pub fn new(code: MemoryErrorCode, message: impl Into<String>, retryable: bool) -> Self {
        Self {
            code,
            message: message.into(),
            retryable,
            operation_id: None,
            provider: None,
            details: BTreeMap::new(),
        }
    }
}

impl Display for MemoryOperationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for MemoryOperationError {}

/// Tenant and subject partition for a memory operation.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct MemoryNamespace {
    /// Tenant isolation identifier.
    pub tenant_id: String,
    /// Person, entity, or other durable memory subject.
    pub subject_id: String,
    /// Session that produced or contextualizes the memory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Agent that produced or contextualizes the memory.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
}

impl MemoryNamespace {
    /// Create and validate a tenant and subject namespace.
    pub fn new(
        tenant_id: impl Into<String>,
        subject_id: impl Into<String>,
    ) -> Result<Self, MemoryOperationError> {
        let namespace = Self {
            tenant_id: tenant_id.into(),
            subject_id: subject_id.into(),
            session_id: None,
            agent_id: None,
        };
        namespace.validate()?;
        Ok(namespace)
    }

    /// Validate required and optional namespace identifiers.
    pub fn validate(&self) -> Result<(), MemoryOperationError> {
        validate_identifier("tenant_id", &self.tenant_id)?;
        validate_identifier("subject_id", &self.subject_id)?;
        validate_optional_identifier("session_id", self.session_id.as_deref())?;
        validate_optional_identifier("agent_id", self.agent_id.as_deref())
    }
}

/// Namespace fields used to narrow a search within one tenant and subject.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum MemorySearchScope {
    /// Search the subject across its sessions and agents.
    #[default]
    Subject,
    /// Additionally require the namespace agent identifier.
    Agent,
    /// Additionally require the namespace session identifier.
    Session,
    /// Match all optional identifiers present in the query namespace.
    Exact,
}

impl MemorySearchScope {
    /// Validate that the namespace contains fields required by this scope.
    pub fn validate(self, namespace: &MemoryNamespace) -> Result<(), MemoryOperationError> {
        namespace.validate()?;
        match self {
            Self::Agent if namespace.agent_id.is_none() => Err(
                MemoryOperationError::invalid_request("agent scope requires agent_id"),
            ),
            Self::Session if namespace.session_id.is_none() => Err(
                MemoryOperationError::invalid_request("session scope requires session_id"),
            ),
            Self::Subject | Self::Agent | Self::Session | Self::Exact => Ok(()),
        }
    }
}

/// Structured memory content or an opaque provider reference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MemoryContent {
    /// Plain searchable text.
    Text {
        /// Text value.
        text: String,
    },
    /// Structured JSON content.
    Json {
        /// JSON value.
        value: Json,
    },
    /// Opaque provider or object-store reference.
    Reference {
        /// Reference value. Relay does not require URI semantics.
        reference: String,
        /// Optional searchable or display-safe preview.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        preview: Option<String>,
    },
}

impl MemoryContent {
    /// Validate nonempty text and reference fields.
    pub fn validate(&self) -> Result<(), MemoryOperationError> {
        match self {
            Self::Text { text } if text.trim().is_empty() => Err(
                MemoryOperationError::invalid_request("memory text must not be empty"),
            ),
            Self::Reference { reference, .. } if reference.trim().is_empty() => Err(
                MemoryOperationError::invalid_request("memory reference must not be empty"),
            ),
            Self::Text { .. } | Self::Json { .. } | Self::Reference { .. } => Ok(()),
        }
    }
}

/// Origin and derivation facts for one memory.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct MemoryProvenance {
    /// Provider-neutral source classification such as `user` or `derived`.
    pub source: String,
    /// Source event or external record identifiers.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub source_ids: Vec<String>,
    /// Parent memories used to derive this record.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parent_memory_ids: Vec<String>,
    /// Additional neutral provenance fields.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Json>,
}

impl MemoryProvenance {
    /// Validate that the source classification is present.
    pub fn validate(&self) -> Result<(), MemoryOperationError> {
        validate_identifier("provenance.source", &self.source)
    }
}

/// Durable provider-neutral memory record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct MemoryRecord {
    /// Stable provider-owned item identifier.
    pub id: String,
    /// Stable provider name.
    pub provider: String,
    /// Original namespace and origin context.
    pub namespace: MemoryNamespace,
    /// Stored content or opaque reference.
    pub content: MemoryContent,
    /// Time represented by the source event or content.
    pub event_timestamp: DateTime<Utc>,
    /// Time at which the provider ingested the record.
    pub ingested_at: DateTime<Utc>,
    /// Source and derivation facts.
    pub provenance: MemoryProvenance,
    /// Provider-neutral user or application metadata.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Json>,
    /// Lossless provider-specific metadata escape hatch.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub provider_metadata: BTreeMap<String, Json>,
}

/// Ranked result returned by a memory search.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct MemoryMatch {
    /// Durable matched record.
    pub record: MemoryRecord,
    /// Provider or reference-provider relevance score.
    pub score: f64,
    /// One-based rank after filtering and limiting.
    pub rank: usize,
}

/// Exact provider-neutral filters for a search.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct MemoryFilter {
    /// Exact metadata key/value requirements.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Json>,
    /// Earliest accepted source event timestamp, inclusive.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub event_after: Option<DateTime<Utc>>,
    /// Latest accepted source event timestamp, inclusive.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub event_before: Option<DateTime<Utc>>,
    /// Earliest accepted ingestion timestamp, inclusive.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ingested_after: Option<DateTime<Utc>>,
    /// Latest accepted ingestion timestamp, inclusive.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ingested_before: Option<DateTime<Utc>>,
}

impl MemoryFilter {
    /// Validate timestamp range ordering.
    pub fn validate(&self) -> Result<(), MemoryOperationError> {
        validate_range("event", self.event_after, self.event_before)?;
        validate_range("ingested", self.ingested_after, self.ingested_before)
    }
}

/// Correlation and deadline data shared by provider operations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct MemoryRequestContext {
    /// Caller-supplied operation correlation identifier.
    pub operation_id: String,
    /// Absolute UTC deadline propagated to local or remote providers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub deadline: Option<DateTime<Utc>>,
}

impl MemoryRequestContext {
    /// Create a context without a deadline.
    pub fn new(operation_id: impl Into<String>) -> Result<Self, MemoryOperationError> {
        let context = Self {
            operation_id: operation_id.into(),
            deadline: None,
        };
        context.validate()?;
        Ok(context)
    }

    /// Validate the correlation identifier.
    pub fn validate(&self) -> Result<(), MemoryOperationError> {
        validate_identifier("operation_id", &self.operation_id)
    }
}

/// Provider-neutral memory search request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct MemorySearchRequest {
    /// Correlation and deadline data.
    pub context: MemoryRequestContext,
    /// Tenant and subject partition plus optional origin context.
    pub namespace: MemoryNamespace,
    /// Natural-language or provider-interpreted query text.
    pub query: String,
    /// Namespace fields that narrow the subject partition.
    #[serde(default)]
    pub scope: MemorySearchScope,
    /// Provider-neutral exact filters.
    #[serde(default)]
    pub filter: MemoryFilter,
    /// Maximum number of returned matches.
    pub limit: usize,
}

impl MemorySearchRequest {
    /// Create a subject-scope search with the default result limit.
    pub fn new(
        context: MemoryRequestContext,
        namespace: MemoryNamespace,
        query: impl Into<String>,
    ) -> Result<Self, MemoryOperationError> {
        let request = Self {
            context,
            namespace,
            query: query.into(),
            scope: MemorySearchScope::Subject,
            filter: MemoryFilter::default(),
            limit: DEFAULT_SEARCH_LIMIT,
        };
        request.validate()?;
        Ok(request)
    }

    /// Validate identity, scope, query, filters, and result limit.
    pub fn validate(&self) -> Result<(), MemoryOperationError> {
        self.context.validate()?;
        self.scope.validate(&self.namespace)?;
        if self.query.trim().is_empty() {
            return Err(MemoryOperationError::invalid_request(
                "memory search query must not be empty",
            ));
        }
        if !(1..=MAX_SEARCH_LIMIT).contains(&self.limit) {
            return Err(MemoryOperationError::invalid_request(format!(
                "memory search limit must be in 1..={MAX_SEARCH_LIMIT}",
            )));
        }
        self.filter.validate()
    }
}

/// Successful matches plus non-fatal provider shard or source failures.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct MemorySearchResult {
    /// Ordered matches.
    #[serde(default)]
    pub matches: Vec<MemoryMatch>,
    /// Ordered failures that did not prevent all results.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub partial_errors: Vec<MemoryOperationError>,
}

/// Provider-neutral memory store request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct MemoryStoreRequest {
    /// Correlation and deadline data.
    pub context: MemoryRequestContext,
    /// Tenant, subject, and origin context.
    pub namespace: MemoryNamespace,
    /// Content or opaque reference to store.
    pub content: MemoryContent,
    /// Time represented by the source event or content.
    pub event_timestamp: DateTime<Utc>,
    /// Source and derivation facts.
    pub provenance: MemoryProvenance,
    /// Provider-neutral metadata.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Json>,
    /// Optional caller key for replay-safe storage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idempotency_key: Option<String>,
}

impl MemoryStoreRequest {
    /// Validate identity, content, provenance, and optional idempotency key.
    pub fn validate(&self) -> Result<(), MemoryOperationError> {
        self.context.validate()?;
        self.namespace.validate()?;
        self.content.validate()?;
        self.provenance.validate()?;
        validate_optional_identifier("idempotency_key", self.idempotency_key.as_deref())
    }
}

/// Outcome of an idempotent store operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum MemoryStoreDisposition {
    /// A new record was created.
    Created,
    /// The original record was returned for an idempotent replay.
    Existing,
}

/// Result returned by a successful store operation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct MemoryStoreResult {
    /// Created or replayed durable record.
    pub record: MemoryRecord,
    /// Whether the operation created or replayed a record.
    pub disposition: MemoryStoreDisposition,
}

/// Optional capabilities advertised by a provider.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct MemoryCapabilities {
    /// Provider can update existing records.
    #[serde(default)]
    pub update: bool,
    /// Provider can delete or forget existing records.
    #[serde(default)]
    pub delete: bool,
    /// Provider can store a batch with per-item results.
    #[serde(default)]
    pub batch_store: bool,
    /// Provider can reflect or consolidate memories.
    #[serde(default)]
    pub maintenance: bool,
    /// Provider accepts explicit relevance or attribution feedback.
    #[serde(default)]
    pub feedback: bool,
    /// Provider exposes an explicit health operation.
    #[serde(default)]
    pub health: bool,
}

/// Update request for providers that advertise update capability.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct MemoryUpdateRequest {
    /// Correlation and deadline data.
    pub context: MemoryRequestContext,
    /// Tenant and subject partition.
    pub namespace: MemoryNamespace,
    /// Stable provider item identifier.
    pub id: String,
    /// Replacement content, when changing content.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<MemoryContent>,
    /// Replacement neutral metadata, when changing metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<BTreeMap<String, Json>>,
}

/// Delete request for providers that advertise delete capability.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct MemoryDeleteRequest {
    /// Correlation and deadline data.
    pub context: MemoryRequestContext,
    /// Tenant and subject partition.
    pub namespace: MemoryNamespace,
    /// Stable provider item identifier.
    pub id: String,
}

/// Result of a provider delete operation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct MemoryDeleteResult {
    /// Stable provider item identifier.
    pub id: String,
    /// Whether a record was deleted.
    pub deleted: bool,
}

/// Batch store request for providers that advertise batch capability.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct MemoryBatchStoreRequest {
    /// Correlation and deadline for the batch.
    pub context: MemoryRequestContext,
    /// Individual store requests.
    pub requests: Vec<MemoryStoreRequest>,
}

/// Per-item batch results and partial failures.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct MemoryBatchStoreResult {
    /// Successful item results.
    #[serde(default)]
    pub results: Vec<MemoryStoreResult>,
    /// Item failures that did not abort the entire batch.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub partial_errors: Vec<MemoryOperationError>,
}

/// Provider maintenance operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum MemoryMaintenanceAction {
    /// Generate reflective or inferential memories.
    Reflect,
    /// Consolidate redundant or related memories.
    Consolidate,
}

/// Maintenance request for providers that advertise maintenance capability.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct MemoryMaintenanceRequest {
    /// Correlation and deadline data.
    pub context: MemoryRequestContext,
    /// Tenant and subject partition.
    pub namespace: MemoryNamespace,
    /// Requested maintenance action.
    pub action: MemoryMaintenanceAction,
    /// Provider-neutral maintenance parameters.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub parameters: BTreeMap<String, Json>,
}

/// Maintenance job or immediate derived-record result.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct MemoryMaintenanceResult {
    /// Provider job identifier for asynchronous maintenance.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,
    /// Immediately committed derived records.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub records: Vec<MemoryRecord>,
    /// Non-fatal provider failures.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub partial_errors: Vec<MemoryOperationError>,
}

/// Feedback classification for providers that accept relevance signals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum MemoryFeedbackKind {
    /// The response explicitly cited the record.
    Cited,
    /// An evaluator or user marked the record relevant.
    Relevant,
    /// An evaluator or user marked the record irrelevant.
    Irrelevant,
}

/// Feedback request for providers that advertise feedback capability.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct MemoryFeedbackRequest {
    /// Correlation and deadline data.
    pub context: MemoryRequestContext,
    /// Tenant and subject partition.
    pub namespace: MemoryNamespace,
    /// Stable provider item identifier.
    pub id: String,
    /// Feedback classification.
    pub kind: MemoryFeedbackKind,
    /// Optional confidence from zero to one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    /// Provider-neutral feedback details.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, Json>,
}

/// Result of accepted provider feedback.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct MemoryFeedbackResult {
    /// Whether the provider accepted the feedback.
    pub accepted: bool,
}

/// Health request for providers that advertise health capability.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct MemoryHealthRequest {
    /// Correlation and deadline data.
    pub context: MemoryRequestContext,
}

/// Provider health status.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct MemoryHealthResult {
    /// Whether the provider considers itself healthy.
    pub healthy: bool,
    /// Optional provider-safe diagnostic summary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

fn validate_identifier(name: &str, value: &str) -> Result<(), MemoryOperationError> {
    if value.trim().is_empty() {
        return Err(MemoryOperationError::invalid_request(format!(
            "{name} must not be empty",
        )));
    }
    Ok(())
}

fn validate_optional_identifier(
    name: &str,
    value: Option<&str>,
) -> Result<(), MemoryOperationError> {
    if let Some(value) = value {
        validate_identifier(name, value)?;
    }
    Ok(())
}

fn validate_range(
    name: &str,
    after: Option<DateTime<Utc>>,
    before: Option<DateTime<Utc>>,
) -> Result<(), MemoryOperationError> {
    if let (Some(after), Some(before)) = (after, before)
        && after > before
    {
        return Err(MemoryOperationError::invalid_request(format!(
            "{name}_after must not be later than {name}_before",
        )));
    }
    Ok(())
}
