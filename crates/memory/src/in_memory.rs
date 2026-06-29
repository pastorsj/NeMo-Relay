// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic, instance-local reference memory provider.
//!
//! Search tokenizes case-normalized alphanumeric words, removes duplicates,
//! and scores each record as the fraction of unique query tokens present in the
//! searchable record content. Zero-overlap records are omitted. Ties sort by
//! stable record ID before one-based ranks are assigned.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::SystemTime;

use async_trait::async_trait;
use nemo_relay_types::memory::{
    MemoryContent, MemoryErrorCode, MemoryMatch, MemoryNamespace, MemoryOperationError,
    MemoryRecord, MemorySearchRequest, MemorySearchResult, MemorySearchScope,
    MemoryStoreDisposition, MemoryStoreRequest, MemoryStoreResult,
};
use tokio::sync::Mutex;

use crate::provider::{MemoryProvider, MemoryProviderResult};

/// Network-free reference provider with deterministic search and storage.
///
/// State is private to each provider instance. Clones share that instance's
/// state through an async mutex; no process-global registry or background task
/// is created.
#[derive(Clone, Default)]
pub struct InMemoryProvider {
    state: Arc<Mutex<InMemoryState>>,
}

#[derive(Default)]
struct InMemoryState {
    next_id: u64,
    records: Vec<MemoryRecord>,
    idempotency: BTreeMap<(MemoryNamespace, String), IdempotencyEntry>,
}

struct IdempotencyEntry {
    fingerprint: Vec<u8>,
    record: MemoryRecord,
}

impl InMemoryProvider {
    /// Create an empty provider instance.
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl MemoryProvider for InMemoryProvider {
    fn name(&self) -> &str {
        "in_memory"
    }

    async fn search(
        &self,
        request: MemorySearchRequest,
    ) -> MemoryProviderResult<MemorySearchResult> {
        request.validate()?;
        let query_tokens = tokenize(&request.query);
        let records = self.state.lock().await.records.clone();
        let mut matches = records
            .into_iter()
            .filter(|record| namespace_matches(record, &request))
            .filter(|record| filter_matches(record, &request))
            .filter_map(|record| {
                let score = overlap_score(&query_tokens, &searchable_tokens(&record.content));
                (score > 0.0).then_some(MemoryMatch {
                    record,
                    score,
                    rank: 0,
                })
            })
            .collect::<Vec<_>>();

        matches.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.record.id.cmp(&right.record.id))
        });
        matches.truncate(request.limit);
        for (index, memory_match) in matches.iter_mut().enumerate() {
            memory_match.rank = index + 1;
        }

        Ok(MemorySearchResult {
            matches,
            partial_errors: vec![],
        })
    }

    async fn store(&self, request: MemoryStoreRequest) -> MemoryProviderResult<MemoryStoreResult> {
        request.validate()?;
        let fingerprint = fingerprint(&request)?;
        let idempotency = request
            .idempotency_key
            .as_ref()
            .map(|key| (request.namespace.clone(), key.clone()));
        let mut state = self.state.lock().await;

        if let Some(key) = idempotency.as_ref()
            && let Some(existing) = state.idempotency.get(key)
        {
            if existing.fingerprint == fingerprint {
                return Ok(MemoryStoreResult {
                    record: existing.record.clone(),
                    disposition: MemoryStoreDisposition::Existing,
                });
            }
            return Err(MemoryOperationError::conflict(
                "idempotency key was already used with a different memory payload",
            )
            .with_operation_id(&request.context.operation_id)
            .with_provider(self.name()));
        }

        let sequence = state.next_id.checked_add(1).ok_or_else(|| {
            MemoryOperationError::new(
                MemoryErrorCode::Internal,
                "in-memory provider exhausted its record ID space",
                false,
            )
            .with_operation_id(&request.context.operation_id)
            .with_provider(self.name())
        })?;
        let record = MemoryRecord {
            id: format!("memory-{sequence:016}"),
            provider: self.name().to_string(),
            namespace: request.namespace,
            content: request.content,
            event_timestamp: request.event_timestamp,
            ingested_at: SystemTime::now().into(),
            provenance: request.provenance,
            metadata: request.metadata,
            provider_metadata: BTreeMap::new(),
        };

        state.next_id = sequence;
        state.records.push(record.clone());
        if let Some(key) = idempotency {
            state.idempotency.insert(
                key,
                IdempotencyEntry {
                    fingerprint,
                    record: record.clone(),
                },
            );
        }

        Ok(MemoryStoreResult {
            record,
            disposition: MemoryStoreDisposition::Created,
        })
    }
}

fn namespace_matches(record: &MemoryRecord, request: &MemorySearchRequest) -> bool {
    let record_namespace = &record.namespace;
    let query_namespace = &request.namespace;
    if record_namespace.tenant_id != query_namespace.tenant_id
        || record_namespace.subject_id != query_namespace.subject_id
    {
        return false;
    }

    match request.scope {
        MemorySearchScope::Subject => true,
        MemorySearchScope::Agent => record_namespace.agent_id == query_namespace.agent_id,
        MemorySearchScope::Session => record_namespace.session_id == query_namespace.session_id,
        MemorySearchScope::Exact => {
            query_namespace
                .session_id
                .as_ref()
                .is_none_or(|session| record_namespace.session_id.as_ref() == Some(session))
                && query_namespace
                    .agent_id
                    .as_ref()
                    .is_none_or(|agent| record_namespace.agent_id.as_ref() == Some(agent))
        }
    }
}

fn filter_matches(record: &MemoryRecord, request: &MemorySearchRequest) -> bool {
    let filter = &request.filter;
    filter
        .metadata
        .iter()
        .all(|(key, value)| record.metadata.get(key) == Some(value))
        && filter
            .event_after
            .is_none_or(|after| record.event_timestamp >= after)
        && filter
            .event_before
            .is_none_or(|before| record.event_timestamp <= before)
        && filter
            .ingested_after
            .is_none_or(|after| record.ingested_at >= after)
        && filter
            .ingested_before
            .is_none_or(|before| record.ingested_at <= before)
}

fn searchable_tokens(content: &MemoryContent) -> BTreeSet<String> {
    match content {
        MemoryContent::Text { text } => tokenize(text),
        MemoryContent::Json { value } => tokenize(&value.to_string()),
        MemoryContent::Reference { reference, preview } => {
            tokenize(preview.as_deref().unwrap_or(reference))
        }
    }
}

fn tokenize(value: &str) -> BTreeSet<String> {
    value
        .split(|character: char| !character.is_alphanumeric())
        .filter(|token| !token.is_empty())
        .map(str::to_lowercase)
        .collect()
}

fn overlap_score(query: &BTreeSet<String>, content: &BTreeSet<String>) -> f64 {
    if query.is_empty() {
        return 0.0;
    }
    query.intersection(content).count() as f64 / query.len() as f64
}

fn fingerprint(request: &MemoryStoreRequest) -> MemoryProviderResult<Vec<u8>> {
    serde_json::to_vec(&(
        &request.content,
        &request.event_timestamp,
        &request.provenance,
        &request.metadata,
    ))
    .map_err(|error| {
        MemoryOperationError::new(
            MemoryErrorCode::Internal,
            format!("failed to fingerprint memory payload: {error}"),
            false,
        )
        .with_operation_id(&request.context.operation_id)
        .with_provider("in_memory")
    })
}
