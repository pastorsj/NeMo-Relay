// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reusable provider conformance coverage.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use nemo_relay_types::memory::{
    MemoryBatchStoreRequest, MemoryContent, MemoryDeleteRequest, MemoryErrorCode,
    MemoryFeedbackKind, MemoryFeedbackRequest, MemoryFilter, MemoryHealthRequest,
    MemoryMaintenanceAction, MemoryMaintenanceRequest, MemoryNamespace, MemoryOperationError,
    MemoryProvenance, MemoryRequestContext, MemorySearchRequest, MemorySearchResult,
    MemorySearchScope, MemoryStoreDisposition, MemoryStoreRequest, MemoryStoreResult,
    MemoryUpdateRequest,
};

use crate::provider::{MemoryProvider, MemoryProviderResult};
use crate::runtime::MemoryRuntime;

/// Result of one named provider conformance case.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConformanceCase {
    /// Stable case name suitable for test and CI output.
    pub name: &'static str,
    /// Whether every assertion in the case passed.
    pub passed: bool,
    /// Diagnostic summary when the case failed.
    pub message: Option<String>,
}

/// Structured result of one provider conformance run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConformanceReport {
    /// Provider identifier reported by the implementation.
    pub provider: String,
    /// Caller-supplied identifier used to isolate synthetic test data.
    pub run_id: String,
    /// Named cases in execution order.
    pub cases: Vec<ConformanceCase>,
}

impl ConformanceReport {
    /// Return `true` when every conformance case passed.
    pub fn passed(&self) -> bool {
        self.cases.iter().all(|case| case.passed)
    }

    /// Return only failed cases.
    pub fn failures(&self) -> impl Iterator<Item = &ConformanceCase> {
        self.cases.iter().filter(|case| !case.passed)
    }

    fn record(&mut self, name: &'static str, result: Result<(), String>) {
        match result {
            Ok(()) => self.cases.push(ConformanceCase {
                name,
                passed: true,
                message: None,
            }),
            Err(message) => self.cases.push(ConformanceCase {
                name,
                passed: false,
                message: Some(message),
            }),
        }
    }
}

/// Run provider-neutral acceptance cases against one isolated provider namespace.
///
/// `run_id` must be unique for each invocation against a durable or shared
/// backend. The harness reports failures as named data instead of panicking.
/// It tests required store/search behavior, record preservation, namespace
/// isolation, deterministic order, idempotency, optional capability agreement,
/// and direct-runtime deadline/partial-result invariants.
pub async fn run_provider_conformance(
    provider: Arc<dyn MemoryProvider>,
    run_id: impl Into<String>,
) -> ConformanceReport {
    let run_id = run_id.into();
    let mut report = ConformanceReport {
        provider: provider.name().to_string(),
        run_id: run_id.clone(),
        cases: vec![],
    };
    if run_id.trim().is_empty() {
        report.record("valid_run_id", Err("run_id must not be empty".to_string()));
        return report;
    }

    let runtime = MemoryRuntime::from_arc(provider);
    let origin = namespace(&run_id, "subject", "session-a", "assistant");
    let cross_session = namespace(&run_id, "subject", "session-b", "assistant");
    let stored_request = store_request(
        &run_id,
        "required-store",
        origin.clone(),
        "conformance solarized editor preference",
        Some("required-store-key"),
    );
    let stored = runtime.store(stored_request.clone()).await;
    report.record(
        "required_store",
        stored
            .as_ref()
            .map(|result| {
                require(
                    result.disposition == MemoryStoreDisposition::Created,
                    "initial store did not report created",
                )
            })
            .unwrap_or_else(|error| Err(error.to_string())),
    );

    let Some(stored) = stored.ok() else {
        for name in [
            "required_search",
            "record_preservation",
            "subject_isolation_cross_session",
            "deterministic_order",
            "idempotency",
            "capability_agreement",
        ] {
            report.record(name, Err("required seed store failed".to_string()));
        }
        record_runtime_invariants(&mut report).await;
        return report;
    };

    let search = runtime
        .search(search_request(
            &run_id,
            "required-search",
            cross_session.clone(),
            "solarized editor preference",
        ))
        .await;
    report.record(
        "required_search",
        search
            .as_ref()
            .map_err(ToString::to_string)
            .and_then(|result| require(!result.matches.is_empty(), "search returned no matches")),
    );
    report.record(
        "record_preservation",
        search
            .as_ref()
            .map_err(ToString::to_string)
            .and_then(|result| {
                result
                    .matches
                    .iter()
                    .find(|memory_match| memory_match.record.id == stored.record.id)
                    .ok_or_else(|| "stored record was not returned".to_string())
                    .and_then(|memory_match| {
                        require(
                            memory_match.record == stored.record
                                && memory_match.rank > 0
                                && memory_match.score > 0.0,
                            "record fields, rank, or score changed",
                        )
                    })
            }),
    );

    let isolated = runtime
        .search(search_request(
            &run_id,
            "isolated-search",
            namespace(&run_id, "other-subject", "session-b", "assistant"),
            "solarized editor preference",
        ))
        .await;
    report.record(
        "subject_isolation_cross_session",
        search
            .as_ref()
            .map_err(ToString::to_string)
            .and_then(|result| {
                require(
                    result
                        .matches
                        .iter()
                        .any(|memory_match| memory_match.record.id == stored.record.id),
                    "subject-scope search did not cross sessions",
                )
            })
            .and_then(|()| {
                isolated
                    .as_ref()
                    .map_err(ToString::to_string)
                    .and_then(|result| {
                        require(result.matches.is_empty(), "search crossed subject boundary")
                    })
            }),
    );

    let deterministic_request = search_request(
        &run_id,
        "deterministic-search",
        cross_session,
        "solarized editor preference",
    );
    let deterministic_first = runtime.search(deterministic_request.clone()).await;
    let deterministic_second = runtime.search(deterministic_request).await;
    report.record(
        "deterministic_order",
        match (deterministic_first, deterministic_second) {
            (Ok(first), Ok(second)) => require(first == second, "repeated search changed order"),
            (Err(error), _) | (_, Err(error)) => Err(error.to_string()),
        },
    );

    let mut replay_request = stored_request.clone();
    replay_request.context = context(&run_id, "idempotent-replay");
    let replay = runtime.store(replay_request).await;
    let mut conflict_request = stored_request;
    conflict_request.context = context(&run_id, "idempotent-conflict");
    conflict_request.content = MemoryContent::Text {
        text: "different conformance payload".to_string(),
    };
    let conflict = runtime.store(conflict_request).await;
    report.record(
        "idempotency",
        replay
            .as_ref()
            .map_err(ToString::to_string)
            .and_then(|result| {
                require(
                    result.disposition == MemoryStoreDisposition::Existing
                        && result.record == stored.record,
                    "idempotent replay did not return the original record",
                )
            })
            .and_then(|()| match conflict {
                Err(error) if error.code == MemoryErrorCode::Conflict => Ok(()),
                Err(error) => Err(format!("expected conflict, got {:?}", error.code)),
                Ok(_) => Err("conflicting replay unexpectedly succeeded".to_string()),
            }),
    );

    report.record(
        "capability_agreement",
        check_capability_agreement(&runtime, &run_id, origin, &stored.record.id).await,
    );
    record_runtime_invariants(&mut report).await;
    report
}

async fn check_capability_agreement(
    runtime: &MemoryRuntime,
    run_id: &str,
    namespace: MemoryNamespace,
    record_id: &str,
) -> Result<(), String> {
    let capabilities = runtime.capabilities();
    let update = runtime
        .update(MemoryUpdateRequest {
            context: context(run_id, "capability-update"),
            namespace: namespace.clone(),
            id: record_id.to_string(),
            content: None,
            metadata: Some(BTreeMap::new()),
        })
        .await
        .map(|_| ());
    agreement("update", capabilities.update, update)?;

    let batch = runtime
        .batch_store(MemoryBatchStoreRequest {
            context: context(run_id, "capability-batch"),
            requests: vec![store_request(
                run_id,
                "capability-batch-item",
                namespace.clone(),
                "conformance batch item",
                Some("capability-batch-key"),
            )],
        })
        .await
        .map(|_| ());
    agreement("batch_store", capabilities.batch_store, batch)?;

    let maintenance = runtime
        .maintain(MemoryMaintenanceRequest {
            context: context(run_id, "capability-maintenance"),
            namespace: namespace.clone(),
            action: MemoryMaintenanceAction::Reflect,
            parameters: BTreeMap::new(),
        })
        .await
        .map(|_| ());
    agreement("maintenance", capabilities.maintenance, maintenance)?;

    let feedback = runtime
        .feedback(MemoryFeedbackRequest {
            context: context(run_id, "capability-feedback"),
            namespace: namespace.clone(),
            id: record_id.to_string(),
            kind: MemoryFeedbackKind::Relevant,
            confidence: Some(1.0),
            metadata: BTreeMap::new(),
        })
        .await
        .map(|_| ());
    agreement("feedback", capabilities.feedback, feedback)?;

    let health = runtime
        .health(MemoryHealthRequest {
            context: context(run_id, "capability-health"),
        })
        .await
        .map(|_| ());
    agreement("health", capabilities.health, health)?;

    let delete = runtime
        .delete(MemoryDeleteRequest {
            context: context(run_id, "capability-delete"),
            namespace,
            id: record_id.to_string(),
        })
        .await
        .map(|_| ());
    agreement("delete", capabilities.delete, delete)
}

fn agreement(
    capability: &str,
    advertised: bool,
    result: Result<(), MemoryOperationError>,
) -> Result<(), String> {
    match (advertised, result) {
        (false, Err(error)) if error.code == MemoryErrorCode::Unsupported => Ok(()),
        (false, Err(error)) => Err(format!(
            "{capability} is disabled but returned {:?}",
            error.code
        )),
        (false, Ok(())) => Err(format!("{capability} is disabled but succeeded")),
        (true, Err(error)) if error.code == MemoryErrorCode::Unsupported => Err(format!(
            "{capability} is advertised but returned unsupported"
        )),
        (true, Err(_) | Ok(())) => Ok(()),
    }
}

async fn record_runtime_invariants(report: &mut ConformanceReport) {
    let partial_error = MemoryOperationError::new(
        MemoryErrorCode::ProviderUnavailable,
        "synthetic partial failure",
        true,
    );
    let partial_result = MemorySearchResult {
        matches: vec![],
        partial_errors: vec![partial_error],
    };
    let partial_runtime = MemoryRuntime::new(RuntimeProbe {
        response: partial_result.clone(),
        block: false,
        calls: Arc::new(AtomicUsize::new(0)),
        dropped: Arc::new(AtomicBool::new(false)),
    });
    report.record(
        "partial_result_preservation",
        partial_runtime
            .search(search_request(
                "runtime",
                "partial",
                namespace("runtime", "subject", "session", "agent"),
                "synthetic query",
            ))
            .await
            .map_err(|error| error.to_string())
            .and_then(|actual| require(actual == partial_result, "partial result changed")),
    );

    let calls = Arc::new(AtomicUsize::new(0));
    let dropped = Arc::new(AtomicBool::new(false));
    let deadline_runtime = MemoryRuntime::new(RuntimeProbe {
        response: MemorySearchResult::default(),
        block: true,
        calls: calls.clone(),
        dropped: dropped.clone(),
    });
    let mut deadline_request = search_request(
        "runtime",
        "deadline",
        namespace("runtime", "subject", "session", "agent"),
        "synthetic query",
    );
    deadline_request.context.deadline =
        Some((SystemTime::now() + Duration::from_millis(50)).into());
    let deadline_result = deadline_runtime.search(deadline_request).await;
    report.record(
        "deadline_cancellation",
        match deadline_result {
            Err(error) => require(
                error.code == MemoryErrorCode::DeadlineExceeded
                    && calls.load(Ordering::SeqCst) == 1
                    && dropped.load(Ordering::SeqCst),
                "deadline did not drop exactly one provider future",
            ),
            Ok(_) => Err("blocking provider completed before deadline".to_string()),
        },
    );
}

struct RuntimeProbe {
    response: MemorySearchResult,
    block: bool,
    calls: Arc<AtomicUsize>,
    dropped: Arc<AtomicBool>,
}

#[async_trait]
impl MemoryProvider for RuntimeProbe {
    fn name(&self) -> &str {
        "conformance_runtime_probe"
    }

    async fn search(
        &self,
        _request: MemorySearchRequest,
    ) -> MemoryProviderResult<MemorySearchResult> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.block {
            let _guard = DropFlag(self.dropped.clone());
            std::future::pending().await
        } else {
            Ok(self.response.clone())
        }
    }

    async fn store(&self, _request: MemoryStoreRequest) -> MemoryProviderResult<MemoryStoreResult> {
        Err(MemoryOperationError::new(
            MemoryErrorCode::Internal,
            "runtime probe does not store",
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

fn namespace(run_id: &str, subject: &str, session: &str, agent: &str) -> MemoryNamespace {
    MemoryNamespace {
        tenant_id: format!("conformance-{run_id}"),
        subject_id: subject.to_string(),
        session_id: Some(session.to_string()),
        agent_id: Some(agent.to_string()),
    }
}

fn context(run_id: &str, operation: &str) -> MemoryRequestContext {
    MemoryRequestContext {
        operation_id: format!("{run_id}-{operation}"),
        deadline: None,
    }
}

fn store_request(
    run_id: &str,
    operation: &str,
    namespace: MemoryNamespace,
    text: &str,
    idempotency_key: Option<&str>,
) -> MemoryStoreRequest {
    MemoryStoreRequest {
        context: context(run_id, operation),
        namespace,
        content: MemoryContent::Text {
            text: text.to_string(),
        },
        event_timestamp: (SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000)).into(),
        provenance: MemoryProvenance {
            source: "conformance".to_string(),
            source_ids: vec![format!("source-{run_id}-{operation}")],
            ..MemoryProvenance::default()
        },
        metadata: BTreeMap::from([("suite".to_string(), "memory".into())]),
        idempotency_key: idempotency_key.map(|key| format!("{run_id}-{key}")),
    }
}

fn search_request(
    run_id: &str,
    operation: &str,
    namespace: MemoryNamespace,
    query: &str,
) -> MemorySearchRequest {
    MemorySearchRequest {
        context: context(run_id, operation),
        namespace,
        query: query.to_string(),
        scope: MemorySearchScope::Subject,
        filter: MemoryFilter::default(),
        limit: 10,
    }
}

fn require(condition: bool, message: &str) -> Result<(), String> {
    condition.then_some(()).ok_or_else(|| message.to_string())
}
