// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic bounded queue, retry, and lifecycle tests.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use nemo_relay_memory::memory::{
    MemoryContent, MemoryErrorCode, MemoryFilter, MemoryMaintenanceAction,
    MemoryMaintenanceRequest, MemoryMaintenanceResult, MemoryMaintenanceWindow, MemoryNamespace,
    MemoryOperationError, MemoryProvenance, MemoryRequestContext, MemorySearchRequest,
    MemorySearchResult, MemorySearchScope, MemoryStoreRequest, MemoryStoreResult,
};
use nemo_relay_memory::{
    InMemoryProvider, MemoryBackpressurePolicy, MemoryJobState, MemoryMaintainer,
    MemoryMaintainerResult, MemoryProvider, MemoryProviderResult, MemoryRuntime,
    MemoryWorkObserver, MemoryWorkQueue, MemoryWorkQueueConfig, MemoryWorkTransition,
};
use tokio::sync::{Barrier, Semaphore};

fn namespace(subject: &str) -> MemoryNamespace {
    MemoryNamespace {
        tenant_id: "tenant-a".to_string(),
        subject_id: subject.to_string(),
        session_id: Some("session-a".to_string()),
        agent_id: Some("agent-a".to_string()),
    }
}

fn store_request(job_id: &str, text: &str) -> MemoryStoreRequest {
    MemoryStoreRequest {
        context: MemoryRequestContext::new(job_id).expect("valid context"),
        namespace: namespace("subject-a"),
        content: MemoryContent::Text {
            text: text.to_string(),
        },
        event_timestamp: serde_json::from_str("\"2026-06-29T12:00:00Z\"").expect("valid timestamp"),
        provenance: MemoryProvenance {
            source: "test".to_string(),
            source_ids: vec![job_id.to_string()],
            parent_memory_ids: vec![],
            metadata: BTreeMap::new(),
        },
        metadata: BTreeMap::new(),
        idempotency_key: None,
    }
}

fn queue_config(capacity: usize) -> MemoryWorkQueueConfig {
    MemoryWorkQueueConfig {
        capacity,
        max_attempts: 1,
        attempt_timeout_millis: 10_000,
        ..MemoryWorkQueueConfig::default()
    }
}

#[derive(Clone)]
struct GateProvider {
    inner: InMemoryProvider,
    gate: Arc<Semaphore>,
    calls: Arc<AtomicUsize>,
}

impl GateProvider {
    fn new() -> Self {
        Self {
            inner: InMemoryProvider::new(),
            gate: Arc::new(Semaphore::new(0)),
            calls: Arc::new(AtomicUsize::new(0)),
        }
    }
}

#[async_trait]
impl MemoryProvider for GateProvider {
    fn name(&self) -> &str {
        "gate"
    }

    async fn search(
        &self,
        request: MemorySearchRequest,
    ) -> MemoryProviderResult<MemorySearchResult> {
        self.inner.search(request).await
    }

    async fn store(&self, request: MemoryStoreRequest) -> MemoryProviderResult<MemoryStoreResult> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.gate
            .acquire()
            .await
            .expect("test semaphore stays open")
            .forget();
        self.inner.store(request).await
    }
}

#[derive(Clone)]
struct FlakyProvider {
    inner: InMemoryProvider,
    failures_left: Arc<AtomicUsize>,
    calls: Arc<AtomicUsize>,
    retryable: bool,
}

#[async_trait]
impl MemoryProvider for FlakyProvider {
    fn name(&self) -> &str {
        "flaky"
    }

    async fn search(
        &self,
        request: MemorySearchRequest,
    ) -> MemoryProviderResult<MemorySearchResult> {
        self.inner.search(request).await
    }

    async fn store(&self, request: MemoryStoreRequest) -> MemoryProviderResult<MemoryStoreResult> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self
            .failures_left
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |value| {
                value.checked_sub(1)
            })
            .is_ok()
        {
            return Err(MemoryOperationError::new(
                if self.retryable {
                    MemoryErrorCode::ProviderUnavailable
                } else {
                    MemoryErrorCode::InvalidRequest
                },
                "controlled failure",
                self.retryable,
            )
            .with_operation_id(request.context.operation_id)
            .with_provider(self.name()));
        }
        self.inner.store(request).await
    }
}

#[derive(Default)]
struct RecordingObserver {
    transitions: Mutex<Vec<MemoryWorkTransition>>,
}

impl MemoryWorkObserver for RecordingObserver {
    fn on_transition(&self, transition: &MemoryWorkTransition) {
        self.transitions
            .lock()
            .expect("observer lock")
            .push(transition.clone());
    }
}

async fn wait_for_state(queue: &MemoryWorkQueue, job_id: &str, state: MemoryJobState) {
    for _ in 0..100 {
        if queue.job(job_id).is_some_and(|job| job.state == state) {
            return;
        }
        tokio::task::yield_now().await;
    }
    panic!("job {job_id} did not reach {state:?}");
}

async fn wait_for_attempt_state(
    queue: &MemoryWorkQueue,
    job_id: &str,
    attempts: u32,
    state: MemoryJobState,
) {
    for _ in 0..100 {
        if queue
            .job(job_id)
            .is_some_and(|job| job.attempts == attempts && job.state == state)
        {
            return;
        }
        tokio::task::yield_now().await;
    }
    panic!("job {job_id} attempt {attempts} did not reach {state:?}");
}

#[test]
fn queue_config_rejects_unbounded_or_incoherent_values() {
    let invalid = [
        MemoryWorkQueueConfig {
            capacity: 0,
            ..MemoryWorkQueueConfig::default()
        },
        MemoryWorkQueueConfig {
            backpressure: MemoryBackpressurePolicy::Wait,
            enqueue_timeout_millis: 0,
            ..MemoryWorkQueueConfig::default()
        },
        MemoryWorkQueueConfig {
            max_attempts: 0,
            ..MemoryWorkQueueConfig::default()
        },
        MemoryWorkQueueConfig {
            attempt_timeout_millis: 0,
            ..MemoryWorkQueueConfig::default()
        },
        MemoryWorkQueueConfig {
            retry_initial_delay_millis: 2,
            retry_max_delay_millis: 1,
            ..MemoryWorkQueueConfig::default()
        },
        MemoryWorkQueueConfig {
            terminal_history_capacity: 0,
            ..MemoryWorkQueueConfig::default()
        },
    ];
    for config in invalid {
        assert_eq!(
            config.validate().expect_err("config must fail").code,
            MemoryErrorCode::InvalidRequest
        );
    }
}

#[test]
fn queue_construction_is_lazy_and_does_not_require_a_runtime() {
    let queue = MemoryWorkQueue::new(MemoryRuntime::new(InMemoryProvider::new()), queue_config(1))
        .expect("queue construction is synchronous");
    let snapshot = queue.snapshot();
    assert!(snapshot.accepting);
    assert_eq!(snapshot.accepted_total, 0);
    assert_eq!(snapshot.queued, 0);
    assert_eq!(snapshot.running, 0);
}

#[tokio::test]
async fn first_submission_and_shutdown_race_cannot_restart_or_hang_the_queue() {
    for index in 0..50 {
        let queue =
            MemoryWorkQueue::new(MemoryRuntime::new(InMemoryProvider::new()), queue_config(1))
                .expect("valid queue");
        let barrier = Arc::new(Barrier::new(3));
        let submit_queue = queue.clone();
        let submit_barrier = barrier.clone();
        let submit = tokio::spawn(async move {
            submit_barrier.wait().await;
            submit_queue
                .submit_store(store_request(&format!("race-{index}"), "raced"))
                .await
        });
        let shutdown_queue = queue.clone();
        let shutdown_barrier = barrier.clone();
        let shutdown = tokio::spawn(async move {
            shutdown_barrier.wait().await;
            shutdown_queue.shutdown(Duration::from_secs(1)).await
        });
        barrier.wait().await;

        let (submitted, shut_down) = tokio::time::timeout(Duration::from_secs(1), async {
            (
                submit.await.expect("submit task joins"),
                shutdown.await.expect("shutdown task joins"),
            )
        })
        .await
        .expect("race must not deadlock");
        shut_down.expect("shutdown drains any accepted job");
        if let Err(error) = submitted {
            assert_eq!(error.code, MemoryErrorCode::Cancelled);
        }
        assert!(!queue.snapshot().accepting);
    }
}

#[tokio::test]
async fn reject_backpressure_never_exceeds_pending_capacity() {
    let provider = GateProvider::new();
    let queue = MemoryWorkQueue::new(MemoryRuntime::new(provider.clone()), queue_config(1))
        .expect("valid queue");
    queue
        .submit_store(store_request("job-1", "first"))
        .await
        .expect("first accepted");
    wait_for_state(&queue, "job-1", MemoryJobState::Running).await;
    queue
        .submit_store(store_request("job-2", "second"))
        .await
        .expect("second fills pending capacity");
    let error = queue
        .submit_store(store_request("job-3", "third"))
        .await
        .expect_err("third must be rejected");
    assert_eq!(error.code, MemoryErrorCode::ProviderUnavailable);
    assert!(error.retryable);
    let snapshot = queue.snapshot();
    assert_eq!(snapshot.running, 1);
    assert_eq!(snapshot.queued, 1);
    assert_eq!(snapshot.rejected_total, 1);
    assert_eq!(snapshot.accepted_total, 2);

    provider.gate.add_permits(2);
    queue
        .shutdown(Duration::from_secs(1))
        .await
        .expect("queue drains");
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
    assert_eq!(queue.snapshot().succeeded_total, 2);
}

#[tokio::test]
async fn identical_job_is_deduplicated_and_changed_payload_conflicts() {
    let provider = GateProvider::new();
    let queue = MemoryWorkQueue::new(MemoryRuntime::new(provider.clone()), queue_config(2))
        .expect("valid queue");
    let request = store_request("same-job", "original");
    let first = queue
        .submit_store(request.clone())
        .await
        .expect("first accepted");
    wait_for_state(&queue, "same-job", MemoryJobState::Running).await;
    let duplicate = queue
        .submit_store(request)
        .await
        .expect("identical replay returns retained receipt");
    assert_eq!(duplicate.sequence, first.sequence);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);

    let conflict = queue
        .submit_store(store_request("same-job", "changed"))
        .await
        .expect_err("changed payload must conflict");
    assert_eq!(conflict.code, MemoryErrorCode::Conflict);

    provider.gate.add_permits(1);
    queue
        .shutdown(Duration::from_secs(1))
        .await
        .expect("queue drains");
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test(start_paused = true)]
async fn wait_backpressure_has_a_deterministic_admission_deadline() {
    let provider = GateProvider::new();
    let config = MemoryWorkQueueConfig {
        capacity: 1,
        backpressure: MemoryBackpressurePolicy::Wait,
        enqueue_timeout_millis: 100,
        max_attempts: 1,
        attempt_timeout_millis: 10_000,
        ..MemoryWorkQueueConfig::default()
    };
    let queue =
        MemoryWorkQueue::new(MemoryRuntime::new(provider.clone()), config).expect("valid queue");
    queue
        .submit_store(store_request("wait-1", "first"))
        .await
        .expect("first accepted");
    wait_for_state(&queue, "wait-1", MemoryJobState::Running).await;
    queue
        .submit_store(store_request("wait-2", "second"))
        .await
        .expect("second queued");
    let waiting_queue = queue.clone();
    let waiting = tokio::spawn(async move {
        waiting_queue
            .submit_store(store_request("wait-3", "third"))
            .await
    });
    tokio::task::yield_now().await;
    tokio::time::advance(Duration::from_millis(100)).await;
    let error = waiting
        .await
        .expect("submission task joins")
        .expect_err("capacity wait times out");
    assert_eq!(error.code, MemoryErrorCode::DeadlineExceeded);
    assert_eq!(queue.snapshot().rejected_total, 1);

    provider.gate.add_permits(2);
    queue
        .shutdown(Duration::from_secs(1))
        .await
        .expect("queue drains");
}

#[tokio::test]
async fn wait_backpressure_admits_after_capacity_and_drain_closes_the_queue() {
    let provider = GateProvider::new();
    let config = MemoryWorkQueueConfig {
        capacity: 1,
        backpressure: MemoryBackpressurePolicy::Wait,
        enqueue_timeout_millis: 1_000,
        max_attempts: 1,
        attempt_timeout_millis: 10_000,
        ..MemoryWorkQueueConfig::default()
    };
    let queue =
        MemoryWorkQueue::new(MemoryRuntime::new(provider.clone()), config).expect("valid queue");
    queue
        .submit_store(store_request("admit-1", "first"))
        .await
        .expect("first accepted");
    wait_for_state(&queue, "admit-1", MemoryJobState::Running).await;
    queue
        .submit_store(store_request("admit-2", "second"))
        .await
        .expect("second queued");
    let waiting_queue = queue.clone();
    let waiting = tokio::spawn(async move {
        waiting_queue
            .submit_store(store_request("admit-3", "third"))
            .await
    });
    tokio::task::yield_now().await;
    provider.gate.add_permits(1);
    let receipt = waiting
        .await
        .expect("submission task joins")
        .expect("capacity release admits third job");
    assert_eq!(receipt.sequence, 3);

    provider.gate.add_permits(2);
    queue
        .drain(Duration::from_secs(1))
        .await
        .expect("drain completes");
    assert!(!queue.snapshot().accepting);
    assert_eq!(queue.snapshot().succeeded_total, 3);
    assert_eq!(
        queue
            .submit_store(store_request("after-drain", "closed"))
            .await
            .expect_err("drained queue is closed")
            .code,
        MemoryErrorCode::Cancelled
    );
    queue
        .shutdown(Duration::from_secs(1))
        .await
        .expect("shutdown joins a drained worker");
}

#[tokio::test(start_paused = true)]
async fn retries_only_retryable_errors_and_preserves_idempotency() {
    let provider = FlakyProvider {
        inner: InMemoryProvider::new(),
        failures_left: Arc::new(AtomicUsize::new(2)),
        calls: Arc::new(AtomicUsize::new(0)),
        retryable: true,
    };
    let observer = Arc::new(RecordingObserver::default());
    let config = MemoryWorkQueueConfig {
        capacity: 2,
        max_attempts: 3,
        retry_initial_delay_millis: 10,
        retry_max_delay_millis: 20,
        attempt_timeout_millis: 1_000,
        ..MemoryWorkQueueConfig::default()
    };
    let queue =
        MemoryWorkQueue::new(MemoryRuntime::new(provider.clone()), config).expect("valid queue");
    queue
        .submit_store_observed(
            store_request("retry-job", "eventual"),
            Some(observer.clone()),
        )
        .await
        .expect("job accepted");
    wait_for_attempt_state(&queue, "retry-job", 1, MemoryJobState::Retrying).await;
    tokio::time::advance(Duration::from_millis(10)).await;
    wait_for_attempt_state(&queue, "retry-job", 2, MemoryJobState::Retrying).await;
    tokio::time::advance(Duration::from_millis(20)).await;
    wait_for_state(&queue, "retry-job", MemoryJobState::Succeeded).await;
    queue
        .shutdown(Duration::from_secs(1))
        .await
        .expect("queue shuts down");

    let job = queue.job("retry-job").expect("job retained");
    assert_eq!(job.attempts, 3);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 3);
    assert_eq!(job.outcome.expect("success outcome").memory_ids.len(), 1);
    let states = observer
        .transitions
        .lock()
        .expect("observer lock")
        .iter()
        .map(|transition| transition.state)
        .collect::<Vec<_>>();
    assert_eq!(
        states,
        [
            MemoryJobState::Queued,
            MemoryJobState::Running,
            MemoryJobState::Retrying,
            MemoryJobState::Running,
            MemoryJobState::Retrying,
            MemoryJobState::Running,
            MemoryJobState::Succeeded,
        ]
    );
}

#[tokio::test]
async fn non_retryable_failure_is_terminal_after_one_attempt() {
    let provider = FlakyProvider {
        inner: InMemoryProvider::new(),
        failures_left: Arc::new(AtomicUsize::new(1)),
        calls: Arc::new(AtomicUsize::new(0)),
        retryable: false,
    };
    let config = MemoryWorkQueueConfig {
        max_attempts: 5,
        ..queue_config(1)
    };
    let queue =
        MemoryWorkQueue::new(MemoryRuntime::new(provider.clone()), config).expect("valid queue");
    queue
        .submit_store(store_request("no-retry", "invalid"))
        .await
        .expect("job accepted");
    queue
        .flush(Duration::from_secs(1))
        .await
        .expect("failed job is terminal");
    let job = queue.job("no-retry").expect("job retained");
    assert_eq!(job.state, MemoryJobState::Failed);
    assert_eq!(job.attempts, 1);
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    queue
        .shutdown(Duration::from_secs(1))
        .await
        .expect("queue shuts down");
}

#[tokio::test(start_paused = true)]
async fn attempt_timeout_retries_then_exhausts() {
    let provider = GateProvider::new();
    let config = MemoryWorkQueueConfig {
        capacity: 1,
        max_attempts: 2,
        retry_initial_delay_millis: 5,
        retry_max_delay_millis: 5,
        attempt_timeout_millis: 10,
        ..MemoryWorkQueueConfig::default()
    };
    let queue =
        MemoryWorkQueue::new(MemoryRuntime::new(provider.clone()), config).expect("valid queue");
    queue
        .submit_store(store_request("timeout-job", "blocked"))
        .await
        .expect("job accepted");
    wait_for_state(&queue, "timeout-job", MemoryJobState::Running).await;
    tokio::time::advance(Duration::from_millis(10)).await;
    wait_for_state(&queue, "timeout-job", MemoryJobState::Retrying).await;
    tokio::time::advance(Duration::from_millis(5)).await;
    wait_for_state(&queue, "timeout-job", MemoryJobState::Running).await;
    tokio::time::advance(Duration::from_millis(10)).await;
    wait_for_state(&queue, "timeout-job", MemoryJobState::Failed).await;
    let job = queue.job("timeout-job").expect("job retained");
    assert_eq!(job.attempts, 2);
    assert_eq!(
        job.error.expect("timeout error").code,
        MemoryErrorCode::DeadlineExceeded
    );
    assert_eq!(provider.calls.load(Ordering::SeqCst), 2);
    queue
        .shutdown(Duration::from_secs(1))
        .await
        .expect("queue shuts down");
}

#[tokio::test]
async fn flush_uses_the_acceptance_watermark_and_ignores_later_work() {
    let provider = GateProvider::new();
    let queue = MemoryWorkQueue::new(MemoryRuntime::new(provider.clone()), queue_config(2))
        .expect("valid queue");
    queue
        .submit_store(store_request("watermark-1", "first"))
        .await
        .expect("first accepted");
    wait_for_state(&queue, "watermark-1", MemoryJobState::Running).await;
    let flush_queue = queue.clone();
    let flush = tokio::spawn(async move { flush_queue.flush(Duration::from_secs(1)).await });
    tokio::task::yield_now().await;
    queue
        .submit_store(store_request("watermark-2", "second"))
        .await
        .expect("later job accepted");
    provider.gate.add_permits(1);
    flush
        .await
        .expect("flush task joins")
        .expect("watermark completes after first job");
    wait_for_state(&queue, "watermark-2", MemoryJobState::Running).await;

    provider.gate.add_permits(1);
    queue
        .shutdown(Duration::from_secs(1))
        .await
        .expect("remaining work drains");
}

#[tokio::test(start_paused = true)]
async fn flush_timeout_does_not_close_admission_or_abandon_work() {
    let provider = GateProvider::new();
    let queue = MemoryWorkQueue::new(MemoryRuntime::new(provider.clone()), queue_config(1))
        .expect("valid queue");
    queue
        .submit_store(store_request("flush-timeout", "blocked"))
        .await
        .expect("job accepted");
    wait_for_state(&queue, "flush-timeout", MemoryJobState::Running).await;
    let flush_queue = queue.clone();
    let flush = tokio::spawn(async move { flush_queue.flush(Duration::from_millis(10)).await });
    tokio::task::yield_now().await;
    tokio::time::advance(Duration::from_millis(10)).await;
    assert_eq!(
        flush
            .await
            .expect("flush joins")
            .expect_err("flush times out")
            .code,
        MemoryErrorCode::DeadlineExceeded
    );
    assert!(queue.snapshot().accepting);
    assert_eq!(
        queue.job("flush-timeout").expect("job retained").state,
        MemoryJobState::Running
    );

    provider.gate.add_permits(1);
    queue
        .shutdown(Duration::from_secs(1))
        .await
        .expect("work remains drainable");
}

#[tokio::test]
async fn maintenance_jobs_share_queue_lifecycle_and_return_derived_ids() {
    let provider = InMemoryProvider::new();
    let runtime = MemoryRuntime::new(provider.clone());
    runtime
        .store(store_request(
            "source-for-maintenance",
            "editor preference dark",
        ))
        .await
        .expect("source stored");
    let queue = MemoryWorkQueue::new(runtime, queue_config(2)).expect("valid queue");
    queue
        .submit_maintenance(MemoryMaintenanceRequest {
            context: MemoryRequestContext::new("maintenance-job").expect("valid context"),
            namespace: namespace("subject-a"),
            action: MemoryMaintenanceAction::Reflect,
            window: Some(MemoryMaintenanceWindow {
                checkpoint_id: "checkpoint-a".to_string(),
                previous_checkpoint_id: None,
                query: "editor preference".to_string(),
                scope: MemorySearchScope::Subject,
                filter: MemoryFilter::default(),
                limit: 10,
            }),
            parameters: BTreeMap::new(),
        })
        .await
        .expect("maintenance accepted");
    queue
        .flush(Duration::from_secs(1))
        .await
        .expect("maintenance completes");
    let job = queue.job("maintenance-job").expect("job retained");
    assert_eq!(job.state, MemoryJobState::Succeeded);
    assert_eq!(job.outcome.expect("outcome").memory_ids.len(), 1);
    queue
        .shutdown(Duration::from_secs(1))
        .await
        .expect("queue shuts down");
}

struct HangingMaintainer;

#[async_trait]
impl MemoryMaintainer for HangingMaintainer {
    fn name(&self) -> &str {
        "hanging"
    }

    async fn maintain(
        &self,
        _runtime: &MemoryRuntime,
        _request: MemoryMaintenanceRequest,
    ) -> MemoryMaintainerResult<MemoryMaintenanceResult> {
        std::future::pending().await
    }
}

#[tokio::test(start_paused = true)]
async fn attempt_timeout_bounds_a_maintainer_that_ignores_provider_deadlines() {
    let config = MemoryWorkQueueConfig {
        capacity: 1,
        max_attempts: 1,
        attempt_timeout_millis: 10,
        ..MemoryWorkQueueConfig::default()
    };
    let queue = MemoryWorkQueue::with_maintainer(
        MemoryRuntime::new(InMemoryProvider::new()),
        HangingMaintainer,
        config,
    )
    .expect("valid queue");
    queue
        .submit_maintenance(MemoryMaintenanceRequest {
            context: MemoryRequestContext::new("hanging-maintenance").expect("valid context"),
            namespace: namespace("subject-a"),
            action: MemoryMaintenanceAction::Reflect,
            window: Some(MemoryMaintenanceWindow {
                checkpoint_id: "checkpoint-hanging".to_string(),
                previous_checkpoint_id: None,
                query: "anything".to_string(),
                scope: MemorySearchScope::Subject,
                filter: MemoryFilter::default(),
                limit: 1,
            }),
            parameters: BTreeMap::new(),
        })
        .await
        .expect("maintenance accepted");
    wait_for_state(&queue, "hanging-maintenance", MemoryJobState::Running).await;
    tokio::time::advance(Duration::from_millis(10)).await;
    wait_for_state(&queue, "hanging-maintenance", MemoryJobState::Failed).await;
    assert_eq!(
        queue
            .job("hanging-maintenance")
            .expect("job retained")
            .error
            .expect("timeout retained")
            .code,
        MemoryErrorCode::DeadlineExceeded
    );
    queue
        .shutdown(Duration::from_secs(1))
        .await
        .expect("queue shuts down");
}

#[tokio::test]
async fn terminal_history_is_bounded_without_losing_lifetime_counters() {
    let config = MemoryWorkQueueConfig {
        terminal_history_capacity: 2,
        ..queue_config(2)
    };
    let queue = MemoryWorkQueue::new(MemoryRuntime::new(InMemoryProvider::new()), config)
        .expect("valid queue");
    for index in 1..=3 {
        let job_id = format!("history-{index}");
        queue
            .submit_store(store_request(&job_id, &format!("item {index}")))
            .await
            .expect("job accepted");
        queue
            .flush(Duration::from_secs(1))
            .await
            .expect("job completes");
    }
    assert!(queue.job("history-1").is_none());
    assert!(queue.job("history-2").is_some());
    assert!(queue.job("history-3").is_some());
    let snapshot = queue.snapshot();
    assert_eq!(snapshot.succeeded_total, 3);
    assert_eq!(snapshot.retained_terminal_jobs, 2);
    queue
        .shutdown(Duration::from_secs(1))
        .await
        .expect("queue shuts down");
}

struct PanicObserver;

impl MemoryWorkObserver for PanicObserver {
    fn on_transition(&self, _transition: &MemoryWorkTransition) {
        panic!("controlled observer panic");
    }
}

#[tokio::test]
async fn observer_panics_do_not_kill_the_worker() {
    let queue = MemoryWorkQueue::new(MemoryRuntime::new(InMemoryProvider::new()), queue_config(1))
        .expect("valid queue");
    queue
        .submit_store_observed(
            store_request("panic-observer", "safe"),
            Some(Arc::new(PanicObserver)),
        )
        .await
        .expect("job accepted despite observer panic");
    queue
        .flush(Duration::from_secs(1))
        .await
        .expect("worker survives observer panic");
    assert_eq!(
        queue.job("panic-observer").expect("job retained").state,
        MemoryJobState::Succeeded
    );
    queue
        .shutdown(Duration::from_secs(1))
        .await
        .expect("queue shuts down");
}

#[tokio::test(start_paused = true)]
async fn shutdown_timeout_cancels_nonterminal_work_and_is_idempotent() {
    let provider = GateProvider::new();
    let queue =
        MemoryWorkQueue::new(MemoryRuntime::new(provider), queue_config(1)).expect("valid queue");
    let observer = Arc::new(RecordingObserver::default());
    queue
        .submit_store_observed(
            store_request("cancelled-job", "blocked"),
            Some(observer.clone()),
        )
        .await
        .expect("job accepted");
    wait_for_state(&queue, "cancelled-job", MemoryJobState::Running).await;
    let shutdown_queue = queue.clone();
    let shutdown =
        tokio::spawn(async move { shutdown_queue.shutdown(Duration::from_millis(10)).await });
    tokio::task::yield_now().await;
    tokio::time::advance(Duration::from_millis(10)).await;
    let error = shutdown
        .await
        .expect("shutdown task joins")
        .expect_err("shutdown times out");
    assert_eq!(error.code, MemoryErrorCode::DeadlineExceeded);
    assert_eq!(
        queue.job("cancelled-job").expect("job retained").state,
        MemoryJobState::Cancelled
    );
    assert!(!queue.snapshot().accepting);
    assert_eq!(queue.snapshot().cancelled_total, 1);
    assert_eq!(
        observer
            .transitions
            .lock()
            .expect("observer lock")
            .last()
            .expect("cancellation transition")
            .state,
        MemoryJobState::Cancelled
    );
    assert_eq!(
        queue
            .submit_store(store_request("after-close", "rejected"))
            .await
            .expect_err("closed queue rejects")
            .code,
        MemoryErrorCode::Cancelled
    );
    queue
        .shutdown(Duration::from_millis(1))
        .await
        .expect("second shutdown is a no-op");
}
