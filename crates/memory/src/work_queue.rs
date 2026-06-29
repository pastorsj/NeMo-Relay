// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded process-local execution for memory writes and maintenance.

use std::collections::{HashMap, VecDeque};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant, SystemTime};

use nemo_relay_types::memory::{
    MemoryErrorCode, MemoryMaintenanceRequest, MemoryOperationError, MemoryStoreDisposition,
    MemoryStoreRequest,
};
use serde::{Deserialize, Serialize};
use tokio::sync::{Notify, mpsc, oneshot};
use tokio::task::JoinHandle;

use crate::{MemoryMaintainer, MemoryRuntime, ReferenceMaintainer};

const QUEUE_PROVIDER: &str = "memory_work_queue";

/// Admission behavior when the pending queue is full.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryBackpressurePolicy {
    /// Reject the new job immediately.
    #[default]
    Reject,
    /// Wait up to `enqueue_timeout_millis` for pending capacity.
    Wait,
}

/// Bounded local worker policy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct MemoryWorkQueueConfig {
    /// Maximum pending jobs. The currently running job is not counted.
    pub capacity: usize,
    /// Behavior when all pending slots are occupied.
    pub backpressure: MemoryBackpressurePolicy,
    /// Maximum capacity wait for the `wait` policy.
    pub enqueue_timeout_millis: u64,
    /// Maximum executions of one job, including the first attempt.
    pub max_attempts: u32,
    /// Delay before the first retry.
    pub retry_initial_delay_millis: u64,
    /// Maximum delay between attempts.
    pub retry_max_delay_millis: u64,
    /// Fresh provider deadline assigned to each attempt.
    pub attempt_timeout_millis: u64,
    /// Maximum number of terminal jobs retained for inspection/deduplication.
    pub terminal_history_capacity: usize,
}

impl Default for MemoryWorkQueueConfig {
    fn default() -> Self {
        Self {
            capacity: 64,
            backpressure: MemoryBackpressurePolicy::Reject,
            enqueue_timeout_millis: 250,
            max_attempts: 3,
            retry_initial_delay_millis: 25,
            retry_max_delay_millis: 1_000,
            attempt_timeout_millis: 2_000,
            terminal_history_capacity: 1_024,
        }
    }
}

impl MemoryWorkQueueConfig {
    /// Validate queue bounds and timing policy.
    pub fn validate(&self) -> Result<(), MemoryOperationError> {
        if self.capacity == 0 {
            return Err(invalid_config("capacity must be positive"));
        }
        if self.backpressure == MemoryBackpressurePolicy::Wait && self.enqueue_timeout_millis == 0 {
            return Err(invalid_config(
                "enqueue_timeout_millis must be positive for wait backpressure",
            ));
        }
        if self.max_attempts == 0 {
            return Err(invalid_config("max_attempts must be positive"));
        }
        if self.attempt_timeout_millis == 0 {
            return Err(invalid_config("attempt_timeout_millis must be positive"));
        }
        if self.retry_initial_delay_millis > self.retry_max_delay_millis {
            return Err(invalid_config(
                "retry_initial_delay_millis must not exceed retry_max_delay_millis",
            ));
        }
        if self.terminal_history_capacity == 0 {
            return Err(invalid_config("terminal_history_capacity must be positive"));
        }
        Ok(())
    }
}

/// Kind of mutation executed by the local worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryWorkKind {
    /// Persist one memory record.
    Store,
    /// Derive memory through a [`MemoryMaintainer`].
    Maintenance,
}

/// Monotonic lifecycle state of one queued job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryJobState {
    /// Accepted and waiting for the single consumer.
    Queued,
    /// One provider/maintainer attempt is running.
    Running,
    /// A retryable attempt failed and another attempt is scheduled.
    Retrying,
    /// Work completed successfully.
    Succeeded,
    /// Work reached a terminal execution failure.
    Failed,
    /// Admission failed before the job was accepted.
    Rejected,
    /// Shutdown timed out and cancelled accepted work.
    Cancelled,
}

impl MemoryJobState {
    /// Return whether no further transition is possible.
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Succeeded | Self::Failed | Self::Rejected | Self::Cancelled
        )
    }
}

/// Successful mutation references retained in job status.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryWorkOutcome {
    /// Stable provider record IDs created or replayed by the job.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub memory_ids: Vec<String>,
    /// Store disposition for direct store work.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disposition: Option<MemoryStoreDisposition>,
    /// Number of non-fatal provider errors returned with successful work.
    #[serde(default)]
    pub partial_error_count: usize,
}

/// Inspectable status for one accepted job.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryJobStatus {
    /// Immutable caller operation/job identifier.
    pub job_id: String,
    /// Monotonic accepted-job sequence.
    pub sequence: u64,
    /// Store or maintenance work.
    pub kind: MemoryWorkKind,
    /// Current lifecycle state.
    pub state: MemoryJobState,
    /// Attempts started so far.
    pub attempts: u32,
    /// Last or terminal typed failure.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<MemoryOperationError>,
    /// Successful provider references.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome: Option<MemoryWorkOutcome>,
}

/// Receipt returned for accepted work or an identical retained replay.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryWorkReceipt {
    /// Immutable job identifier.
    pub job_id: String,
    /// Monotonic accepted-job sequence.
    pub sequence: u64,
    /// State observed when the receipt was returned.
    pub state: MemoryJobState,
}

/// Aggregate local queue state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryWorkQueueSnapshot {
    /// Whether new jobs may still be admitted.
    pub accepting: bool,
    /// Configured pending capacity.
    pub capacity: usize,
    /// Jobs currently waiting.
    pub queued: usize,
    /// Jobs currently executing.
    pub running: usize,
    /// Jobs currently waiting for retry delay.
    pub retrying: usize,
    /// Total accepted jobs over this queue lifetime.
    pub accepted_total: u64,
    /// Total admission rejections over this queue lifetime.
    pub rejected_total: u64,
    /// Total successful jobs over this queue lifetime.
    pub succeeded_total: u64,
    /// Total execution failures over this queue lifetime.
    pub failed_total: u64,
    /// Total jobs cancelled by shutdown timeout.
    pub cancelled_total: u64,
    /// Most recent accepted sequence.
    pub last_accepted_sequence: u64,
    /// Highest contiguous terminal sequence.
    pub last_terminal_sequence: u64,
    /// Terminal jobs currently retained for inspection.
    pub retained_terminal_jobs: usize,
}

/// One state transition delivered to an optional observer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryWorkTransition {
    /// Job identifier, even for rejected admission.
    pub job_id: String,
    /// Accepted sequence, or zero for rejected admission.
    pub sequence: u64,
    /// Store or maintenance work.
    pub kind: MemoryWorkKind,
    /// New state.
    pub state: MemoryJobState,
    /// Attempts started so far.
    pub attempts: u32,
    /// Time since admission for an accepted job.
    pub elapsed_millis: u64,
    /// Typed failure for retry, rejection, failure, or cancellation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<MemoryOperationError>,
    /// Successful provider references.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome: Option<MemoryWorkOutcome>,
}

/// Non-blocking observer for queue state transitions.
pub trait MemoryWorkObserver: Send + Sync {
    /// Observe one immutable transition. Panics are contained by the queue.
    fn on_transition(&self, transition: &MemoryWorkTransition);
}

/// Bounded process-local queue over one provider runtime and maintainer.
#[derive(Clone)]
pub struct MemoryWorkQueue {
    inner: Arc<QueueInner>,
}

impl MemoryWorkQueue {
    /// Create a queue using the deterministic reference maintainer.
    pub fn new(
        runtime: MemoryRuntime,
        config: MemoryWorkQueueConfig,
    ) -> Result<Self, MemoryOperationError> {
        Self::with_maintainer(runtime, ReferenceMaintainer, config)
    }

    /// Create a queue with a concrete maintenance policy.
    pub fn with_maintainer<M>(
        runtime: MemoryRuntime,
        maintainer: M,
        config: MemoryWorkQueueConfig,
    ) -> Result<Self, MemoryOperationError>
    where
        M: MemoryMaintainer + 'static,
    {
        Self::from_maintainer_arc(runtime, Arc::new(maintainer), config)
    }

    /// Create a queue with a shared maintenance policy object.
    pub fn from_maintainer_arc(
        runtime: MemoryRuntime,
        maintainer: Arc<dyn MemoryMaintainer>,
        config: MemoryWorkQueueConfig,
    ) -> Result<Self, MemoryOperationError> {
        config.validate()?;
        Ok(Self {
            inner: Arc::new(QueueInner {
                runtime,
                maintainer,
                status: Arc::new(StatusCell {
                    data: Mutex::new(StatusData::default()),
                    changed: Notify::new(),
                }),
                control: Mutex::new(QueueControl::default()),
                config,
            }),
        })
    }

    /// Submit one store using its operation ID as immutable job identity.
    pub async fn submit_store(
        &self,
        request: MemoryStoreRequest,
    ) -> Result<MemoryWorkReceipt, MemoryOperationError> {
        self.submit_store_observed(request, None).await
    }

    /// Submit a store and observe its state transitions.
    pub async fn submit_store_observed(
        &self,
        mut request: MemoryStoreRequest,
        observer: Option<Arc<dyn MemoryWorkObserver>>,
    ) -> Result<MemoryWorkReceipt, MemoryOperationError> {
        request.validate()?;
        let job_id = request.context.operation_id.clone();
        if request.idempotency_key.is_none() {
            request.idempotency_key = Some(format!("memory-work:{job_id}"));
        }
        self.submit(WorkPayload::Store(request), observer).await
    }

    /// Submit one maintenance request using its operation ID as job identity.
    pub async fn submit_maintenance(
        &self,
        request: MemoryMaintenanceRequest,
    ) -> Result<MemoryWorkReceipt, MemoryOperationError> {
        self.submit_maintenance_observed(request, None).await
    }

    /// Submit maintenance and observe its state transitions.
    pub async fn submit_maintenance_observed(
        &self,
        request: MemoryMaintenanceRequest,
        observer: Option<Arc<dyn MemoryWorkObserver>>,
    ) -> Result<MemoryWorkReceipt, MemoryOperationError> {
        request.validate()?;
        self.submit(WorkPayload::Maintenance(request), observer)
            .await
    }

    /// Return aggregate state without blocking on work.
    pub fn snapshot(&self) -> MemoryWorkQueueSnapshot {
        let data = self.inner.status.lock();
        data.snapshot(&self.inner.config)
    }

    /// Return retained state for one accepted job.
    pub fn job(&self, job_id: &str) -> Option<MemoryJobStatus> {
        self.inner
            .status
            .lock()
            .jobs
            .get(job_id)
            .map(|entry| entry.status.clone())
    }

    /// Wait until every job accepted before this call is terminal.
    pub async fn flush(&self, timeout: Duration) -> Result<(), MemoryOperationError> {
        let target = self.inner.status.lock().last_accepted_sequence;
        self.wait_for_terminal(target, timeout, "flush").await
    }

    /// Stop admission and wait for all accepted work.
    pub async fn drain(&self, timeout: Duration) -> Result<(), MemoryOperationError> {
        let target = self.close_admission();
        self.wait_for_terminal(target, timeout, "drain").await
    }

    /// Stop admission, drain within a deadline, and join the worker.
    pub async fn shutdown(&self, timeout: Duration) -> Result<(), MemoryOperationError> {
        let target = self.close_admission();
        let drained = self.wait_for_terminal(target, timeout, "shutdown").await;
        if drained.is_err() {
            if let Some(worker) = self.inner.take_worker() {
                worker.abort();
                let _ = worker.await;
            }
            self.cancel_nonterminal("memory queue shutdown deadline exceeded");
            return drained;
        }
        if let Some(worker) = self.inner.take_worker() {
            let _ = worker.await;
        }
        Ok(())
    }

    async fn submit(
        &self,
        payload: WorkPayload,
        observer: Option<Arc<dyn MemoryWorkObserver>>,
    ) -> Result<MemoryWorkReceipt, MemoryOperationError> {
        let job_id = payload.job_id().to_string();
        let kind = payload.kind();
        let fingerprint = payload.fingerprint()?;
        if let Some(existing) = self.existing_receipt(&job_id, &fingerprint)? {
            return Ok(existing);
        }

        let sender = match self.inner.ensure_started() {
            Ok(sender) => sender,
            Err(error) => {
                self.reject(&job_id, kind, &observer, error.clone());
                return Err(error);
            }
        };
        let permit = match self.inner.config.backpressure {
            MemoryBackpressurePolicy::Reject => {
                sender.clone().try_reserve_owned().map_err(|error| {
                    let failure = match error {
                        mpsc::error::TrySendError::Full(_) => queue_error(
                            MemoryErrorCode::ProviderUnavailable,
                            "memory work queue is full",
                            true,
                            &job_id,
                        ),
                        mpsc::error::TrySendError::Closed(_) => closed_error(&job_id),
                    };
                    self.reject(&job_id, kind, &observer, failure.clone());
                    failure
                })?
            }
            MemoryBackpressurePolicy::Wait => {
                let wait = sender.clone().reserve_owned();
                match tokio::time::timeout(
                    Duration::from_millis(self.inner.config.enqueue_timeout_millis),
                    wait,
                )
                .await
                {
                    Ok(Ok(permit)) => permit,
                    Ok(Err(_)) => {
                        let failure = closed_error(&job_id);
                        self.reject(&job_id, kind, &observer, failure.clone());
                        return Err(failure);
                    }
                    Err(_) => {
                        let failure = queue_error(
                            MemoryErrorCode::DeadlineExceeded,
                            "memory work queue admission deadline exceeded",
                            true,
                            &job_id,
                        );
                        self.reject(&job_id, kind, &observer, failure.clone());
                        return Err(failure);
                    }
                }
            }
        };

        if let Some(existing) = self.existing_receipt(&job_id, &fingerprint)? {
            drop(permit);
            return Ok(existing);
        }
        let (sequence, admitted_at) = {
            let mut data = self.inner.status.lock();
            if !data.accepting {
                drop(data);
                drop(permit);
                let failure = closed_error(&job_id);
                self.reject(&job_id, kind, &observer, failure.clone());
                return Err(failure);
            }
            let sequence = data.next_sequence.checked_add(1).ok_or_else(|| {
                queue_error(
                    MemoryErrorCode::Internal,
                    "memory work queue exhausted its sequence space",
                    false,
                    &job_id,
                )
            })?;
            data.next_sequence = sequence;
            data.last_accepted_sequence = sequence;
            data.accepted_total += 1;
            let admitted_at = Instant::now();
            data.jobs.insert(
                job_id.clone(),
                JobEntry {
                    fingerprint,
                    admitted_at,
                    observer: observer.clone(),
                    status: MemoryJobStatus {
                        job_id: job_id.clone(),
                        sequence,
                        kind,
                        state: MemoryJobState::Queued,
                        attempts: 0,
                        error: None,
                        outcome: None,
                    },
                },
            );
            (sequence, admitted_at)
        };
        let transition = MemoryWorkTransition {
            job_id: job_id.clone(),
            sequence,
            kind,
            state: MemoryJobState::Queued,
            attempts: 0,
            elapsed_millis: 0,
            error: None,
            outcome: None,
        };
        notify_observer(observer.as_ref(), &transition);
        permit.send(WorkEnvelope {
            job_id: job_id.clone(),
            sequence,
            admitted_at,
            payload,
            observer,
        });
        self.inner.status.changed.notify_waiters();
        Ok(MemoryWorkReceipt {
            job_id,
            sequence,
            state: MemoryJobState::Queued,
        })
    }

    fn existing_receipt(
        &self,
        job_id: &str,
        fingerprint: &[u8],
    ) -> Result<Option<MemoryWorkReceipt>, MemoryOperationError> {
        let data = self.inner.status.lock();
        let Some(existing) = data.jobs.get(job_id) else {
            return Ok(None);
        };
        if existing.fingerprint != fingerprint {
            return Err(queue_error(
                MemoryErrorCode::Conflict,
                "memory work job ID was reused with a different payload",
                false,
                job_id,
            ));
        }
        Ok(Some(MemoryWorkReceipt {
            job_id: job_id.to_string(),
            sequence: existing.status.sequence,
            state: existing.status.state,
        }))
    }

    fn reject(
        &self,
        job_id: &str,
        kind: MemoryWorkKind,
        observer: &Option<Arc<dyn MemoryWorkObserver>>,
        error: MemoryOperationError,
    ) {
        self.inner.status.lock().rejected_total += 1;
        notify_observer(
            observer.as_ref(),
            &MemoryWorkTransition {
                job_id: job_id.to_string(),
                sequence: 0,
                kind,
                state: MemoryJobState::Rejected,
                attempts: 0,
                elapsed_millis: 0,
                error: Some(error),
                outcome: None,
            },
        );
    }

    fn close_admission(&self) -> u64 {
        // Use the same control -> status order as `ensure_started` so no
        // submission can start a fresh worker after shutdown closes an
        // as-yet-unstarted queue.
        let mut control = self.inner.control_lock();
        let target = {
            let mut data = self.inner.status.lock();
            data.accepting = false;
            data.last_accepted_sequence
        };
        control.sender.take();
        if let Some(stop) = control.stop.take() {
            let _ = stop.send(());
        }
        drop(control);
        self.inner.status.changed.notify_waiters();
        target
    }

    async fn wait_for_terminal(
        &self,
        target: u64,
        timeout: Duration,
        operation: &str,
    ) -> Result<(), MemoryOperationError> {
        let wait = async {
            loop {
                let changed = self.inner.status.changed.notified();
                if self.inner.status.lock().last_terminal_sequence >= target {
                    return;
                }
                changed.await;
            }
        };
        tokio::time::timeout(timeout, wait).await.map_err(|_| {
            queue_error(
                MemoryErrorCode::DeadlineExceeded,
                format!("memory work queue {operation} deadline exceeded"),
                true,
                operation,
            )
        })
    }

    fn cancel_nonterminal(&self, message: &str) {
        let mut transitions = Vec::new();
        {
            let mut data = self.inner.status.lock();
            let ids = data
                .jobs
                .iter()
                .filter(|(_, entry)| !entry.status.state.is_terminal())
                .map(|(id, _)| id.clone())
                .collect::<Vec<_>>();
            for id in ids {
                let error = queue_error(MemoryErrorCode::Cancelled, message, false, &id);
                if let Some(entry) = data.jobs.get_mut(&id) {
                    entry.status.state = MemoryJobState::Cancelled;
                    entry.status.error = Some(error.clone());
                    transitions.push((
                        entry.observer.clone(),
                        MemoryWorkTransition {
                            job_id: id.clone(),
                            sequence: entry.status.sequence,
                            kind: entry.status.kind,
                            state: MemoryJobState::Cancelled,
                            attempts: entry.status.attempts,
                            elapsed_millis: duration_millis(entry.admitted_at.elapsed()),
                            error: Some(error),
                            outcome: None,
                        },
                    ));
                    entry.observer = None;
                }
                data.cancelled_total += 1;
                data.record_terminal(id);
            }
            data.last_terminal_sequence = data.last_accepted_sequence;
            data.prune(self.inner.config.terminal_history_capacity);
        }
        for (observer, transition) in transitions {
            notify_observer(observer.as_ref(), &transition);
        }
        self.inner.status.changed.notify_waiters();
    }
}

struct QueueInner {
    runtime: MemoryRuntime,
    maintainer: Arc<dyn MemoryMaintainer>,
    config: MemoryWorkQueueConfig,
    status: Arc<StatusCell>,
    control: Mutex<QueueControl>,
}

impl QueueInner {
    fn ensure_started(&self) -> Result<mpsc::Sender<WorkEnvelope>, MemoryOperationError> {
        let mut control = self.control_lock();
        if !self.status.lock().accepting {
            return Err(closed_error("start"));
        }
        if let Some(sender) = &control.sender {
            return Ok(sender.clone());
        }
        tokio::runtime::Handle::try_current().map_err(|_| {
            invalid_config("memory work queue submission requires an active Tokio runtime")
        })?;
        let (sender, receiver) = mpsc::channel(self.config.capacity);
        let (stop, stop_receiver) = oneshot::channel();
        let worker = tokio::spawn(worker_loop(
            receiver,
            stop_receiver,
            WorkerContext {
                runtime: self.runtime.clone(),
                maintainer: self.maintainer.clone(),
                config: self.config.clone(),
                status: self.status.clone(),
            },
        ));
        control.sender = Some(sender.clone());
        control.stop = Some(stop);
        control.worker = Some(worker);
        Ok(sender)
    }

    fn control_lock(&self) -> MutexGuard<'_, QueueControl> {
        self.control
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }

    fn take_worker(&self) -> Option<JoinHandle<()>> {
        self.control_lock().worker.take()
    }
}

impl Drop for QueueInner {
    fn drop(&mut self) {
        if let Ok(control) = self.control.get_mut() {
            control.sender.take();
            control.stop.take();
        }
    }
}

#[derive(Default)]
struct QueueControl {
    sender: Option<mpsc::Sender<WorkEnvelope>>,
    stop: Option<oneshot::Sender<()>>,
    worker: Option<JoinHandle<()>>,
}

struct StatusCell {
    data: Mutex<StatusData>,
    changed: Notify,
}

impl StatusCell {
    fn lock(&self) -> MutexGuard<'_, StatusData> {
        self.data.lock().unwrap_or_else(|error| error.into_inner())
    }
}

struct JobEntry {
    fingerprint: Vec<u8>,
    admitted_at: Instant,
    observer: Option<Arc<dyn MemoryWorkObserver>>,
    status: MemoryJobStatus,
}

struct StatusData {
    accepting: bool,
    next_sequence: u64,
    last_accepted_sequence: u64,
    last_terminal_sequence: u64,
    accepted_total: u64,
    rejected_total: u64,
    succeeded_total: u64,
    failed_total: u64,
    cancelled_total: u64,
    jobs: HashMap<String, JobEntry>,
    terminal_order: VecDeque<String>,
}

impl Default for StatusData {
    fn default() -> Self {
        Self {
            accepting: true,
            next_sequence: 0,
            last_accepted_sequence: 0,
            last_terminal_sequence: 0,
            accepted_total: 0,
            rejected_total: 0,
            succeeded_total: 0,
            failed_total: 0,
            cancelled_total: 0,
            jobs: HashMap::new(),
            terminal_order: VecDeque::new(),
        }
    }
}

impl StatusData {
    fn snapshot(&self, config: &MemoryWorkQueueConfig) -> MemoryWorkQueueSnapshot {
        MemoryWorkQueueSnapshot {
            accepting: self.accepting,
            capacity: config.capacity,
            queued: self.count(MemoryJobState::Queued),
            running: self.count(MemoryJobState::Running),
            retrying: self.count(MemoryJobState::Retrying),
            accepted_total: self.accepted_total,
            rejected_total: self.rejected_total,
            succeeded_total: self.succeeded_total,
            failed_total: self.failed_total,
            cancelled_total: self.cancelled_total,
            last_accepted_sequence: self.last_accepted_sequence,
            last_terminal_sequence: self.last_terminal_sequence,
            retained_terminal_jobs: self
                .jobs
                .values()
                .filter(|entry| entry.status.state.is_terminal())
                .count(),
        }
    }

    fn count(&self, state: MemoryJobState) -> usize {
        self.jobs
            .values()
            .filter(|entry| entry.status.state == state)
            .count()
    }

    fn record_terminal(&mut self, id: String) {
        self.terminal_order.push_back(id);
    }

    fn prune(&mut self, capacity: usize) {
        while self.terminal_order.len() > capacity {
            if let Some(id) = self.terminal_order.pop_front()
                && self
                    .jobs
                    .get(&id)
                    .is_some_and(|entry| entry.status.state.is_terminal())
            {
                self.jobs.remove(&id);
            }
        }
    }
}

enum WorkPayload {
    Store(MemoryStoreRequest),
    Maintenance(MemoryMaintenanceRequest),
}

impl WorkPayload {
    fn job_id(&self) -> &str {
        match self {
            Self::Store(request) => &request.context.operation_id,
            Self::Maintenance(request) => &request.context.operation_id,
        }
    }

    const fn kind(&self) -> MemoryWorkKind {
        match self {
            Self::Store(_) => MemoryWorkKind::Store,
            Self::Maintenance(_) => MemoryWorkKind::Maintenance,
        }
    }

    fn fingerprint(&self) -> Result<Vec<u8>, MemoryOperationError> {
        match self {
            Self::Store(request) => serde_json::to_vec(&(self.kind(), request)),
            Self::Maintenance(request) => serde_json::to_vec(&(self.kind(), request)),
        }
        .map_err(|error| {
            queue_error(
                MemoryErrorCode::Internal,
                format!("failed to fingerprint memory work: {error}"),
                false,
                self.job_id(),
            )
        })
    }
}

struct WorkEnvelope {
    job_id: String,
    sequence: u64,
    admitted_at: Instant,
    payload: WorkPayload,
    observer: Option<Arc<dyn MemoryWorkObserver>>,
}

struct WorkerContext {
    runtime: MemoryRuntime,
    maintainer: Arc<dyn MemoryMaintainer>,
    config: MemoryWorkQueueConfig,
    status: Arc<StatusCell>,
}

async fn worker_loop(
    mut receiver: mpsc::Receiver<WorkEnvelope>,
    mut stop: oneshot::Receiver<()>,
    context: WorkerContext,
) {
    let mut stopping = false;
    loop {
        let work = if stopping {
            receiver.recv().await
        } else {
            tokio::select! {
                _ = &mut stop => {
                    receiver.close();
                    stopping = true;
                    continue;
                }
                work = receiver.recv() => work,
            }
        };
        let Some(mut work) = work else {
            break;
        };
        let mut retry_delay = Duration::from_millis(context.config.retry_initial_delay_millis);
        for attempt in 1..=context.config.max_attempts {
            transition(
                &context,
                &work,
                MemoryJobState::Running,
                attempt,
                None,
                None,
            );
            refresh_deadline(&mut work.payload, context.config.attempt_timeout_millis);
            let result = tokio::time::timeout(
                Duration::from_millis(context.config.attempt_timeout_millis),
                execute(&context, &work.payload),
            )
            .await
            .unwrap_or_else(|_| {
                Err(queue_error(
                    MemoryErrorCode::DeadlineExceeded,
                    "memory work attempt deadline exceeded",
                    true,
                    &work.job_id,
                ))
            });
            match result {
                Ok(outcome) => {
                    terminal_transition(
                        &context,
                        &work,
                        MemoryJobState::Succeeded,
                        attempt,
                        None,
                        Some(outcome),
                    );
                    break;
                }
                Err(error) if error.retryable && attempt < context.config.max_attempts => {
                    transition(
                        &context,
                        &work,
                        MemoryJobState::Retrying,
                        attempt,
                        Some(error),
                        None,
                    );
                    tokio::time::sleep(retry_delay).await;
                    retry_delay = retry_delay
                        .saturating_mul(2)
                        .min(Duration::from_millis(context.config.retry_max_delay_millis));
                }
                Err(error) => {
                    terminal_transition(
                        &context,
                        &work,
                        MemoryJobState::Failed,
                        attempt,
                        Some(error),
                        None,
                    );
                    break;
                }
            }
        }
    }
}

async fn execute(
    context: &WorkerContext,
    payload: &WorkPayload,
) -> Result<MemoryWorkOutcome, MemoryOperationError> {
    match payload {
        WorkPayload::Store(request) => {
            let result = context.runtime.store(request.clone()).await?;
            Ok(MemoryWorkOutcome {
                memory_ids: vec![result.record.id],
                disposition: Some(result.disposition),
                partial_error_count: 0,
            })
        }
        WorkPayload::Maintenance(request) => {
            let result = context
                .maintainer
                .maintain(&context.runtime, request.clone())
                .await?;
            Ok(MemoryWorkOutcome {
                memory_ids: result.records.into_iter().map(|record| record.id).collect(),
                disposition: None,
                partial_error_count: result.partial_errors.len(),
            })
        }
    }
}

fn refresh_deadline(payload: &mut WorkPayload, timeout_millis: u64) {
    let deadline = (SystemTime::now() + Duration::from_millis(timeout_millis)).into();
    match payload {
        WorkPayload::Store(request) => request.context.deadline = Some(deadline),
        WorkPayload::Maintenance(request) => request.context.deadline = Some(deadline),
    }
}

fn transition(
    context: &WorkerContext,
    work: &WorkEnvelope,
    state: MemoryJobState,
    attempts: u32,
    error: Option<MemoryOperationError>,
    outcome: Option<MemoryWorkOutcome>,
) {
    {
        let mut data = context.status.lock();
        if let Some(entry) = data.jobs.get_mut(&work.job_id) {
            entry.status.state = state;
            entry.status.attempts = attempts;
            entry.status.error = error.clone();
            entry.status.outcome.clone_from(&outcome);
        }
    }
    notify_observer(
        work.observer.as_ref(),
        &MemoryWorkTransition {
            job_id: work.job_id.clone(),
            sequence: work.sequence,
            kind: work.payload.kind(),
            state,
            attempts,
            elapsed_millis: duration_millis(work.admitted_at.elapsed()),
            error,
            outcome,
        },
    );
    context.status.changed.notify_waiters();
}

fn terminal_transition(
    context: &WorkerContext,
    work: &WorkEnvelope,
    state: MemoryJobState,
    attempts: u32,
    error: Option<MemoryOperationError>,
    outcome: Option<MemoryWorkOutcome>,
) {
    transition(context, work, state, attempts, error, outcome);
    {
        let mut data = context.status.lock();
        if let Some(entry) = data.jobs.get_mut(&work.job_id) {
            entry.observer = None;
        }
        match state {
            MemoryJobState::Succeeded => data.succeeded_total += 1,
            MemoryJobState::Failed => data.failed_total += 1,
            _ => {}
        }
        data.last_terminal_sequence = work.sequence;
        data.record_terminal(work.job_id.clone());
        data.prune(context.config.terminal_history_capacity);
    }
    context.status.changed.notify_waiters();
}

fn notify_observer(
    observer: Option<&Arc<dyn MemoryWorkObserver>>,
    transition: &MemoryWorkTransition,
) {
    if let Some(observer) = observer {
        let _ = catch_unwind(AssertUnwindSafe(|| observer.on_transition(transition)));
    }
}

fn duration_millis(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

fn invalid_config(message: impl Into<String>) -> MemoryOperationError {
    queue_error(
        MemoryErrorCode::InvalidRequest,
        message,
        false,
        "configuration",
    )
}

fn closed_error(operation_id: &str) -> MemoryOperationError {
    queue_error(
        MemoryErrorCode::Cancelled,
        "memory work queue is not accepting jobs",
        false,
        operation_id,
    )
}

fn queue_error(
    code: MemoryErrorCode,
    message: impl Into<String>,
    retryable: bool,
    operation_id: &str,
) -> MemoryOperationError {
    MemoryOperationError::new(code, message, retryable)
        .with_operation_id(operation_id)
        .with_provider(QUEUE_PROVIDER)
}
