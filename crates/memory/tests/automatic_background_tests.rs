// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Automatic managed-LLM background write-back and evidence tests.

#![cfg(feature = "relay")]
#![allow(clippy::await_holding_lock)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use nemo_relay::api::event::Event;
use nemo_relay::api::llm::{LlmCallExecuteParams, LlmRequest, llm_call_execute};
use nemo_relay::api::runtime::{
    LlmExecutionNextFn, NemoRelayContextState, create_scope_stack, global_context,
    set_thread_scope_stack,
};
use nemo_relay::api::scope::{PopScopeParams, PushScopeParams, ScopeType, pop_scope, push_scope};
use nemo_relay::api::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
use nemo_relay::codec::openai_chat::OpenAIChatCodec;
use nemo_relay::error::FlowError;
use nemo_relay_memory::memory::{
    MemoryErrorCode, MemoryNamespace, MemoryOperationError, MemorySearchRequest,
    MemorySearchResult, MemoryStoreRequest, MemoryStoreResult,
};
use nemo_relay_memory::{
    AutomaticMemoryConfig, FailurePolicy, InMemoryProvider, MemoryBackpressurePolicy,
    MemoryComponent, MemoryJobState, MemoryProvider, MemoryProviderResult, MemoryRuntime,
    MemoryWorkQueueConfig, WriteDelivery,
};
use serde_json::{Value as Json, json};
use tokio::sync::Semaphore;

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn reset_runtime() {
    *global_context().write().expect("runtime lock") = NemoRelayContextState::new();
    set_thread_scope_stack(create_scope_stack());
}

fn namespace(session: &str) -> MemoryNamespace {
    MemoryNamespace {
        tenant_id: "tenant-background".to_string(),
        subject_id: "subject-alex".to_string(),
        session_id: Some(session.to_string()),
        agent_id: Some("assistant".to_string()),
    }
}

fn request(user: &str) -> LlmRequest {
    LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": user}],
        }),
    }
}

fn response(assistant: &str) -> Json {
    json!({
        "id": "response-background",
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": assistant},
            "finish_reason": "stop",
        }],
    })
}

async fn execute_turn(
    namespace: &MemoryNamespace,
    user: &str,
    assistant: &str,
    seen_requests: Arc<Mutex<Vec<LlmRequest>>>,
    callbacks: Option<Arc<AtomicUsize>>,
) -> Result<Json, FlowError> {
    let scope = push_scope(
        PushScopeParams::builder()
            .name("background-memory-session")
            .scope_type(ScopeType::Agent)
            .metadata(json!({"memory": {"namespace": namespace}}))
            .build(),
    )?;
    let answer = assistant.to_string();
    let callback: LlmExecutionNextFn = Arc::new(move |request| {
        seen_requests.lock().expect("request lock").push(request);
        if let Some(callbacks) = &callbacks {
            callbacks.fetch_add(1, Ordering::SeqCst);
        }
        let answer = answer.clone();
        Box::pin(async move { Ok(response(&answer)) })
    });
    let result = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("background-memory-agent")
            .request(request(user))
            .func(callback)
            .codec(Arc::new(OpenAIChatCodec))
            .response_codec(Arc::new(OpenAIChatCodec))
            .build(),
    )
    .await;
    pop_scope(PopScopeParams::builder().handle_uuid(&scope.uuid).build())?;
    result
}

fn background_config() -> AutomaticMemoryConfig {
    AutomaticMemoryConfig {
        write_delivery: WriteDelivery::Background,
        background_queue: MemoryWorkQueueConfig {
            capacity: 4,
            max_attempts: 2,
            retry_initial_delay_millis: 1,
            retry_max_delay_millis: 1,
            attempt_timeout_millis: 1_000,
            ..MemoryWorkQueueConfig::default()
        },
        ..AutomaticMemoryConfig::default()
    }
}

#[derive(Clone)]
struct GateProvider {
    inner: InMemoryProvider,
    gate: Arc<Semaphore>,
    store_calls: Arc<AtomicUsize>,
}

impl GateProvider {
    fn new() -> Self {
        Self {
            inner: InMemoryProvider::new(),
            gate: Arc::new(Semaphore::new(0)),
            store_calls: Arc::new(AtomicUsize::new(0)),
        }
    }
}

#[async_trait]
impl MemoryProvider for GateProvider {
    fn name(&self) -> &str {
        "gated_memory"
    }

    async fn search(
        &self,
        request: MemorySearchRequest,
    ) -> MemoryProviderResult<MemorySearchResult> {
        self.inner.search(request).await
    }

    async fn store(&self, request: MemoryStoreRequest) -> MemoryProviderResult<MemoryStoreResult> {
        self.store_calls.fetch_add(1, Ordering::SeqCst);
        self.gate
            .acquire()
            .await
            .expect("test gate remains open")
            .forget();
        self.inner.store(request).await
    }
}

#[derive(Clone)]
struct RetryFailureProvider {
    inner: InMemoryProvider,
    store_calls: Arc<AtomicUsize>,
}

#[async_trait]
impl MemoryProvider for RetryFailureProvider {
    fn name(&self) -> &str {
        "retry_failure"
    }

    async fn search(
        &self,
        request: MemorySearchRequest,
    ) -> MemoryProviderResult<MemorySearchResult> {
        self.inner.search(request).await
    }

    async fn store(&self, request: MemoryStoreRequest) -> MemoryProviderResult<MemoryStoreResult> {
        self.store_calls.fetch_add(1, Ordering::SeqCst);
        Err(MemoryOperationError::new(
            MemoryErrorCode::ProviderUnavailable,
            "controlled background failure",
            true,
        )
        .with_operation_id(request.context.operation_id)
        .with_provider(self.name()))
    }
}

async fn wait_for_snapshot(component: &MemoryComponent, running: usize, queued: usize) {
    for _ in 0..100 {
        let snapshot = component
            .background_snapshot()
            .expect("background queue exists");
        if snapshot.running == running && snapshot.queued == queued {
            return;
        }
        tokio::task::yield_now().await;
    }
    panic!("background queue did not reach running={running}, queued={queued}");
}

fn storage_statuses(events: &[Event]) -> Vec<String> {
    events
        .iter()
        .filter(|event| event.name() == "memory.storage")
        .filter_map(|event| {
            event
                .data()
                .and_then(|data| data.get("status"))
                .and_then(Json::as_str)
                .map(str::to_string)
        })
        .collect()
}

#[tokio::test]
async fn queued_write_back_returns_before_store_then_flush_enables_next_session_recall() {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|error| error.into_inner());
    reset_runtime();
    let provider = GateProvider::new();
    let direct = MemoryRuntime::new(provider.clone());
    let component = MemoryComponent::new(provider.clone(), background_config()).expect("component");
    let mut installation = component
        .install_global("background-memory", 0)
        .expect("installation");
    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let event_sink = events.clone();
    register_subscriber(
        "background-memory-events",
        Arc::new(move |event| event_sink.lock().expect("event lock").push(event.clone())),
    )
    .expect("subscriber");

    let first_seen = Arc::new(Mutex::new(Vec::new()));
    execute_turn(
        &namespace("session-a"),
        "BACKGROUND_SENTINEL prefers a solarized dark editor",
        "Preference acknowledged",
        first_seen.clone(),
        None,
    )
    .await
    .expect("LLM response returns after queue admission");
    wait_for_snapshot(&component, 1, 0).await;
    assert_eq!(provider.store_calls.load(Ordering::SeqCst), 1);
    assert!(
        direct
            .search(
                MemorySearchRequest::new(
                    nemo_relay_memory::memory::MemoryRequestContext::new("before-flush")
                        .expect("context"),
                    namespace("session-b"),
                    "solarized editor",
                )
                .expect("search request")
            )
            .await
            .expect("search")
            .matches
            .is_empty()
    );
    assert!(
        !first_seen.lock().expect("request lock")[0]
            .content
            .to_string()
            .contains("<relay_memory")
    );

    provider.gate.add_permits(1);
    assert!(
        component
            .flush_background(Duration::from_secs(1))
            .await
            .expect("flush")
    );

    let second_seen = Arc::new(Mutex::new(Vec::new()));
    execute_turn(
        &namespace("session-b"),
        "Which editor does BACKGROUND_SENTINEL prefer?",
        "Solarized dark.",
        second_seen.clone(),
        None,
    )
    .await
    .expect("second LLM response");
    let injected = second_seen.lock().expect("request lock")[0]
        .content
        .to_string();
    assert!(injected.contains("<relay_memory version=\\\"0.1\\\">"));
    assert!(injected.contains("solarized dark editor"));

    provider.gate.add_permits(1);
    component
        .flush_background(Duration::from_secs(1))
        .await
        .expect("second flush");
    flush_subscribers().expect("subscriber flush");
    let captured = events.lock().expect("event lock").clone();
    let statuses = storage_statuses(&captured);
    assert!(statuses.contains(&"queued".to_string()));
    assert!(statuses.contains(&"running".to_string()));
    assert!(statuses.contains(&"stored".to_string()));
    let llm_uuids = captured
        .iter()
        .filter(|event| event.name() == "background-memory-agent")
        .map(Event::uuid)
        .collect::<Vec<_>>();
    assert!(
        captured
            .iter()
            .filter(|event| event.name() == "memory.storage")
            .all(|event| event
                .parent_uuid()
                .is_some_and(|parent| llm_uuids.contains(&parent)))
    );
    let memory_events = captured
        .iter()
        .filter(|event| {
            event
                .category()
                .is_some_and(|category| category.as_str() == "memory")
        })
        .collect::<Vec<_>>();
    let evidence = serde_json::to_string(&memory_events).expect("event serialization");
    assert!(!evidence.contains("BACKGROUND_SENTINEL"));
    assert!(!evidence.contains("solarized dark editor"));
    assert!(!evidence.contains("tenant-background"));
    assert!(!evidence.contains("subject-alex"));

    installation.close().expect("deregister hook");
    component
        .shutdown_background(Duration::from_secs(1))
        .await
        .expect("shutdown");
    deregister_subscriber("background-memory-events").expect("deregister subscriber");
}

#[tokio::test]
async fn provider_failure_after_admission_is_state_and_evidence_not_response_failure() {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|error| error.into_inner());
    reset_runtime();
    let provider = RetryFailureProvider {
        inner: InMemoryProvider::new(),
        store_calls: Arc::new(AtomicUsize::new(0)),
    };
    let mut config = background_config();
    config.storage_policy = FailurePolicy::FailClosed;
    let component = MemoryComponent::new(provider.clone(), config).expect("component");
    let mut installation = component
        .install_global("background-failure", 0)
        .expect("installation");
    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let event_sink = events.clone();
    register_subscriber(
        "background-failure-events",
        Arc::new(move |event| event_sink.lock().expect("event lock").push(event.clone())),
    )
    .expect("subscriber");

    execute_turn(
        &namespace("session-failure"),
        "FAILURE_SENTINEL should not leak",
        "The callback still succeeds.",
        Arc::new(Mutex::new(Vec::new())),
        None,
    )
    .await
    .expect("admitted background work cannot retroactively fail the response");
    component
        .flush_background(Duration::from_secs(1))
        .await
        .expect("failed work is terminal and flushable");
    let snapshot = component.background_snapshot().expect("queue exists");
    assert_eq!(snapshot.failed_total, 1);
    assert_eq!(provider.store_calls.load(Ordering::SeqCst), 2);

    flush_subscribers().expect("subscriber flush");
    let captured = events.lock().expect("event lock").clone();
    let statuses = storage_statuses(&captured);
    assert!(statuses.contains(&"retrying".to_string()));
    assert!(statuses.contains(&"failed".to_string()));
    let job_id = captured
        .iter()
        .filter(|event| event.name() == "memory.storage")
        .find_map(|event| {
            event
                .data()
                .filter(|data| data["status"] == "failed")
                .and_then(|data| data["job_id"].as_str())
        })
        .expect("failed transition has a job id");
    let job = component
        .background_job(job_id)
        .expect("failed job retained");
    assert_eq!(job.state, MemoryJobState::Failed);
    assert_eq!(job.attempts, 2);
    let memory_events = captured
        .iter()
        .filter(|event| {
            event
                .category()
                .is_some_and(|category| category.as_str() == "memory")
        })
        .collect::<Vec<_>>();
    let evidence = serde_json::to_string(&memory_events).expect("event serialization");
    assert!(!evidence.contains("FAILURE_SENTINEL"));

    installation.close().expect("deregister hook");
    component
        .shutdown_background(Duration::from_secs(1))
        .await
        .expect("shutdown");
    deregister_subscriber("background-failure-events").expect("deregister subscriber");
}

#[tokio::test]
async fn fail_closed_admission_rejection_fails_completion_after_callback() {
    let _guard = TEST_MUTEX.lock().unwrap_or_else(|error| error.into_inner());
    reset_runtime();
    let provider = GateProvider::new();
    let mut config = background_config();
    config.storage_policy = FailurePolicy::FailClosed;
    config.background_queue = MemoryWorkQueueConfig {
        capacity: 1,
        backpressure: MemoryBackpressurePolicy::Reject,
        max_attempts: 1,
        attempt_timeout_millis: 10_000,
        ..MemoryWorkQueueConfig::default()
    };
    let component = MemoryComponent::new(provider.clone(), config).expect("component");
    let mut installation = component
        .install_global("background-rejection", 0)
        .expect("installation");
    let callbacks = Arc::new(AtomicUsize::new(0));
    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let event_sink = events.clone();
    register_subscriber(
        "background-rejection-events",
        Arc::new(move |event| event_sink.lock().expect("event lock").push(event.clone())),
    )
    .expect("subscriber");

    execute_turn(
        &namespace("session-1"),
        "first queued turn",
        "first",
        Arc::new(Mutex::new(Vec::new())),
        Some(callbacks.clone()),
    )
    .await
    .expect("first admitted");
    wait_for_snapshot(&component, 1, 0).await;
    execute_turn(
        &namespace("session-2"),
        "second queued turn",
        "second",
        Arc::new(Mutex::new(Vec::new())),
        Some(callbacks.clone()),
    )
    .await
    .expect("second admitted to pending capacity");
    wait_for_snapshot(&component, 1, 1).await;
    let error = execute_turn(
        &namespace("session-3"),
        "third rejected turn",
        "third",
        Arc::new(Mutex::new(Vec::new())),
        Some(callbacks.clone()),
    )
    .await
    .expect_err("fail-closed admission rejection fails lifecycle completion");
    assert!(error.to_string().contains("storage admission"));
    assert_eq!(callbacks.load(Ordering::SeqCst), 3);
    assert_eq!(
        component
            .background_snapshot()
            .expect("queue")
            .rejected_total,
        1
    );
    flush_subscribers().expect("subscriber flush");
    assert!(
        storage_statuses(&events.lock().expect("event lock")).contains(&"rejected".to_string())
    );

    provider.gate.add_permits(2);
    installation.close().expect("deregister hook");
    component
        .shutdown_background(Duration::from_secs(1))
        .await
        .expect("accepted work drains");
    deregister_subscriber("background-rejection-events").expect("deregister subscriber");
}

#[tokio::test]
async fn inline_mode_ignores_unused_queue_config_and_reports_no_background_worker() {
    let mut config = AutomaticMemoryConfig::default();
    config.background_queue.capacity = 0;
    let inline = MemoryComponent::new(InMemoryProvider::new(), config.clone())
        .expect("unused background config does not change inline compatibility");
    assert!(inline.background_snapshot().is_none());
    assert!(
        !inline
            .flush_background(Duration::from_millis(1))
            .await
            .expect("inline flush is a no-op")
    );

    config.write_delivery = WriteDelivery::Background;
    let error = match MemoryComponent::new(InMemoryProvider::new(), config) {
        Ok(_) => panic!("background mode must validate queue bounds"),
        Err(error) => error,
    };
    assert_eq!(error.code, MemoryErrorCode::InvalidRequest);
}

#[test]
fn background_status_types_remain_serializable_for_bindings() {
    let status = MemoryJobState::Queued;
    assert_eq!(
        serde_json::to_value(status).expect("serialize"),
        json!("queued")
    );
}
