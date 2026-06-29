// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests for automatic managed-LLM memory.

#![cfg(feature = "relay")]
#![allow(clippy::await_holding_lock)]

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

use async_trait::async_trait;
use nemo_relay::api::event::Event;
use nemo_relay::api::llm::{LlmCallExecuteParams, LlmRequest, llm_call_execute};
use nemo_relay::api::runtime::{
    LlmExecutionNextFn, NemoRelayContextState, TASK_SCOPE_STACK, create_scope_stack,
    global_context, set_thread_scope_stack,
};
use nemo_relay::api::scope::{PopScopeParams, PushScopeParams, ScopeType, pop_scope, push_scope};
use nemo_relay::api::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
use nemo_relay::codec::openai_chat::OpenAIChatCodec;
use nemo_relay::error::FlowError;
use nemo_relay::plugin::{
    PluginConfig, clear_plugin_configuration, initialize_plugins_exact, list_plugin_kinds,
    validate_plugin_config,
};
use nemo_relay_memory::memory::{
    MemoryContent, MemoryErrorCode, MemoryFilter, MemoryNamespace, MemoryOperationError,
    MemoryProvenance, MemoryRequestContext, MemorySearchRequest, MemorySearchResult,
    MemorySearchScope, MemoryStoreRequest, MemoryStoreResult,
};
use nemo_relay_memory::{
    AutomaticMemoryConfig, FailurePolicy, InMemoryProvider, MemoryComponent, MemoryProvider,
    MemoryProviderResult, MemoryRuntime, register_memory_component,
};
use serde_json::{Value as Json, json};

static TEST_MUTEX: Mutex<()> = Mutex::new(());

fn reset_runtime() {
    let context = global_context();
    *context.write().unwrap() = NemoRelayContextState::new();
    set_thread_scope_stack(create_scope_stack());
}

fn namespace(tenant: &str, subject: &str, session: &str) -> MemoryNamespace {
    MemoryNamespace {
        tenant_id: tenant.to_string(),
        subject_id: subject.to_string(),
        session_id: Some(session.to_string()),
        agent_id: Some("assistant".to_string()),
    }
}

fn memory_metadata(namespace: &MemoryNamespace) -> Json {
    json!({"memory": {"namespace": namespace}})
}

fn request(user: &str) -> LlmRequest {
    LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": user}],
            "preserved": {"provider": true},
        }),
    }
}

fn response(assistant: &str) -> Json {
    json!({
        "id": "response-1",
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": assistant},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    })
}

async fn execute_turn(
    namespace: &MemoryNamespace,
    user: &str,
    assistant: &str,
    seen_requests: Arc<Mutex<Vec<LlmRequest>>>,
) -> Result<Json, FlowError> {
    let scope = push_scope(
        PushScopeParams::builder()
            .name("automatic-memory-session")
            .scope_type(ScopeType::Agent)
            .metadata(memory_metadata(namespace))
            .build(),
    )?;
    let answer = assistant.to_string();
    let callback: LlmExecutionNextFn = Arc::new(move |request| {
        seen_requests.lock().unwrap().push(request);
        let answer = answer.clone();
        Box::pin(async move { Ok(response(&answer)) })
    });
    let result = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("automatic-memory-agent")
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

fn search_request(namespace: MemoryNamespace, query: &str) -> MemorySearchRequest {
    MemorySearchRequest {
        context: MemoryRequestContext::new("test-search").unwrap(),
        namespace,
        query: query.to_string(),
        scope: MemorySearchScope::Subject,
        filter: MemoryFilter::default(),
        limit: 20,
    }
}

#[tokio::test]
async fn two_sessions_recall_inject_store_and_emit_private_evidence_automatically() {
    let _guard = TEST_MUTEX.lock().unwrap();
    reset_runtime();

    let provider = InMemoryProvider::new();
    let direct = MemoryRuntime::new(provider.clone());
    let component = MemoryComponent::new(provider, AutomaticMemoryConfig::default()).unwrap();
    let mut installation = component.install_global("automatic-memory", 0).unwrap();
    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let event_sink = events.clone();
    register_subscriber(
        "automatic-memory-events",
        Arc::new(move |event| event_sink.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    let session_a_requests = Arc::new(Mutex::new(Vec::new()));
    execute_turn(
        &namespace("tenant-a", "alex", "session-a"),
        "SENTINEL_USER prefers solarized dark editor theme",
        "SENTINEL_ASSISTANT acknowledged the preference",
        session_a_requests.clone(),
    )
    .await
    .unwrap();
    assert!(
        !session_a_requests.lock().unwrap()[0]
            .content
            .to_string()
            .contains("<relay_memory")
    );

    let session_b_requests = Arc::new(Mutex::new(Vec::new()));
    execute_turn(
        &namespace("tenant-a", "alex", "session-b"),
        "Which editor theme does SENTINEL_USER prefer?",
        "The preference is solarized dark.",
        session_b_requests.clone(),
    )
    .await
    .unwrap();
    let injected = session_b_requests.lock().unwrap()[0].content.to_string();
    assert!(injected.contains("<relay_memory version=\\\"0.1\\\">"));
    assert!(injected.contains("solarized dark editor theme"));
    assert!(injected.contains("Which editor theme"));
    assert_eq!(
        session_b_requests.lock().unwrap()[0].content["preserved"],
        json!({"provider": true})
    );

    let records = direct
        .search(search_request(
            namespace("tenant-a", "alex", "session-b"),
            "solarized dark editor theme",
        ))
        .await
        .unwrap();
    assert!(records.matches.len() >= 2);
    let stored_wire = serde_json::to_string(&records.matches[0].record.content).unwrap();
    assert!(!stored_wire.contains("<relay_memory"));
    assert!(records.matches.iter().any(|memory_match| {
        serde_json::to_string(&memory_match.record.content)
            .unwrap()
            .contains("SENTINEL_USER prefers solarized")
    }));

    flush_subscribers().unwrap();
    let memory_events = events
        .lock()
        .unwrap()
        .iter()
        .filter(|event| {
            event
                .category()
                .is_some_and(|category| category.as_str() == "memory")
        })
        .cloned()
        .collect::<Vec<_>>();
    assert!(
        memory_events
            .iter()
            .any(|event| event.name() == "memory.retrieval")
    );
    assert!(
        memory_events
            .iter()
            .any(|event| event.name() == "memory.injection")
    );
    assert!(
        memory_events
            .iter()
            .any(|event| event.name() == "memory.storage")
    );
    let evidence = serde_json::to_string(&memory_events).unwrap();
    assert!(!evidence.contains("SENTINEL_USER"));
    assert!(!evidence.contains("SENTINEL_ASSISTANT"));
    assert!(!evidence.contains("solarized dark editor theme"));
    assert!(evidence.contains("content_hash"));
    assert_eq!(component.active_turns(), 0);

    assert!(installation.close().unwrap());
    assert!(!installation.close().unwrap());
    deregister_subscriber("automatic-memory-events").unwrap();
}

#[tokio::test]
async fn call_opt_out_preserves_request_and_skips_storage() {
    let _guard = TEST_MUTEX.lock().unwrap();
    reset_runtime();

    let provider = InMemoryProvider::new();
    let direct = MemoryRuntime::new(provider.clone());
    let component = MemoryComponent::new(
        provider,
        AutomaticMemoryConfig {
            namespace: Some(namespace("tenant", "subject", "fallback")),
            ..AutomaticMemoryConfig::default()
        },
    )
    .unwrap();
    let _installation = component
        .install_global("automatic-memory-opt-out", 0)
        .unwrap();
    let original = request("do not remember this sentinel");
    let seen = Arc::new(Mutex::new(None));
    let seen_sink = seen.clone();
    let callback: LlmExecutionNextFn = Arc::new(move |request| {
        *seen_sink.lock().unwrap() = Some(request);
        Box::pin(async { Ok(response("not stored")) })
    });
    llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("opt-out-agent")
            .request(original.clone())
            .func(callback)
            .metadata(json!({"memory": {"enabled": false}}))
            .codec(Arc::new(OpenAIChatCodec))
            .response_codec(Arc::new(OpenAIChatCodec))
            .build(),
    )
    .await
    .unwrap();

    assert_eq!(
        seen.lock().unwrap().as_ref().unwrap().content,
        original.content
    );
    let search = direct
        .search(search_request(
            namespace("tenant", "subject", "other"),
            "remember sentinel",
        ))
        .await
        .unwrap();
    assert!(search.matches.is_empty());
    assert_eq!(component.active_turns(), 0);
}

#[derive(Clone)]
struct ControlledProvider {
    inner: InMemoryProvider,
    fail_search: bool,
    fail_store: bool,
}

#[async_trait]
impl MemoryProvider for ControlledProvider {
    fn name(&self) -> &str {
        "controlled"
    }

    async fn search(
        &self,
        request: MemorySearchRequest,
    ) -> MemoryProviderResult<MemorySearchResult> {
        if self.fail_search {
            Err(provider_failure("search"))
        } else {
            self.inner.search(request).await
        }
    }

    async fn store(&self, request: MemoryStoreRequest) -> MemoryProviderResult<MemoryStoreResult> {
        if self.fail_store {
            Err(provider_failure("store"))
        } else {
            self.inner.store(request).await
        }
    }
}

fn provider_failure(stage: &str) -> MemoryOperationError {
    MemoryOperationError::new(
        MemoryErrorCode::ProviderUnavailable,
        format!("synthetic {stage} failure"),
        true,
    )
    .with_provider("controlled")
}

#[tokio::test]
async fn fail_open_search_continues_but_fail_closed_search_blocks_callback() {
    let _guard = TEST_MUTEX.lock().unwrap();
    for (policy, expected_calls, should_succeed) in [
        (FailurePolicy::FailOpen, 1, true),
        (FailurePolicy::FailClosed, 0, false),
    ] {
        reset_runtime();
        let provider = ControlledProvider {
            inner: InMemoryProvider::new(),
            fail_search: true,
            fail_store: false,
        };
        let component = MemoryComponent::new(
            provider,
            AutomaticMemoryConfig {
                namespace: Some(namespace("tenant", "subject", "session")),
                retrieval_policy: policy,
                ..AutomaticMemoryConfig::default()
            },
        )
        .unwrap();
        let _installation = component
            .install_global(format!("search-policy-{policy:?}"), 0)
            .unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_sink = calls.clone();
        let result = llm_call_execute(
            LlmCallExecuteParams::builder()
                .name("search-policy-agent")
                .request(request("remember policy"))
                .func(Arc::new(move |_| {
                    calls_sink.fetch_add(1, Ordering::SeqCst);
                    Box::pin(async { Ok(response("done")) })
                }))
                .codec(Arc::new(OpenAIChatCodec))
                .response_codec(Arc::new(OpenAIChatCodec))
                .build(),
        )
        .await;
        assert_eq!(result.is_ok(), should_succeed);
        assert_eq!(calls.load(Ordering::SeqCst), expected_calls);
        assert_eq!(component.active_turns(), 0);
    }
}

#[tokio::test]
async fn fail_closed_store_returns_error_after_callback_and_cleans_state() {
    let _guard = TEST_MUTEX.lock().unwrap();
    reset_runtime();
    let component = MemoryComponent::new(
        ControlledProvider {
            inner: InMemoryProvider::new(),
            fail_search: false,
            fail_store: true,
        },
        AutomaticMemoryConfig {
            namespace: Some(namespace("tenant", "subject", "session")),
            storage_policy: FailurePolicy::FailClosed,
            ..AutomaticMemoryConfig::default()
        },
    )
    .unwrap();
    let _installation = component.install_global("store-fail-closed", 0).unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_sink = calls.clone();
    let error = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("store-policy-agent")
            .request(request("store this turn"))
            .func(Arc::new(move |_| {
                calls_sink.fetch_add(1, Ordering::SeqCst);
                Box::pin(async { Ok(response("completed before store")) })
            }))
            .codec(Arc::new(OpenAIChatCodec))
            .response_codec(Arc::new(OpenAIChatCodec))
            .build(),
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("automatic memory storage"));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(component.active_turns(), 0);
}

#[tokio::test]
async fn missing_identity_defaults_fail_closed_before_callback() {
    let _guard = TEST_MUTEX.lock().unwrap();
    reset_runtime();
    let component =
        MemoryComponent::new(InMemoryProvider::new(), AutomaticMemoryConfig::default()).unwrap();
    let _installation = component.install_global("missing-identity", 0).unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let calls_sink = calls.clone();
    let result = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("missing-identity-agent")
            .request(request("hello"))
            .func(Arc::new(move |_| {
                calls_sink.fetch_add(1, Ordering::SeqCst);
                Box::pin(async { Ok(response("unreachable")) })
            }))
            .codec(Arc::new(OpenAIChatCodec))
            .build(),
    )
    .await;
    assert!(result.is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(component.active_turns(), 0);
}

#[tokio::test]
async fn callback_failure_skips_storage_and_releases_turn_state() {
    let _guard = TEST_MUTEX.lock().unwrap();
    reset_runtime();
    let provider = InMemoryProvider::new();
    let direct = MemoryRuntime::new(provider.clone());
    let component = MemoryComponent::new(
        provider,
        AutomaticMemoryConfig {
            namespace: Some(namespace("tenant", "subject", "session")),
            ..AutomaticMemoryConfig::default()
        },
    )
    .unwrap();
    let _installation = component.install_global("callback-failure", 0).unwrap();
    let error = llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("callback-failure-agent")
            .request(request("do not store a failed turn"))
            .func(Arc::new(|_| {
                Box::pin(async { Err(FlowError::Internal("synthetic callback failure".into())) })
            }))
            .codec(Arc::new(OpenAIChatCodec))
            .build(),
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("synthetic callback failure"));
    assert_eq!(component.active_turns(), 0);
    assert!(
        direct
            .search(search_request(
                namespace("tenant", "subject", "other"),
                "failed turn",
            ))
            .await
            .unwrap()
            .matches
            .is_empty()
    );
}

async fn seed(runtime: &MemoryRuntime, namespace: MemoryNamespace, text: &str, id: &str) {
    runtime
        .store(MemoryStoreRequest {
            context: MemoryRequestContext::new(format!("seed-{id}")).unwrap(),
            namespace,
            content: MemoryContent::Text {
                text: text.to_string(),
            },
            event_timestamp: SystemTime::now().into(),
            provenance: MemoryProvenance {
                source: "test".to_string(),
                source_ids: vec![id.to_string()],
                ..MemoryProvenance::default()
            },
            metadata: BTreeMap::new(),
            idempotency_key: Some(id.to_string()),
        })
        .await
        .unwrap();
}

#[tokio::test]
async fn selection_enforces_item_budget_with_deterministic_stable_ties() {
    let _guard = TEST_MUTEX.lock().unwrap();
    reset_runtime();
    let provider = InMemoryProvider::new();
    let direct = MemoryRuntime::new(provider.clone());
    let identity = namespace("tenant", "subject", "session");
    seed(
        &direct,
        identity.clone(),
        "shared first-only record",
        "first",
    )
    .await;
    seed(
        &direct,
        identity.clone(),
        "shared second-only record",
        "second",
    )
    .await;
    let component = MemoryComponent::new(
        provider,
        AutomaticMemoryConfig {
            namespace: Some(identity),
            max_candidates: 2,
            max_items: 1,
            ..AutomaticMemoryConfig::default()
        },
    )
    .unwrap();
    let _installation = component.install_global("item-budget", 0).unwrap();
    let seen = Arc::new(Mutex::new(None));
    let seen_sink = seen.clone();
    llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("item-budget-agent")
            .request(request("shared record"))
            .func(Arc::new(move |request| {
                *seen_sink.lock().unwrap() = Some(request);
                Box::pin(async { Ok(response("done")) })
            }))
            .codec(Arc::new(OpenAIChatCodec))
            .response_codec(Arc::new(OpenAIChatCodec))
            .build(),
    )
    .await
    .unwrap();
    let wire = seen.lock().unwrap().as_ref().unwrap().content.to_string();
    assert!(wire.contains("first-only"));
    assert!(!wire.contains("second-only"));
}

#[tokio::test]
async fn scope_installation_disappears_on_pop_and_close_is_idempotent() {
    let _guard = TEST_MUTEX.lock().unwrap();
    reset_runtime();
    let provider = InMemoryProvider::new();
    let direct = MemoryRuntime::new(provider.clone());
    let identity = namespace("tenant", "subject", "session");
    seed(
        &direct,
        identity.clone(),
        "scope-only durable preference",
        "scope-only",
    )
    .await;
    let component = MemoryComponent::new(
        provider,
        AutomaticMemoryConfig {
            namespace: Some(identity),
            ..AutomaticMemoryConfig::default()
        },
    )
    .unwrap();
    let scope = push_scope(
        PushScopeParams::builder()
            .name("memory-owner")
            .scope_type(ScopeType::Agent)
            .build(),
    )
    .unwrap();
    let mut installation = component.install_scope(&scope, "scope-memory", 0).unwrap();
    let inside = Arc::new(Mutex::new(None));
    let inside_sink = inside.clone();
    llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("inside-scope-memory")
            .request(request("durable preference"))
            .func(Arc::new(move |request| {
                *inside_sink.lock().unwrap() = Some(request);
                Box::pin(async { Ok(response("inside")) })
            }))
            .codec(Arc::new(OpenAIChatCodec))
            .response_codec(Arc::new(OpenAIChatCodec))
            .build(),
    )
    .await
    .unwrap();
    assert!(
        inside
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .content
            .to_string()
            .contains("scope-only")
    );
    pop_scope(PopScopeParams::builder().handle_uuid(&scope.uuid).build()).unwrap();

    let outside = Arc::new(Mutex::new(None));
    let outside_sink = outside.clone();
    llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("outside-scope-memory")
            .request(request("durable preference"))
            .func(Arc::new(move |request| {
                *outside_sink.lock().unwrap() = Some(request);
                Box::pin(async { Ok(response("outside")) })
            }))
            .codec(Arc::new(OpenAIChatCodec))
            .response_codec(Arc::new(OpenAIChatCodec))
            .build(),
    )
    .await
    .unwrap();
    assert!(
        !outside
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .content
            .to_string()
            .contains("<relay_memory")
    );
    assert!(!installation.close().unwrap());
    assert!(!installation.close().unwrap());
}

#[tokio::test]
async fn concurrent_scope_stacks_keep_automatic_identity_isolated() {
    let _guard = TEST_MUTEX.lock().unwrap();
    reset_runtime();
    let provider = InMemoryProvider::new();
    let direct = MemoryRuntime::new(provider.clone());
    seed(
        &direct,
        namespace("tenant", "alex", "seed"),
        "alex-only solarized preference",
        "alex",
    )
    .await;
    seed(
        &direct,
        namespace("tenant", "blair", "seed"),
        "blair-only emacs preference",
        "blair",
    )
    .await;
    let component = MemoryComponent::new(provider, AutomaticMemoryConfig::default()).unwrap();
    let _installation = component.install_global("concurrent-memory", 0).unwrap();

    let alex_seen = Arc::new(Mutex::new(Vec::new()));
    let blair_seen = Arc::new(Mutex::new(Vec::new()));
    let alex_namespace = namespace("tenant", "alex", "query");
    let blair_namespace = namespace("tenant", "blair", "query");
    let alex = TASK_SCOPE_STACK.scope(
        create_scope_stack(),
        execute_turn(
            &alex_namespace,
            "What is my solarized preference?",
            "alex answer",
            alex_seen.clone(),
        ),
    );
    let blair = TASK_SCOPE_STACK.scope(
        create_scope_stack(),
        execute_turn(
            &blair_namespace,
            "What is my emacs preference?",
            "blair answer",
            blair_seen.clone(),
        ),
    );
    let (alex_result, blair_result) = tokio::join!(alex, blair);
    alex_result.unwrap();
    blair_result.unwrap();

    let alex_wire = alex_seen.lock().unwrap()[0].content.to_string();
    let blair_wire = blair_seen.lock().unwrap()[0].content.to_string();
    assert!(alex_wire.contains("alex-only"));
    assert!(!alex_wire.contains("blair-only"));
    assert!(blair_wire.contains("blair-only"));
    assert!(!blair_wire.contains("alex-only"));
    assert_eq!(component.active_turns(), 0);
}

#[tokio::test]
async fn plugin_config_installs_reference_automatic_memory_and_rejects_unknown_provider() {
    let _guard = TEST_MUTEX.lock().unwrap();
    reset_runtime();
    register_memory_component().unwrap();
    register_memory_component().unwrap();
    assert!(list_plugin_kinds().contains(&"memory".to_string()));

    let invalid: PluginConfig = serde_json::from_value(json!({
        "version": 1,
        "components": [{"kind": "memory", "config": {"provider": "unknown"}}]
    }))
    .unwrap();
    assert!(validate_plugin_config(&invalid).has_errors());

    let config: PluginConfig = serde_json::from_value(json!({
        "version": 1,
        "components": [{
            "kind": "memory",
            "config": {
                "provider": "in_memory",
                "namespace": {
                    "tenant_id": "plugin-tenant",
                    "subject_id": "plugin-subject",
                    "session_id": "plugin-session",
                    "agent_id": "assistant"
                }
            }
        }]
    }))
    .unwrap();
    initialize_plugins_exact(config).await.unwrap();
    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let event_sink = events.clone();
    register_subscriber(
        "plugin-memory-events",
        Arc::new(move |event| event_sink.lock().unwrap().push(event.clone())),
    )
    .unwrap();
    llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("plugin-memory-agent")
            .request(request("plugin memory turn"))
            .func(Arc::new(|_| {
                Box::pin(async { Ok(response("plugin answer")) })
            }))
            .codec(Arc::new(OpenAIChatCodec))
            .response_codec(Arc::new(OpenAIChatCodec))
            .build(),
    )
    .await
    .unwrap();
    flush_subscribers().unwrap();
    assert!(
        events
            .lock()
            .unwrap()
            .iter()
            .any(|event| event.name() == "memory.storage")
    );
    deregister_subscriber("plugin-memory-events").unwrap();
    clear_plugin_configuration().unwrap();
}
