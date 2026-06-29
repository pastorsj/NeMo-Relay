// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Behavioral tests for the deterministic in-memory reference provider.

use std::collections::BTreeMap;
use std::sync::Arc;

use nemo_relay_memory::memory::{
    MemoryContent, MemoryErrorCode, MemoryFilter, MemoryNamespace, MemoryProvenance,
    MemoryRequestContext, MemorySearchRequest, MemorySearchScope, MemoryStoreDisposition,
    MemoryStoreRequest,
};
use nemo_relay_memory::{InMemoryProvider, MemoryRuntime};

fn namespace(tenant: &str, subject: &str, session: &str, agent: &str) -> MemoryNamespace {
    MemoryNamespace {
        tenant_id: tenant.to_string(),
        subject_id: subject.to_string(),
        session_id: Some(session.to_string()),
        agent_id: Some(agent.to_string()),
    }
}

fn store_request(
    operation: &str,
    namespace: MemoryNamespace,
    text: &str,
    event_timestamp: &str,
    idempotency_key: Option<&str>,
) -> MemoryStoreRequest {
    MemoryStoreRequest {
        context: MemoryRequestContext::new(operation).unwrap(),
        namespace,
        content: MemoryContent::Text {
            text: text.to_string(),
        },
        event_timestamp: serde_json::from_str(&format!("\"{event_timestamp}\""))
            .expect("valid timestamp"),
        provenance: MemoryProvenance {
            source: "test".to_string(),
            source_ids: vec![format!("event-{operation}")],
            ..MemoryProvenance::default()
        },
        metadata: BTreeMap::from([("category".to_string(), "preference".into())]),
        idempotency_key: idempotency_key.map(str::to_string),
    }
}

fn search_request(
    operation: &str,
    namespace: MemoryNamespace,
    query: &str,
    scope: MemorySearchScope,
) -> MemorySearchRequest {
    MemorySearchRequest {
        context: MemoryRequestContext::new(operation).unwrap(),
        namespace,
        query: query.to_string(),
        scope,
        filter: MemoryFilter::default(),
        limit: 10,
    }
}

#[tokio::test]
async fn subject_scope_crosses_sessions_without_crossing_subject_or_tenant() {
    let runtime = MemoryRuntime::new(InMemoryProvider::new());
    let origin = namespace("tenant-a", "alex", "session-a", "assistant");
    let stored = runtime
        .store(store_request(
            "store-preference",
            origin.clone(),
            "Alex prefers solarized dark editor theme",
            "2026-01-02T00:00:00Z",
            None,
        ))
        .await
        .unwrap();

    let recalled = runtime
        .search(search_request(
            "search-session-b",
            namespace("tenant-a", "alex", "session-b", "assistant"),
            "solarized editor theme",
            MemorySearchScope::Subject,
        ))
        .await
        .unwrap();
    assert_eq!(recalled.matches.len(), 1);
    assert_eq!(recalled.matches[0].record.id, stored.record.id);
    assert_eq!(recalled.matches[0].record.namespace, origin);

    for isolated in [
        namespace("tenant-a", "blair", "session-b", "assistant"),
        namespace("tenant-b", "alex", "session-b", "assistant"),
    ] {
        let result = runtime
            .search(search_request(
                "isolated-search",
                isolated,
                "solarized editor theme",
                MemorySearchScope::Subject,
            ))
            .await
            .unwrap();
        assert!(result.matches.is_empty());
    }
}

#[tokio::test]
async fn agent_session_and_exact_scopes_only_narrow_the_subject_partition() {
    let runtime = MemoryRuntime::new(InMemoryProvider::new());
    for (operation, session, agent, text) in [
        ("one", "session-a", "assistant", "shared alpha fact"),
        ("two", "session-b", "assistant", "shared beta fact"),
        ("three", "session-a", "researcher", "shared gamma fact"),
    ] {
        runtime
            .store(store_request(
                operation,
                namespace("tenant", "subject", session, agent),
                text,
                "2026-01-02T00:00:00Z",
                None,
            ))
            .await
            .unwrap();
    }

    let cases = [
        (MemorySearchScope::Subject, "session-z", "assistant", 3),
        (MemorySearchScope::Agent, "session-z", "assistant", 2),
        (MemorySearchScope::Session, "session-a", "assistant", 2),
        (MemorySearchScope::Exact, "session-a", "assistant", 1),
    ];
    for (scope, session, agent, expected) in cases {
        let result = runtime
            .search(search_request(
                "scope-search",
                namespace("tenant", "subject", session, agent),
                "shared fact",
                scope,
            ))
            .await
            .unwrap();
        assert_eq!(result.matches.len(), expected, "scope {scope:?}");
    }
}

#[tokio::test]
async fn ranking_ties_limits_and_filters_are_deterministic() {
    let runtime = MemoryRuntime::new(InMemoryProvider::new());
    let namespace = namespace("tenant", "subject", "session", "assistant");
    for (operation, text, time) in [
        ("first", "alpha beta", "2026-01-01T00:00:00Z"),
        ("second", "alpha gamma", "2026-01-02T00:00:00Z"),
        ("third", "alpha beta gamma", "2026-01-03T00:00:00Z"),
    ] {
        runtime
            .store(store_request(
                operation,
                namespace.clone(),
                text,
                time,
                None,
            ))
            .await
            .unwrap();
    }

    let mut request = search_request(
        "rank-search",
        namespace.clone(),
        "alpha beta gamma",
        MemorySearchScope::Subject,
    );
    request.limit = 2;
    let first = runtime.search(request.clone()).await.unwrap();
    let second = runtime.search(request).await.unwrap();
    assert_eq!(first, second);
    assert_eq!(first.matches.len(), 2);
    assert_eq!(first.matches[0].rank, 1);
    assert_eq!(first.matches[0].score, 1.0);
    assert_eq!(first.matches[1].rank, 2);
    assert_eq!(first.matches[1].record.id, "memory-0000000000000001");

    let mut filtered = search_request(
        "filtered-search",
        namespace,
        "alpha",
        MemorySearchScope::Subject,
    );
    filtered.filter.event_after =
        Some(serde_json::from_str("\"2026-01-02T00:00:00Z\"").expect("valid timestamp"));
    filtered.filter.event_before =
        Some(serde_json::from_str("\"2026-01-02T00:00:00Z\"").expect("valid timestamp"));
    filtered.filter.metadata.insert(
        "category".to_string(),
        serde_json::Value::String("preference".to_string()),
    );
    let result = runtime.search(filtered).await.unwrap();
    assert_eq!(result.matches.len(), 1);
    assert_eq!(result.matches[0].record.id, "memory-0000000000000002");
}

#[tokio::test]
async fn idempotency_replays_original_and_conflicts_without_mutating() {
    let runtime = MemoryRuntime::new(InMemoryProvider::new());
    let namespace = namespace("tenant", "subject", "session", "assistant");
    let original = store_request(
        "store-original",
        namespace.clone(),
        "durable preference",
        "2026-01-01T00:00:00Z",
        Some("turn-1"),
    );
    let created = runtime.store(original.clone()).await.unwrap();
    let replayed = runtime.store(original).await.unwrap();
    assert_eq!(created.disposition, MemoryStoreDisposition::Created);
    assert_eq!(replayed.disposition, MemoryStoreDisposition::Existing);
    assert_eq!(created.record, replayed.record);

    let conflict = runtime
        .store(store_request(
            "store-conflict",
            namespace.clone(),
            "different preference",
            "2026-01-01T00:00:00Z",
            Some("turn-1"),
        ))
        .await
        .expect_err("different payload must conflict");
    assert_eq!(conflict.code, MemoryErrorCode::Conflict);

    let result = runtime
        .search(search_request(
            "search-after-conflict",
            namespace,
            "preference",
            MemorySearchScope::Subject,
        ))
        .await
        .unwrap();
    assert_eq!(result.matches.len(), 1);
    assert_eq!(result.matches[0].record.id, created.record.id);
}

#[tokio::test]
async fn concurrent_stores_get_unique_monotonic_ids() {
    let runtime = Arc::new(MemoryRuntime::new(InMemoryProvider::new()));
    let mut tasks = Vec::new();
    for index in 0..32 {
        let runtime = runtime.clone();
        tasks.push(tokio::spawn(async move {
            let tenant = if index % 2 == 0 { "tenant" } else { "other" };
            runtime
                .store(store_request(
                    &format!("store-{index}"),
                    namespace(tenant, "subject", "session", "assistant"),
                    &format!("shared concurrent item {index}"),
                    "2026-01-01T00:00:00Z",
                    None,
                ))
                .await
                .unwrap()
                .record
                .id
        }));
    }

    let mut ids = Vec::new();
    for task in tasks {
        ids.push(task.await.unwrap());
    }
    ids.sort();
    ids.dedup();
    assert_eq!(ids.len(), 32);

    let mut request = search_request(
        "concurrent-search",
        namespace("tenant", "subject", "different", "assistant"),
        "concurrent item",
        MemorySearchScope::Subject,
    );
    request.limit = 100;
    let result = runtime.search(request).await.unwrap();
    assert_eq!(result.matches.len(), 16);
    assert!(
        result
            .matches
            .iter()
            .all(|memory_match| memory_match.record.namespace.tenant_id == "tenant")
    );
    assert_eq!(result.matches[0].rank, 1);
}
