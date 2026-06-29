// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Serialization and validation tests for provider-neutral memory DTOs.

use chrono::{TimeZone, Utc};
use nemo_relay_types::memory::{
    MAX_SEARCH_LIMIT, MemoryCapabilities, MemoryContent, MemoryErrorCode, MemoryFilter,
    MemoryMaintenanceAction, MemoryMaintenanceRequest, MemoryMaintenanceWindow, MemoryNamespace,
    MemoryOperationError, MemoryProvenance, MemoryRequestContext, MemorySearchRequest,
    MemorySearchResult, MemorySearchScope, MemoryStoreResult,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, PartialEq)]
struct MemoryContractFixture {
    search_request: MemorySearchRequest,
    search_result: MemorySearchResult,
    store_result: MemoryStoreResult,
    capabilities: MemoryCapabilities,
}

#[test]
fn canonical_fixture_round_trips_without_semantic_loss() {
    let source = include_str!("fixtures/memory_contract_v0_1.json");
    let fixture: MemoryContractFixture =
        serde_json::from_str(source).expect("fixture should deserialize");

    fixture
        .search_request
        .validate()
        .expect("fixture request should validate");
    assert_eq!(fixture.search_result.matches[0].rank, 1);
    assert_eq!(
        fixture.search_result.matches[0]
            .record
            .namespace
            .session_id
            .as_deref(),
        Some("session-a")
    );
    assert_eq!(
        fixture.search_result.partial_errors[0].code,
        MemoryErrorCode::ProviderUnavailable
    );

    let expected: serde_json::Value =
        serde_json::from_str(source).expect("fixture should be valid JSON");
    let encoded = serde_json::to_value(&fixture).expect("fixture should serialize");
    assert_eq!(encoded, expected);

    let decoded: MemoryContractFixture =
        serde_json::from_value(encoded).expect("serialized fixture should deserialize");
    assert_eq!(decoded, fixture);
}

#[test]
fn namespace_rejects_empty_required_and_optional_identifiers() {
    let error = MemoryNamespace::new("", "subject").expect_err("tenant is required");
    assert_eq!(error.code, MemoryErrorCode::InvalidRequest);

    let error = MemoryNamespace::new("tenant", " ").expect_err("subject is required");
    assert_eq!(error.code, MemoryErrorCode::InvalidRequest);

    let namespace = MemoryNamespace {
        tenant_id: "tenant".into(),
        subject_id: "subject".into(),
        session_id: Some("".into()),
        agent_id: None,
    };
    assert_eq!(
        namespace
            .validate()
            .expect_err("empty session is invalid")
            .code,
        MemoryErrorCode::InvalidRequest
    );
}

#[test]
fn narrow_search_scopes_require_their_namespace_identifier() {
    let namespace = MemoryNamespace::new("tenant", "subject").expect("valid namespace");
    assert!(MemorySearchScope::Subject.validate(&namespace).is_ok());
    assert!(MemorySearchScope::Exact.validate(&namespace).is_ok());
    assert_eq!(
        MemorySearchScope::Agent
            .validate(&namespace)
            .expect_err("agent scope needs an agent")
            .code,
        MemoryErrorCode::InvalidRequest
    );
    assert_eq!(
        MemorySearchScope::Session
            .validate(&namespace)
            .expect_err("session scope needs a session")
            .code,
        MemoryErrorCode::InvalidRequest
    );
}

#[test]
fn search_validates_query_limit_and_filter_ranges() {
    let context = MemoryRequestContext::new("search").expect("valid context");
    let namespace = MemoryNamespace::new("tenant", "subject").expect("valid namespace");
    let mut request =
        MemorySearchRequest::new(context, namespace, "preference").expect("valid request");

    request.limit = 0;
    assert_eq!(
        request.validate().expect_err("zero limit is invalid").code,
        MemoryErrorCode::InvalidRequest
    );
    request.limit = MAX_SEARCH_LIMIT + 1;
    assert_eq!(
        request
            .validate()
            .expect_err("oversized limit is invalid")
            .code,
        MemoryErrorCode::InvalidRequest
    );

    request.limit = 1;
    request.query = " ".into();
    assert_eq!(
        request.validate().expect_err("empty query is invalid").code,
        MemoryErrorCode::InvalidRequest
    );

    request.query = "preference".into();
    request.filter = MemoryFilter {
        event_after: Some(Utc.with_ymd_and_hms(2026, 2, 1, 0, 0, 0).unwrap()),
        event_before: Some(Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap()),
        ..MemoryFilter::default()
    };
    assert_eq!(
        request
            .validate()
            .expect_err("reversed event range is invalid")
            .code,
        MemoryErrorCode::InvalidRequest
    );
}

#[test]
fn content_and_provenance_validate_required_text() {
    assert_eq!(
        MemoryContent::Text { text: "".into() }
            .validate()
            .expect_err("empty text is invalid")
            .code,
        MemoryErrorCode::InvalidRequest
    );
    assert_eq!(
        MemoryContent::Reference {
            reference: " ".into(),
            preview: None,
        }
        .validate()
        .expect_err("empty reference is invalid")
        .code,
        MemoryErrorCode::InvalidRequest
    );
    assert_eq!(
        MemoryProvenance::default()
            .validate()
            .expect_err("source is required")
            .code,
        MemoryErrorCode::InvalidRequest
    );
}

#[test]
fn maintenance_window_round_trips_and_validates_bounds() {
    let namespace = MemoryNamespace {
        tenant_id: "tenant".into(),
        subject_id: "subject".into(),
        session_id: Some("session".into()),
        agent_id: None,
    };
    let mut request = MemoryMaintenanceRequest {
        context: MemoryRequestContext::new("maintenance-1").expect("valid context"),
        namespace,
        action: MemoryMaintenanceAction::Consolidate,
        window: Some(MemoryMaintenanceWindow {
            checkpoint_id: "checkpoint-2".into(),
            previous_checkpoint_id: Some("checkpoint-1".into()),
            query: "shared project preferences".into(),
            scope: MemorySearchScope::Session,
            filter: MemoryFilter::default(),
            limit: 25,
        }),
        parameters: Default::default(),
    };

    request.validate().expect("valid maintenance request");
    let encoded = serde_json::to_value(&request).expect("request should serialize");
    let decoded: MemoryMaintenanceRequest =
        serde_json::from_value(encoded).expect("request should deserialize");
    assert_eq!(decoded, request);

    let window = request.window.as_mut().expect("window is present");
    window.previous_checkpoint_id = Some("checkpoint-2".into());
    assert_eq!(
        request
            .validate()
            .expect_err("a checkpoint cannot advance from itself")
            .code,
        MemoryErrorCode::InvalidRequest
    );

    let window = request.window.as_mut().expect("window is present");
    window.previous_checkpoint_id = None;
    window.limit = MAX_SEARCH_LIMIT + 1;
    assert_eq!(
        request
            .validate()
            .expect_err("an oversized window is invalid")
            .code,
        MemoryErrorCode::InvalidRequest
    );

    let window = request.window.as_mut().expect("window is present");
    window.limit = 1;
    window.query.clear();
    assert_eq!(
        request
            .validate()
            .expect_err("an empty window query is invalid")
            .code,
        MemoryErrorCode::InvalidRequest
    );
}

#[test]
fn error_helpers_keep_stable_wire_codes() {
    let errors = [
        MemoryOperationError::invalid_request("bad request"),
        MemoryOperationError::unsupported("update"),
        MemoryOperationError::deadline_exceeded("late"),
        MemoryOperationError::conflict("duplicate"),
    ];
    let encoded = serde_json::to_value(errors).expect("errors should serialize");
    let codes: Vec<_> = encoded
        .as_array()
        .expect("encoded errors should be an array")
        .iter()
        .map(|error| error["code"].as_str().expect("code should be a string"))
        .collect();
    assert_eq!(
        codes,
        [
            "invalid_request",
            "unsupported",
            "deadline_exceeded",
            "conflict"
        ]
    );
}

#[cfg(feature = "schema")]
#[test]
fn memory_contract_generates_json_schema() {
    let schema = schemars::schema_for!(MemorySearchRequest);
    let encoded = serde_json::to_string(&schema).expect("schema should serialize");
    assert!(encoded.contains("tenant_id"));
    assert!(encoded.contains("MemorySearchScope"));

    let maintenance_schema = schemars::schema_for!(MemoryMaintenanceRequest);
    let maintenance_encoded =
        serde_json::to_string(&maintenance_schema).expect("schema should serialize");
    assert!(maintenance_encoded.contains("checkpoint_id"));
    assert!(maintenance_encoded.contains("MemoryMaintenanceWindow"));
}
