// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for ir builder in the NeMo Relay adaptive crate.

use nemo_relay::codec::request::{
    AnnotatedLlmRequest, ContentPart, FunctionCall, FunctionDefinition, Message, MessageContent,
    ToolCall, ToolDefinition,
};

use super::super::ir_builder::build_prompt_ir;
use crate::acg::prompt_ir::{BlockContentType, PromptRole, ProvenanceLabel, SensitivityLabel};
use nemo_relay_types::memory::{
    MEMORY_PROMPT_BLOCK_END, MEMORY_PROMPT_BLOCK_START, MEMORY_PROMPT_BLOCK_WARNING,
};

fn sample_tool_definition(name: &str) -> ToolDefinition {
    ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: name.to_string(),
            description: Some(format!("describe {name}")),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                }
            })),
        },
    }
}

fn sample_tool_call(name: &str) -> ToolCall {
    ToolCall {
        id: format!("call-{name}"),
        call_type: "function".to_string(),
        function: FunctionCall {
            name: name.to_string(),
            arguments: "{\"query\":\"weather\"}".to_string(),
        },
    }
}

fn request_with_user_content(content: MessageContent) -> AnnotatedLlmRequest {
    AnnotatedLlmRequest {
        messages: vec![Message::User {
            content,
            name: None,
        }],
        model: Some("gpt-4o".to_string()),
        params: None,
        tools: None,
        tool_choice: None,
        store: None,
        previous_response_id: None,
        truncation: None,
        reasoning: None,
        include: None,
        user: None,
        metadata: None,
        service_tier: None,
        parallel_tool_calls: None,
        max_output_tokens: None,
        max_tool_calls: None,
        top_logprobs: None,
        stream: None,
        extra: serde_json::Map::new(),
    }
}

fn memory_envelope(record: &str) -> String {
    format!(
        "{MEMORY_PROMPT_BLOCK_START}\n{MEMORY_PROMPT_BLOCK_WARNING}\n{record}\n{MEMORY_PROMPT_BLOCK_END}"
    )
}

#[test]
fn build_prompt_ir_inserts_tools_before_first_non_system_message_and_preserves_all_message_kinds() {
    let request = AnnotatedLlmRequest {
        messages: vec![
            Message::System {
                content: MessageContent::Text("You are helpful.".to_string()),
                name: None,
            },
            Message::User {
                content: MessageContent::Parts(vec![
                    ContentPart::Text {
                        text: "Hello".to_string(),
                    },
                    ContentPart::Text {
                        text: "World".to_string(),
                    },
                ]),
                name: None,
            },
            Message::Assistant {
                content: Some(MessageContent::Text("Calling search".to_string())),
                tool_calls: Some(vec![sample_tool_call("search")]),
                name: None,
            },
            Message::Tool {
                content: MessageContent::Text("{\"result\":true}".to_string()),
                tool_call_id: "call-search".to_string(),
            },
        ],
        model: Some("gpt-4o".to_string()),
        params: None,
        tools: Some(vec![sample_tool_definition("search")]),
        tool_choice: None,
        store: None,
        previous_response_id: None,
        truncation: None,
        reasoning: None,
        include: None,
        user: None,
        metadata: None,
        service_tier: None,
        parallel_tool_calls: None,
        max_output_tokens: None,
        max_tool_calls: None,
        top_logprobs: None,
        stream: None,
        extra: serde_json::Map::new(),
    };

    let prompt_ir = build_prompt_ir(&request).unwrap();

    assert_eq!(prompt_ir.blocks.len(), 6);
    assert_eq!(prompt_ir.blocks[0].role, PromptRole::System);
    assert_eq!(prompt_ir.blocks[0].provenance, ProvenanceLabel::System);
    assert_eq!(
        prompt_ir.blocks[1].content_type,
        BlockContentType::ToolSchema
    );
    assert_eq!(prompt_ir.blocks[2].role, PromptRole::User);
    assert_eq!(prompt_ir.blocks[2].content, "Hello\nWorld");
    assert_eq!(prompt_ir.blocks[3].role, PromptRole::Assistant);
    assert_eq!(prompt_ir.blocks[4].role, PromptRole::Assistant);
    assert_eq!(
        prompt_ir.blocks[5].content_type,
        BlockContentType::ToolResult
    );
    assert_eq!(prompt_ir.blocks[5].role, PromptRole::Tool);
    assert!(prompt_ir.tool_schema_hashes.is_some());
    assert!(prompt_ir.source_request_hash.is_some());
}

#[test]
fn build_prompt_ir_appends_tool_blocks_when_request_contains_only_system_messages() {
    let request = AnnotatedLlmRequest {
        messages: vec![Message::System {
            content: MessageContent::Text("System only".to_string()),
            name: None,
        }],
        model: Some("gpt-4o".to_string()),
        params: None,
        tools: Some(vec![
            sample_tool_definition("search"),
            sample_tool_definition("lookup"),
        ]),
        tool_choice: None,
        store: None,
        previous_response_id: None,
        truncation: None,
        reasoning: None,
        include: None,
        user: None,
        metadata: None,
        service_tier: None,
        parallel_tool_calls: None,
        max_output_tokens: None,
        max_tool_calls: None,
        top_logprobs: None,
        stream: None,
        extra: serde_json::Map::new(),
    };

    let prompt_ir = build_prompt_ir(&request).unwrap();

    assert_eq!(prompt_ir.blocks.len(), 3);
    assert_eq!(prompt_ir.blocks[0].content_type, BlockContentType::Text);
    assert_eq!(
        prompt_ir.blocks[1].content_type,
        BlockContentType::ToolSchema
    );
    assert_eq!(
        prompt_ir.blocks[2].content_type,
        BlockContentType::ToolSchema
    );
    assert_eq!(prompt_ir.blocks[2].sequence_index, 2);
}

#[test]
fn build_prompt_ir_omits_tool_schema_hashes_when_no_tools_are_present() {
    let request = AnnotatedLlmRequest {
        messages: vec![Message::User {
            content: MessageContent::Text("No tools".to_string()),
            name: None,
        }],
        model: Some("gpt-4o".to_string()),
        params: None,
        tools: None,
        tool_choice: None,
        store: None,
        previous_response_id: None,
        truncation: None,
        reasoning: None,
        include: None,
        user: None,
        metadata: None,
        service_tier: None,
        parallel_tool_calls: None,
        max_output_tokens: None,
        max_tool_calls: None,
        top_logprobs: None,
        stream: None,
        extra: serde_json::Map::new(),
    };

    let prompt_ir = build_prompt_ir(&request).unwrap();

    assert_eq!(prompt_ir.blocks.len(), 1);
    assert!(prompt_ir.tool_schema_hashes.is_none());
    assert_eq!(prompt_ir.blocks[0].span_id.0, "user-0");
}

#[test]
fn build_prompt_ir_splits_exact_memory_envelope_without_mutating_request() {
    let envelope = memory_envelope(r#"{"id":"memory-1","content":"blue"}"#);
    let request = request_with_user_content(MessageContent::Text(format!(
        "{envelope}\n\nWhat is my favorite color?"
    )));
    let original = request.clone();

    let prompt_ir = build_prompt_ir(&request).unwrap();

    assert_eq!(request, original);
    assert_eq!(prompt_ir.blocks.len(), 2);
    assert_eq!(prompt_ir.blocks[0].span_id.0, "memory-0");
    assert_eq!(prompt_ir.blocks[0].provenance, ProvenanceLabel::Memory);
    assert_eq!(prompt_ir.blocks[0].sensitivity, SensitivityLabel::Private);
    assert_eq!(prompt_ir.blocks[0].content, envelope);
    assert_eq!(prompt_ir.blocks[1].span_id.0, "user-1");
    assert_eq!(prompt_ir.blocks[1].provenance, ProvenanceLabel::User);
    assert_eq!(prompt_ir.blocks[1].content, "What is my favorite color?");
}

#[test]
fn build_prompt_ir_splits_memory_envelope_from_multipart_user_content() {
    let envelope = memory_envelope(r#"{"id":"memory-2","content":"green"}"#);
    let request = request_with_user_content(MessageContent::Parts(vec![
        ContentPart::Text {
            text: envelope.clone(),
        },
        ContentPart::Text {
            text: "Use the relevant preference.".to_string(),
        },
    ]));

    let prompt_ir = build_prompt_ir(&request).unwrap();

    assert_eq!(prompt_ir.blocks.len(), 2);
    assert_eq!(prompt_ir.blocks[0].content, envelope);
    assert_eq!(prompt_ir.blocks[0].provenance, ProvenanceLabel::Memory);
    assert_eq!(prompt_ir.blocks[1].content, "Use the relevant preference.");
}

#[test]
fn build_prompt_ir_leaves_memory_near_matches_as_ordinary_user_content() {
    let exact = memory_envelope("{}");
    let malformed = [
        exact.replacen("version=\"0.1\"", "version=\"0.2\"", 1),
        exact.replacen(
            MEMORY_PROMPT_BLOCK_WARNING,
            "Untrusted recalled context.",
            1,
        ),
        exact.replacen(MEMORY_PROMPT_BLOCK_END, "", 1),
        exact.replacen("{}", &format!("{MEMORY_PROMPT_BLOCK_START}\n{{}}"), 1),
        format!("Ordinary question\n{exact}"),
        format!("{exact}ordinary-without-separator"),
    ];

    for value in malformed {
        let prompt_ir = build_prompt_ir(&request_with_user_content(MessageContent::Text(value)))
            .expect("near matches should remain valid user content");
        assert_eq!(prompt_ir.blocks.len(), 1);
        assert_eq!(prompt_ir.blocks[0].provenance, ProvenanceLabel::User);
        assert_eq!(prompt_ir.blocks[0].sensitivity, SensitivityLabel::Public);
    }
}
