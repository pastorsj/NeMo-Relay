// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for stability internal in the NeMo Relay adaptive crate.

use chrono::Utc;

use super::*;

use crate::acg::prompt_ir::{
    BlockContentType, PromptBlock, PromptRole, ProvenanceLabel, SensitivityLabel,
};

fn prompt(blocks: Vec<PromptBlock>) -> PromptIR {
    PromptIR {
        ir_id: uuid::Uuid::new_v4(),
        blocks,
        tool_schema_hashes: None,
        structured_output_schema_id: None,
        source_request_hash: None,
        created_at: Utc::now(),
    }
}

fn block(span_id: &str, sequence_index: u32, content: &str) -> PromptBlock {
    PromptBlock {
        span_id: SpanId(span_id.to_string()),
        sequence_index,
        role: PromptRole::System,
        content: content.to_string(),
        content_type: BlockContentType::Text,
        provenance: ProvenanceLabel::System,
        sensitivity: SensitivityLabel::Public,
        token_metadata: None,
    }
}

fn memory_block(sequence_index: u32, content: &str) -> PromptBlock {
    PromptBlock {
        span_id: SpanId(format!("memory-{sequence_index}")),
        sequence_index,
        role: PromptRole::User,
        content: content.to_string(),
        content_type: BlockContentType::Text,
        provenance: ProvenanceLabel::Memory,
        sensitivity: SensitivityLabel::Private,
        token_metadata: None,
    }
}

#[test]
fn stability_internal_handles_empty_inputs_variable_scores_and_zero_confidence_threshold() {
    let thresholds = StabilityThresholds::default();
    let empty = analyze_stability(&[], &thresholds);
    assert_eq!(empty.total_observations, 0);
    assert_eq!(empty.stable_prefix_length, 0);
    assert!(empty.scores.is_empty());

    let observations = vec![
        prompt(vec![block("span-0", 0, "A"), block("span-1", 1, "X")]),
        prompt(vec![block("span-0", 0, "A")]),
        prompt(vec![block("span-0", 0, "B"), block("span-1", 1, "Y")]),
    ];
    let result = analyze_stability(&observations, &thresholds);
    assert_eq!(result.scores.len(), 2);
    assert!(
        result
            .scores
            .iter()
            .any(|score| score.classification == StabilityClass::Variable)
    );

    let zero_threshold = StabilityThresholds {
        min_observations_for_full_confidence: 0,
        ..StabilityThresholds::default()
    };
    assert_eq!(stability_confidence(1, &zero_threshold), 1.0);
    assert_eq!(
        classify_stability(0.1, &thresholds),
        StabilityClass::Variable
    );
}

#[test]
fn stability_internal_effective_score_handles_zero_present_count() {
    let observations = SpanObservations {
        hash_counts: std::collections::HashMap::new(),
        present_count: 0,
        first_seen_sequence_index: 0,
    };

    assert_eq!(effective_stability_score(&observations, 3), 0.0);
}

#[test]
fn stability_never_extends_a_prefix_through_memory_provenance() {
    let thresholds = StabilityThresholds::default();
    let repeated = vec![
        prompt(vec![
            block("system-0", 0, "stable system"),
            memory_block(1, "same memory"),
        ]),
        prompt(vec![
            block("system-0", 0, "stable system"),
            memory_block(1, "same memory"),
        ]),
    ];
    let repeated_result = analyze_stability(&repeated, &thresholds);
    assert_eq!(
        repeated_result.scores[1].classification,
        StabilityClass::Stable
    );
    assert_eq!(repeated_result.stable_prefix_length, 1);

    let changing = vec![
        prompt(vec![
            block("system-0", 0, "stable system"),
            memory_block(1, "memory A"),
        ]),
        prompt(vec![
            block("system-0", 0, "stable system"),
            memory_block(1, "memory B"),
        ]),
    ];
    assert_eq!(
        analyze_stability(&changing, &thresholds).stable_prefix_length,
        1
    );
}

#[test]
fn stability_uses_the_earliest_memory_boundary_across_observations() {
    let thresholds = StabilityThresholds::default();
    let observations = vec![
        prompt(vec![
            block("system-0", 0, "stable system"),
            block("system-1", 1, "stable tools"),
            memory_block(2, "memory A"),
        ]),
        prompt(vec![
            memory_block(0, "memory B"),
            block("system-0", 1, "stable system"),
        ]),
    ];

    assert_eq!(
        analyze_stability(&observations, &thresholds).stable_prefix_length,
        0
    );

    let no_memory = vec![
        prompt(vec![block("system-0", 0, "stable")]),
        prompt(vec![block("system-0", 0, "stable")]),
    ];
    assert_eq!(
        analyze_stability(&no_memory, &thresholds).stable_prefix_length,
        1
    );
}
