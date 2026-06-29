// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Callback type aliases used by the runtime middleware pipeline.
//!
//! The public middleware registration APIs accept callback closures with the
//! signatures defined in this module. These aliases centralize those signatures
//! so the runtime can compose tool and LLM middleware consistently across
//! bindings.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use tokio_stream::Stream;

use crate::api::event::Event;
use crate::api::llm::{LlmHandle, LlmRequest};
use crate::api::scope::ScopeHandle;
use crate::codec::request::AnnotatedLlmRequest;
use crate::codec::response::AnnotatedLlmResponse;
use crate::codec::traits::{LlmCodec, LlmResponseCodec};
use crate::error::Result;
use crate::json::Json;

/// Sanitize a tool request payload before the runtime records it.
///
/// Tool sanitize callbacks are used only for observability payloads. They can
/// rewrite the JSON arguments recorded on tool-start events without changing
/// the caller-owned request that is passed to the tool implementation.
///
/// # Parameters
/// - First argument: Tool name associated with the request payload.
/// - Second argument: JSON payload to sanitize for observability.
///
/// # Returns
/// Sanitized JSON payload for the emitted event.
pub type ToolSanitizeFn = Arc<dyn Fn(&str, Json) -> Json + Send + Sync>;
/// Decide whether a tool call is allowed to continue.
///
/// The callback receives the tool name and the current argument payload. It can
/// return `Ok(None)` to allow execution, `Ok(Some(reason))` to reject the call
/// with a guardrail message, or an error to abort evaluation entirely.
///
/// This alias is [`Arc`]-backed so the runtime can clone conditional
/// guardrails into an evaluation snapshot and invoke them after registry locks
/// are released.
///
/// # Parameters
/// - First argument: Tool name being evaluated.
/// - Second argument: Current tool argument payload.
///
/// # Returns
/// A [`Result`] containing `Ok(None)` when execution is allowed or
/// `Ok(Some(reason))` when the guardrail rejects the call.
///
/// # Errors
/// The callback can return any [`FlowError`](crate::error::FlowError) to abort
/// guardrail evaluation.
pub type ToolConditionalFn = Arc<dyn Fn(&str, &Json) -> Result<Option<String>> + Send + Sync>;
/// Rewrite tool arguments before execution.
///
/// Tool request intercepts run in priority order and can transform the JSON
/// payload that is eventually passed into the tool execution callback.
///
/// # Parameters
/// - First argument: Tool name associated with the request.
/// - Second argument: JSON argument payload to transform.
///
/// # Returns
/// A [`Result`] containing the transformed JSON argument payload.
///
/// # Errors
/// The callback can return any [`FlowError`](crate::error::FlowError) to abort
/// the request-intercept chain.
pub type ToolInterceptFn = Arc<dyn Fn(&str, Json) -> Result<Json> + Send + Sync>;
/// Continuation type invoked by tool execution intercepts.
///
/// Execution intercepts receive this callable as their `next` continuation and
/// can call it with modified arguments, wrap it, or skip it entirely.
///
/// # Parameters
/// - First argument: JSON argument payload to pass to the remaining execution
///   chain.
///
/// # Returns
/// A future resolving to the tool result JSON.
///
/// # Errors
/// The future resolves to an error when the remaining execution chain fails.
pub type ToolExecutionNextFn =
    Arc<dyn Fn(Json) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>> + Send + Sync>;
/// Wrap or replace tool execution.
///
/// A tool execution intercept receives the tool name, the current argument
/// payload, and the continuation representing the rest of the chain.
///
/// # Parameters
/// - First argument: Tool name associated with the execution.
/// - Second argument: Current JSON argument payload.
/// - Third argument: Continuation for the remaining execution chain.
///
/// # Returns
/// A future resolving to the tool result JSON.
///
/// # Errors
/// The future resolves to an error when the intercept or remaining execution
/// chain fails.
pub type ToolExecutionFn = Arc<
    dyn Fn(&str, Json, ToolExecutionNextFn) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>>
        + Send
        + Sync,
>;

/// Sanitize an LLM request before the runtime records it.
///
/// LLM request sanitizers affect the serialized request payload emitted on
/// start events. They do not mutate the caller-owned [`LlmRequest`] unless a
/// separate request intercept does so.
///
/// # Parameters
/// - First argument: LLM request payload to sanitize for observability.
///
/// # Returns
/// Sanitized [`LlmRequest`] for the emitted event.
pub type LlmSanitizeRequestFn = Arc<dyn Fn(LlmRequest) -> LlmRequest + Send + Sync>;
/// Sanitize an LLM response before the runtime records it.
///
/// These callbacks rewrite the JSON response payload captured on LLM-end
/// events, which is useful for redaction or payload normalization.
///
/// # Parameters
/// - First argument: JSON response payload to sanitize for observability.
///
/// # Returns
/// Sanitized JSON response payload for the emitted event.
pub type LlmSanitizeResponseFn = Arc<dyn Fn(Json) -> Json + Send + Sync>;
/// Decide whether an LLM call is allowed to continue.
///
/// The callback receives the current [`LlmRequest`] and can allow execution,
/// reject it with a guardrail reason, or return an error.
///
/// This alias is [`Arc`]-backed so the runtime can clone conditional
/// guardrails into an evaluation snapshot and invoke them after registry locks
/// are released.
///
/// # Parameters
/// - First argument: Current [`LlmRequest`] being evaluated.
///
/// # Returns
/// A [`Result`] containing `Ok(None)` when execution is allowed or
/// `Ok(Some(reason))` when the guardrail rejects the call.
///
/// # Errors
/// The callback can return any [`FlowError`](crate::error::FlowError) to abort
/// guardrail evaluation.
pub type LlmConditionalFn = Arc<dyn Fn(&LlmRequest) -> Result<Option<String>> + Send + Sync>;
/// Rewrite or annotate an LLM request before execution.
///
/// Request intercepts can transform the wire request, attach or replace a
/// normalized [`AnnotatedLlmRequest`], or both.
///
/// # Parameters
/// - First argument: Logical provider or model family name.
/// - Second argument: LLM request to transform.
/// - Third argument: Optional normalized request annotation to carry forward.
///
/// # Returns
/// A [`Result`] containing the transformed request and optional annotation.
///
/// # Errors
/// The callback can return any [`FlowError`](crate::error::FlowError) to abort
/// the request-intercept chain.
pub type LlmRequestInterceptFn = Arc<
    dyn Fn(
            &str,
            LlmRequest,
            Option<AnnotatedLlmRequest>,
        ) -> Result<(LlmRequest, Option<AnnotatedLlmRequest>)>
        + Send
        + Sync,
>;
/// Continuation type invoked by non-streaming LLM execution intercepts.
///
/// Execution intercepts use this callable to continue the non-streaming LLM
/// pipeline after applying their own logic.
///
/// # Parameters
/// - First argument: LLM request to pass to the remaining execution chain.
///
/// # Returns
/// A future resolving to the provider response JSON.
///
/// # Errors
/// The future resolves to an error when the remaining execution chain fails.
pub type LlmExecutionNextFn =
    Arc<dyn Fn(LlmRequest) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>> + Send + Sync>;
/// Wrap or replace non-streaming LLM execution.
///
/// A non-streaming execution intercept receives the logical provider name, the
/// current request, and the continuation representing the rest of the chain.
///
/// # Parameters
/// - First argument: Logical provider or model family name.
/// - Second argument: Current LLM request.
/// - Third argument: Continuation for the remaining execution chain.
///
/// # Returns
/// A future resolving to the provider response JSON.
///
/// # Errors
/// The future resolves to an error when the intercept or remaining execution
/// chain fails.
pub type LlmExecutionFn = Arc<
    dyn Fn(
            &str,
            LlmRequest,
            LlmExecutionNextFn,
        ) -> Pin<Box<dyn Future<Output = Result<Json>> + Send>>
        + Send
        + Sync,
>;

/// Immutable context shared with one managed LLM lifecycle hook invocation.
#[derive(Clone)]
pub struct LlmLifecycleContext {
    /// Runtime-owned handle for the managed LLM call.
    pub handle: LlmHandle,
    /// Ordered active scopes from the implicit root to the current scope.
    pub scopes: Vec<ScopeHandle>,
    /// Request codec supplied to the managed call, if any.
    pub request_codec: Option<Arc<dyn LlmCodec>>,
    /// Response codec supplied to the managed call, if any.
    pub response_codec: Option<Arc<dyn LlmResponseCodec>>,
}

/// Request state passed through managed LLM lifecycle preparation hooks.
#[derive(Clone)]
pub struct LlmLifecycleRequest {
    /// Provider request that later middleware and the callback will receive.
    pub request: LlmRequest,
    /// Normalized request produced by the call's codec, when available.
    pub annotated_request: Option<Arc<AnnotatedLlmRequest>>,
}

/// Provider outcome observed by managed LLM lifecycle completion hooks.
#[derive(Clone)]
pub enum LlmLifecycleOutcome {
    /// The provider execution chain returned a response.
    Success {
        /// Raw response returned to the managed runtime.
        response: Json,
        /// Best-effort normalized response produced with the configured codec.
        annotated_response: Option<Arc<AnnotatedLlmResponse>>,
    },
    /// The provider execution chain returned an error.
    Failure {
        /// Stable display form of the original execution error.
        error: String,
    },
}

/// Asynchronous observer and request preparer for a managed non-streaming LLM call.
///
/// Hooks cannot replace the provider continuation. They may transform the
/// prepared request before the LLM start event and observe the final outcome.
/// Completion is invoked in reverse preparation order, including when a later
/// preparation hook or provider execution fails.
pub trait LlmLifecycleHook: Send + Sync {
    /// Prepare a request before the runtime emits the LLM start event.
    fn prepare<'a>(
        &'a self,
        context: &'a LlmLifecycleContext,
        request: LlmLifecycleRequest,
    ) -> Pin<Box<dyn Future<Output = Result<LlmLifecycleRequest>> + Send + 'a>>;

    /// Observe completion and release any per-call state.
    fn complete<'a>(
        &'a self,
        _context: &'a LlmLifecycleContext,
        _request: &'a LlmLifecycleRequest,
        _outcome: &'a LlmLifecycleOutcome,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async { Ok(()) })
    }
}

/// Shared trait object stored in lifecycle-hook registries.
pub type LlmLifecycleHookFn = Arc<dyn LlmLifecycleHook>;
/// Stream of JSON chunks produced by the managed streaming LLM pipeline.
pub type LlmJsonStream = Pin<Box<dyn Stream<Item = Result<Json>> + Send>>;
/// Per-chunk collector used by the streaming LLM runtime.
///
/// # Parameters
/// - First argument: One JSON chunk emitted by the provider stream.
///
/// # Returns
/// A [`Result`] that is `Ok(())` when the chunk was collected.
///
/// # Errors
/// The callback can return any [`FlowError`](crate::error::FlowError) to abort
/// stream processing.
pub type LlmCollectorFn = Box<dyn FnMut(Json) -> Result<()> + Send>;
/// Finalizer used to synthesize the aggregate streaming response payload.
///
/// # Parameters
/// This callback takes no arguments.
///
/// # Returns
/// Aggregate response JSON synthesized from collected stream chunks.
pub type LlmFinalizerFn = Box<dyn FnOnce() -> Json + Send>;
/// Scope-local registry references passed into streaming execution-chain builders.
///
/// # Returns
/// A shared reference to a scope-local streaming execution registry.
pub(crate) type LlmStreamExecutionRegistryRef<'a> = &'a crate::registry::SortedRegistry<
    crate::api::registry::ExecutionIntercept<LlmStreamExecutionFn>,
>;
/// Slice of scope-local streaming execution registries.
///
/// # Returns
/// A borrowed slice of scope-local streaming execution registry references.
pub(crate) type LlmStreamExecutionRegistryRefs<'a> = &'a [LlmStreamExecutionRegistryRef<'a>];

/// Continuation type invoked by streaming LLM execution intercepts.
///
/// This callable represents the remainder of the streaming LLM execution chain
/// and resolves to a stream of JSON response chunks.
///
/// # Parameters
/// - First argument: LLM request to pass to the remaining streaming execution
///   chain.
///
/// # Returns
/// A future resolving to a JSON chunk stream.
///
/// # Errors
/// The future resolves to an error when the remaining streaming execution
/// chain fails.
pub type LlmStreamExecutionNextFn = Arc<
    dyn Fn(LlmRequest) -> Pin<Box<dyn Future<Output = Result<LlmJsonStream>> + Send>> + Send + Sync,
>;
/// Wrap or replace streaming LLM execution.
///
/// A streaming execution intercept can observe or modify the request before
/// invoking the continuation, and it can also replace the returned stream.
///
/// # Parameters
/// - First argument: Logical provider or model family name.
/// - Second argument: Current LLM request.
/// - Third argument: Continuation for the remaining streaming execution chain.
///
/// # Returns
/// A future resolving to a JSON chunk stream.
///
/// # Errors
/// The future resolves to an error when the intercept or remaining streaming
/// execution chain fails.
pub type LlmStreamExecutionFn = Arc<
    dyn Fn(
            &str,
            LlmRequest,
            LlmStreamExecutionNextFn,
        ) -> Pin<Box<dyn Future<Output = Result<LlmJsonStream>> + Send>>
        + Send
        + Sync,
>;

/// Consume runtime lifecycle events after they are emitted.
///
/// Event subscribers are invoked for scope, tool, LLM, and mark events after
/// the runtime has built the final event payload.
///
/// # Parameters
/// - First argument: Runtime event that was just emitted.
///
/// # Returns
/// `()`.
pub type EventSubscriberFn = Arc<dyn Fn(&Event) + Send + Sync>;
