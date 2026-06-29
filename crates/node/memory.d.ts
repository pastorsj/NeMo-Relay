// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Json, ScopeHandle } from './index';

/** Failure behavior for one automatic-memory stage. */
export type FailurePolicy = 'fail_open' | 'fail_closed';

/** Completed-turn content written by automatic memory. */
export type WriteProjection = 'user' | 'user_and_assistant';

/** Completed-turn storage delivery path. */
export type WriteDelivery = 'inline' | 'background';

/** Admission behavior when the pending memory queue is full. */
export type MemoryBackpressurePolicy = 'reject' | 'wait';

/** Bounded local queue policy for background memory work. */
export interface MemoryWorkQueueConfig {
  capacity?: number;
  backpressure?: MemoryBackpressurePolicy;
  enqueueTimeoutMillis?: number;
  maxAttempts?: number;
  retryInitialDelayMillis?: number;
  retryMaxDelayMillis?: number;
  attemptTimeoutMillis?: number;
  terminalHistoryCapacity?: number;
}

/** Supported evidence capture mode. */
export type EvidenceMode = 'references';

/** Stable machine-readable memory operation error code. */
export type MemoryErrorCode =
  | 'invalid_request'
  | 'unsupported'
  | 'deadline_exceeded'
  | 'cancelled'
  | 'conflict'
  | 'provider_unavailable'
  | 'internal';

/** Serializable provider or contract failure. */
export interface MemoryOperationError {
  code: MemoryErrorCode;
  message: string;
  retryable: boolean;
  operationId?: string;
  provider?: string;
  details?: Record<string, Json>;
}

/** Tenant and subject partition for a memory operation. */
export interface MemoryNamespace {
  tenantId: string;
  subjectId: string;
  sessionId?: string;
  agentId?: string;
}

/** Configuration for the native reference automatic-memory component. */
export interface AutomaticMemoryConfig {
  namespace?: MemoryNamespace;
  searchScope?: MemorySearchScope;
  maxCandidates?: number;
  maxItems?: number;
  maxEstimatedTokens?: number;
  operationTimeoutMillis?: number;
  identityPolicy?: FailurePolicy;
  retrievalPolicy?: FailurePolicy;
  storagePolicy?: FailurePolicy;
  writeProjection?: WriteProjection;
  writeDelivery?: WriteDelivery;
  backgroundQueue?: MemoryWorkQueueConfig;
  evidenceMode?: EvidenceMode;
}

/** Aggregate native background queue state. */
export interface MemoryWorkQueueStatus {
  accepting: boolean;
  capacity: number;
  queued: number;
  running: number;
  retrying: number;
  acceptedTotal: number;
  rejectedTotal: number;
  succeededTotal: number;
  failedTotal: number;
  cancelledTotal: number;
  lastAcceptedSequence: number;
  lastTerminalSequence: number;
  retainedTerminalJobs: number;
}

/** Retained state for one accepted background job. */
export interface MemoryJobStatus {
  jobId: string;
  sequence: number;
  kind: 'store' | 'maintenance';
  state: 'queued' | 'running' | 'retrying' | 'succeeded' | 'failed' | 'rejected' | 'cancelled';
  attempts: number;
  error?: MemoryOperationError;
  outcome?: {
    memoryIds?: string[];
    disposition?: MemoryStoreDisposition;
    partialErrorCount: number;
  };
}

/** Installation target and ordering for automatic memory. */
export interface AutomaticMemoryInstallOptions {
  name?: string;
  priority?: number;
  scope?: ScopeHandle;
}

/** Dependency-free native reference automatic memory. */
export declare class InMemoryAutomaticMemory {
  constructor(config?: AutomaticMemoryConfig);
  /** Number of prepared calls still awaiting lifecycle completion. */
  readonly activeTurns: number;
  /** Aggregate queue state, or null when write-back is inline. */
  readonly backgroundStatus: MemoryWorkQueueStatus | null;
  /** Return retained state for one background job. */
  backgroundJobStatus(jobId: string): MemoryJobStatus | null;
  /** Install globally, or only within `scope` when supplied. */
  install(options?: AutomaticMemoryInstallOptions): this;
  /** Deregister once and report whether a registration was removed. */
  close(): boolean;
  /** Wait for work accepted before this call without closing admission. */
  flush(timeoutMillis?: number): Promise<boolean>;
  /** Stop background admission and wait for all accepted work. */
  drain(timeoutMillis?: number): Promise<boolean>;
  /** Deregister, drain, and join the optional background worker. */
  shutdown(timeoutMillis?: number): Promise<boolean>;
}

/** Namespace fields used to narrow a search. */
export type MemorySearchScope = 'subject' | 'agent' | 'session' | 'exact';

/** Structured memory content or an opaque provider reference. */
export type MemoryContent =
  | { kind: 'text'; text: string }
  | { kind: 'json'; value: Json }
  | { kind: 'reference'; reference: string; preview?: string };

/** Origin and derivation facts for one memory. */
export interface MemoryProvenance {
  source: string;
  sourceIds?: string[];
  parentMemoryIds?: string[];
  metadata?: Record<string, Json>;
}

/** Canonical provider-neutral memory record. */
export interface MemoryRecord {
  id: string;
  provider: string;
  namespace: MemoryNamespace;
  content: MemoryContent;
  eventTimestamp: string;
  ingestedAt: string;
  provenance: MemoryProvenance;
  metadata?: Record<string, Json>;
  providerMetadata?: Record<string, Json>;
}

/** One ranked match from a memory search. */
export interface MemoryMatch {
  record: MemoryRecord;
  score: number;
  rank: number;
}

/** Provider-neutral filters applied to a memory search. */
export interface MemoryFilter {
  metadata?: Record<string, Json>;
  eventAfter?: string;
  eventBefore?: string;
  ingestedAfter?: string;
  ingestedBefore?: string;
}

/** Correlation and deadline context for one memory operation. */
export interface MemoryRequestContext {
  operationId: string;
  deadline?: string;
}

/** Search input for a memory provider. */
export interface MemorySearchRequest {
  context: MemoryRequestContext;
  namespace: MemoryNamespace;
  query: string;
  scope?: MemorySearchScope;
  filter?: MemoryFilter;
  limit?: number;
}

/** Search output from a memory provider. */
export interface MemorySearchResult {
  matches: MemoryMatch[];
  partialErrors?: MemoryOperationError[];
}

/** Store input for a memory provider. */
export interface MemoryStoreRequest {
  context: MemoryRequestContext;
  namespace: MemoryNamespace;
  content: MemoryContent;
  eventTimestamp: string;
  provenance: MemoryProvenance;
  metadata?: Record<string, Json>;
  idempotencyKey?: string;
}

/** Outcome classification for a store operation. */
export type MemoryStoreDisposition = 'created' | 'existing';

/** Store output from a memory provider. */
export interface MemoryStoreResult {
  record: MemoryRecord;
  disposition: MemoryStoreDisposition;
}

/** Optional operations supported by a memory provider. */
export interface MemoryCapabilities {
  update: boolean;
  delete: boolean;
  batchStore: boolean;
  maintenance: boolean;
  feedback: boolean;
  health: boolean;
}

/** Minimal provider contract required by the Relay memory runtime. */
export interface MemoryProvider {
  readonly name: string;
  readonly capabilities: MemoryCapabilities;
  search(request: MemorySearchRequest): Promise<MemorySearchResult>;
  store(request: MemoryStoreRequest): Promise<MemoryStoreResult>;
}

/** Error raised when a memory value violates the provider-neutral contract. */
export declare class MemoryContractError extends TypeError {
  readonly code: 'invalid_request';
}

/**
 * Validate a public Node memory namespace and its requested search scope.
 *
 * @param namespace - Camel-case namespace to validate.
 * @param scope - Search scope whose required fields must be present.
 * @returns The unchanged namespace after successful validation.
 */
export declare function validateMemoryNamespace<T extends MemoryNamespace>(namespace: T, scope?: MemorySearchScope): T;

/** Return the capability set implemented by a search-and-store-only provider. */
export declare function defaultMemoryCapabilities(): MemoryCapabilities;

/** Convert public Node memory values to the canonical snake-case wire shape. */
export declare function toMemoryWire(value: Json): Json;

/** Convert canonical snake-case wire values to public Node memory values. */
export declare function fromMemoryWire(value: Json): Json;
