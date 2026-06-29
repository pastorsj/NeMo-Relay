// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Json } from './index';

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
export type MemoryStoreDisposition = 'created' | 'updated' | 'existing';

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
