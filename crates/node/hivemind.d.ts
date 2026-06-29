// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Json, ScopeHandle } from './index';
import type { MemoryNamespace } from './memory';

/** Operations exposed by the Hivemind organizational-learning profile. */
export interface HivemindCapabilities {
  traceCapture: true;
  search: true;
  summaries: true;
  skills: true;
  store: false;
  update: false;
  delete: false;
  vectorCrud: false;
}

/** Explicit content-sharing policy required before any Hivemind capture. */
export interface HivemindCapturePolicy {
  consent: 'explicit';
  captureContent: true;
  categories?: string[];
  /** Return a sanitized event or null to suppress it. */
  redact?: (event: Record<string, Json>) => Record<string, Json> | null;
}

/** Required Relay identity for one captured Hivemind source session. */
export interface HivemindNamespace extends MemoryNamespace {
  sessionId: string;
  agentId: string;
}

/** Bounded canonical event handed to a Hivemind transport client. */
export interface HivemindCaptureInput {
  sessionId: string;
  cwd: string;
  model: string;
  event: Record<string, Json>;
}

/** Source-session finalization input. */
export interface HivemindFinishInput {
  sessionId: string;
  cwd: string;
  model: string;
  assistantMessage: string;
}

/** Supported organizational artifact classifications. */
export type HivemindArtifactKind = 'trace' | 'summary' | 'skill';

/** Raw provider artifact returned by an injected Hivemind client. */
export interface HivemindClientArtifact {
  kind: HivemindArtifactKind;
  reference: string;
  preview?: string;
  sourceSessionIds?: string[];
  providerMetadata?: Record<string, Json>;
}

/** Search input handed to an injected Hivemind client. */
export interface HivemindClientSearchRequest {
  query: string;
  kinds: HivemindArtifactKind[];
  limit: number;
}

/** Search output from an injected Hivemind client. */
export interface HivemindClientSearchResult {
  artifacts: HivemindClientArtifact[];
  partialErrors?: string[];
  truncated?: boolean;
}

/** Minimal async client contract used by the organizational profile. */
export interface HivemindClient {
  capture(input: HivemindCaptureInput): Promise<void>;
  finish(input: HivemindFinishInput): Promise<void>;
  search(input: HivemindClientSearchRequest): Promise<HivemindClientSearchResult>;
}

/** Provenance retained for one normalized organizational artifact. */
export interface HivemindArtifactProvenance {
  source: 'hivemind';
  derivation: 'captured_trace' | 'session_summary' | 'trace_skillification';
  sourceSessionIds: string[];
  provenanceUnavailable: boolean;
}

/** Ordered trace, summary, or skill discovered through Hivemind. */
export interface HivemindArtifact {
  id: string;
  kind: HivemindArtifactKind;
  reference: string;
  preview?: string;
  rank: number;
  provenance: HivemindArtifactProvenance;
  providerMetadata: Record<string, Json> & { scoreSemantics: 'rank_only' };
}

/** Public organizational search result. */
export interface HivemindSearchResult {
  artifacts: HivemindArtifact[];
  partialErrors: string[];
  truncated: boolean;
}

/** Asynchronous capture status. */
export interface HivemindCaptureStatus {
  accepting: boolean;
  pending: number;
  capacity: number;
  accepted: number;
  succeeded: number;
  failed: number;
  rejected: number;
  lastError: { kind: string; message: string } | null;
}

/** Organizational profile construction options. */
export interface HivemindOrganizationalMemoryConfig {
  client: HivemindClient;
  namespace: HivemindNamespace;
  capturePolicy: HivemindCapturePolicy;
  cwd?: string;
  model?: string;
  maxEventBytes?: number;
  maxTextBytes?: number;
  queueCapacity?: number;
}

/** Error raised when Hivemind organizational-memory input is invalid. */
export declare class HivemindContractError extends TypeError {
  readonly code: 'invalid_request';
}

/** Error raised when an installed Hivemind hook or MCP process fails. */
export declare class HivemindTransportError extends Error {
  readonly code: 'provider_unavailable';
  readonly details?: object;
}

/**
 * Explicit-consent Relay lifecycle capture and organizational artifact search.
 *
 * This is intentionally not a generic `MemoryProvider` implementation.
 */
export declare class HivemindOrganizationalMemory {
  constructor(config: HivemindOrganizationalMemoryConfig);
  readonly capabilities: HivemindCapabilities;
  readonly sessionKey: string;
  readonly status: HivemindCaptureStatus;
  capture(event: Record<string, Json>): boolean;
  /** Install globally, or only under one Relay scope for multi-agent isolation. */
  install(options?: { name?: string; scope?: ScopeHandle }): this;
  finish(input?: { assistantMessage?: string }): Promise<void>;
  search(input: { query: string; kinds?: HivemindArtifactKind[]; limit?: number }): Promise<HivemindSearchResult>;
  flush(): Promise<void>;
  close(): boolean;
  shutdown(): Promise<void>;
}

/** Configuration for an independently installed Hivemind process bridge. */
export interface InstalledHivemindClientConfig {
  homeDirectory?: string;
  captureHook?: string;
  stopHook?: string;
  mcpServer?: string;
  skillRoots?: string[];
  timeoutMillis?: number;
  environment?: Record<string, string>;
}

/** Process bridge to installed Codex hooks, read-only MCP, and local skills. */
export declare class InstalledHivemindClient implements HivemindClient {
  constructor(config?: InstalledHivemindClientConfig);
  capture(input: HivemindCaptureInput): Promise<void>;
  finish(input: HivemindFinishInput): Promise<void>;
  search(input: HivemindClientSearchRequest): Promise<HivemindClientSearchResult>;
}

/** Return the exact supported and unsupported Hivemind capability profile. */
export declare function hivemindCapabilities(): HivemindCapabilities;

/** Derive a versioned, non-reversible Hivemind source-session key. */
export declare function opaqueHivemindSessionKey(namespace: HivemindNamespace): string;
