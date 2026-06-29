// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use strict';

const TO_WIRE_KEYS = Object.freeze({
  agentId: 'agent_id',
  batchStore: 'batch_store',
  eventAfter: 'event_after',
  eventBefore: 'event_before',
  eventTimestamp: 'event_timestamp',
  idempotencyKey: 'idempotency_key',
  ingestedAfter: 'ingested_after',
  ingestedAt: 'ingested_at',
  ingestedBefore: 'ingested_before',
  jobId: 'job_id',
  operationId: 'operation_id',
  parentMemoryIds: 'parent_memory_ids',
  partialErrors: 'partial_errors',
  providerMetadata: 'provider_metadata',
  sessionId: 'session_id',
  sourceIds: 'source_ids',
  subjectId: 'subject_id',
  tenantId: 'tenant_id',
});

const FROM_WIRE_KEYS = Object.freeze(
  Object.fromEntries(Object.entries(TO_WIRE_KEYS).map(([publicKey, wireKey]) => [wireKey, publicKey])),
);

const PRESERVED_PUBLIC_KEYS = new Set(['details', 'metadata', 'parameters', 'providerMetadata', 'value']);
const PRESERVED_WIRE_KEYS = new Set(['details', 'metadata', 'parameters', 'provider_metadata', 'value']);
const SEARCH_SCOPES = new Set(['subject', 'agent', 'session', 'exact']);

/** Error raised when a memory value violates the provider-neutral contract. */
class MemoryContractError extends TypeError {
  /**
   * Create an invalid-request error.
   *
   * @param {string} message - Human-readable contract failure.
   */
  constructor(message) {
    super(message);
    this.name = 'MemoryContractError';
    this.code = 'invalid_request';
  }
}

function cloneJson(value) {
  if (Array.isArray(value)) {
    return value.map(cloneJson);
  }
  if (value !== null && typeof value === 'object') {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, cloneJson(item)]));
  }
  return value;
}

function transformKeys(value, keyMap, preservedKeys) {
  if (Array.isArray(value)) {
    return value.map((item) => transformKeys(item, keyMap, preservedKeys));
  }
  if (value !== null && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value).map(([key, item]) => [
        keyMap[key] ?? key,
        preservedKeys.has(key) ? cloneJson(item) : transformKeys(item, keyMap, preservedKeys),
      ]),
    );
  }
  return value;
}

function requireIdentifier(value, name) {
  if (typeof value !== 'string' || value.trim().length === 0) {
    throw new MemoryContractError(`${name} must not be empty`);
  }
}

/**
 * Validate a public Node memory namespace and its requested search scope.
 *
 * @param {object} namespace - Camel-case namespace to validate.
 * @param {string} [scope='subject'] - Search scope whose required fields must be present.
 * @returns {object} The unchanged namespace after successful validation.
 * @throws {MemoryContractError} If an identifier or search scope is invalid.
 */
function validateMemoryNamespace(namespace, scope = 'subject') {
  if (namespace === null || typeof namespace !== 'object' || Array.isArray(namespace)) {
    throw new MemoryContractError('namespace must be an object');
  }

  requireIdentifier(namespace.tenantId, 'tenantId');
  requireIdentifier(namespace.subjectId, 'subjectId');
  if (namespace.sessionId !== undefined) {
    requireIdentifier(namespace.sessionId, 'sessionId');
  }
  if (namespace.agentId !== undefined) {
    requireIdentifier(namespace.agentId, 'agentId');
  }
  if (!SEARCH_SCOPES.has(scope)) {
    throw new MemoryContractError(`unsupported memory search scope: ${scope}`);
  }
  if (scope === 'agent' && namespace.agentId === undefined) {
    throw new MemoryContractError('agent scope requires agentId');
  }
  if (scope === 'session' && namespace.sessionId === undefined) {
    throw new MemoryContractError('session scope requires sessionId');
  }
  return namespace;
}

/**
 * Return the capability set implemented by a search-and-store-only provider.
 *
 * @returns {object} A fresh capability object with every optional operation disabled.
 */
function defaultMemoryCapabilities() {
  return {
    update: false,
    delete: false,
    batchStore: false,
    maintenance: false,
    feedback: false,
    health: false,
  };
}

/**
 * Convert camel-case public Node memory values to the canonical snake-case wire shape.
 *
 * Only provider-neutral contract keys are renamed. JSON content, metadata,
 * provider metadata, error details, and maintenance parameters remain opaque.
 *
 * @param {*} value - Public memory DTO, array, or scalar to convert.
 * @returns {*} A detached canonical wire value.
 */
function toMemoryWire(value) {
  return transformKeys(value, TO_WIRE_KEYS, PRESERVED_PUBLIC_KEYS);
}

/**
 * Convert canonical snake-case memory wire values to camel-case public Node values.
 *
 * Only provider-neutral contract keys are renamed. JSON content, metadata,
 * provider metadata, error details, and maintenance parameters remain opaque.
 *
 * @param {*} value - Canonical memory wire DTO, array, or scalar to convert.
 * @returns {*} A detached public Node value.
 */
function fromMemoryWire(value) {
  return transformKeys(value, FROM_WIRE_KEYS, PRESERVED_WIRE_KEYS);
}

module.exports = {
  MemoryContractError,
  validateMemoryNamespace,
  defaultMemoryCapabilities,
  toMemoryWire,
  fromMemoryWire,
};
