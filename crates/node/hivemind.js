// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use strict';

const { createHash } = require('node:crypto');
const { existsSync } = require('node:fs');
const { mkdtemp, readdir, readFile, rm, stat, writeFile } = require('node:fs/promises');
const { homedir, tmpdir } = require('node:os');
const path = require('node:path');
const { spawn } = require('node:child_process');

const DEFAULT_TIMEOUT_MILLIS = 10_000;
const DEFAULT_MAX_EVENT_BYTES = 256 * 1024;
const DEFAULT_MAX_TEXT_BYTES = 64 * 1024;
const DEFAULT_QUEUE_CAPACITY = 1024;
const MAX_PROCESS_OUTPUT_BYTES = 1024 * 1024;
const MAX_SKILL_BYTES = 1024 * 1024;
const MAX_PENDING_TOOL_STARTS = 1024;
const PACKAGE_VERSION = require('./package.json').version;
const DEFAULT_CATEGORIES = Object.freeze(['llm', 'tool', 'memory']);
const ORGANIZATIONAL_CAPABILITIES = Object.freeze({
  traceCapture: true,
  search: true,
  summaries: true,
  skills: true,
  store: false,
  update: false,
  delete: false,
  vectorCrud: false,
});

/** Error raised when Hivemind organizational-memory configuration is invalid. */
class HivemindContractError extends TypeError {
  /**
   * Create an invalid-request error.
   *
   * @param {string} message - Human-readable contract failure.
   */
  constructor(message) {
    super(message);
    this.name = 'HivemindContractError';
    this.code = 'invalid_request';
  }
}

/** Error raised when an installed Hivemind hook or MCP process fails. */
class HivemindTransportError extends Error {
  /**
   * Create a transport error with a stable code.
   *
   * @param {string} message - Human-readable transport failure.
   * @param {object} [options] - Optional cause and diagnostic details.
   */
  constructor(message, options = {}) {
    super(message, options.cause === undefined ? undefined : { cause: options.cause });
    this.name = 'HivemindTransportError';
    this.code = 'provider_unavailable';
    this.details = options.details;
  }
}

/** Return the exact partial capability profile implemented by this integration. */
function hivemindCapabilities() {
  return { ...ORGANIZATIONAL_CAPABILITIES };
}

function requireIdentifier(value, name) {
  if (typeof value !== 'string' || value.trim().length === 0) {
    throw new HivemindContractError(`${name} must not be empty`);
  }
}

function validateNamespace(namespace) {
  if (namespace === null || typeof namespace !== 'object' || Array.isArray(namespace)) {
    throw new HivemindContractError('namespace must be an object');
  }
  requireIdentifier(namespace.tenantId, 'tenantId');
  requireIdentifier(namespace.subjectId, 'subjectId');
  requireIdentifier(namespace.sessionId, 'sessionId');
  requireIdentifier(namespace.agentId, 'agentId');
  return namespace;
}

/**
 * Derive a non-reversible Hivemind session key from an explicit Relay namespace.
 *
 * The Deeplake workspace remains the security boundary. This key prevents raw
 * Relay tenant and subject identifiers from appearing in Hivemind paths.
 *
 * @param {object} namespace - Tenant, subject, session, and agent identifiers.
 * @returns {string} A stable versioned opaque session key.
 */
function opaqueHivemindSessionKey(namespace) {
  validateNamespace(namespace);
  const canonical = JSON.stringify([
    namespace.tenantId.trim(),
    namespace.subjectId.trim(),
    namespace.sessionId.trim(),
    namespace.agentId.trim(),
  ]);
  return `nmr1-${createHash('sha256').update('nemo-relay-hivemind-session-v1\0').update(canonical).digest('hex').slice(0, 32)}`;
}

function validateCapturePolicy(policy) {
  if (policy === null || typeof policy !== 'object' || Array.isArray(policy)) {
    throw new HivemindContractError('capturePolicy must be an object');
  }
  if (policy.consent !== 'explicit' || policy.captureContent !== true) {
    throw new HivemindContractError(
      "Hivemind capture requires consent='explicit' and captureContent=true because workspace members can read captured content",
    );
  }
  if (policy.categories !== undefined) {
    if (
      !Array.isArray(policy.categories) ||
      policy.categories.length === 0 ||
      policy.categories.some((item) => typeof item !== 'string' || item.trim().length === 0)
    ) {
      throw new HivemindContractError('capturePolicy.categories must be a non-empty array of non-empty strings');
    }
  }
  if (policy.redact !== undefined && typeof policy.redact !== 'function') {
    throw new HivemindContractError('capturePolicy.redact must be a function');
  }
}

function cloneJson(value) {
  try {
    return JSON.parse(JSON.stringify(value));
  } catch {
    throw new HivemindContractError('captured values must be JSON-serializable');
  }
}

function boundedEvent(event, maxEventBytes) {
  const snapshot = cloneJson(event);
  const encoded = JSON.stringify(snapshot);
  const size = Buffer.byteLength(encoded, 'utf8');
  if (size <= maxEventBytes) {
    return snapshot;
  }
  return {
    kind: snapshot.kind,
    name: snapshot.name,
    category: snapshot.category,
    scope_category: snapshot.scope_category,
    timestamp: snapshot.timestamp,
    data: {
      omitted: true,
      reason: 'event_size_limit',
      original_bytes: size,
    },
  };
}

function sourceSessionsFromReference(reference) {
  const matches = String(reference).match(/nmr1-[a-f0-9]{32}/g);
  return matches === null ? [] : [...new Set(matches)];
}

function stableArtifactId(kind, reference) {
  return `hivemind-${createHash('sha256').update(`${kind}\0${reference}`).digest('hex').slice(0, 32)}`;
}

function normalizeArtifact(candidate, rank) {
  if (candidate === null || typeof candidate !== 'object' || Array.isArray(candidate)) {
    throw new HivemindContractError('Hivemind search returned a non-object artifact');
  }
  const kind = candidate.kind;
  if (!['trace', 'summary', 'skill'].includes(kind)) {
    throw new HivemindContractError(`unsupported Hivemind artifact kind: ${kind}`);
  }
  requireIdentifier(candidate.reference, 'artifact.reference');
  if (
    candidate.providerMetadata !== undefined &&
    (candidate.providerMetadata === null ||
      typeof candidate.providerMetadata !== 'object' ||
      Array.isArray(candidate.providerMetadata))
  ) {
    throw new HivemindContractError('artifact.providerMetadata must be an object');
  }
  const providedSessions = Array.isArray(candidate.sourceSessionIds)
    ? candidate.sourceSessionIds
        .filter((item) => typeof item === 'string' && item.trim().length > 0)
        .map((item) => item.trim())
    : [];
  const sourceSessionIds = [...new Set([...providedSessions, ...sourceSessionsFromReference(candidate.reference)])];
  const derivation =
    kind === 'summary' ? 'session_summary' : kind === 'skill' ? 'trace_skillification' : 'captured_trace';
  return {
    id: stableArtifactId(kind, candidate.reference),
    kind,
    reference: candidate.reference,
    ...(typeof candidate.preview === 'string' && candidate.preview.length > 0 ? { preview: candidate.preview } : {}),
    rank,
    provenance: {
      source: 'hivemind',
      derivation,
      sourceSessionIds,
      provenanceUnavailable: sourceSessionIds.length === 0,
    },
    providerMetadata: {
      ...(candidate.providerMetadata ?? {}),
      scoreSemantics: 'rank_only',
    },
  };
}

/**
 * Explicit-consent Relay lifecycle capture and organizational artifact search.
 *
 * This class intentionally does not implement `MemoryProvider`: Hivemind trace
 * capture and derived artifacts are not generic fact storage or vector CRUD.
 */
class HivemindOrganizationalMemory {
  #client;
  #sessionKey;
  #redact;
  #categories;
  #cwd;
  #model;
  #maxEventBytes;
  #maxTextBytes;
  #queueCapacity;
  #tail = Promise.resolve();
  #pending = 0;
  #accepted = 0;
  #succeeded = 0;
  #failed = 0;
  #rejected = 0;
  #lastError = null;
  #closed = false;
  #subscriberName = null;
  #subscriberScopeUuid = null;

  /**
   * Create an organizational-memory profile.
   *
   * @param {object} config - Client, identity, explicit capture policy, and process metadata.
   */
  constructor(config) {
    if (config === null || typeof config !== 'object' || Array.isArray(config)) {
      throw new HivemindContractError('config must be an object');
    }
    if (
      config.client === null ||
      typeof config.client !== 'object' ||
      typeof config.client.capture !== 'function' ||
      typeof config.client.finish !== 'function' ||
      typeof config.client.search !== 'function'
    ) {
      throw new HivemindContractError('client must implement async capture, finish, and search methods');
    }
    validateNamespace(config.namespace);
    validateCapturePolicy(config.capturePolicy);
    const maxEventBytes = config.maxEventBytes ?? DEFAULT_MAX_EVENT_BYTES;
    if (!Number.isInteger(maxEventBytes) || maxEventBytes < 1024) {
      throw new HivemindContractError('maxEventBytes must be an integer of at least 1024');
    }
    const maxTextBytes = config.maxTextBytes ?? DEFAULT_MAX_TEXT_BYTES;
    if (!Number.isInteger(maxTextBytes) || maxTextBytes < 1024) {
      throw new HivemindContractError('maxTextBytes must be an integer of at least 1024');
    }
    const queueCapacity = config.queueCapacity ?? DEFAULT_QUEUE_CAPACITY;
    if (!Number.isInteger(queueCapacity) || queueCapacity < 1) {
      throw new HivemindContractError('queueCapacity must be a positive integer');
    }
    if (config.cwd !== undefined) requireIdentifier(config.cwd, 'cwd');
    if (config.model !== undefined) requireIdentifier(config.model, 'model');

    this.#client = config.client;
    this.#sessionKey = opaqueHivemindSessionKey(config.namespace);
    this.#redact = config.capturePolicy.redact;
    this.#categories = new Set((config.capturePolicy.categories ?? DEFAULT_CATEGORIES).map((item) => item.trim()));
    this.#cwd = config.cwd ?? process.cwd();
    this.#model = config.model ?? 'unknown';
    this.#maxEventBytes = maxEventBytes;
    this.#maxTextBytes = maxTextBytes;
    this.#queueCapacity = queueCapacity;
  }

  /** Exact supported and unsupported organizational-memory operations. */
  get capabilities() {
    return hivemindCapabilities();
  }

  /** Opaque Hivemind session correlation key; it contains no raw Relay identity. */
  get sessionKey() {
    return this.#sessionKey;
  }

  /** Snapshot of accepted asynchronous capture work and its latest failure. */
  get status() {
    return {
      accepting: !this.#closed,
      pending: this.#pending,
      capacity: this.#queueCapacity,
      accepted: this.#accepted,
      succeeded: this.#succeeded,
      failed: this.#failed,
      rejected: this.#rejected,
      lastError: this.#lastError === null ? null : { ...this.#lastError },
    };
  }

  #enqueue(kind, operation) {
    if (this.#closed) {
      throw new HivemindContractError('Hivemind organizational memory is closed');
    }
    if (this.#pending >= this.#queueCapacity) {
      this.#rejected += 1;
      this.#lastError = {
        kind: 'backpressure',
        message: `Hivemind capture queue is full (${this.#queueCapacity})`,
      };
      throw new HivemindTransportError(this.#lastError.message, {
        details: { kind: 'backpressure', capacity: this.#queueCapacity },
      });
    }
    this.#accepted += 1;
    this.#pending += 1;
    const current = this.#tail.then(operation);
    this.#tail = current
      .then(
        () => {
          this.#succeeded += 1;
        },
        (error) => {
          this.#failed += 1;
          this.#lastError = {
            kind,
            message: error instanceof Error ? error.message : String(error),
          };
        },
      )
      .then(() => {
        this.#pending -= 1;
      });
    return current;
  }

  #deregisterSubscriber() {
    if (this.#subscriberName === null) return;
    const relay = require('./index.js');
    try {
      if (this.#subscriberScopeUuid === null) {
        relay.deregisterSubscriber(this.#subscriberName);
      } else {
        relay.scopeDeregisterSubscriber(this.#subscriberScopeUuid, this.#subscriberName);
      }
    } catch (error) {
      const alreadyCleaned =
        this.#subscriberScopeUuid !== null &&
        error instanceof Error &&
        /scope [0-9a-f-]+ not found/i.test(error.message);
      if (!alreadyCleaned) throw error;
    }
    this.#subscriberName = null;
    this.#subscriberScopeUuid = null;
  }

  /**
   * Accept one canonical Relay event for asynchronous Hivemind capture.
   *
   * @param {object} event - Canonical ATOF event object.
   * @returns {boolean} Whether the event category was accepted.
   */
  capture(event) {
    if (this.#closed) {
      throw new HivemindContractError('Hivemind organizational memory is closed');
    }
    if (event === null || typeof event !== 'object' || Array.isArray(event)) {
      throw new HivemindContractError('event must be an object');
    }
    if (!this.#categories.has(event.category)) {
      return false;
    }
    const transformed = this.#redact === undefined ? event : this.#redact(cloneJson(event));
    if (transformed === null) {
      return false;
    }
    if (typeof transformed !== 'object' || Array.isArray(transformed)) {
      throw new HivemindContractError('capturePolicy.redact must return an event object or null');
    }
    const snapshot = boundedEvent(transformed, this.#maxEventBytes);
    void this.#enqueue('capture', () =>
      this.#client.capture({
        sessionId: this.#sessionKey,
        cwd: this.#cwd,
        model: this.#model,
        event: snapshot,
      }),
    );
    return true;
  }

  /**
   * Register this profile as a non-blocking global or scope-local Relay event subscriber.
   *
   * @param {object} [options] - Optional unique subscriber name and scope.
   * @returns {HivemindOrganizationalMemory} This profile for fluent setup.
   */
  install({ name = 'hivemind-organizational-memory', scope } = {}) {
    if (this.#closed) {
      throw new HivemindContractError('Hivemind organizational memory is closed');
    }
    if (this.#subscriberName !== null) {
      throw new HivemindContractError('Hivemind organizational memory is already installed');
    }
    requireIdentifier(name, 'name');
    if (
      scope !== undefined &&
      (scope === null || typeof scope !== 'object' || typeof scope.uuid !== 'string' || scope.uuid.length === 0)
    ) {
      throw new HivemindContractError('scope must be a Relay ScopeHandle');
    }
    const relay = require('./index.js');
    const subscriber = (event) => {
      try {
        this.capture(event);
      } catch (error) {
        if (!(error instanceof HivemindTransportError && error.details?.kind === 'backpressure')) {
          this.#failed += 1;
        }
        this.#lastError = {
          kind: 'subscriber',
          message: error instanceof Error ? error.message : String(error),
        };
      }
    };
    if (scope === undefined) {
      relay.registerSubscriber(name, subscriber);
    } else {
      relay.scopeRegisterSubscriber(scope.uuid, name, subscriber);
      this.#subscriberScopeUuid = scope.uuid;
    }
    this.#subscriberName = name;
    return this;
  }

  /**
   * Finish the source session and trigger Hivemind's detached derivation workers.
   *
   * @param {object} [input] - Optional final assistant content.
   * @returns {Promise<void>} Resolves when the finish transport acknowledges.
   */
  async finish({ assistantMessage = '' } = {}) {
    if (typeof assistantMessage !== 'string') {
      throw new HivemindContractError('assistantMessage must be a string');
    }
    if (Buffer.byteLength(assistantMessage, 'utf8') > this.#maxTextBytes) {
      throw new HivemindContractError(`assistantMessage exceeds maxTextBytes (${this.#maxTextBytes})`);
    }
    await this.#enqueue('finish', () =>
      this.#client.finish({
        sessionId: this.#sessionKey,
        cwd: this.#cwd,
        model: this.#model,
        assistantMessage,
      }),
    );
  }

  /**
   * Search ordered Hivemind traces, summaries, or skills.
   *
   * @param {object} request - Query, optional kinds, and result limit.
   * @returns {Promise<object>} Ranked artifacts and non-fatal provider errors.
   */
  async search({ query, kinds = ['trace', 'summary', 'skill'], limit = 10 }) {
    requireIdentifier(query, 'query');
    if (Buffer.byteLength(query, 'utf8') > this.#maxTextBytes) {
      throw new HivemindContractError(`query exceeds maxTextBytes (${this.#maxTextBytes})`);
    }
    if (
      !Array.isArray(kinds) ||
      kinds.length === 0 ||
      kinds.some((kind) => !['trace', 'summary', 'skill'].includes(kind))
    ) {
      throw new HivemindContractError('kinds must contain trace, summary, or skill');
    }
    if (!Number.isInteger(limit) || limit < 1 || limit > 50) {
      throw new HivemindContractError('limit must be an integer from 1 through 50');
    }
    await this.flush();
    const result = await this.#client.search({ query, kinds: [...new Set(kinds)], limit });
    if (result === null || typeof result !== 'object' || !Array.isArray(result.artifacts)) {
      throw new HivemindTransportError('Hivemind client returned an invalid search result');
    }
    return {
      artifacts: result.artifacts.slice(0, limit).map((candidate, index) => normalizeArtifact(candidate, index + 1)),
      partialErrors: Array.isArray(result.partialErrors)
        ? result.partialErrors.filter((error) => typeof error === 'string')
        : [],
      truncated: result.truncated === true,
    };
  }

  /** Wait for all accepted capture work and surface the latest transport failure. */
  async flush() {
    await this.#tail;
    if (this.#lastError !== null) {
      throw new HivemindTransportError(`Hivemind ${this.#lastError.kind} failed: ${this.#lastError.message}`, {
        details: this.#lastError,
      });
    }
  }

  /** Deregister this profile once while allowing accepted work to drain. */
  close() {
    if (this.#closed) {
      return false;
    }
    if (this.#subscriberName !== null) {
      this.#deregisterSubscriber();
    }
    this.#closed = true;
    return true;
  }

  /** Deregister and wait for accepted work without implicitly ending a session. */
  async shutdown() {
    if (!this.#closed && this.#subscriberName !== null) {
      const relay = require('./index.js');
      this.#deregisterSubscriber();
      relay.flushSubscribers();
      await new Promise((resolve) => setImmediate(resolve));
    }
    this.#closed = true;
    await this.flush();
  }
}

function defaultInstalledPaths(homeDirectory = homedir()) {
  return {
    captureHook: path.join(homeDirectory, '.codex', 'hivemind', 'bundle', 'capture.js'),
    stopHook: path.join(homeDirectory, '.codex', 'hivemind', 'bundle', 'stop.js'),
    mcpServer: path.join(homeDirectory, '.hivemind', 'mcp', 'server.js'),
  };
}

function processError(label, message, details) {
  return new HivemindTransportError(`${label}: ${message}`, { details });
}

function runHook(entry, payload, timeoutMillis, environment) {
  return new Promise((resolve, reject) => {
    if (!existsSync(entry)) {
      reject(processError('Hivemind hook unavailable', `file does not exist: ${entry}`));
      return;
    }
    const child = spawn(process.execPath, [entry], {
      env: { ...process.env, HIVEMIND_CAPTURE: 'true', ...environment },
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';
    let settled = false;
    const finish = (callback) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      callback();
    };
    const timer = setTimeout(() => {
      child.kill('SIGKILL');
      finish(() => reject(processError('Hivemind hook timed out', entry)));
    }, timeoutMillis);
    child.stdout.on('data', (chunk) => {
      stdout = (stdout + chunk.toString()).slice(-16_384);
    });
    child.stderr.on('data', (chunk) => {
      stderr = (stderr + chunk.toString()).slice(-16_384);
    });
    child.once('error', (error) => {
      finish(() =>
        reject(new HivemindTransportError(`failed to start Hivemind hook: ${error.message}`, { cause: error })),
      );
    });
    child.stdin.on('error', (error) => {
      finish(() =>
        reject(new HivemindTransportError(`failed to write Hivemind hook input: ${error.message}`, { cause: error })),
      );
    });
    child.once('exit', (code, signal) => {
      if (code === 0) {
        finish(() => resolve({ stdout, stderr }));
      } else {
        finish(() =>
          reject(
            processError('Hivemind hook failed', `exit=${code ?? 'null'} signal=${signal ?? 'none'}`, {
              entry,
              stdout,
              stderr,
            }),
          ),
        );
      }
    });
    child.stdin.end(`${JSON.stringify(payload)}\n`);
  });
}

function callMcpTool(entry, toolName, args, timeoutMillis, environment) {
  return new Promise((resolve, reject) => {
    if (!existsSync(entry)) {
      reject(processError('Hivemind MCP server unavailable', `file does not exist: ${entry}`));
      return;
    }
    const child = spawn(process.execPath, [entry], {
      env: { ...process.env, ...environment },
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    let buffer = '';
    let stderr = '';
    let settled = false;
    const finish = (callback) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      child.kill('SIGTERM');
      callback();
    };
    const timer = setTimeout(() => {
      finish(() => reject(processError('Hivemind MCP call timed out', toolName, { stderr })));
    }, timeoutMillis);
    child.stderr.on('data', (chunk) => {
      stderr = (stderr + chunk.toString()).slice(-16_384);
    });
    child.once('error', (error) => {
      finish(() =>
        reject(new HivemindTransportError(`failed to start Hivemind MCP server: ${error.message}`, { cause: error })),
      );
    });
    child.stdin.on('error', (error) => {
      finish(() =>
        reject(new HivemindTransportError(`failed to write Hivemind MCP input: ${error.message}`, { cause: error })),
      );
    });
    child.once('exit', (code, signal) => {
      if (!settled) {
        finish(() =>
          reject(
            processError('Hivemind MCP server exited', `exit=${code ?? 'null'} signal=${signal ?? 'none'}`, { stderr }),
          ),
        );
      }
    });
    const send = (message) => child.stdin.write(`${JSON.stringify(message)}\n`);
    child.stdout.on('data', (chunk) => {
      buffer += chunk.toString();
      if (Buffer.byteLength(buffer, 'utf8') > MAX_PROCESS_OUTPUT_BYTES) {
        finish(() =>
          reject(processError('Hivemind MCP output exceeded limit', `${MAX_PROCESS_OUTPUT_BYTES} bytes`, { stderr })),
        );
        return;
      }
      let newline;
      while ((newline = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, newline).trim();
        buffer = buffer.slice(newline + 1);
        if (line.length === 0) continue;
        let message;
        try {
          message = JSON.parse(line);
        } catch {
          finish(() => reject(processError('Hivemind MCP returned invalid JSON', line.slice(0, 300), { stderr })));
          return;
        }
        if (message.id === 1 && message.result !== undefined) {
          send({ jsonrpc: '2.0', method: 'notifications/initialized' });
          send({ jsonrpc: '2.0', id: 2, method: 'tools/call', params: { name: toolName, arguments: args } });
        } else if (message.id === 1 && message.error !== undefined) {
          finish(() =>
            reject(processError('Hivemind MCP initialization failed', JSON.stringify(message.error), { stderr })),
          );
        } else if (message.id === 2 && message.result !== undefined) {
          finish(() => resolve(message.result));
        } else if (message.id === 2 && message.error !== undefined) {
          finish(() => reject(processError('Hivemind MCP tool failed', JSON.stringify(message.error), { stderr })));
        }
      }
    });
    send({
      jsonrpc: '2.0',
      id: 1,
      method: 'initialize',
      params: {
        protocolVersion: '2025-06-18',
        capabilities: {},
        clientInfo: { name: 'nemo-relay', version: PACKAGE_VERSION },
      },
    });
  });
}

function textFromMcpResult(result) {
  if (result === null || typeof result !== 'object' || !Array.isArray(result.content)) {
    throw new HivemindTransportError('Hivemind MCP returned an invalid tool result');
  }
  return result.content
    .filter(
      (item) => item !== null && typeof item === 'object' && item.type === 'text' && typeof item.text === 'string',
    )
    .map((item) => item.text)
    .join('\n');
}

function parseMcpSearchText(text) {
  const partialErrors = [];
  const artifacts = [];
  let truncated = false;
  for (const block of text.split(/\n\n---\n\n/)) {
    const trimmed = block.trim();
    if (trimmed.length === 0) continue;
    const match = /^\[([^\]]+)]\n([\s\S]*)$/.exec(trimmed);
    if (match !== null) {
      const reference = match[1];
      const kind = reference.startsWith('/summaries/')
        ? 'summary'
        : reference.startsWith('/sessions/')
          ? 'trace'
          : null;
      if (kind !== null) {
        artifacts.push({ kind, reference, preview: match[2].trim() });
      }
      continue;
    }
    if (/truncat|additional matches|result limit/i.test(trimmed)) {
      truncated = true;
    } else if (/^No matches/i.test(trimmed)) {
      continue;
    } else if (/^(Search failed|Not authenticated|Hivemind memory is empty)/i.test(trimmed)) {
      partialErrors.push(trimmed);
    } else {
      partialErrors.push(`Unparsed Hivemind search response: ${trimmed.slice(0, 300)}`);
    }
  }
  return { artifacts, partialErrors, truncated };
}

function parseSkillFrontmatter(content) {
  if (!content.startsWith('---\n')) return [];
  const end = content.indexOf('\n---', 4);
  if (end < 0) return [];
  const lines = content.slice(4, end).split('\n');
  const sessions = [];
  let inSources = false;
  for (const line of lines) {
    if (/^source_sessions:\s*$/.test(line)) {
      inSources = true;
      continue;
    }
    if (/^[A-Za-z_][A-Za-z0-9_-]*:/.test(line)) {
      inSources = false;
    }
    if (inSources) {
      const match = /^\s*-\s*["']?([^"']+?)["']?\s*$/.exec(line);
      if (match !== null) sessions.push(match[1]);
    }
  }
  return [...new Set(sessions)];
}

async function findSkillFiles(root) {
  const output = [];
  let children;
  try {
    children = await readdir(root, { withFileTypes: true });
  } catch (error) {
    if (error !== null && typeof error === 'object' && error.code === 'ENOENT') return output;
    throw error;
  }
  children.sort((left, right) => left.name.localeCompare(right.name));
  for (const child of children) {
    if (!child.isDirectory()) continue;
    const skillPath = path.join(root, child.name, 'SKILL.md');
    if (existsSync(skillPath)) output.push(skillPath);
  }
  return output;
}

async function searchLocalSkills(roots, query, limit) {
  const needle = query.toLocaleLowerCase();
  const artifacts = [];
  const partialErrors = [];
  for (const root of roots) {
    for (const skillPath of await findSkillFiles(root)) {
      const skillStat = await stat(skillPath);
      if (skillStat.size > MAX_SKILL_BYTES) {
        partialErrors.push(`Skipped oversized Hivemind skill (${skillStat.size} bytes): ${skillPath}`);
        continue;
      }
      const content = await readFile(skillPath, 'utf8');
      if (!content.toLocaleLowerCase().includes(needle)) continue;
      artifacts.push({
        kind: 'skill',
        reference: skillPath,
        preview: content.slice(0, 600),
        sourceSessionIds: parseSkillFrontmatter(content),
      });
      if (artifacts.length >= limit) return { artifacts, partialErrors };
    }
  }
  return { artifacts, partialErrors };
}

/**
 * Process bridge for an independently installed Activeloop Hivemind.
 *
 * It invokes documented installed Codex hooks for capture/finalization and the
 * read-only Hivemind MCP server for trace and summary search. Optional local
 * skill roots expose generated `SKILL.md` files without modifying them.
 */
class InstalledHivemindClient {
  #captureHook;
  #stopHook;
  #mcpServer;
  #skillRoots;
  #timeoutMillis;
  #environment;
  #toolStarts = new Map();

  /**
   * Create a bridge with explicit paths or documented install defaults.
   *
   * @param {object} [config] - Hook paths, MCP path, skill roots, timeout, and environment.
   */
  constructor(config = {}) {
    if (config === null || typeof config !== 'object' || Array.isArray(config)) {
      throw new HivemindContractError('config must be an object');
    }
    if (config.skillRoots !== undefined && !Array.isArray(config.skillRoots)) {
      throw new HivemindContractError('skillRoots must be an array');
    }
    if (
      config.environment !== undefined &&
      (config.environment === null || typeof config.environment !== 'object' || Array.isArray(config.environment))
    ) {
      throw new HivemindContractError('environment must be an object');
    }
    for (const field of ['homeDirectory', 'captureHook', 'stopHook', 'mcpServer']) {
      if (config[field] !== undefined) requireIdentifier(config[field], field);
    }
    if (
      config.environment !== undefined &&
      Object.entries(config.environment).some(([key, value]) => key.trim().length === 0 || typeof value !== 'string')
    ) {
      throw new HivemindContractError('environment keys must be non-empty and values must be strings');
    }
    const defaults = defaultInstalledPaths(config.homeDirectory);
    this.#captureHook = config.captureHook ?? defaults.captureHook;
    this.#stopHook = config.stopHook ?? defaults.stopHook;
    this.#mcpServer = config.mcpServer ?? defaults.mcpServer;
    this.#skillRoots = [...(config.skillRoots ?? [])];
    this.#timeoutMillis = config.timeoutMillis ?? DEFAULT_TIMEOUT_MILLIS;
    this.#environment = { ...(config.environment ?? {}) };
    if (!Number.isInteger(this.#timeoutMillis) || this.#timeoutMillis < 1) {
      throw new HivemindContractError('timeoutMillis must be a positive integer');
    }
    if (this.#skillRoots.some((root) => typeof root !== 'string' || root.length === 0)) {
      throw new HivemindContractError('skillRoots must contain non-empty paths');
    }
  }

  /** Capture one bounded Relay event through the installed Codex hook. */
  async capture({ sessionId, cwd, model, event }) {
    const eventId = event.uuid ?? event.id;
    const toolKey = event.category === 'tool' && eventId !== undefined ? `${sessionId}\0${eventId}` : null;
    const toolStartOverflow =
      toolKey !== null && event.scope_category === 'start' && this.#toolStarts.size >= MAX_PENDING_TOOL_STARTS;
    if (toolKey !== null && event.scope_category === 'start' && this.#toolStarts.size < MAX_PENDING_TOOL_STARTS) {
      this.#toolStarts.set(toolKey, event);
      return;
    }
    const isPrompt = event.category === 'llm' && event.scope_category === 'start';
    const toolStart = toolKey === null ? undefined : this.#toolStarts.get(toolKey);
    const payload = isPrompt
      ? {
          session_id: sessionId,
          transcript_path: null,
          cwd,
          hook_event_name: 'UserPromptSubmit',
          model,
          prompt: `[NeMo Relay ATOF llm/start]\n${JSON.stringify(event)}`,
        }
      : {
          session_id: sessionId,
          transcript_path: null,
          cwd,
          hook_event_name: 'PostToolUse',
          model,
          tool_name:
            event.category === 'tool'
              ? (event.name ?? 'relay_tool')
              : `nemo_relay_${event.category ?? 'event'}_${event.scope_category ?? event.kind ?? 'event'}`,
          tool_use_id: eventId,
          tool_input: { relay_event: toolStart ?? event },
          tool_response: toolStartOverflow
            ? { relay_event: null, incomplete: true, reason: 'pairing_capacity' }
            : { relay_event: event.data ?? null },
        };
    await runHook(this.#captureHook, payload, this.#timeoutMillis, this.#environment);
    if (toolKey !== null) this.#toolStarts.delete(toolKey);
  }

  /** Finish one source session and trigger Hivemind's detached workers. */
  async finish({ sessionId, cwd, model, assistantMessage }) {
    const prefix = `${sessionId}\0`;
    for (const [toolKey, event] of this.#toolStarts) {
      if (!toolKey.startsWith(prefix)) continue;
      await runHook(
        this.#captureHook,
        {
          session_id: sessionId,
          transcript_path: null,
          cwd,
          hook_event_name: 'PostToolUse',
          model,
          tool_name: event.name ?? 'relay_tool',
          tool_use_id: event.uuid ?? event.id,
          tool_input: { relay_event: event },
          tool_response: { relay_event: null, incomplete: true },
        },
        this.#timeoutMillis,
        this.#environment,
      );
      this.#toolStarts.delete(toolKey);
    }
    const directory = await mkdtemp(path.join(tmpdir(), 'nemo-relay-hivemind-'));
    const transcriptPath = path.join(directory, 'transcript.jsonl');
    const transcript = {
      type: 'response_item',
      payload: {
        type: 'message',
        role: 'assistant',
        content: [{ type: 'output_text', text: assistantMessage }],
      },
    };
    try {
      await writeFile(transcriptPath, `${JSON.stringify(transcript)}\n`, { encoding: 'utf8', mode: 0o600 });
      await runHook(
        this.#stopHook,
        {
          session_id: sessionId,
          transcript_path: transcriptPath,
          cwd,
          hook_event_name: 'Stop',
          model,
        },
        this.#timeoutMillis,
        this.#environment,
      );
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  }

  /** Search installed Hivemind memory and configured local skill roots. */
  async search({ query, kinds, limit }) {
    const artifacts = [];
    const partialErrors = [];
    let truncated = false;
    if (kinds.includes('trace') || kinds.includes('summary')) {
      const result = await callMcpTool(
        this.#mcpServer,
        'hivemind_search',
        { query, limit },
        this.#timeoutMillis,
        this.#environment,
      );
      const parsed = parseMcpSearchText(textFromMcpResult(result));
      artifacts.push(...parsed.artifacts.filter((artifact) => kinds.includes(artifact.kind)));
      partialErrors.push(...parsed.partialErrors);
      truncated ||= parsed.truncated;
    }
    if (kinds.includes('skill') && artifacts.length < limit) {
      const skills = await searchLocalSkills(this.#skillRoots, query, limit - artifacts.length);
      artifacts.push(...skills.artifacts);
      partialErrors.push(...skills.partialErrors);
    }
    return { artifacts: artifacts.slice(0, limit), partialErrors, truncated };
  }
}

module.exports = {
  HivemindContractError,
  HivemindTransportError,
  HivemindOrganizationalMemory,
  InstalledHivemindClient,
  hivemindCapabilities,
  opaqueHivemindSessionKey,
};
