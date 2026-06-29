// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { describe, it } from 'node:test';

const require = createRequire(import.meta.url);
const {
  MemoryContractError,
  defaultMemoryCapabilities,
  fromMemoryWire,
  toMemoryWire,
  validateMemoryNamespace,
} = require('../memory.js');
const fixture = JSON.parse(
  readFileSync(new URL('../../types/tests/fixtures/memory_contract_v0_1.json', import.meta.url), 'utf8'),
);

describe('memory contract', () => {
  it('round trips the shared fixture without changing wire data', () => {
    for (const value of Object.values(fixture)) {
      assert.deepEqual(toMemoryWire(fromMemoryWire(value)), value);
    }

    const request = fromMemoryWire(fixture.search_request);
    assert.equal(request.namespace.tenantId, 'tenant-demo');
    assert.equal(request.namespace.subjectId, 'subject-alex');
    assert.equal(request.context.operationId, 'search-1');
    assert.equal(request.filter.eventAfter, '2026-01-01T00:00:00Z');
  });

  it('preserves opaque JSON and provider-native keys', () => {
    const publicValue = {
      providerMetadata: {
        native_collection: { revision_id: 7 },
      },
      metadata: { user_key: { nested_key: true } },
      details: { provider_error_code: 'archived' },
      parameters: { compact_before: '2026-01-01T00:00:00Z' },
      content: { kind: 'json', value: { favorite_theme: 'solarized_dark' } },
    };

    const wireValue = toMemoryWire(publicValue);
    assert.deepEqual(wireValue.provider_metadata, publicValue.providerMetadata);
    assert.deepEqual(wireValue.metadata, publicValue.metadata);
    assert.deepEqual(wireValue.details, publicValue.details);
    assert.deepEqual(wireValue.parameters, publicValue.parameters);
    assert.deepEqual(wireValue.content.value, publicValue.content.value);
    assert.deepEqual(fromMemoryWire(wireValue), publicValue);
    assert.notEqual(wireValue.provider_metadata, publicValue.providerMetadata);
  });

  it('validates namespace identifiers and scope requirements', () => {
    const namespace = {
      tenantId: 'tenant-demo',
      subjectId: 'subject-alex',
      sessionId: 'session-b',
      agentId: 'assistant',
    };

    assert.equal(validateMemoryNamespace(namespace, 'agent'), namespace);
    assert.equal(validateMemoryNamespace(namespace, 'session'), namespace);
    assert.throws(() => validateMemoryNamespace({ ...namespace, tenantId: '  ' }), MemoryContractError);
    assert.throws(
      () => validateMemoryNamespace({ tenantId: 'tenant-demo', subjectId: 'subject-alex' }, 'agent'),
      /agent scope requires agentId/,
    );
    assert.throws(
      () => validateMemoryNamespace({ tenantId: 'tenant-demo', subjectId: 'subject-alex' }, 'session'),
      /session scope requires sessionId/,
    );
    assert.throws(() => validateMemoryNamespace(namespace, 'global'), /unsupported memory search scope/);
  });

  it('returns a fresh minimal capability set in public and wire naming', () => {
    const first = defaultMemoryCapabilities();
    const second = defaultMemoryCapabilities();
    assert.deepEqual(first, {
      update: false,
      delete: false,
      batchStore: false,
      maintenance: false,
      feedback: false,
      health: false,
    });
    assert.deepEqual(toMemoryWire(first), fixture.capabilities);
    assert.notEqual(first, second);
  });

  it('supports a minimal asynchronous provider implementation', async () => {
    const provider = {
      name: 'fixture',
      capabilities: defaultMemoryCapabilities(),
      async search() {
        return fromMemoryWire(fixture.search_result);
      },
      async store() {
        return fromMemoryWire(fixture.store_result);
      },
    };

    assert.equal((await provider.search()).matches[0].record.provider, 'in_memory');
    assert.equal((await provider.store()).disposition, 'existing');
  });
});
