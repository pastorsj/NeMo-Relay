// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { describe, it } from 'node:test';

const require = createRequire(import.meta.url);
const {
  HivemindContractError,
  HivemindOrganizationalMemory,
  HivemindTransportError,
  hivemindCapabilities,
  opaqueHivemindSessionKey,
} = require('../hivemind.js');
const relay = require('../index.js');

const namespace = {
  tenantId: 'tenant-private',
  subjectId: 'subject-alex',
  sessionId: 'session-a',
  agentId: 'agent-a',
};

const consent = {
  consent: 'explicit',
  captureContent: true,
};

class FakeClient {
  captures = [];
  finishes = [];
  searches = [];
  artifacts = [];

  async capture(input) {
    this.captures.push(input);
  }

  async finish(input) {
    this.finishes.push(input);
  }

  async search(input) {
    this.searches.push(input);
    return { artifacts: this.artifacts };
  }
}

describe('Hivemind organizational-memory contract', () => {
  it('declares a truthful partial capability profile without generic CRUD', () => {
    assert.deepEqual(hivemindCapabilities(), {
      traceCapture: true,
      search: true,
      summaries: true,
      skills: true,
      store: false,
      update: false,
      delete: false,
      vectorCrud: false,
    });
    const first = hivemindCapabilities();
    first.store = true;
    assert.equal(hivemindCapabilities().store, false);
  });

  it('requires explicit consent and complete identity before capture exists', () => {
    const client = new FakeClient();
    assert.throws(
      () =>
        new HivemindOrganizationalMemory({
          client,
          namespace,
          capturePolicy: { consent: 'implicit', captureContent: true },
        }),
      (error) => error instanceof HivemindContractError && /explicit/.test(error.message),
    );
    assert.throws(
      () =>
        new HivemindOrganizationalMemory({
          client,
          namespace: { ...namespace, subjectId: '' },
          capturePolicy: consent,
        }),
      /subjectId/,
    );
    assert.throws(
      () =>
        new HivemindOrganizationalMemory({
          client,
          namespace,
          capturePolicy: { ...consent, categories: [] },
        }),
      /non-empty array/,
    );
  });

  it('derives a stable opaque key without exposing Relay identifiers', () => {
    const first = opaqueHivemindSessionKey(namespace);
    const second = opaqueHivemindSessionKey({ ...namespace });
    assert.equal(first, second);
    assert.match(first, /^nmr1-[a-f0-9]{32}$/);
    for (const value of Object.values(namespace)) {
      assert.doesNotMatch(first, new RegExp(value));
    }
    assert.notEqual(first, opaqueHivemindSessionKey({ ...namespace, sessionId: 'session-b' }));
  });

  it('filters, redacts, bounds, and serializes accepted Relay events', async () => {
    const client = new FakeClient();
    const profile = new HivemindOrganizationalMemory({
      client,
      namespace,
      capturePolicy: {
        ...consent,
        categories: ['llm'],
        redact(event) {
          return { ...event, data: { safe: true } };
        },
      },
      cwd: '/workspace/demo',
      model: 'model-a',
      maxEventBytes: 1024,
    });

    assert.equal(profile.capture({ category: 'tool', name: 'ignored', data: { secret: true } }), false);
    assert.equal(profile.capture({ category: 'llm', name: 'answer', data: { secret: true } }), true);
    await profile.flush();
    assert.equal(client.captures.length, 1);
    assert.deepEqual(client.captures[0].event.data, { safe: true });
    assert.equal(client.captures[0].cwd, '/workspace/demo');
    assert.equal(client.captures[0].model, 'model-a');
    assert.equal(client.captures[0].sessionId, profile.sessionKey);

    const largeProfile = new HivemindOrganizationalMemory({
      client,
      namespace,
      capturePolicy: consent,
      maxEventBytes: 1024,
    });
    largeProfile.capture({ category: 'llm', data: { content: 'x'.repeat(2048) } });
    await largeProfile.flush();
    assert.deepEqual(client.captures.at(-1).event.data, {
      omitted: true,
      reason: 'event_size_limit',
      original_bytes: 2088,
    });

    const invalidRedaction = new HivemindOrganizationalMemory({
      client,
      namespace,
      capturePolicy: { ...consent, redact: () => 'not-an-event' },
    });
    assert.throws(() => invalidRedaction.capture({ category: 'llm' }), /event object or null/);
  });

  it('captures asynchronously, finishes explicitly, and has deterministic close semantics', async () => {
    const client = new FakeClient();
    const profile = new HivemindOrganizationalMemory({ client, namespace, capturePolicy: consent });
    profile.capture({ category: 'memory', name: 'memory.retrieval', data: { memory_ids: ['m1'] } });
    await profile.finish({ assistantMessage: 'Used the retrieved design pattern.' });
    assert.equal(client.captures.length, 1);
    assert.equal(client.finishes.length, 1);
    assert.equal(client.finishes[0].assistantMessage, 'Used the retrieved design pattern.');
    assert.deepEqual(profile.status, {
      accepting: true,
      pending: 0,
      capacity: 1024,
      accepted: 2,
      succeeded: 2,
      failed: 0,
      rejected: 0,
      lastError: null,
    });
    assert.equal(profile.close(), true);
    assert.equal(profile.close(), false);
    await profile.shutdown();
    assert.throws(() => profile.capture({ category: 'llm' }), /closed/);
  });

  it('captures ordinary Relay-managed lifecycle events when installed', async () => {
    const client = new FakeClient();
    const profile = new HivemindOrganizationalMemory({
      client,
      namespace,
      capturePolicy: { ...consent, categories: ['llm'] },
    }).install({ name: `hivemind-contract-${Date.now()}` });
    try {
      const handle = relay.llmCall('hivemind-managed-call', {
        headers: {},
        content: { prompt: 'ordinary agent request' },
      });
      relay.llmCallEnd(handle, { response: 'ordinary agent response' });
      await profile.shutdown();
      assert.deepEqual(
        client.captures.map((capture) => [capture.event.category, capture.event.scope_category]),
        [
          ['llm', 'start'],
          ['llm', 'end'],
        ],
      );
    } finally {
      assert.equal(profile.close(), false);
      await profile.shutdown();
    }
  });

  it('supports scope-local capture so concurrent agent identities do not mix', async () => {
    const client = new FakeClient();
    const scope = relay.pushScope('hivemind-agent-scope', relay.ScopeType.Agent, null, null);
    const profile = new HivemindOrganizationalMemory({
      client,
      namespace,
      capturePolicy: { ...consent, categories: ['llm'] },
    }).install({ name: `hivemind-scope-${Date.now()}`, scope });
    try {
      const inside = relay.llmCall('inside-agent-scope', { headers: {}, content: {} });
      relay.llmCallEnd(inside, {});
      relay.popScope(scope);
      const outside = relay.llmCall('outside-agent-scope', { headers: {}, content: {} });
      relay.llmCallEnd(outside, {});
      await profile.shutdown();
      assert.deepEqual(
        client.captures.map((capture) => capture.event.name),
        ['inside-agent-scope', 'inside-agent-scope'],
      );
    } finally {
      await profile.shutdown();
    }
  });

  it('bounds pending capture work and reports backpressure', async () => {
    const client = new FakeClient();
    let release;
    client.capture = () =>
      new Promise((resolve) => {
        release = resolve;
      });
    const profile = new HivemindOrganizationalMemory({
      client,
      namespace,
      capturePolicy: consent,
      queueCapacity: 1,
    });
    profile.capture({ category: 'llm', name: 'first' });
    assert.throws(
      () => profile.capture({ category: 'llm', name: 'second' }),
      (error) => error instanceof HivemindTransportError && /queue is full/.test(error.message),
    );
    await new Promise((resolve) => setImmediate(resolve));
    release();
    await assert.rejects(profile.flush(), /queue is full/);
    assert.equal(profile.status.accepted, 1);
    assert.equal(profile.status.rejected, 1);
    assert.equal(profile.status.pending, 0);
  });

  it('bounds final assistant content and search input before process transport', async () => {
    const client = new FakeClient();
    const profile = new HivemindOrganizationalMemory({
      client,
      namespace,
      capturePolicy: consent,
      maxTextBytes: 1024,
    });
    await assert.rejects(profile.finish({ assistantMessage: 'x'.repeat(1025) }), /maxTextBytes/);
    await assert.rejects(profile.search({ query: 'x'.repeat(1025) }), /maxTextBytes/);
    assert.equal(client.finishes.length, 0);
    assert.equal(client.searches.length, 0);
  });

  it('surfaces asynchronous transport failures through status and flush', async () => {
    const client = new FakeClient();
    client.capture = async () => {
      throw new Error('backend unavailable');
    };
    const profile = new HivemindOrganizationalMemory({ client, namespace, capturePolicy: consent });
    profile.capture({ category: 'llm', name: 'failed' });
    await assert.rejects(profile.flush(), (error) => {
      assert.ok(error instanceof HivemindTransportError);
      assert.match(error.message, /backend unavailable/);
      return true;
    });
    assert.equal(profile.status.failed, 1);
    assert.equal(profile.status.lastError.kind, 'capture');
  });

  it('normalizes ordered artifacts with derivation and source-session provenance', async () => {
    const client = new FakeClient();
    const sourceSession = opaqueHivemindSessionKey(namespace);
    client.artifacts = [
      {
        kind: 'summary',
        reference: `/summaries/alice/${sourceSession}.md`,
        preview: 'Learned how to rotate credentials.',
        providerMetadata: { scoreSemantics: 'vendor-confidence' },
      },
      {
        kind: 'skill',
        reference: '/workspace/.agents/skills/rotate/SKILL.md',
        preview: 'Rotate safely.',
        sourceSessionIds: [sourceSession],
      },
      {
        kind: 'trace',
        reference: '/sessions/alice/unattributed.jsonl',
      },
    ];
    const profile = new HivemindOrganizationalMemory({ client, namespace, capturePolicy: consent });
    const result = await profile.search({ query: 'rotate credentials' });
    assert.deepEqual(
      result.artifacts.map((artifact) => [artifact.kind, artifact.rank, artifact.provenance.derivation]),
      [
        ['summary', 1, 'session_summary'],
        ['skill', 2, 'trace_skillification'],
        ['trace', 3, 'captured_trace'],
      ],
    );
    assert.deepEqual(result.artifacts[0].provenance.sourceSessionIds, [sourceSession]);
    assert.deepEqual(result.artifacts[1].provenance.sourceSessionIds, [sourceSession]);
    assert.equal(result.artifacts[2].provenance.provenanceUnavailable, true);
    assert.equal(result.artifacts[0].providerMetadata.scoreSemantics, 'rank_only');
    assert.equal('score' in result.artifacts[0], false);
    assert.deepEqual(client.searches[0].kinds, ['trace', 'summary', 'skill']);
  });
});
