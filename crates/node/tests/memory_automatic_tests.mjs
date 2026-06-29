// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { describe, it } from 'node:test';

const require = createRequire(import.meta.url);
const relay = require('../index.js');
const { InMemoryAutomaticMemory } = require('../memory.js');

function request(user) {
  return {
    headers: {},
    content: {
      model: 'test-model',
      messages: [{ role: 'user', content: user }],
      preserved: { provider: true },
    },
  };
}

function response(assistant) {
  return {
    id: 'response-1',
    model: 'test-model',
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: assistant },
        finish_reason: 'stop',
      },
    ],
    usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
  };
}

async function flushSubscriberCallbacks() {
  relay.flushSubscribers();
  for (let index = 0; index < 10; index += 1) {
    await new Promise((resolve) => setImmediate(resolve));
  }
}

async function turn(sessionId, user, assistant, seen, { enabled = true } = {}) {
  const codec = new relay.OpenAIChatCodec();
  return relay.llmCallExecuteAsync(
    'node-memory-agent',
    request(user),
    async (providerRequest) => {
      seen.push(providerRequest);
      return response(assistant);
    },
    null,
    null,
    null,
    {
      memory: {
        enabled,
        namespace: {
          tenant_id: 'tenant-node',
          subject_id: 'subject-alex',
          session_id: sessionId,
          agent_id: 'assistant',
        },
      },
    },
    null,
    codec.decode.bind(codec),
    (payload) => codec.encode(payload.annotated, payload.original),
    codec.decodeResponse.bind(codec),
  );
}

describe('native reference automatic memory', () => {
  it('recalls across sessions and emits private evidence without memory tools', async () => {
    const events = [];
    relay.registerSubscriber('node-automatic-memory-events', (event) => events.push(event));
    const component = new InMemoryAutomaticMemory().install({ name: 'node-automatic-memory' });
    try {
      const firstSeen = [];
      await turn(
        'session-a',
        'SENTINEL_USER prefers solarized dark editor theme',
        'SENTINEL_ASSISTANT acknowledged the preference',
        firstSeen,
      );
      assert.doesNotMatch(JSON.stringify(firstSeen[0]), /<relay_memory/);

      const secondSeen = [];
      await turn(
        'session-b',
        'Which editor theme does SENTINEL_USER prefer?',
        'The preference is solarized dark.',
        secondSeen,
      );
      const injected = secondSeen[0].content.messages.at(-1).content;
      assert.match(injected, /<relay_memory version="0\.1">/);
      assert.match(injected, /solarized dark editor theme/);
      assert.deepEqual(secondSeen[0].content.preserved, { provider: true });

      await flushSubscriberCallbacks();
      const memoryEvents = events.filter((event) => event.category === 'memory');
      const names = new Set(memoryEvents.map((event) => event.name));
      assert.ok(names.has('memory.retrieval'));
      assert.ok(names.has('memory.injection'));
      assert.ok(names.has('memory.storage'));
      const evidence = JSON.stringify(memoryEvents.map((event) => event.data));
      assert.doesNotMatch(evidence, /SENTINEL_USER/);
      assert.doesNotMatch(evidence, /SENTINEL_ASSISTANT/);
      assert.doesNotMatch(evidence, /solarized dark editor theme/);
      assert.match(evidence, /content_hash/);
      assert.equal(component.activeTurns, 0);
    } finally {
      assert.equal(component.close(), true);
      assert.equal(component.close(), false);
      await flushSubscriberCallbacks();
      relay.deregisterSubscriber('node-automatic-memory-events');
    }
  });

  it('honors opt-out and scope cleanup while preserving the normal request', async () => {
    const parent = relay.pushScope('node-memory-scope', relay.ScopeType.Agent, null, null);
    const component = new InMemoryAutomaticMemory({
      namespace: {
        tenantId: 'tenant-node',
        subjectId: 'subject-alex',
        sessionId: 'fallback',
        agentId: 'assistant',
      },
    }).install({ name: 'node-scope-memory', scope: parent });
    try {
      const seen = [];
      await turn('session-opt-out', 'do not remember this sentinel', 'not stored', seen, { enabled: false });
      assert.deepEqual(seen, [request('do not remember this sentinel')]);
    } finally {
      relay.popScope(parent);
    }

    assert.equal(component.close(), false);
    assert.equal(component.activeTurns, 0);
  });

  it('rejects invalid native configuration', () => {
    assert.throws(() => new InMemoryAutomaticMemory({ maxCandidates: 1, maxItems: 2 }), /max_items/);
  });

  it('flushes background write-back and exposes status before deterministic shutdown', async () => {
    const events = [];
    relay.registerSubscriber('node-background-memory-events', (event) => events.push(event));
    const component = new InMemoryAutomaticMemory({
      writeDelivery: 'background',
      backgroundQueue: {
        capacity: 2,
        maxAttempts: 2,
        retryInitialDelayMillis: 1,
        retryMaxDelayMillis: 1,
      },
    }).install({ name: 'node-background-memory' });
    let shutDown = false;
    try {
      const firstSeen = [];
      await turn('background-a', 'BACKGROUND_USER prefers a nord editor theme', 'Preference acknowledged', firstSeen);
      assert.doesNotMatch(JSON.stringify(firstSeen[0]), /<relay_memory/);
      assert.equal(component.backgroundStatus.acceptedTotal, 1);
      assert.equal(await component.flush(), true);

      const secondSeen = [];
      await turn('background-b', 'Which editor theme does BACKGROUND_USER prefer?', 'Nord.', secondSeen);
      const injected = secondSeen[0].content.messages.at(-1).content;
      assert.match(injected, /<relay_memory version="0\.1">/);
      assert.match(injected, /nord editor theme/);
      assert.equal(await component.flush(), true);

      await flushSubscriberCallbacks();
      const storageEvents = events.filter((event) => event.name === 'memory.storage');
      const statuses = new Set(storageEvents.map((event) => event.data.status));
      assert.ok(statuses.has('queued'));
      assert.ok(statuses.has('running'));
      assert.ok(statuses.has('stored'));
      const job = component.backgroundJobStatus(storageEvents[0].data.job_id);
      assert.equal(job.state, 'succeeded');
      assert.equal(job.attempts, 1);
      const evidence = JSON.stringify(storageEvents.map((event) => event.data));
      assert.doesNotMatch(evidence, /BACKGROUND_USER/);
      assert.doesNotMatch(evidence, /nord editor theme/);

      await assert.rejects(component.flush(0), /timeoutMillis/);
      assert.equal(await component.shutdown(), true);
      shutDown = true;
      assert.equal(component.backgroundStatus.accepting, false);
    } finally {
      if (!shutDown) {
        component.close();
        await component.shutdown();
      }
      await flushSubscriberCallbacks();
      relay.deregisterSubscriber('node-background-memory-events');
    }
  });
});
