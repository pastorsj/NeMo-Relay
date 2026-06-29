// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { describe, it } from 'node:test';

const require = createRequire(import.meta.url);
const { HivemindOrganizationalMemory } = require('../hivemind.js');

class OrganizationalFake {
  traces = new Map();
  artifacts = [];

  async capture(input) {
    const trace = this.traces.get(input.sessionId) ?? [];
    trace.push(input.event);
    this.traces.set(input.sessionId, trace);
  }

  async finish(input) {
    const trace = this.traces.get(input.sessionId) ?? [];
    const content = JSON.stringify(trace);
    if (content.includes('expand-and-contract')) {
      this.artifacts.push({
        kind: 'summary',
        reference: `/summaries/team/${input.sessionId}.md`,
        preview: 'Use expand-and-contract migrations to preserve compatibility during a schema rollout.',
        sourceSessionIds: [input.sessionId],
      });
      this.artifacts.push({
        kind: 'skill',
        reference: `/team/skills/safe-schema-rollout-${input.sessionId}/SKILL.md`,
        preview: 'Apply additive schema changes before removing old fields.',
        sourceSessionIds: [input.sessionId],
      });
    }
  }

  async search({ query, kinds, limit }) {
    const terms = query.toLocaleLowerCase().split(/\s+/);
    const artifacts = this.artifacts.filter(
      (artifact) =>
        kinds.includes(artifact.kind) && terms.some((term) => artifact.preview.toLocaleLowerCase().includes(term)),
    );
    return { artifacts: artifacts.slice(0, limit) };
  }
}

function profile(client, sessionId, agentId) {
  return new HivemindOrganizationalMemory({
    client,
    namespace: {
      tenantId: 'demo-tenant',
      subjectId: 'engineering-team',
      sessionId,
      agentId,
    },
    capturePolicy: {
      consent: 'explicit',
      captureContent: true,
      categories: ['llm'],
    },
    cwd: '/workspace/service',
    model: 'deterministic-demo',
  });
}

describe('two-agent Hivemind organizational learning', () => {
  it('makes an artifact derived from agent A discoverable to agent B', async () => {
    const hivemind = new OrganizationalFake();
    const agentA = profile(hivemind, 'session-a', 'migration-agent');
    const agentB = profile(hivemind, 'session-b', 'review-agent');

    agentA.capture({
      kind: 'scope',
      category: 'llm',
      scope_category: 'start',
      name: 'plan-migration',
      data: {
        prompt: 'Plan an expand-and-contract database migration before removing the legacy column.',
      },
    });
    await agentA.finish({
      assistantMessage: 'First add the replacement column, backfill it, then remove the legacy column.',
    });

    const discovered = await agentB.search({
      query: 'safe schema migration',
      kinds: ['summary', 'skill'],
      limit: 5,
    });

    assert.deepEqual(
      discovered.artifacts.map((artifact) => artifact.kind),
      ['summary', 'skill'],
    );
    for (const artifact of discovered.artifacts) {
      assert.deepEqual(artifact.provenance.sourceSessionIds, [agentA.sessionKey]);
      assert.notEqual(artifact.provenance.sourceSessionIds[0], agentB.sessionKey);
      assert.equal(artifact.provenance.provenanceUnavailable, false);
      assert.equal(artifact.providerMetadata.scoreSemantics, 'rank_only');
    }
    assert.equal(agentB.capabilities.store, false);
    assert.equal(agentB.capabilities.delete, false);
    assert.equal(agentB.capabilities.vectorCrud, false);
  });
});
