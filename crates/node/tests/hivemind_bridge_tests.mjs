// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { afterEach, describe, it } from 'node:test';

const require = createRequire(import.meta.url);
const { HivemindContractError, HivemindTransportError, InstalledHivemindClient } = require('../hivemind.js');
const fixture = new URL('./fixtures/hivemind_fixture.mjs', import.meta.url).pathname;
const temporaryDirectories = [];

async function temporaryDirectory() {
  const directory = await mkdtemp(path.join(tmpdir(), 'nemo-relay-hivemind-test-'));
  temporaryDirectories.push(directory);
  return directory;
}

afterEach(async () => {
  await Promise.all(temporaryDirectories.splice(0).map((directory) => rm(directory, { recursive: true, force: true })));
});

describe('installed Hivemind process bridge', () => {
  it('validates process bridge collection options', () => {
    assert.throws(
      () => new InstalledHivemindClient({ skillRoots: 'not-an-array' }),
      (error) => error instanceof HivemindContractError && /skillRoots/.test(error.message),
    );
    assert.throws(
      () => new InstalledHivemindClient({ environment: null }),
      (error) => error instanceof HivemindContractError && /environment/.test(error.message),
    );
    assert.throws(
      () => new InstalledHivemindClient({ environment: { HIVEMIND_TOKEN: 123 } }),
      (error) => error instanceof HivemindContractError && /values must be strings/.test(error.message),
    );
    assert.throws(
      () => new InstalledHivemindClient({ mcpServer: '' }),
      (error) => error instanceof HivemindContractError && /mcpServer/.test(error.message),
    );
    assert.throws(
      () => new InstalledHivemindClient({ environment: { TOKEN: 42 } }),
      (error) => error instanceof HivemindContractError && /strings/.test(error.message),
    );
  });

  it('maps Relay lifecycle events to installed capture hooks', async () => {
    const directory = await temporaryDirectory();
    const log = path.join(directory, 'hooks.jsonl');
    const client = new InstalledHivemindClient({
      captureHook: fixture,
      stopHook: fixture,
      mcpServer: fixture,
      environment: { HIVEMIND_FIXTURE_LOG: log },
    });

    await client.capture({
      sessionId: 'nmr1-11111111111111111111111111111111',
      cwd: '/workspace/demo',
      model: 'model-a',
      event: { category: 'llm', scope_category: 'start', name: 'answer', data: { prompt: 'hello' } },
    });
    await client.capture({
      sessionId: 'nmr1-11111111111111111111111111111111',
      cwd: '/workspace/demo',
      model: 'model-a',
      event: { uuid: 'tool-1', category: 'tool', scope_category: 'start', name: 'lookup', data: { query: 42 } },
    });
    await client.capture({
      sessionId: 'nmr1-11111111111111111111111111111111',
      cwd: '/workspace/demo',
      model: 'model-a',
      event: { id: 'tool-1', category: 'tool', scope_category: 'end', name: 'lookup', data: { answer: 42 } },
    });
    await client.finish({
      sessionId: 'nmr1-11111111111111111111111111111111',
      cwd: '/workspace/demo',
      model: 'model-a',
      assistantMessage: 'The answer is 42.',
    });

    const records = (await readFile(log, 'utf8'))
      .trim()
      .split('\n')
      .map((line) => JSON.parse(line));
    assert.equal(records.length, 3);
    assert.equal(records[0].input.hook_event_name, 'UserPromptSubmit');
    assert.match(records[0].input.prompt, /NeMo Relay ATOF llm\/start/);
    assert.equal(records[1].input.hook_event_name, 'PostToolUse');
    assert.equal(records[1].input.tool_name, 'lookup');
    assert.deepEqual(records[1].input.tool_input.relay_event.data, { query: 42 });
    assert.deepEqual(records[1].input.tool_response, { relay_event: { answer: 42 } });
    assert.equal(records[2].input.hook_event_name, 'Stop');
    const transcript = JSON.parse(records[2].transcript.trim());
    assert.equal(transcript.payload.content[0].text, 'The answer is 42.');
  });

  it('retains an unmatched tool start as explicitly incomplete at session finish', async () => {
    const directory = await temporaryDirectory();
    const log = path.join(directory, 'incomplete.jsonl');
    const client = new InstalledHivemindClient({
      captureHook: fixture,
      stopHook: fixture,
      mcpServer: fixture,
      environment: { HIVEMIND_FIXTURE_LOG: log },
    });
    await client.capture({
      sessionId: 'nmr1-33333333333333333333333333333333',
      cwd: '/workspace/demo',
      model: 'model-a',
      event: { uuid: 'tool-incomplete', category: 'tool', scope_category: 'start', name: 'long-task', data: {} },
    });
    await client.finish({
      sessionId: 'nmr1-33333333333333333333333333333333',
      cwd: '/workspace/demo',
      model: 'model-a',
      assistantMessage: 'Session stopped.',
    });
    const records = (await readFile(log, 'utf8'))
      .trim()
      .split('\n')
      .map((line) => JSON.parse(line));
    assert.equal(records[0].input.hook_event_name, 'PostToolUse');
    assert.deepEqual(records[0].input.tool_response, { relay_event: null, incomplete: true });
    assert.equal(records[1].input.hook_event_name, 'Stop');
  });

  it('retains paired tool input when a hook failure is retried', async () => {
    const directory = await temporaryDirectory();
    const log = path.join(directory, 'retry.jsonl');
    const failOnce = path.join(directory, 'fail-once');
    const client = new InstalledHivemindClient({
      captureHook: fixture,
      stopHook: fixture,
      mcpServer: fixture,
      environment: { HIVEMIND_FIXTURE_LOG: log, HIVEMIND_FIXTURE_FAIL_ONCE: failOnce },
    });
    const start = {
      uuid: 'tool-retry',
      category: 'tool',
      scope_category: 'start',
      name: 'retry-tool',
      data: { original: true },
    };
    const end = {
      uuid: 'tool-retry',
      category: 'tool',
      scope_category: 'end',
      name: 'retry-tool',
      data: { completed: true },
    };
    await client.capture({ sessionId: 'retry-session', cwd: '/workspace/demo', model: 'model-a', event: start });
    await assert.rejects(
      client.capture({ sessionId: 'retry-session', cwd: '/workspace/demo', model: 'model-a', event: end }),
      /exit=7/,
    );
    await client.capture({ sessionId: 'retry-session', cwd: '/workspace/demo', model: 'model-a', event: end });
    const record = JSON.parse((await readFile(log, 'utf8')).trim());
    assert.deepEqual(record.input.tool_input.relay_event.data, { original: true });
  });

  it('parses ordered MCP traces/summaries and local skill provenance', async () => {
    const directory = await temporaryDirectory();
    const skillsRoot = path.join(directory, 'skills');
    const skillDirectory = path.join(skillsRoot, 'safe-migration');
    await mkdir(skillDirectory, { recursive: true });
    await writeFile(
      path.join(skillDirectory, 'SKILL.md'),
      [
        '---',
        'name: safe-migration',
        'source_sessions:',
        '  - nmr1-22222222222222222222222222222222',
        '---',
        '',
        '# Safe migration',
        '',
        'Use an expand-and-contract migration.',
      ].join('\n'),
      'utf8',
    );
    const searchText = [
      '[/summaries/alice/nmr1-11111111111111111111111111111111.md]\nSafe migration summary.',
      '[/sessions/alice/org_ws_nmr1-11111111111111111111111111111111.jsonl]\nRaw trace excerpt.',
      'Results truncated; additional matches are available.',
    ].join('\n\n---\n\n');
    const client = new InstalledHivemindClient({
      captureHook: fixture,
      stopHook: fixture,
      mcpServer: fixture,
      skillRoots: [skillsRoot],
      environment: { HIVEMIND_FIXTURE_SEARCH_TEXT: searchText },
    });

    const result = await client.search({
      query: 'safe migration',
      kinds: ['summary', 'trace', 'skill'],
      limit: 10,
    });
    assert.deepEqual(
      result.artifacts.map((artifact) => artifact.kind),
      ['summary', 'trace', 'skill'],
    );
    assert.equal(result.truncated, true);
    assert.deepEqual(result.artifacts[2].sourceSessionIds, ['nmr1-22222222222222222222222222222222']);
    assert.match(result.artifacts[2].preview, /expand-and-contract/);
  });

  it('filters MCP result kinds without reclassifying unknown paths', async () => {
    const text = [
      '[/summaries/alice/one.md]\nSummary.',
      '[/sessions/alice/two.jsonl]\nTrace.',
      '[/other/not-a-supported-artifact]\nOther.',
    ].join('\n\n---\n\n');
    const client = new InstalledHivemindClient({
      captureHook: fixture,
      stopHook: fixture,
      mcpServer: fixture,
      environment: { HIVEMIND_FIXTURE_SEARCH_TEXT: text },
    });
    const result = await client.search({ query: 'one', kinds: ['summary'], limit: 5 });
    assert.deepEqual(
      result.artifacts.map((artifact) => artifact.reference),
      ['/summaries/alice/one.md'],
    );
  });

  it('treats an ordinary no-match response as an empty result, not a provider error', async () => {
    const client = new InstalledHivemindClient({
      captureHook: fixture,
      stopHook: fixture,
      mcpServer: fixture,
      environment: { HIVEMIND_FIXTURE_SEARCH_TEXT: 'No matches for "unseen".' },
    });
    const result = await client.search({ query: 'unseen', kinds: ['summary'], limit: 5 });
    assert.deepEqual(result, { artifacts: [], partialErrors: [], truncated: false });
  });

  it('reports missing hook files and bounded MCP timeouts', async () => {
    const missing = new InstalledHivemindClient({
      captureHook: '/does/not/exist/capture.js',
      stopHook: fixture,
      mcpServer: fixture,
    });
    await assert.rejects(
      missing.capture({ sessionId: 's', cwd: '/', model: 'm', event: { category: 'llm' } }),
      (error) => error instanceof HivemindTransportError && /does not exist/.test(error.message),
    );

    const hanging = new InstalledHivemindClient({
      captureHook: fixture,
      stopHook: fixture,
      mcpServer: fixture,
      timeoutMillis: 30,
      environment: { HIVEMIND_FIXTURE_HANG: '1' },
    });
    await assert.rejects(
      hanging.search({ query: 'anything', kinds: ['summary'], limit: 1 }),
      (error) => error instanceof HivemindTransportError && /timed out/.test(error.message),
    );
  });

  it('rejects malformed MCP protocol output', async () => {
    const client = new InstalledHivemindClient({
      captureHook: fixture,
      stopHook: fixture,
      mcpServer: fixture,
      environment: { HIVEMIND_FIXTURE_INVALID_JSON: '1' },
    });
    await assert.rejects(
      client.search({ query: 'anything', kinds: ['summary'], limit: 1 }),
      (error) => error instanceof HivemindTransportError && /invalid JSON/.test(error.message),
    );
  });

  it('bounds unterminated MCP output from an incompatible process', async () => {
    const client = new InstalledHivemindClient({
      captureHook: fixture,
      stopHook: fixture,
      mcpServer: fixture,
      environment: { HIVEMIND_FIXTURE_LARGE_OUTPUT: '1' },
    });
    await assert.rejects(
      client.search({ query: 'anything', kinds: ['summary'], limit: 1 }),
      (error) => error instanceof HivemindTransportError && /output exceeded limit/.test(error.message),
    );
  });
});
