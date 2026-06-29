// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { access, appendFile, readFile, writeFile } from 'node:fs/promises';
import { createInterface } from 'node:readline';

const lines = createInterface({ input: process.stdin, crlfDelay: Infinity });

async function record(value) {
  if (process.env.HIVEMIND_FIXTURE_LOG) {
    await appendFile(process.env.HIVEMIND_FIXTURE_LOG, `${JSON.stringify(value)}\n`, 'utf8');
  }
}

function respond(value) {
  process.stdout.write(`${JSON.stringify(value)}\n`);
}

for await (const line of lines) {
  if (line.trim().length === 0) continue;
  if (process.env.HIVEMIND_FIXTURE_FAIL_ONCE) {
    try {
      await access(process.env.HIVEMIND_FIXTURE_FAIL_ONCE);
    } catch {
      await writeFile(process.env.HIVEMIND_FIXTURE_FAIL_ONCE, 'failed\n', 'utf8');
      process.exitCode = 7;
      break;
    }
  }
  if (process.env.HIVEMIND_FIXTURE_LARGE_OUTPUT === '1') {
    process.stdout.write('x'.repeat(1024 * 1024 + 1));
    continue;
  }
  if (process.env.HIVEMIND_FIXTURE_INVALID_JSON === '1') {
    process.stdout.write('not-json\n');
    continue;
  }
  const input = JSON.parse(line);
  if (input.jsonrpc === '2.0') {
    if (process.env.HIVEMIND_FIXTURE_HANG === '1') continue;
    if (input.method === 'initialize') {
      respond({
        jsonrpc: '2.0',
        id: input.id,
        result: {
          protocolVersion: '2025-06-18',
          capabilities: { tools: {} },
          serverInfo: { name: 'hivemind-fixture', version: '0.0.0' },
        },
      });
    } else if (input.method === 'tools/call') {
      respond({
        jsonrpc: '2.0',
        id: input.id,
        result: {
          content: [{ type: 'text', text: process.env.HIVEMIND_FIXTURE_SEARCH_TEXT ?? 'No matches.' }],
        },
      });
    }
    continue;
  }

  const recorded = { type: 'hook', input };
  if (input.transcript_path) {
    recorded.transcript = await readFile(input.transcript_path, 'utf8');
  }
  await record(recorded);
}
