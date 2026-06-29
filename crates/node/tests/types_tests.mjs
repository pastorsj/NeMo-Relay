// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const lib = require('../index.js');

const { ScopeType, LlmRequest, ScopeStack } = lib;

// ===========================================================================
// Type constants
// ===========================================================================

describe('Type constants', () => {
  it('generates exported recursive JSON declarations for package entry points', () => {
    const declarations = readFileSync(new URL('../index.d.ts', import.meta.url), 'utf8');
    assert.match(declarations, /export type Json = JsonPrimitive \| JsonObject \| JsonArray/);
    assert.equal((declarations.match(/\/\* nemo-relay-json-types \*\//g) ?? []).length, 1);
  });

  it('exports canonical non-Js binding names', () => {
    assert.equal(typeof lib.ScopeStack, 'function');
    assert.equal(typeof lib.ScopeHandle, 'function');
    assert.equal(typeof lib.ToolHandle, 'function');
    assert.equal(typeof lib.LlmRequest, 'function');
    assert.equal(typeof lib.OpenAIChatCodec, 'function');
    assert.equal(typeof lib.OpenAIResponsesCodec, 'function');
    assert.equal(typeof lib.AnthropicMessagesCodec, 'function');
  });

  it('scope type enum values', () => {
    assert.equal(ScopeType.Agent, 0);
    assert.equal(ScopeType.Function, 1);
    assert.equal(ScopeType.Tool, 2);
    assert.equal(ScopeType.Llm, 3);
    assert.equal(ScopeType.Retriever, 4);
    assert.equal(ScopeType.Embedder, 5);
    assert.equal(ScopeType.Reranker, 6);
    assert.equal(ScopeType.Guardrail, 7);
    assert.equal(ScopeType.Evaluator, 8);
    assert.equal(ScopeType.Custom, 9);
    assert.equal(ScopeType.Unknown, 10);
  });
});

// ===========================================================================
// LlmRequest
// ===========================================================================

describe('LlmRequest', () => {
  it('construction and getters', () => {
    const req = new LlmRequest(
      {
        'Content-Type': 'application/json',
      },
      {
        model: 'gpt-4',
      },
    );
    assert.deepEqual(req.headers, {
      'Content-Type': 'application/json',
    });
    assert.deepEqual(req.content, {
      model: 'gpt-4',
    });
  });

  it('coerces non-object headers to an empty object', () => {
    const req = new LlmRequest(null, {
      model: 'gpt-4',
    });
    assert.deepEqual(req.headers, {});
    assert.deepEqual(req.content, {
      model: 'gpt-4',
    });
  });
});

describe('ScopeStack', () => {
  it('constructs a scope stack instance', () => {
    const stack = new ScopeStack();
    assert.ok(stack instanceof ScopeStack);
  });
});
