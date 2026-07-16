import test from 'node:test';
import assert from 'node:assert/strict';
import { run } from './src/m2lib8.ts';
test('run', () => assert.equal(run(2), 4));
test('types', () => assert.equal(typeof 1, 'number'));
