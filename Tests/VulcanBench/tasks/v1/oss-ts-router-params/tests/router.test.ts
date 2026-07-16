import test from 'node:test';
import assert from 'node:assert/strict';
import { match } from './src/router.ts';
test('staticSegment', () => {
  assert.deepEqual(match('/users/:id', '/users/1'), { id: '1' });
});
test('decodeParam', () => {
  assert.deepEqual(match('/q/:s', '/q/a%20b'), { s: 'a b' });
});
