import { test } from "node:test";
import assert from "node:assert/strict";

import { deepMerge } from "../src/merge.ts";

// --- fail_to_pass: behaviors the buggy version gets wrong ---

test("arrays are replaced, not merged element-by-element", () => {
  const result = deepMerge({ list: [1, 2, 3] }, { list: [9] });
  assert.deepEqual(result, { list: [9] });
});

test("does not mutate the target object", () => {
  const target = { a: { x: 1 } };
  deepMerge(target, { a: { y: 2 } });
  assert.deepEqual(target, { a: { x: 1 } }, "target must be left unchanged");
});

test("a __proto__ payload does not pollute Object.prototype", () => {
  const malicious = JSON.parse('{"__proto__": {"polluted": "yes"}}');
  deepMerge({}, malicious);
  const probe = {} as Record<string, unknown>;
  assert.equal(probe.polluted, undefined, "Object.prototype must not be polluted");
});

// --- pass_to_pass: behaviors that already work (regression guard) ---

test("merges nested plain objects", () => {
  const result = deepMerge({ a: { x: 1 } }, { a: { z: 2 } });
  assert.deepEqual(result, { a: { x: 1, z: 2 } });
});

test("source primitives overwrite target", () => {
  assert.deepEqual(deepMerge({ a: 1 }, { a: 2 }), { a: 2 });
});

test("new keys from source are added", () => {
  assert.deepEqual(deepMerge({ a: 1 }, { b: 2 }), { a: 1, b: 2 });
});
