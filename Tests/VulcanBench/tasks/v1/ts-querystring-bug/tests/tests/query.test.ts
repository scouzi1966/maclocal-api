import test from "node:test";
import assert from "node:assert/strict";

import { parseQuery } from "../src/parse.ts";

test("single key=value", () => {
  assert.deepEqual(parseQuery("a=1"), { a: "1" });
});

test("strips leading question mark", () => {
  assert.deepEqual(parseQuery("?a=1&b=2"), { a: "1", b: "2" });
});

test("decodes spaces", () => {
  assert.deepEqual(parseQuery("q=hello%20world"), { q: "hello world" });
});

test("decodes plus as space", () => {
  assert.deepEqual(parseQuery("q=a+b"), { q: "a b" });
});

test("decodes keys", () => {
  assert.deepEqual(parseQuery("first%20name=Ada"), { "first name": "Ada" });
});

test("collects repeated keys into array", () => {
  assert.deepEqual(parseQuery("tag=a&tag=b&tag=c"), { tag: ["a", "b", "c"] });
});

test("repeated keys keep decoded values", () => {
  assert.deepEqual(parseQuery("x=a%20b&x=c%2Fd"), { x: ["a b", "c/d"] });
});
