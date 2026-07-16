import { test } from "node:test";
import assert from "node:assert/strict";

import { Emitter } from "../src/emitter.ts";

// --- fail_to_pass: off and once are broken in the starting repo ---

test("once fires its listener only on the first emit", () => {
  const e = new Emitter();
  const calls: unknown[][] = [];
  e.once("ping", (...args) => calls.push(args));
  e.emit("ping", 1);
  e.emit("ping", 2);
  assert.deepEqual(calls, [[1]]);
});

test("off removes a listener", () => {
  const e = new Emitter();
  let count = 0;
  const fn = () => {
    count += 1;
  };
  e.on("x", fn);
  e.off("x", fn);
  e.emit("x");
  assert.equal(count, 0);
});

test("off only removes the targeted listener", () => {
  const e = new Emitter();
  const seen: string[] = [];
  const a = () => seen.push("a");
  const b = () => seen.push("b");
  e.on("x", a);
  e.on("x", b);
  e.off("x", a);
  e.emit("x");
  assert.deepEqual(seen, ["b"]);
});

test("once is unsubscribed after it fires", () => {
  const e = new Emitter();
  e.once("x", () => {});
  assert.equal(e.listenerCount("x"), 1);
  e.emit("x");
  assert.equal(e.listenerCount("x"), 0);
});

// --- pass_to_pass: on / emit / listenerCount already work (regression guard) ---

test("on + emit invokes the listener with args", () => {
  const e = new Emitter();
  const calls: unknown[][] = [];
  e.on("x", (...args) => calls.push(args));
  e.emit("x", 1, 2);
  assert.deepEqual(calls, [[1, 2]]);
});

test("emit calls listeners in subscription order", () => {
  const e = new Emitter();
  const seen: number[] = [];
  e.on("x", () => seen.push(1));
  e.on("x", () => seen.push(2));
  e.emit("x");
  assert.deepEqual(seen, [1, 2]);
});

test("listenerCount reflects subscriptions", () => {
  const e = new Emitter();
  e.on("x", () => {});
  e.on("x", () => {});
  assert.equal(e.listenerCount("x"), 2);
});
