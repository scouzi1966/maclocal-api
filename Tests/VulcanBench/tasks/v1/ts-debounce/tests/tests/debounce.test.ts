import { test, mock } from "node:test";
import assert from "node:assert/strict";

import { debounce } from "../src/debounce.ts";

function tracker() {
  const calls: unknown[][] = [];
  const fn = (...args: unknown[]): void => {
    calls.push(args);
  };
  return { fn, calls };
}

// --- fail_to_pass: real debouncing behavior (the skeleton never invokes fn) ---

test("invokes once on the trailing edge with the latest args", () => {
  mock.timers.enable({ apis: ["setTimeout"] });
  try {
    const { fn, calls } = tracker();
    const d = debounce(fn, 100);
    d(1);
    mock.timers.tick(50);
    d(2);
    mock.timers.tick(100);
    assert.equal(calls.length, 1);
    assert.deepEqual(calls[0], [2]);
  } finally {
    mock.timers.reset();
  }
});

test("does not invoke before the wait elapses", () => {
  mock.timers.enable({ apis: ["setTimeout"] });
  try {
    const { fn, calls } = tracker();
    const d = debounce(fn, 100);
    d("x");
    mock.timers.tick(99);
    assert.equal(calls.length, 0);
    mock.timers.tick(1);
    assert.deepEqual(calls, [["x"]]);
  } finally {
    mock.timers.reset();
  }
});

test("collapses a burst of calls into one", () => {
  mock.timers.enable({ apis: ["setTimeout"] });
  try {
    const { fn, calls } = tracker();
    const d = debounce(fn, 100);
    d(1);
    d(2);
    d(3);
    mock.timers.tick(100);
    assert.equal(calls.length, 1);
    assert.deepEqual(calls[0], [3]);
  } finally {
    mock.timers.reset();
  }
});

test("leading invokes immediately and not on the trailing edge", () => {
  mock.timers.enable({ apis: ["setTimeout"] });
  try {
    const { fn, calls } = tracker();
    const d = debounce(fn, 100, { leading: true, trailing: false });
    d(1);
    assert.deepEqual(calls, [[1]], "leading call should fire immediately");
    mock.timers.tick(100);
    assert.equal(calls.length, 1, "no trailing call expected");
  } finally {
    mock.timers.reset();
  }
});

test("cancel discards a pending call but the debouncer still works after", () => {
  mock.timers.enable({ apis: ["setTimeout"] });
  try {
    const { fn, calls } = tracker();
    const d = debounce(fn, 100);
    d(1);
    d.cancel();
    mock.timers.tick(100);
    assert.equal(calls.length, 0, "cancel must discard the pending call");
    d(2);
    mock.timers.tick(100);
    assert.deepEqual(calls, [[2]]);
  } finally {
    mock.timers.reset();
  }
});

// --- pass_to_pass: shape of the returned debouncer (holds before and after) ---

test("returns a callable", () => {
  const d = debounce(() => {}, 100);
  assert.equal(typeof d, "function");
});

test("exposes a cancel method", () => {
  const d = debounce(() => {}, 100);
  assert.equal(typeof d.cancel, "function");
});
