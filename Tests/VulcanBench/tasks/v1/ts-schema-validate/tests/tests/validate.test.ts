import { test } from "node:test";
import assert from "node:assert/strict";

import { validate } from "../src/validate.ts";
import type { Schema } from "../src/types.ts";

const userSchema: Schema = {
  name: { type: "string", required: true },
  age: { type: "number", required: true },
  active: { type: "boolean" },
  nickname: { type: "string" },
};

test("reports missing required field", () => {
  const errors = validate({ name: "Ada" }, userSchema);
  const fields = errors.map((e) => e.field);
  assert.ok(fields.includes("age"), "expected an error for missing 'age'");
  const ageErr = errors.find((e) => e.field === "age");
  assert.equal(ageErr?.kind, "missing");
});

test("reports wrong type for present field", () => {
  const errors = validate(
    { name: "Ada", age: "not a number" },
    userSchema,
  );
  const ageErr = errors.find((e) => e.field === "age");
  assert.ok(ageErr, "expected a type error for 'age'");
  assert.equal(ageErr?.kind, "type");
});

test("reports both missing and type errors together", () => {
  const errors = validate({ active: 1 }, userSchema);
  const byField = new Map(errors.map((e) => [e.field, e]));
  assert.equal(byField.get("name")?.kind, "missing");
  assert.equal(byField.get("age")?.kind, "missing");
  assert.equal(byField.get("active")?.kind, "type");
});

test("optional field of correct type is accepted", () => {
  const errors = validate(
    { name: "Ada", age: 30, nickname: "A" },
    userSchema,
  );
  assert.deepEqual(errors, []);
});

test("fully valid object yields no errors", () => {
  const errors = validate(
    { name: "Ada", age: 30, active: true },
    userSchema,
  );
  assert.deepEqual(errors, []);
});
