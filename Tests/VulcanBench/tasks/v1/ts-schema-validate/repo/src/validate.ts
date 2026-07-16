import type { Schema, ValidationError } from "./types.ts";

/**
 * Validate a plain object against a schema.
 *
 * Returns a list of {@link ValidationError}; an empty list means the object is
 * valid. See issue.md for the exact semantics this should implement.
 *
 * NOTE: this is currently a stub that accepts everything.
 */
export function validate(
  obj: Record<string, unknown>,
  schema: Schema,
): ValidationError[] {
  // TODO: implement required-field and field-type checks.
  return [];
}
