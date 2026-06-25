// Schema definitions for the tiny object validator.

/** Primitive field types the validator understands. */
export type FieldType = "string" | "number" | "boolean";

/** Description of a single field in a schema. */
export interface FieldSpec {
  /** The expected primitive type of the field's value. */
  type: FieldType;
  /** Whether the field must be present on the object. Defaults to false. */
  required?: boolean;
}

/** A schema maps field names to their specs. */
export type Schema = Record<string, FieldSpec>;

/** A single problem found while validating an object against a schema. */
export interface ValidationError {
  /** The field the error is about. */
  field: string;
  /** Why this field failed: "missing" or "type". */
  kind: "missing" | "type";
  /** Human-readable explanation. */
  message: string;
}
