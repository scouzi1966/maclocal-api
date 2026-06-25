/**
 * Recursively merge `source` into `target`.
 *
 * NOTE: this implementation is buggy — see issue.md. It merges arrays
 * element-by-element, mutates its inputs, and copies dangerous keys such as
 * `__proto__`.
 */

function isObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null;
}

export function deepMerge(
  target: Record<string, unknown>,
  source: Record<string, unknown>,
): Record<string, unknown> {
  for (const key of Object.keys(source)) {
    const sv = (source as Record<string, unknown>)[key];
    const tv = (target as Record<string, unknown>)[key];
    if (isObject(sv) && isObject(tv)) {
      (target as Record<string, unknown>)[key] = deepMerge(
        tv as Record<string, unknown>,
        sv as Record<string, unknown>,
      );
    } else {
      (target as Record<string, unknown>)[key] = sv;
    }
  }
  return target;
}
