import { decodeComponent } from "./decode.ts";

// Parse a URL query string into a map of keys to values.
//
// A leading "?" is ignored. Pairs are separated by "&"; within a pair the key
// and value are separated by the first "=". A key with no "=" maps to "".
//
// When the same key appears more than once, all of its values should be
// collected into an array (in the order they appear). Keys and values should be
// URL-decoded.
export function parseQuery(qs: string): Record<string, string | string[]> {
  const result: Record<string, string | string[]> = {};
  if (qs.startsWith("?")) {
    qs = qs.slice(1);
  }
  if (qs === "") {
    return result;
  }

  for (const pair of qs.split("&")) {
    if (pair === "") {
      continue;
    }
    const eq = pair.indexOf("=");
    let rawKey: string;
    let rawValue: string;
    if (eq === -1) {
      rawKey = pair;
      rawValue = "";
    } else {
      rawKey = pair.slice(0, eq);
      rawValue = pair.slice(eq + 1);
    }

    const key = decodeComponent(rawKey);
    // BUG: the raw value is stored without decoding, so "%20" and "+" survive
    // literally instead of becoming spaces.
    const value = rawValue;
    // BUG: a repeated key overwrites the earlier value instead of collecting
    // all values into an array.
    result[key] = value;
  }

  return result;
}
