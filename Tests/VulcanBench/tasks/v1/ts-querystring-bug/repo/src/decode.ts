// Helpers for turning raw query-string components into their decoded form.

// In an x-www-form-urlencoded query string, spaces are encoded as "+" and all
// other reserved characters are percent-encoded (e.g. "%20", "%2F").
// `decodeComponent` reverses that: "+" -> " ", then percent-decoding.
export function decodeComponent(raw: string): string {
  const withSpaces = raw.replace(/\+/g, " ");
  return decodeURIComponent(withSpaces);
}
