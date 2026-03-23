// Custom promptfoo assertion: checks X-Grammar-Constraints response header
// Usage in YAML:
//   - type: javascript
//     value: file://judges/assert-grammar-header.mjs
//     config:
//       expectDowngraded: true   # or false
export default function({ output, metadata, config }) {
  const header = metadata?.responseHeaders?.['x-grammar-constraints'];
  const expectDowngraded = config?.expectDowngraded === true;

  if (expectDowngraded) {
    const pass = header === 'downgraded';
    return {
      pass,
      score: pass ? 1 : 0,
      reason: pass
        ? 'Header present: downgraded'
        : `Expected X-Grammar-Constraints: downgraded, got: ${header || 'absent'}`,
    };
  } else {
    const pass = !header;
    return {
      pass,
      score: pass ? 1 : 0,
      reason: pass
        ? 'Header absent (grammar active or not requested)'
        : `Expected no X-Grammar-Constraints header, got: ${header}`,
    };
  }
}
