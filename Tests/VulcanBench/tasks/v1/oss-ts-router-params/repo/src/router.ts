export type Params = Record<string, string>;

export function match(pattern: string, path: string): Params | null {
  const pp = pattern.split('/').filter(Boolean);
  const ps = path.split('/').filter(Boolean);
  if (pp.length !== ps.length) return null;
  const out: Params = {};
  for (let i = 0; i < pp.length; i++) {
    if (pp[i].startsWith(':')) out[pp[i].slice(1)] = ps[i];
    else if (pp[i] !== ps[i]) return null;
  }
  return out;
}
