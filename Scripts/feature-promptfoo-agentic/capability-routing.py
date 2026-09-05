#!/usr/bin/env python3
"""Route qualification contracts using advertised capabilities, never model names."""
import json
from pathlib import Path
import sys


def snapshot(payload, model):
    entries = payload.get('models')
    entries = [entry for entry in entries if isinstance(entry, dict)] if isinstance(entries, list) else []
    selected = [entry for entry in entries if entry.get('model') == model]
    if not selected:
        selected = [entry for entry in entries if isinstance(entry.get('capabilities'), list)
                    and 'embeddings' not in entry['capabilities']]
    caps = selected[0].get('capabilities') if len(selected) == 1 else None
    known = (isinstance(caps, list) and all(isinstance(cap, str) for cap in caps)
             and bool({'mlx_runtime', 'dwarfstar_runtime'}.intersection(caps)))
    return dict(requested_model=model, selected_model=selected[0].get('model') if len(selected) == 1 else None,
                status='known' if known else 'unknown',
                capabilities=sorted(set(caps)) if known else None)


def missing_capabilities(evidence, required):
    # Unknown or ambiguous metadata must never manufacture SKIP results.
    if evidence['status'] != 'known':
        return []
    return sorted(set(required) - set(evidence['capabilities']))


def main():
    action, source, *args = sys.argv[1:]
    payload = json.loads(Path(source).read_text())
    if action == 'snapshot':
        print(json.dumps(snapshot(payload, args[0]), indent=2))
        return 0
    if action != 'check':
        raise ValueError(f'Unknown action: {action}')
    destination, scope, *required = args
    missing = missing_capabilities(payload, required)
    if not missing:
        return 0
    path = Path(destination)
    records = json.loads(path.read_text()) if path.exists() else []
    records.append(dict(scope=scope, status='SKIP', required_capabilities=required,
                        missing_capabilities=missing, evidence=payload,
                        reason='Contract requires capabilities not advertised by selected engine; not a conformance pass'))
    path.write_text(json.dumps(records, indent=2))
    print(f'SKIP {scope}: missing advertised capabilities {", ".join(missing)} (not a conformance pass)')
    return 3


if __name__ == '__main__':
    sys.exit(main())
