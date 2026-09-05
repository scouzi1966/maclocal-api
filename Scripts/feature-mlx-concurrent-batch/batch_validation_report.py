"""Opt-in snapshots of batch validation evidence, outside the request path."""
import json
import os
from pathlib import Path
import uuid


class BatchReport:
    def __init__(self, suite, model, endpoint):
        root = os.environ.get('AFM_REPORT_DIR')
        self.path = Path(root) / f'{suite}-{uuid.uuid4().hex}.json' if root else None
        self.metadata = dict(suite=suite, model=model, endpoint=endpoint,
                             failure_attribution='unattributed')

    def save(self, batches):
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.partial')
        temporary.write_text(json.dumps(dict(self.metadata, batches=batches), indent=2))
        temporary.replace(self.path)
        print(f'  Raw evidence: {self.path}')
