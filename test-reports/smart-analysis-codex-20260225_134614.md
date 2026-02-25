## AFM Compatibility QA Report

Parsed file: `/tmp/mlx-test-results.jsonl`  
Result: **1 line total, metadata only, 0 model test records**.

### 1) Broken Models
No model executions found, so no load/crash failures could be evaluated.

| Error Type | Models | AFM Bug vs Model Incompatibility |
|---|---|---|
| N/A (no test records) | None | Unknown |

### 2) Anomalies & Red Flags
No generated outputs were present, so repetition/degeneration/truncation/relevance checks are not possible.

### 3) Variant Comparison
No `@ variant` model records found; cannot compare A/B settings.

### 4) Quality Assessment
No model outputs available to score coherence/relevance per model.

### 5) Performance Summary (tokens/sec)
No performance rows available.

| Model | Variant | Tokens/sec | Notes |
|---|---|---:|---|
| N/A | N/A | N/A | No test records in file |

### 6) Recommendations
Priority action is to rerun/export the JSONL with actual result rows (lines containing model name, status, timings, tokens, and content fields). Current file appears to contain only run metadata:
- `afm_version: v0.9.5-0cfba17`
- `timestamp: 2026-02-25T13:46:06Z`
- `test_command: mlx-model-test.sh ...`

<!-- AI_SCORES [{"i":0,"s":3}] -->
