# Upstream provenance

This directory's `afm-dashboard.json` is a one-time port of vLLM's
example Grafana dashboard. It is **not** a live dependency — there is
no submodule, no fetch step, no remote reference at runtime or build
time. The file below records the exact upstream snapshot it was
derived from so future re-ports can diff against a known baseline.

## Snapshot point

| Field | Value |
|---|---|
| Repository | [`vllm-project/vllm`](https://github.com/vllm-project/vllm) |
| Branch when ported | `main` |
| Repo HEAD at port time | `2ee8c2a56e41fbd00b4fb52f29464fb7fca48dba` |
| Source file | `examples/observability/prometheus_grafana/grafana.json` |
| Source blob hash | `1c89d459383094f52b8848d9aa82ba04ff500d37` |
| Port date | 2026-05-09 |

## Transformations applied during port

The script that did the port isn't checked in (it's mechanical) but
is reproducible from these rules applied in this order:

1. Global string replace: `vllm:` → `afm:` (every metric-name reference)
2. Metric-name remappings where afm and vLLM diverged:
   - `afm:inter_token_latency_seconds*` → `afm:time_per_output_token_seconds*`
   - `afm:kv_cache_usage_perc` → `afm:gpu_cache_usage_perc`
3. Brand replace: `vLLM` → `AFM` (ran AFTER the metric remaps so it
   doesn't accidentally rewrite metric names; the dashboard's title is
   restored to `AFM (vLLM-compatible)` afterwards)
4. Drop the panel `Max Generation Token in Sequence Group` —
   `vllm:request_max_num_generation_tokens` has no afm equivalent
5. Hard-code datasource UID `${DS_PROMETHEUS}` → `PROM-AFM` for
   Grafana auto-provisioning, and drop the corresponding `templating`
   entry
6. Append two afm-native panels:
   - **Radix Prefix Cache Hit Rate** —
     `afm:radix_cache_hits_total / (afm:radix_cache_hits_total + afm:radix_cache_misses_total)`
   - **Batch Utilization (in-flight vs cap)** —
     `afm:num_requests_running` / `afm:batch_size_peak` / `afm:max_concurrent_slots`
7. Set dashboard title to `AFM (vLLM-compatible)`, uid `afm-vllm-compat`

## Updating the snapshot

When upstream improves their dashboard or adds new metrics worth
mirroring:

```sh
gh api repos/vllm-project/vllm/contents/examples/observability/prometheus_grafana/grafana.json \
   --jq '.content' | base64 -d > /tmp/vllm-grafana-new.json

# Diff to see what's new:
diff <(jq -S . Scripts/grafana/afm-dashboard.json) \
     <(jq -S . /tmp/vllm-grafana-new.json) | head -60

# Re-apply the 7 transformations above against the new file, replace
# Scripts/grafana/afm-dashboard.json, update this file's "Repo HEAD",
# "Source blob hash", and "Port date" rows.
```

## Related: bucket boundaries in `Sources/MacLocalAPI/Models/StatsAggregator.swift`

The histogram bucket arrays in `StatsAggregator.Buckets` are also
copied verbatim from upstream vLLM. The provenance comment is
attached directly to the `Buckets` enum in that file and references
this document.

| Field | Value |
|---|---|
| Source file | `vllm/v1/metrics/loggers.py` |
| Source blob hash | `6855efd9f54c6f8ac5b95704455f64d6e456b4c8` |
| Port date | 2026-05-09 |

## What this is and isn't

**Is**: a documented snapshot, kept up to date manually whenever
someone wants to re-mirror upstream.

**Isn't**: a dependency. Nothing in afm's build, runtime, or release
pipeline reaches out to `vllm-project/vllm`. You can delete the
upstream repo from existence and afm's `/metrics`, in-webui dashboard,
and Grafana stack all keep working unchanged.
