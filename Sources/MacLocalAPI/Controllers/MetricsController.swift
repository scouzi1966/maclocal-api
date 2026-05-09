import Vapor
import Foundation

/// `GET /metrics` — Prometheus text exposition format.
///
/// Mirrors vLLM's metric taxonomy (`vllm:` → `afm:` prefix, same
/// gauge / counter / histogram semantics with identical bucket boundaries)
/// so Grafana dashboards authored for vLLM work against AFM with a
/// search-and-replace of the namespace prefix.
///
/// Reads from `StatsAggregator.shared`, which is populated by
/// `BatchScheduler` (batch path) and `MLXModelService` (serial path).
struct MetricsController: RouteCollection {

    let aggregator: StatsAggregator

    init(aggregator: StatsAggregator = .shared) {
        self.aggregator = aggregator
    }

    func boot(routes: RoutesBuilder) throws {
        routes.get("metrics", use: metrics)
    }

    func metrics(req: Request) async throws -> Response {
        let body = Self.renderPrometheus(aggregator.snapshot())
        let response = Response(status: .ok)
        response.headers.replaceOrAdd(
            name: .contentType,
            value: "text/plain; version=0.0.4; charset=utf-8"
        )
        response.headers.add(name: .accessControlAllowOrigin, value: "*")
        response.body = .init(string: body)
        return response
    }

    // MARK: - Prometheus text format

    /// Escape a string for inclusion in a Prometheus label value.
    /// Spec: backslash, double-quote and newline must be escaped.
    private static func labelEscape(_ s: String) -> String {
        var out = ""
        out.reserveCapacity(s.count)
        for ch in s {
            switch ch {
            case "\\": out += "\\\\"
            case "\"": out += "\\\""
            case "\n": out += "\\n"
            default:   out.append(ch)
            }
        }
        return out
    }

    /// Render a `Double` as a finite, locale-independent decimal. Matches
    /// `repr`-style output for the values we emit (latencies, fractions).
    private static func formatDouble(_ d: Double) -> String {
        if !d.isFinite { return "0" }
        // Use Swift's default which is locale-independent and round-trip safe.
        return String(d)
    }

    static func renderPrometheus(_ s: StatsAggregator.Snapshot) -> String {
        let modelLabelOnly = "model_name=\"\(labelEscape(s.modelName))\""
        let modelLabel = "{\(modelLabelOnly)}"

        var out = ""
        out.reserveCapacity(8192)

        func gauge(_ name: String, _ help: String, _ value: String) {
            out += "# HELP \(name) \(help)\n"
            out += "# TYPE \(name) gauge\n"
            out += "\(name)\(modelLabel) \(value)\n"
        }
        func counter(_ name: String, _ help: String, _ value: UInt64) {
            out += "# HELP \(name) \(help)\n"
            out += "# TYPE \(name) counter\n"
            out += "\(name)\(modelLabel) \(value)\n"
        }

        // ─── Gauges ─────────────────────────────────────────────────────────
        gauge(
            "afm:num_requests_running",
            "Number of requests currently generating on the GPU (active batch size).",
            String(s.numRunning)
        )
        gauge(
            "afm:num_requests_waiting",
            "Number of requests queued behind the --concurrent capacity.",
            String(s.numWaiting)
        )
        gauge(
            "afm:batch_size_peak",
            "Highest num_requests_running observed since server start.",
            String(s.batchSizePeak)
        )
        gauge(
            "afm:max_concurrent_slots",
            "Configured --concurrent capacity of the server.",
            String(s.maxConcurrent)
        )
        if let usage = s.gpuCacheUsage {
            gauge(
                "afm:gpu_cache_usage_perc",
                "GPU KV-cache usage as a fraction in [0, 1] (1.0 = 100%).",
                formatDouble(usage)
            )
        }
        gauge(
            "afm:num_active_connections",
            "Number of HTTP client connections currently being served (excludes /metrics scrapes).",
            String(s.activeConnections)
        )
        gauge(
            "afm:active_connections_peak",
            "All-time-high number of concurrent HTTP client connections since server start.",
            String(s.activeConnectionsPeak)
        )

        // ─── Counters ───────────────────────────────────────────────────────
        counter(
            "afm:generation_tokens_total",
            "Total number of output tokens generated since server start.",
            s.genTokensTotal
        )
        counter(
            "afm:prompt_tokens_total",
            "Total number of prompt tokens processed by prefill since server start.",
            s.promptTokensTotal
        )
        counter(
            "afm:requests_started_total",
            "Total number of requests accepted since server start.",
            s.requestsStartedTotal
        )
        counter(
            "afm:requests_completed_total",
            "Total number of requests fully completed since server start.",
            s.requestsCompletedTotal
        )
        counter(
            "afm:radix_cache_hits_total",
            "Total number of prefix cache hits (radix tree) since server start.",
            s.cacheHitsTotal
        )
        counter(
            "afm:radix_cache_misses_total",
            "Total number of prefix cache misses (radix tree) since server start.",
            s.cacheMissesTotal
        )

        // request_success_total{finished_reason=...} — vLLM-style labeled counter.
        out += "# HELP afm:request_success_total Count of successfully processed requests, broken out by finished_reason (stop|length|tool_calls|abort|error|...).\n"
        out += "# TYPE afm:request_success_total counter\n"
        if s.requestSuccessByReason.isEmpty {
            // Emit a zero-valued sample so scrapers see the series exists.
            out += "afm:request_success_total{\(modelLabelOnly),finished_reason=\"stop\"} 0\n"
        } else {
            for reason in s.requestSuccessByReason.keys.sorted() {
                let v = s.requestSuccessByReason[reason] ?? 0
                out += "afm:request_success_total{\(modelLabelOnly),finished_reason=\"\(labelEscape(reason))\"} \(v)\n"
            }
        }

        // ─── Histograms ─────────────────────────────────────────────────────
        renderHistogram(
            into: &out,
            name: "afm:e2e_request_latency_seconds",
            help: "End-to-end request latency in seconds (queued → completed).",
            labels: modelLabelOnly,
            histogram: s.e2eLatency
        )
        renderHistogram(
            into: &out,
            name: "afm:request_queue_time_seconds",
            help: "Time a request spent waiting in the queue before scheduling.",
            labels: modelLabelOnly,
            histogram: s.queueTime
        )
        renderHistogram(
            into: &out,
            name: "afm:request_inference_time_seconds",
            help: "Time spent generating (started → completed), exclusive of queue time.",
            labels: modelLabelOnly,
            histogram: s.inferenceTime
        )
        renderHistogram(
            into: &out,
            name: "afm:request_prefill_time_seconds",
            help: "Time spent on prefill (started → first token).",
            labels: modelLabelOnly,
            histogram: s.prefillTime
        )
        renderHistogram(
            into: &out,
            name: "afm:request_decode_time_seconds",
            help: "Time spent on decode (first token → completed).",
            labels: modelLabelOnly,
            histogram: s.decodeTime
        )
        renderHistogram(
            into: &out,
            name: "afm:time_to_first_token_seconds",
            help: "Time from request arrival to the first generated token.",
            labels: modelLabelOnly,
            histogram: s.timeToFirstToken
        )
        renderHistogram(
            into: &out,
            name: "afm:time_per_output_token_seconds",
            help: "Average inter-token latency during decode for each completed request.",
            labels: modelLabelOnly,
            histogram: s.timePerOutputToken
        )
        renderHistogram(
            into: &out,
            name: "afm:request_prompt_tokens",
            help: "Number of prompt tokens per request.",
            labels: modelLabelOnly,
            histogram: s.promptTokens
        )
        renderHistogram(
            into: &out,
            name: "afm:request_generation_tokens",
            help: "Number of generated tokens per request.",
            labels: modelLabelOnly,
            histogram: s.generationTokens
        )
        renderHistogram(
            into: &out,
            name: "afm:request_params_n",
            help: "Distribution of the n sampling parameter per request.",
            labels: modelLabelOnly,
            histogram: s.paramsN
        )
        renderHistogram(
            into: &out,
            name: "afm:request_params_best_of",
            help: "Distribution of the best_of sampling parameter per request.",
            labels: modelLabelOnly,
            histogram: s.paramsBestOf
        )

        // ─── Process info (unlabeled, scraper-friendly) ─────────────────────
        out += "# HELP afm:process_start_time_seconds Unix epoch time the afm process started.\n"
        out += "# TYPE afm:process_start_time_seconds gauge\n"
        out += "afm:process_start_time_seconds \(s.processStartEpoch)\n"

        out += "# HELP afm:snapshot_timestamp_ms Unix epoch time (ms) this snapshot was taken.\n"
        out += "# TYPE afm:snapshot_timestamp_ms gauge\n"
        out += "afm:snapshot_timestamp_ms \(s.timestampMs)\n"

        return out
    }

    /// Render one histogram in Prometheus text exposition format:
    ///
    ///   # HELP NAME HELP_TEXT
    ///   # TYPE NAME histogram
    ///   NAME_bucket{LABELS,le="0.3"} 0
    ///   ...
    ///   NAME_bucket{LABELS,le="+Inf"} 1
    ///   NAME_sum{LABELS} 0.42
    ///   NAME_count{LABELS} 1
    private static func renderHistogram(
        into out: inout String,
        name: String,
        help: String,
        labels: String,
        histogram h: StatsAggregator.Histogram
    ) {
        out += "# HELP \(name) \(help)\n"
        out += "# TYPE \(name) histogram\n"
        for i in 0..<h.buckets.count {
            let le = formatDouble(h.buckets[i])
            let count = h.bucketCounts[i]
            out += "\(name)_bucket{\(labels),le=\"\(le)\"} \(count)\n"
        }
        // +Inf bucket is the last entry of bucketCounts
        out += "\(name)_bucket{\(labels),le=\"+Inf\"} \(h.bucketCounts[h.buckets.count])\n"
        out += "\(name)_sum{\(labels)} \(formatDouble(h.sum))\n"
        out += "\(name)_count{\(labels)} \(h.count)\n"
    }
}
