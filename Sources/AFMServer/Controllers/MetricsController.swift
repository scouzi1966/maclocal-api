import Vapor
import AFMKitCore
import Foundation

/// `GET /metrics` in Prometheus text exposition format.
///
/// Existing `afm:*` families remain available. Pinned `vllm:*` compatibility
/// families are emitted from the same immutable process snapshot so an
/// unmodified vLLM Playground can scrape AFM without a second registry.
struct MetricsController: RouteCollection {
    let snapshotSource: any AFMInferenceMetricsSnapshotSource

    init(snapshotSource: any AFMInferenceMetricsSnapshotSource) {
        self.snapshotSource = snapshotSource
    }

    func boot(routes: RoutesBuilder) throws {
        routes.get("metrics", use: metrics)
    }

    func metrics(req: Request) async throws -> Response {
        let body = Self.renderPrometheus(snapshotSource.metricsSnapshot())
        let response = Response(status: .ok)
        response.headers.replaceOrAdd(
            name: .contentType,
            value: "text/plain; version=0.0.4; charset=utf-8"
        )
        response.headers.add(name: .accessControlAllowOrigin, value: "*")
        response.body = .init(string: body)
        return response
    }

    private static func labelEscape(_ value: String) -> String {
        var result = ""
        result.reserveCapacity(value.count)
        for character in value {
            switch character {
            case "\\": result += "\\\\"
            case "\"": result += "\\\""
            case "\n": result += "\\n"
            default: result.append(character)
            }
        }
        return result
    }

    private static func formatDouble(_ value: Double) -> String {
        value.isFinite ? String(value) : "0"
    }

    private static func integerGauge(
        _ name: String,
        in snapshot: AFMInferenceMetricsSnapshot
    ) -> Int {
        snapshot.supplementalIntegerGauges.first(where: { $0.name == name })?.value ?? 0
    }

    private static func doubleGauge(
        _ name: String,
        in snapshot: AFMInferenceMetricsSnapshot
    ) -> Double {
        snapshot.supplementalDoubleGauges.first(where: { $0.name == name })?.value ?? 0
    }

    private static func supplementalCount(
        _ name: String,
        in snapshot: AFMInferenceMetricsSnapshot
    ) -> UInt64 {
        snapshot.supplementalCounts.first(where: { $0.name == name })?.count ?? 0
    }

    private static func terminalCount(
        _ name: String,
        in snapshot: AFMInferenceMetricsSnapshot
    ) -> UInt64 {
        snapshot.terminalCounts.first(where: { $0.name == name })?.count ?? 0
    }

    static func renderPrometheus(_ snapshot: AFMInferenceMetricsSnapshot) -> String {
        let modelLabelOnly = "model_name=\"\(labelEscape(snapshot.modelName))\""
        let modelLabel = "{\(modelLabelOnly)}"

        var output = ""
        output.reserveCapacity(16_384)

        func gauge(_ name: String, _ help: String, _ value: String) {
            output += "# HELP \(name) \(help)\n"
            output += "# TYPE \(name) gauge\n"
            output += "\(name)\(modelLabel) \(value)\n"
        }

        func counter(_ name: String, _ help: String, _ value: UInt64) {
            output += "# HELP \(name) \(help)\n"
            output += "# TYPE \(name) counter\n"
            output += "\(name)\(modelLabel) \(value)\n"
        }

        // Existing AFM families. The AFM prompt counter and histogram retain
        // their computed-prefill semantics rather than aliasing vLLM's full
        // prompt accounting.
        gauge(
            "afm:num_requests_running",
            "Number of requests currently generating on the GPU (active batch size).",
            String(snapshot.runningRequests)
        )
        gauge(
            "afm:num_requests_waiting",
            "Number of requests queued behind the --concurrent capacity.",
            String(snapshot.waitingRequests)
        )
        gauge(
            "afm:batch_size_peak",
            "Highest num_requests_running observed since server start.",
            String(snapshot.peakRunningRequests)
        )
        gauge(
            "afm:max_concurrent_slots",
            "Configured --concurrent capacity of the server.",
            String(snapshot.maximumConcurrentRequests)
        )
        if let usage = snapshot.memoryCacheUsage {
            gauge(
                "afm:gpu_cache_usage_perc",
                "GPU memory pressure as a fraction in [0, 1] of Metal's maxRecommendedWorkingSetSize (model weights + KV cache + intermediate tensors).",
                formatDouble(usage)
            )
        }
        if let fill = snapshot.prefixCacheFill {
            gauge(
                "afm:radix_cache_fill_perc",
                "Radix prefix cache fill as a fraction in [0, 1] (current entries / configured capacity). Omitted when --enable-prefix-caching is off.",
                formatDouble(fill)
            )
        }
        gauge(
            "afm:num_active_connections",
            "Number of HTTP client connections currently being served (excludes /metrics scrapes).",
            String(integerGauge("active_connections", in: snapshot))
        )
        gauge(
            "afm:active_connections_peak",
            "All-time-high number of concurrent HTTP client connections since server start.",
            String(integerGauge("active_connections_peak", in: snapshot))
        )
        counter(
            "afm:generation_tokens_total",
            "Total number of output tokens generated since server start.",
            snapshot.generatedTokensTotal
        )
        counter(
            "afm:prompt_tokens_total",
            "Total number of prompt tokens processed by prefill since server start.",
            snapshot.computedPromptTokensTotal
        )
        counter(
            "afm:requests_started_total",
            "Total number of requests accepted since server start.",
            snapshot.acceptedRequestsTotal
        )
        counter(
            "afm:requests_completed_total",
            "Total number of requests fully completed since server start.",
            snapshot.terminalRequestsTotal
        )
        counter(
            "afm:radix_cache_hits_total",
            "Total number of prefix cache hits (radix tree) since server start.",
            supplementalCount("legacy_cache_hits", in: snapshot)
        )
        counter(
            "afm:radix_cache_misses_total",
            "Total number of prefix cache misses (radix tree) since server start.",
            supplementalCount("legacy_cache_misses", in: snapshot)
        )

        output += "# HELP afm:request_success_total Count of successfully processed requests, broken out by finished_reason (stop|length|tool_calls|abort|error|...).\n"
        output += "# TYPE afm:request_success_total counter\n"
        let nonzeroTerminalCounts = snapshot.terminalCounts.filter { $0.count > 0 }
        if nonzeroTerminalCounts.isEmpty {
            output += "afm:request_success_total{\(modelLabelOnly),finished_reason=\"stop\"} 0\n"
        } else {
            for entry in nonzeroTerminalCounts.sorted(by: { $0.name < $1.name }) {
                output += "afm:request_success_total{\(modelLabelOnly),finished_reason=\"\(labelEscape(entry.name))\"} \(entry.count)\n"
            }
        }

        output += "# HELP afm:request_failures_total Count of provider failures and bounded server rejections by status.\n"
        output += "# TYPE afm:request_failures_total counter\n"
        for status in ["decode", "authentication", "validation", "capacity", "cancelled", "inference", "internal"] {
            let count = snapshot.failureCounts.first(where: { $0.name == status })?.count ?? 0
            output += "afm:request_failures_total{\(modelLabelOnly),status=\"\(status)\"} \(count)\n"
        }

        renderHistogram(into: &output, name: "afm:e2e_request_latency_seconds", help: "End-to-end request latency in seconds (queued -> completed).", labels: modelLabelOnly, histogram: snapshot.endToEndLatency)
        renderHistogram(into: &output, name: "afm:request_queue_time_seconds", help: "Time a request spent waiting in the queue before scheduling.", labels: modelLabelOnly, histogram: snapshot.queueLatency)
        renderHistogram(into: &output, name: "afm:request_inference_time_seconds", help: "Time spent generating (started -> completed), exclusive of queue time.", labels: modelLabelOnly, histogram: snapshot.inferenceLatency)
        renderHistogram(into: &output, name: "afm:request_prefill_time_seconds", help: "Time spent on prefill (started -> first token).", labels: modelLabelOnly, histogram: snapshot.prefillLatency)
        renderHistogram(into: &output, name: "afm:request_decode_time_seconds", help: "Time spent on decode (first token -> completed).", labels: modelLabelOnly, histogram: snapshot.decodeLatency)
        renderHistogram(into: &output, name: "afm:time_to_first_token_seconds", help: "Time from request arrival to the first generated token.", labels: modelLabelOnly, histogram: snapshot.timeToFirstToken)
        renderHistogram(into: &output, name: "afm:time_per_output_token_seconds", help: "Average inter-token latency during decode for each completed request.", labels: modelLabelOnly, histogram: snapshot.timePerOutputToken)
        renderHistogram(into: &output, name: "afm:request_prompt_tokens", help: "Number of computed prompt tokens per request.", labels: modelLabelOnly, histogram: snapshot.computedPromptTokens)
        renderHistogram(into: &output, name: "afm:request_generation_tokens", help: "Number of generated tokens per request.", labels: modelLabelOnly, histogram: snapshot.generatedTokens)
        renderHistogram(into: &output, name: "afm:request_params_n", help: "Distribution of the n sampling parameter per request.", labels: modelLabelOnly, histogram: snapshot.samplingN)
        renderHistogram(into: &output, name: "afm:request_params_best_of", help: "Distribution of the best_of sampling parameter per request.", labels: modelLabelOnly, histogram: snapshot.samplingBestOf)

        // Pinned vLLM/Playground compatibility families.
        gauge("vllm:num_requests_running", "Number of requests currently running on the engine.", String(snapshot.runningRequests))
        gauge("vllm:num_requests_waiting", "Number of requests waiting to be processed.", String(snapshot.waitingRequests))
        gauge("vllm:kv_cache_usage_perc", "Logical KV cache occupancy as a fraction in [0, 1].", formatDouble(snapshot.logicalCacheUsage))
        gauge("vllm:avg_prompt_throughput_toks_per_s", "Average computed prompt throughput over AFM's 10-second rolling window.", formatDouble(doubleGauge("computed_prompt_throughput", in: snapshot)))
        gauge("vllm:avg_generation_throughput_toks_per_s", "Average generation throughput over AFM's 10-second rolling window.", formatDouble(doubleGauge("generation_throughput", in: snapshot)))

        let prefixHitRate = snapshot.prefixCacheQueriesTotal == 0
            ? 0
            : Double(snapshot.prefixCacheHitsTotal) / Double(snapshot.prefixCacheQueriesTotal)
        gauge("vllm:prefix_cache_hit_rate", "Fraction of eligible prompt tokens reused from the prefix cache.", formatDouble(prefixHitRate))

        let speculativeAcceptanceRate = snapshot.speculativeDraftTokensTotal == 0
            ? 0
            : Double(snapshot.speculativeAcceptedTokensTotal) / Double(snapshot.speculativeDraftTokensTotal)
        gauge("vllm:spec_decode_acceptance_rate", "Fraction of speculative draft tokens accepted.", formatDouble(speculativeAcceptanceRate))

        counter("vllm:num_preemptions_total", "Cumulative number of engine preemptions.", snapshot.preemptionsTotal)
        counter("vllm:prompt_tokens_total", "Cumulative number of full prompt tokens received.", snapshot.fullPromptTokensTotal)
        counter("vllm:generation_tokens_total", "Cumulative number of generated output tokens.", snapshot.generatedTokensTotal)
        counter("vllm:prefix_cache_queries_total", "Cumulative number of prompt tokens eligible for prefix-cache lookup.", snapshot.prefixCacheQueriesTotal)
        counter("vllm:prefix_cache_hits_total", "Cumulative number of prompt tokens reused from the prefix cache.", snapshot.prefixCacheHitsTotal)
        counter("vllm:spec_decode_num_draft_tokens_total", "Cumulative number of speculative draft tokens.", snapshot.speculativeDraftTokensTotal)
        counter("vllm:spec_decode_num_accepted_tokens_total", "Cumulative number of accepted speculative draft tokens.", snapshot.speculativeAcceptedTokensTotal)
        counter("vllm:spec_decode_num_drafts_total", "Cumulative number of speculative decode rounds.", snapshot.speculativeDraftRoundsTotal)

        output += "# HELP vllm:request_success_total Count of successfully processed requests by canonical vLLM finished_reason.\n"
        output += "# TYPE vllm:request_success_total counter\n"
        for reason in ["stop", "length", "abort", "error", "repetition"] {
            output += "vllm:request_success_total{\(modelLabelOnly),finished_reason=\"\(reason)\"} \(terminalCount(reason, in: snapshot))\n"
        }

        renderHistogram(into: &output, name: "vllm:e2e_request_latency_seconds", help: "End-to-end request latency in seconds.", labels: modelLabelOnly, histogram: snapshot.endToEndLatency)
        renderHistogram(into: &output, name: "vllm:time_to_first_token_seconds", help: "Time from request acceptance to first output token in seconds.", labels: modelLabelOnly, histogram: snapshot.timeToFirstToken)
        renderHistogram(into: &output, name: "vllm:request_time_per_output_token_seconds", help: "Decode duration per output-token interval in seconds.", labels: modelLabelOnly, histogram: snapshot.timePerOutputToken)
        renderHistogram(into: &output, name: "vllm:inter_token_latency_seconds", help: "Latency between adjacent output tokens in seconds.", labels: modelLabelOnly, histogram: snapshot.interTokenLatency)
        renderHistogram(into: &output, name: "vllm:request_prompt_tokens", help: "Number of full prompt tokens per request.", labels: modelLabelOnly, histogram: snapshot.fullPromptTokens)
        renderHistogram(into: &output, name: "vllm:request_generation_tokens", help: "Number of generated tokens per request.", labels: modelLabelOnly, histogram: snapshot.generatedTokens)

        let rejectedSpeculativeTokens = snapshot.speculativeDraftTokensTotal
            &- min(snapshot.speculativeDraftTokensTotal, snapshot.speculativeAcceptedTokensTotal)
        counter("afm:spec_decode_num_rejected_tokens_total", "Cumulative number of rejected speculative draft tokens.", rejectedSpeculativeTokens)

        output += "# HELP afm:process_start_time_seconds Unix epoch time the afm process started.\n"
        output += "# TYPE afm:process_start_time_seconds gauge\n"
        output += "afm:process_start_time_seconds \(formatDouble(snapshot.processStartEpochSeconds))\n"
        output += "# HELP afm:snapshot_timestamp_ms Unix epoch time (ms) this snapshot was taken.\n"
        output += "# TYPE afm:snapshot_timestamp_ms gauge\n"
        output += "afm:snapshot_timestamp_ms \(snapshot.timestampMilliseconds)\n"

        return output
    }

    private static func renderHistogram(
        into output: inout String,
        name: String,
        help: String,
        labels: String,
        histogram: AFMHistogramSnapshot
    ) {
        output += "# HELP \(name) \(help)\n"
        output += "# TYPE \(name) histogram\n"
        for index in histogram.buckets.indices {
            let count = index < histogram.bucketCounts.count
                ? histogram.bucketCounts[index]
                : 0
            output += "\(name)_bucket{\(labels),le=\"\(formatDouble(histogram.buckets[index]))\"} \(count)\n"
        }
        let infiniteCount = histogram.bucketCounts.indices.contains(histogram.buckets.count)
            ? histogram.bucketCounts[histogram.buckets.count]
            : histogram.count
        output += "\(name)_bucket{\(labels),le=\"+Inf\"} \(infiniteCount)\n"
        output += "\(name)_sum{\(labels)} \(formatDouble(histogram.sum))\n"
        output += "\(name)_count{\(labels)} \(histogram.count)\n"
    }
}
