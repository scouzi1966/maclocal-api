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
        let vLLMLabelOnly = "\(modelLabelOnly),engine=\"0\""

        var output = ""
        output.reserveCapacity(16_384)

        func gauge(
            _ name: String,
            _ help: String,
            _ value: String,
            labels: String = modelLabelOnly
        ) {
            output += "# HELP \(name) \(help)\n"
            output += "# TYPE \(name) gauge\n"
            output += "\(name){\(labels)} \(value)\n"
        }

        func counter(
            _ name: String,
            _ help: String,
            _ value: UInt64,
            labels: String = modelLabelOnly
        ) {
            output += "# HELP \(name) \(help)\n"
            output += "# TYPE \(name) counter\n"
            output += "\(name){\(labels)} \(value)\n"
        }

        func vLLMGauge(_ name: String, _ help: String, _ value: String) {
            gauge(name, help, value, labels: vLLMLabelOnly)
        }

        func vLLMCounter(_ name: String, _ help: String, _ value: UInt64) {
            counter(name, help, value, labels: vLLMLabelOnly)
        }

        // Existing AFM families retain their computed-prefill semantics.
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

        // Pinned vLLM compatibility families. HELP text, metric types, and
        // bounded label values match the pinned vLLM exposition contract.
        vLLMGauge("vllm:num_requests_running", "Number of requests in model execution batches.", String(snapshot.runningRequests))
        vLLMGauge("vllm:num_requests_waiting", "Number of requests waiting to be processed.", String(snapshot.waitingRequests))
        let waitingReasonHelp = "Number of waiting requests by reason. Reason labels: 'capacity' = waiting for scheduling capacity; 'deferred' = deferred by transient constraints (LoRA budget, KV transfer, blocked status). Sum of all reasons equals vllm:num_requests_waiting."
        output += "# HELP vllm:num_requests_waiting_by_reason \(waitingReasonHelp)\n"
        output += "# TYPE vllm:num_requests_waiting_by_reason gauge\n"
        output += "vllm:num_requests_waiting_by_reason{\(vLLMLabelOnly),reason=\"capacity\"} \(snapshot.waitingRequests)\n"
        output += "vllm:num_requests_waiting_by_reason{\(vLLMLabelOnly),reason=\"deferred\"} 0\n"

        let sleepStateHelp = "Engine sleep state; awake = 0 means engine is sleeping; awake = 1 means engine is awake; weights_offloaded = 1 means sleep level 1; discard_all = 1 means sleep level 2."
        output += "# HELP vllm:engine_sleep_state \(sleepStateHelp)\n"
        output += "# TYPE vllm:engine_sleep_state gauge\n"
        output += "vllm:engine_sleep_state{\(vLLMLabelOnly),sleep_state=\"awake\"} 1\n"
        output += "vllm:engine_sleep_state{\(vLLMLabelOnly),sleep_state=\"weights_offloaded\"} 0\n"
        output += "vllm:engine_sleep_state{\(vLLMLabelOnly),sleep_state=\"discard_all\"} 0\n"

        vLLMGauge("vllm:kv_cache_usage_perc", "KV-cache usage. 1 means 100 percent usage.", formatDouble(snapshot.logicalCacheUsage))

        // Legacy vLLM gauges retained for the pinned vLLM Playground. Current
        // vLLM removed them, but their historical HELP contract is stable.
        vLLMGauge("vllm:avg_prompt_throughput_toks_per_s", "Average prefill throughput in tokens/s.", formatDouble(doubleGauge("computed_prompt_throughput", in: snapshot)))
        vLLMGauge("vllm:avg_generation_throughput_toks_per_s", "Average generation throughput in tokens/s.", formatDouble(doubleGauge("generation_throughput", in: snapshot)))

        let prefixHitRate = snapshot.prefixCacheQueriesTotal == 0
            ? 0
            : Double(snapshot.prefixCacheHitsTotal) / Double(snapshot.prefixCacheQueriesTotal)
        vLLMGauge("vllm:prefix_cache_hit_rate", "Fraction of eligible prompt tokens reused from the prefix cache.", formatDouble(prefixHitRate))

        let speculativeAcceptanceRate = snapshot.speculativeDraftTokensTotal == 0
            ? 0
            : Double(snapshot.speculativeAcceptedTokensTotal) / Double(snapshot.speculativeDraftTokensTotal)
        vLLMGauge("vllm:spec_decode_acceptance_rate", "Fraction of speculative draft tokens accepted.", formatDouble(speculativeAcceptanceRate))

        vLLMCounter("vllm:num_preemptions_total", "Cumulative number of preemption from the engine.", snapshot.preemptionsTotal)
        vLLMCounter("vllm:prompt_tokens_total", "Number of prefill tokens processed.", snapshot.fullPromptTokensTotal)

        output += "# HELP vllm:prompt_tokens_by_source_total Number of prompt tokens by source.\n"
        output += "# TYPE vllm:prompt_tokens_by_source_total counter\n"
        output += "vllm:prompt_tokens_by_source_total{\(vLLMLabelOnly),source=\"local_compute\"} \(snapshot.computedPromptTokensTotal)\n"
        output += "vllm:prompt_tokens_by_source_total{\(vLLMLabelOnly),source=\"local_cache_hit\"} \(snapshot.prefixCacheHitsTotal)\n"
        output += "vllm:prompt_tokens_by_source_total{\(vLLMLabelOnly),source=\"external_kv_transfer\"} 0\n"

        vLLMCounter("vllm:prompt_tokens_cached_total", "Number of cached prompt tokens (local + external).", snapshot.prefixCacheHitsTotal)
        vLLMCounter("vllm:generation_tokens_total", "Number of generation tokens processed.", snapshot.generatedTokensTotal)
        vLLMCounter("vllm:prefix_cache_queries_total", "Prefix cache queries, in terms of number of queried tokens.", snapshot.prefixCacheQueriesTotal)
        vLLMCounter("vllm:prefix_cache_hits_total", "Prefix cache hits, in terms of number of cached tokens.", snapshot.prefixCacheHitsTotal)
        vLLMCounter("vllm:external_prefix_cache_queries_total", "External prefix cache queries from KV connector cross-instance cache sharing, in terms of number of queried tokens.", 0)
        vLLMCounter("vllm:external_prefix_cache_hits_total", "External prefix cache hits from KV connector cross-instance cache sharing, in terms of number of cached tokens.", 0)
        vLLMCounter("vllm:mm_cache_queries_total", "Multi-modal cache queries, in terms of number of queried items.", 0)
        vLLMCounter("vllm:mm_cache_hits_total", "Multi-modal cache hits, in terms of number of cached items.", 0)
        vLLMCounter("vllm:spec_decode_num_draft_tokens_total", "Number of draft tokens.", snapshot.speculativeDraftTokensTotal)
        vLLMCounter("vllm:spec_decode_num_accepted_tokens_total", "Number of accepted tokens.", snapshot.speculativeAcceptedTokensTotal)
        vLLMCounter("vllm:spec_decode_num_drafts_total", "Number of spec decoding drafts.", snapshot.speculativeDraftRoundsTotal)

        output += "# HELP vllm:request_success_total Count of successfully processed requests.\n"
        output += "# TYPE vllm:request_success_total counter\n"
        for reason in ["stop", "length", "abort", "error", "repetition"] {
            output += "vllm:request_success_total{\(vLLMLabelOnly),finished_reason=\"\(reason)\"} \(terminalCount(reason, in: snapshot))\n"
        }

        renderHistogram(into: &output, name: "vllm:e2e_request_latency_seconds", help: "Histogram of e2e request latency in seconds.", labels: vLLMLabelOnly, histogram: snapshot.endToEndLatency)
        renderHistogram(into: &output, name: "vllm:request_queue_time_seconds", help: "Histogram of time spent in WAITING phase for request.", labels: vLLMLabelOnly, histogram: snapshot.queueLatency)
        renderHistogram(into: &output, name: "vllm:request_inference_time_seconds", help: "Histogram of time spent in RUNNING phase for request.", labels: vLLMLabelOnly, histogram: snapshot.inferenceLatency)
        renderHistogram(into: &output, name: "vllm:request_prefill_time_seconds", help: "Histogram of time spent in PREFILL phase for request.", labels: vLLMLabelOnly, histogram: snapshot.prefillLatency)
        renderHistogram(into: &output, name: "vllm:request_decode_time_seconds", help: "Histogram of time spent in DECODE phase for request.", labels: vLLMLabelOnly, histogram: snapshot.decodeLatency)
        renderHistogram(into: &output, name: "vllm:time_to_first_token_seconds", help: "Histogram of time to first token in seconds.", labels: vLLMLabelOnly, histogram: snapshot.timeToFirstToken)
        renderHistogram(into: &output, name: "vllm:request_time_per_output_token_seconds", help: "Histogram of time_per_output_token_seconds per request.", labels: vLLMLabelOnly, histogram: snapshot.timePerOutputToken)
        renderHistogram(into: &output, name: "vllm:inter_token_latency_seconds", help: "Histogram of inter-token latency in seconds.", labels: vLLMLabelOnly, histogram: snapshot.interTokenLatency)
        renderHistogram(into: &output, name: "vllm:request_prompt_tokens", help: "Number of prefill tokens processed.", labels: vLLMLabelOnly, histogram: snapshot.fullPromptTokens)
        renderHistogram(into: &output, name: "vllm:request_generation_tokens", help: "Number of generation tokens processed.", labels: vLLMLabelOnly, histogram: snapshot.generatedTokens)
        renderHistogram(into: &output, name: "vllm:request_max_num_generation_tokens", help: "Histogram of maximum number of requested generation tokens.", labels: vLLMLabelOnly, histogram: snapshot.maximumGeneratedTokens)
        renderHistogram(into: &output, name: "vllm:request_params_max_tokens", help: "Histogram of the max_tokens request parameter.", labels: vLLMLabelOnly, histogram: snapshot.maximumOutputTokens)
        renderHistogram(into: &output, name: "vllm:request_params_n", help: "Histogram of the n request parameter.", labels: vLLMLabelOnly, histogram: snapshot.samplingN)
        renderHistogram(into: &output, name: "vllm:request_prefill_kv_computed_tokens", help: "Histogram of new KV tokens computed during prefill (excluding cached tokens).", labels: vLLMLabelOnly, histogram: snapshot.computedPromptTokens)

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
