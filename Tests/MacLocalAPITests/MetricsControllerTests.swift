import AFMKitCore
import AFMKitServices
@testable import AFMServer
import XCTest

final class MetricsControllerTests: XCTestCase {
    func testMiddlewareTracksChatRequestHandlingButNotMetricsScrapes() {
        XCTAssertTrue(
            ActiveConnectionsMiddleware.shouldTrackInMiddleware(
                path: "/v1/chat/completions"
            )
        )
        XCTAssertFalse(
            ActiveConnectionsMiddleware.shouldTrackInMiddleware(path: "/metrics")
        )
    }

    func testRendererPreservesAFMMetricsAndEmitsPinnedVLLMMetrics() {
        let collector = InferenceTelemetryCollector(
            now: { 105 },
            wallTime: { 1_700_000_000 }
        )
        collector.configure(
            modelName: "model\"quoted",
            maximumConcurrentRequests: 8,
            maximumContextTokens: 100
        )
        collector.updateProviderState(
            AFMInferenceProviderState(
                runningRequests: 2,
                waitingRequests: 3,
                activeLogicalCachePositions: 4,
                logicalCacheCapacity: 8,
                memoryCacheUsage: 0.75,
                prefixCacheFill: 0.25
            )
        )
        collector.prefixCacheObserved(queriedTokens: 10, hitTokens: 6)
        collector.speculativeRound(draftTokens: 4, acceptedTokens: 3)
        collector.preemptionObserved()
        collector.recordRejection(.capacity)
        _ = collector.connectionOpened()

        let request = collector.requestAccepted(at: 100)
        collector.requestStarted(request, at: 101)
        collector.outputToken(request, at: 102)
        collector.outputToken(request, at: 103)
        XCTAssertTrue(
            collector.requestFinished(
                request,
                observation: AFMInferenceRequestFinishObservation(
                    reason: .stop,
                    completedAt: 104,
                    fullPromptTokens: 10,
                    computedPromptTokens: 4,
                    generatedTokens: 2,
                    maximumOutputTokens: 64,
                    samplingN: 2,
                    samplingBestOf: 3
                )
            )
        )

        let output = MetricsController.renderPrometheus(collector.metricsSnapshot())
        let modelLabel = #"{model_name="model\"quoted"}"#
        let vLLMLabel = #"{model_name="model\"quoted",engine="0"}"#

        XCTAssertTrue(output.contains("afm:prompt_tokens_total\(modelLabel) 4\n"))
        XCTAssertTrue(output.contains("vllm:prompt_tokens_total\(vLLMLabel) 10\n"))
        XCTAssertTrue(output.contains("vllm:prompt_tokens_cached_total\(vLLMLabel) 6\n"))
        XCTAssertTrue(
            output.contains(
                #"vllm:prompt_tokens_by_source_total{model_name="model\"quoted",engine="0",source="local_compute"} 4"#
            )
        )
        XCTAssertTrue(
            output.contains(
                #"vllm:prompt_tokens_by_source_total{model_name="model\"quoted",engine="0",source="local_cache_hit"} 6"#
            )
        )
        XCTAssertTrue(output.contains("vllm:generation_tokens_total\(vLLMLabel) 2\n"))
        XCTAssertTrue(output.contains("vllm:num_requests_running\(vLLMLabel) 2\n"))
        XCTAssertTrue(output.contains("vllm:num_requests_waiting\(vLLMLabel) 3\n"))
        XCTAssertTrue(output.contains("vllm:kv_cache_usage_perc\(vLLMLabel) 0.5\n"))
        XCTAssertTrue(output.contains("vllm:prefix_cache_hit_rate\(vLLMLabel) 0.6\n"))
        XCTAssertTrue(output.contains("vllm:spec_decode_acceptance_rate\(vLLMLabel) 0.75\n"))
        XCTAssertTrue(output.contains("vllm:num_preemptions_total\(vLLMLabel) 1\n"))
        XCTAssertTrue(output.contains("afm:spec_decode_num_rejected_tokens_total\(modelLabel) 1\n"))
        XCTAssertTrue(output.contains("afm:num_active_connections\(modelLabel) 1\n"))
        XCTAssertTrue(
            output.contains(
                #"afm:request_failures_total{model_name="model\"quoted",status="capacity"} 1"#
            )
        )

        XCTAssertTrue(output.contains("afm:request_prompt_tokens_sum\(modelLabel) 4.0\n"))
        XCTAssertTrue(output.contains("vllm:request_prompt_tokens_sum\(vLLMLabel) 10.0\n"))
        XCTAssertTrue(output.contains("vllm:request_prompt_tokens_count\(vLLMLabel) 1\n"))
        XCTAssertTrue(output.contains("vllm:request_max_num_generation_tokens_sum\(vLLMLabel) 2.0\n"))
        XCTAssertTrue(output.contains("vllm:request_params_max_tokens_sum\(vLLMLabel) 64.0\n"))
        XCTAssertTrue(output.contains("vllm:inter_token_latency_seconds_sum\(vLLMLabel) 1.0\n"))
        XCTAssertTrue(output.contains("vllm:avg_prompt_throughput_toks_per_s\(vLLMLabel) 0.4\n"))
        XCTAssertTrue(output.contains("vllm:avg_generation_throughput_toks_per_s\(vLLMLabel) 0.2\n"))

        let successLines = output.split(separator: "\n").filter {
            $0.hasPrefix("vllm:request_success_total{")
        }
        XCTAssertEqual(successLines.count, 5)
        XCTAssertTrue(successLines.contains { $0.contains("finished_reason=\"stop\"") && $0.hasSuffix(" 1") })
        XCTAssertFalse(successLines.contains { $0.contains("tool_calls") })

        for metric in Self.pinnedVLLMMetrics {
            XCTAssertTrue(
                output.contains("# TYPE \(metric) "),
                "Missing pinned vLLM metric family \(metric)"
            )
        }

        let exposition = PrometheusExposition(output)
        for (family, type) in Self.requiredFamilyTypes {
            XCTAssertEqual(
                exposition.types[family],
                type,
                "Wrong or missing Prometheus type for \(family)"
            )
        }
        for sample in exposition.samples where sample.name.hasPrefix("vllm:") {
            let expected = Self.expectedLabels(for: sample.name)
            XCTAssertEqual(
                Set(sample.labels.keys),
                expected,
                "Unexpected labels for \(sample.name)"
            )
        }

        for family in Self.dashboardHistogramFamilies {
            XCTAssertTrue(exposition.samples.contains { $0.name == "\(family)_bucket" })
            XCTAssertTrue(exposition.samples.contains { $0.name == "\(family)_sum" })
            XCTAssertTrue(exposition.samples.contains { $0.name == "\(family)_count" })
        }

        for family in Self.intentionallyUnsupportedFamilies {
            XCTAssertNil(exposition.types[family], "Unsupported family must not be fabricated")
        }

        XCTAssertEqual(
            exposition.bucketBounds(for: "vllm:request_prompt_tokens"),
            ["1.0", "2.0", "5.0", "10.0", "20.0", "50.0", "100.0", "+Inf"]
        )
    }

    func testRendererMatchesPinnedVLLMFixtureWireSchema() throws {
        let fixtureURL = try XCTUnwrap(
            Bundle.module.url(
                forResource: "vllm-prometheus-contract-9633933",
                withExtension: "prom",
                subdirectory: "Fixtures"
            )
        )
        let fixture = PrometheusExposition(
            try String(contentsOf: fixtureURL, encoding: .utf8)
        )
        let collector = InferenceTelemetryCollector()
        collector.configure(
            modelName: "fixture-model",
            maximumConcurrentRequests: 8,
            maximumContextTokens: 100
        )
        let actual = PrometheusExposition(
            MetricsController.renderPrometheus(collector.metricsSnapshot())
        )

        for (family, expectedType) in fixture.types {
            XCTAssertEqual(actual.types[family], expectedType, "TYPE mismatch for \(family)")
            XCTAssertEqual(actual.helps[family], fixture.helps[family], "HELP mismatch for \(family)")

            let expectedLabelSets = Set(
                fixture.samples(for: family).map { Set($0.labels.keys) }
            )
            let actualLabelSets = Set(
                actual.samples(for: family).map { Set($0.labels.keys) }
            )
            XCTAssertEqual(actualLabelSets, expectedLabelSets, "Label mismatch for \(family)")
        }

        XCTAssertEqual(
            actual.bucketBounds(for: "vllm:time_to_first_token_seconds"),
            fixture.bucketBounds(for: "vllm:time_to_first_token_seconds")
        )
        XCTAssertEqual(
            actual.bucketBounds(for: "vllm:request_params_n"),
            ["1.0", "2.0", "5.0", "10.0", "20.0", "+Inf"]
        )
    }

    func testRendererIsAcceptedByPrometheusClientParser() throws {
        guard let python = ProcessInfo.processInfo.environment[
            "AFM_PROMETHEUS_PARSER_PYTHON"
        ], !python.isEmpty else {
            throw XCTSkip(
                "Set AFM_PROMETHEUS_PARSER_PYTHON to a Python with prometheus-client installed"
            )
        }
        let validatorURL = try XCTUnwrap(
            Bundle.module.url(
                forResource: "prometheus-exposition-validator",
                withExtension: "py",
                subdirectory: "Fixtures"
            )
        )
        let collector = InferenceTelemetryCollector()
        collector.configure(
            modelName: "parser-fixture-model",
            maximumConcurrentRequests: 8,
            maximumContextTokens: 100
        )
        let metricsURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("afm-vllm-metrics-\(UUID().uuidString).prom")
        defer { try? FileManager.default.removeItem(at: metricsURL) }
        try MetricsController.renderPrometheus(collector.metricsSnapshot())
            .write(to: metricsURL, atomically: true, encoding: .utf8)

        let process = Process()
        process.executableURL = URL(fileURLWithPath: python)
        process.arguments = [validatorURL.path, metricsURL.path]
        let errorPipe = Pipe()
        process.standardError = errorPipe
        try process.run()
        process.waitUntilExit()

        let errorData = errorPipe.fileHandleForReading.readDataToEndOfFile()
        let errorText = String(data: errorData, encoding: .utf8) ?? ""
        XCTAssertEqual(process.terminationStatus, 0, errorText)
    }

    func testRendererPrecreatesBoundedVLLMFinishReasonsForAnEmptySnapshot() {
        let collector = InferenceTelemetryCollector()
        let output = MetricsController.renderPrometheus(collector.metricsSnapshot())
        let successLines = output.split(separator: "\n").filter {
            $0.hasPrefix("vllm:request_success_total{")
        }

        XCTAssertEqual(successLines.count, 5)
        for reason in ["stop", "length", "abort", "error", "repetition"] {
            XCTAssertTrue(
                successLines.contains {
                    $0.contains("finished_reason=\"\(reason)\"") && $0.hasSuffix(" 0")
                }
            )
        }
    }

    func testPinnedGrafanaQueriesReferenceSupportedFamiliesOrDocumentedGap() throws {
        let fixtureURL = try XCTUnwrap(
            Bundle.module.url(
                forResource: "vllm-grafana-query-contract-9633933",
                withExtension: "json",
                subdirectory: "Fixtures"
            )
        )
        let data = try Data(contentsOf: fixtureURL)
        let fixture = try XCTUnwrap(
            JSONSerialization.jsonObject(with: data) as? [String: Any]
        )
        let requiredQueries = try XCTUnwrap(fixture["required_queries"] as? [String])
        let unsupportedQueries = try XCTUnwrap(fixture["unsupported_queries"] as? [String])

        let collector = InferenceTelemetryCollector()
        collector.configure(
            modelName: "fixture-model",
            maximumConcurrentRequests: 8,
            maximumContextTokens: 100
        )
        let exposition = PrometheusExposition(
            MetricsController.renderPrometheus(collector.metricsSnapshot())
        )

        for query in requiredQueries {
            for family in Self.vLLMFamilies(inPromQL: query) {
                XCTAssertNotNil(
                    exposition.types[family],
                    "Official dashboard query references missing family \(family): \(query)"
                )
            }
        }
        for query in unsupportedQueries {
            let families = Self.vLLMFamilies(inPromQL: query)
            XCTAssertEqual(families, ["vllm:iteration_tokens_total"])
            XCTAssertNil(exposition.types["vllm:iteration_tokens_total"])
        }
    }

    private static let pinnedVLLMMetrics = [
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:num_requests_waiting_by_reason",
        "vllm:engine_sleep_state",
        "vllm:kv_cache_usage_perc",
        "vllm:avg_prompt_throughput_toks_per_s",
        "vllm:avg_generation_throughput_toks_per_s",
        "vllm:prefix_cache_hit_rate",
        "vllm:spec_decode_acceptance_rate",
        "vllm:num_preemptions_total",
        "vllm:prompt_tokens_total",
        "vllm:prompt_tokens_by_source_total",
        "vllm:prompt_tokens_cached_total",
        "vllm:generation_tokens_total",
        "vllm:prefix_cache_queries_total",
        "vllm:prefix_cache_hits_total",
        "vllm:external_prefix_cache_queries_total",
        "vllm:external_prefix_cache_hits_total",
        "vllm:mm_cache_queries_total",
        "vllm:mm_cache_hits_total",
        "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_accepted_tokens_total",
        "vllm:spec_decode_num_drafts_total",
        "vllm:request_success_total",
        "vllm:e2e_request_latency_seconds",
        "vllm:time_to_first_token_seconds",
        "vllm:request_time_per_output_token_seconds",
        "vllm:inter_token_latency_seconds",
        "vllm:request_prompt_tokens",
        "vllm:request_generation_tokens",
        "vllm:request_max_num_generation_tokens",
        "vllm:request_params_max_tokens",
        "vllm:request_params_n",
        "vllm:request_queue_time_seconds",
        "vllm:request_inference_time_seconds",
        "vllm:request_prefill_time_seconds",
        "vllm:request_decode_time_seconds",
        "vllm:request_prefill_kv_computed_tokens",
    ]

    private static let requiredFamilyTypes: [String: String] = [
        "vllm:num_requests_running": "gauge",
        "vllm:num_requests_waiting": "gauge",
        "vllm:num_requests_waiting_by_reason": "gauge",
        "vllm:engine_sleep_state": "gauge",
        "vllm:kv_cache_usage_perc": "gauge",
        "vllm:prompt_tokens_total": "counter",
        "vllm:prompt_tokens_by_source_total": "counter",
        "vllm:prompt_tokens_cached_total": "counter",
        "vllm:generation_tokens_total": "counter",
        "vllm:request_success_total": "counter",
        "vllm:e2e_request_latency_seconds": "histogram",
        "vllm:request_queue_time_seconds": "histogram",
        "vllm:request_inference_time_seconds": "histogram",
        "vllm:request_prefill_time_seconds": "histogram",
        "vllm:request_decode_time_seconds": "histogram",
        "vllm:time_to_first_token_seconds": "histogram",
        "vllm:inter_token_latency_seconds": "histogram",
        "vllm:request_time_per_output_token_seconds": "histogram",
        "vllm:request_prompt_tokens": "histogram",
        "vllm:request_generation_tokens": "histogram",
        "vllm:request_max_num_generation_tokens": "histogram",
        "vllm:request_params_max_tokens": "histogram",
        "vllm:request_params_n": "histogram",
        "vllm:request_prefill_kv_computed_tokens": "histogram",
    ]

    private static let dashboardHistogramFamilies = [
        "vllm:e2e_request_latency_seconds",
        "vllm:inter_token_latency_seconds",
        "vllm:time_to_first_token_seconds",
        "vllm:request_decode_time_seconds",
        "vllm:request_max_num_generation_tokens",
        "vllm:request_prefill_time_seconds",
        "vllm:request_queue_time_seconds",
        "vllm:request_generation_tokens",
        "vllm:request_prompt_tokens",
        "vllm:request_prefill_kv_computed_tokens",
    ]

    private static let intentionallyUnsupportedFamilies = [
        "vllm:iteration_tokens_total",
        "vllm:spec_decode_num_accepted_tokens_per_pos_total",
        "vllm:corrupted_requests_total",
        "vllm:lora_requests_info",
        "vllm:cache_config_info",
        "vllm:kv_block_lifetime_seconds",
        "vllm:kv_block_idle_before_evict_seconds",
        "vllm:kv_block_reuse_gap_seconds",
        "vllm:estimated_flops_per_gpu_total",
        "vllm:estimated_read_bytes_per_gpu_total",
        "vllm:estimated_write_bytes_per_gpu_total",
    ]

    private static func vLLMFamilies(inPromQL query: String) -> [String] {
        let pattern = #"vllm:[A-Za-z0-9_:]+"#
        guard let expression = try? NSRegularExpression(pattern: pattern) else { return [] }
        let range = NSRange(query.startIndex..<query.endIndex, in: query)
        let names = expression.matches(in: query, range: range).compactMap { match -> String? in
            guard let swiftRange = Range(match.range, in: query) else { return nil }
            var name = String(query[swiftRange])
            for suffix in ["_bucket", "_count", "_sum"] where name.hasSuffix(suffix) {
                name.removeLast(suffix.count)
                break
            }
            return name
        }
        return Array(Set(names)).sorted()
    }

    private static func expectedLabels(for sampleName: String) -> Set<String> {
        if sampleName == "vllm:request_success_total" {
            return ["model_name", "engine", "finished_reason"]
        }
        if sampleName == "vllm:num_requests_waiting_by_reason" {
            return ["model_name", "engine", "reason"]
        }
        if sampleName == "vllm:engine_sleep_state" {
            return ["model_name", "engine", "sleep_state"]
        }
        if sampleName == "vllm:prompt_tokens_by_source_total" {
            return ["model_name", "engine", "source"]
        }
        if sampleName.hasSuffix("_bucket") {
            return ["model_name", "engine", "le"]
        }
        return ["model_name", "engine"]
    }
}

private struct PrometheusExposition {
    struct Sample {
        let name: String
        let labels: [String: String]
    }

    var types: [String: String] = [:]
    var helps: [String: String] = [:]
    var samples: [Sample] = []

    init(_ text: String) {
        for rawLine in text.split(separator: "\n") {
            let line = String(rawLine)
            if line.hasPrefix("# HELP ") {
                let remainder = line.dropFirst("# HELP ".count)
                let fields = remainder.split(separator: " ", maxSplits: 1)
                if fields.count == 2 {
                    helps[String(fields[0])] = String(fields[1])
                }
                continue
            }
            if line.hasPrefix("# TYPE ") {
                let fields = line.split(separator: " ")
                if fields.count == 4 {
                    types[String(fields[2])] = String(fields[3])
                }
                continue
            }
            guard !line.hasPrefix("#"), !line.isEmpty else { continue }
            let metricAndLabels = line.split(separator: " ", maxSplits: 1)[0]
            let token = String(metricAndLabels)
            guard let open = token.firstIndex(of: "{") else {
                samples.append(Sample(name: token, labels: [:]))
                continue
            }
            let name = String(token[..<open])
            let labelStart = token.index(after: open)
            let close = token.lastIndex(of: "}") ?? token.endIndex
            let labelText = token[labelStart..<close]
            var labels: [String: String] = [:]
            for pair in labelText.split(separator: ",") {
                let fields = pair.split(separator: "=", maxSplits: 1)
                if fields.count == 2 {
                    labels[String(fields[0])] = String(fields[1])
                }
            }
            samples.append(Sample(name: name, labels: labels))
        }
    }

    func samples(for family: String) -> [Sample] {
        samples.filter {
            $0.name == family
                || $0.name == "\(family)_bucket"
                || $0.name == "\(family)_sum"
                || $0.name == "\(family)_count"
        }
    }

    func bucketBounds(for family: String) -> [String] {
        samples
            .filter { $0.name == "\(family)_bucket" }
            .compactMap { $0.labels["le"]?.trimmingCharacters(in: CharacterSet(charactersIn: "\"")) }
    }
}
