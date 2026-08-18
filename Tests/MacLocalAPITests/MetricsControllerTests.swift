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
        collector.configure(modelName: "model\"quoted", maximumConcurrentRequests: 8)
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
                    fullPromptTokens: 12,
                    computedPromptTokens: 5,
                    generatedTokens: 2,
                    samplingN: 2,
                    samplingBestOf: 3
                )
            )
        )

        let output = MetricsController.renderPrometheus(collector.metricsSnapshot())
        let modelLabel = #"{model_name="model\"quoted"}"#

        XCTAssertTrue(output.contains("afm:prompt_tokens_total\(modelLabel) 5\n"))
        XCTAssertTrue(output.contains("vllm:prompt_tokens_total\(modelLabel) 12\n"))
        XCTAssertTrue(output.contains("vllm:generation_tokens_total\(modelLabel) 2\n"))
        XCTAssertTrue(output.contains("vllm:num_requests_running\(modelLabel) 2\n"))
        XCTAssertTrue(output.contains("vllm:num_requests_waiting\(modelLabel) 3\n"))
        XCTAssertTrue(output.contains("vllm:kv_cache_usage_perc\(modelLabel) 0.5\n"))
        XCTAssertTrue(output.contains("vllm:prefix_cache_hit_rate\(modelLabel) 0.6\n"))
        XCTAssertTrue(output.contains("vllm:spec_decode_acceptance_rate\(modelLabel) 0.75\n"))
        XCTAssertTrue(output.contains("vllm:num_preemptions_total\(modelLabel) 1\n"))
        XCTAssertTrue(output.contains("afm:spec_decode_num_rejected_tokens_total\(modelLabel) 1\n"))
        XCTAssertTrue(output.contains("afm:num_active_connections\(modelLabel) 1\n"))
        XCTAssertTrue(
            output.contains(
                #"afm:request_failures_total{model_name="model\"quoted",status="capacity"} 1"#
            )
        )

        XCTAssertTrue(output.contains("afm:request_prompt_tokens_sum\(modelLabel) 5.0\n"))
        XCTAssertTrue(output.contains("vllm:request_prompt_tokens_sum\(modelLabel) 12.0\n"))
        XCTAssertTrue(output.contains("vllm:request_prompt_tokens_count\(modelLabel) 1\n"))
        XCTAssertTrue(output.contains("vllm:inter_token_latency_seconds_sum\(modelLabel) 1.0\n"))
        XCTAssertTrue(output.contains("vllm:avg_prompt_throughput_toks_per_s\(modelLabel) 0.5\n"))
        XCTAssertTrue(output.contains("vllm:avg_generation_throughput_toks_per_s\(modelLabel) 0.2\n"))

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

    private static let pinnedVLLMMetrics = [
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:kv_cache_usage_perc",
        "vllm:avg_prompt_throughput_toks_per_s",
        "vllm:avg_generation_throughput_toks_per_s",
        "vllm:prefix_cache_hit_rate",
        "vllm:spec_decode_acceptance_rate",
        "vllm:num_preemptions_total",
        "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
        "vllm:prefix_cache_queries_total",
        "vllm:prefix_cache_hits_total",
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
    ]
}
