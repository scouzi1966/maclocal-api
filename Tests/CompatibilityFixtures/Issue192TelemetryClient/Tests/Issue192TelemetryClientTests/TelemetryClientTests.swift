import AFMKitCore
import AFMKitMLX
import AFMOpenAICompat
import AFMKitServices
import XCTest

final class TelemetryClientTests: XCTestCase {
    func testExternalConsumerCanObserveLifecycleAndReadImmutableSnapshot() {
        let collector = InferenceTelemetryCollector(now: { 20 }, wallTime: { 100 })
        let observer: any AFMInferenceTelemetryObserving = collector
        let source: any AFMInferenceMetricsSnapshotSource = collector
        let token = observer.requestAccepted(at: 10)
        observer.requestStarted(token, at: 12)
        observer.outputToken(token, at: 14)
        XCTAssertTrue(
            observer.requestFinished(
                token,
                observation: AFMInferenceRequestFinishObservation(
                    reason: .stop,
                    completedAt: 18,
                    fullPromptTokens: 8,
                    computedPromptTokens: 3,
                    generatedTokens: 1
                )
            )
        )

        let snapshot = source.metricsSnapshot()
        XCTAssertEqual(snapshot.fullPromptTokensTotal, 8)
        XCTAssertEqual(snapshot.computedPromptTokensTotal, 3)
        XCTAssertEqual(snapshot.generatedTokensTotal, 1)
    }

    func testIngressCannotAllocateOrTerminateProviderRequests() {
        let collector = InferenceTelemetryCollector()
        let ingress: any AFMIngressTelemetryRecording = collector
        ingress.recordRejection(.validation)
        let connection = ingress.connectionOpened()
        ingress.connectionClosed(connection)
        ingress.connectionClosed(connection)

        let snapshot = collector.metricsSnapshot()
        XCTAssertEqual(snapshot.acceptedRequestsTotal, 0)
        XCTAssertEqual(snapshot.terminalRequestsTotal, 0)
        XCTAssertEqual(snapshot.failureCounts.first { $0.name == "validation" }?.count, 1)
    }

    func testGuideLLMExtensionFieldsDecodeWithoutChangingFinalUsagePolicy() throws {
        let data = Data(
            #"{"model":"test","messages":[{"role":"user","content":"hello"}],"ignore_eos":true,"stream":true,"stream_options":{"include_usage":true,"continuous_usage_stats":true}}"#.utf8
        )

        let request = try JSONDecoder().decode(ChatCompletionRequest.self, from: data)

        XCTAssertEqual(request.ignoreEOS, true)
        XCTAssertEqual(request.streamOptions?.includeUsage, true)
        XCTAssertEqual(request.streamOptions?.continuousUsageStats, true)
        XCTAssertEqual(request.includeStreamingUsage, true)
    }

    func testDeprecatedStatsAggregatorRemainsBehavioralWithoutAFMKitComposition() {
        let stats = StatsAggregator.shared
        stats.reset()
        stats.setModel("external-client", maxConcurrent: 2)
        stats.addGenTokens(3)
        stats.addPromptTokens(5)
        stats.requestStarted()
        stats.requestCompleted()
        stats.requestSucceeded(reason: "stop")
        stats.observeGenerationTokens(3)

        var histogram = StatsAggregator.Histogram(buckets: [1, 2])
        histogram.observe(1.5)
        XCTAssertEqual(histogram.count, 1)
        let snapshot = stats.snapshot()
        XCTAssertEqual(snapshot.modelName, "external-client")
        XCTAssertEqual(snapshot.maxConcurrent, 2)
        XCTAssertEqual(snapshot.genTokensTotal, 3)
        XCTAssertEqual(snapshot.promptTokensTotal, 5)
        XCTAssertEqual(snapshot.requestSuccessByReason["stop"], 1)
        XCTAssertEqual(snapshot.generationTokens.count, 1)

        stats.reset()
        XCTAssertEqual(stats.snapshot().genTokensTotal, 0)
    }
}
