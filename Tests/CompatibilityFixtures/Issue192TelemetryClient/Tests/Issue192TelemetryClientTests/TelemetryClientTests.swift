import AFMKitCore
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
}
