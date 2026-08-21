import AFMKitCore
import AFMKitServices
import XCTest

@testable import AFMKitDwarfStar
@testable import AFMKitMLX
@testable import AFMServer

final class ServerDefaultTelemetryCompositionTests: XCTestCase {
    func testDefaultMLXServiceSharesServerCollector() {
        let service = MLXModelService(resolver: MLXCacheResolver())
        let telemetry = Server.composeTelemetry(
            telemetry: nil,
            mlxModelService: service,
            afmModel: nil
        )

        recordCompletedRequest(using: service.telemetryObserver)

        assertProviderMetrics(in: telemetry.metricsSnapshot())
    }

    func testDefaultDwarfStarModelSharesServerCollectorThroughTypeErasure() {
        let model = AFMDwarfStarModel(
            modelID: "default-telemetry-test",
            modelPath: "/nonexistent/default-telemetry-test.gguf"
        )
        let telemetry = Server.composeTelemetry(
            telemetry: nil,
            mlxModelService: nil,
            afmModel: AnyAFMModel(model)
        )

        recordCompletedRequest(using: model.telemetryObserver)

        assertProviderMetrics(in: telemetry.metricsSnapshot())
    }

    private func recordCompletedRequest(
        using observer: any AFMInferenceTelemetryObserving
    ) {
        let token = observer.requestAccepted(at: 10)
        observer.requestStarted(token, at: 11)
        observer.promptTokensProcessed(
            token,
            fullPromptTokens: 3,
            computedPromptTokens: 2,
            at: 12
        )
        observer.outputToken(token, at: 13)
        XCTAssertTrue(observer.requestFinished(
            token,
            observation: AFMInferenceRequestFinishObservation(
                reason: .stop,
                completedAt: 15,
                fullPromptTokens: 3,
                computedPromptTokens: 2,
                generatedTokens: 1
            )
        ))
    }

    private func assertProviderMetrics(
        in snapshot: AFMInferenceMetricsSnapshot,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(snapshot.acceptedRequestsTotal, 1, file: file, line: line)
        XCTAssertEqual(snapshot.fullPromptTokensTotal, 3, file: file, line: line)
        XCTAssertEqual(snapshot.generatedTokensTotal, 1, file: file, line: line)
        XCTAssertEqual(snapshot.endToEndLatency.count, 1, file: file, line: line)
        XCTAssertEqual(snapshot.timeToFirstToken.count, 1, file: file, line: line)

        let metrics = MetricsController.renderPrometheus(snapshot)
        let labels = #"{model_name="",engine="0"}"#
        XCTAssertTrue(
            metrics.contains("vllm:prompt_tokens_total\(labels) 3\n"),
            file: file,
            line: line
        )
        XCTAssertTrue(
            metrics.contains("vllm:generation_tokens_total\(labels) 1\n"),
            file: file,
            line: line
        )
        XCTAssertTrue(
            metrics.contains("vllm:e2e_request_latency_seconds_count\(labels) 1\n"),
            file: file,
            line: line
        )
    }
}
