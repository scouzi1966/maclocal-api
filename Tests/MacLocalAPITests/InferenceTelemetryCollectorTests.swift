import AFMKitCore
import AFMKitServices
import XCTest

final class InferenceTelemetryCollectorTests: XCTestCase {
    private final class Clock: @unchecked Sendable {
        private let lock = NSLock()
        private var value: Double

        init(_ value: Double) { self.value = value }

        func read() -> Double { lock.withLock { value } }
        func set(_ value: Double) { lock.withLock { self.value = value } }
    }

    func testLifecycleSeparatesFullAndComputedPromptTokensAndDeduplicatesTerminal() {
        let clock = Clock(100)
        let collector = InferenceTelemetryCollector(
            now: { clock.read() },
            wallTime: { 1_000 }
        )
        let token = collector.requestAccepted(at: 90)
        collector.requestStarted(token, at: 92)
        collector.outputToken(token, at: 94)
        collector.outputToken(token, at: 96)
        collector.prefixCacheObserved(queriedTokens: 10, hitTokens: 6)

        XCTAssertTrue(
            collector.requestFinished(
                token,
                observation: AFMInferenceRequestFinishObservation(
                    reason: .stop,
                    completedAt: 98,
                    fullPromptTokens: 10,
                    computedPromptTokens: 4,
                    generatedTokens: 2,
                    maximumOutputTokens: 64
                )
            )
        )
        XCTAssertFalse(
            collector.requestFinished(
                token,
                observation: AFMInferenceRequestFinishObservation(
                    reason: .error,
                    completedAt: 99,
                    fullPromptTokens: 10,
                    computedPromptTokens: 4,
                    generatedTokens: 2
                )
            )
        )

        let snapshot = collector.metricsSnapshot()
        XCTAssertEqual(snapshot.acceptedRequestsTotal, 1)
        XCTAssertEqual(snapshot.terminalRequestsTotal, 1)
        XCTAssertEqual(snapshot.fullPromptTokensTotal, 10)
        XCTAssertEqual(snapshot.computedPromptTokensTotal, 4)
        XCTAssertEqual(snapshot.generatedTokensTotal, 2)
        XCTAssertEqual(snapshot.maximumGeneratedTokens.count, 1)
        XCTAssertEqual(snapshot.maximumGeneratedTokens.sum, 2)
        XCTAssertEqual(snapshot.maximumOutputTokens.count, 1)
        XCTAssertEqual(snapshot.maximumOutputTokens.sum, 64)
        XCTAssertEqual(snapshot.prefixCacheQueriesTotal, 10)
        XCTAssertEqual(snapshot.prefixCacheHitsTotal, 6)
        XCTAssertEqual(snapshot.interTokenLatency.count, 1)
        XCTAssertEqual(snapshot.interTokenLatency.sum, 2, accuracy: 0.0001)
        XCTAssertEqual(snapshot.terminalCounts.first { $0.name == "stop" }?.count, 1)
        XCTAssertEqual(snapshot.terminalCounts.first { $0.name == "error" }?.count, 0)
    }

    func testTokenCountersAdvanceDuringRequestAndTerminalDoesNotDoubleCount() {
        let collector = InferenceTelemetryCollector(now: { 110 }, wallTime: { 1_000 })
        let token = collector.requestAccepted(at: 100)
        collector.requestStarted(token, at: 101)

        collector.promptTokensProcessed(
            token,
            fullPromptTokens: 10,
            computedPromptTokens: 4,
            at: 102
        )
        collector.promptTokensProcessed(
            token,
            fullPromptTokens: 12,
            computedPromptTokens: 5,
            at: 103
        )
        collector.promptTokensProcessed(
            token,
            fullPromptTokens: 11,
            computedPromptTokens: 4,
            at: 103.5
        )
        collector.outputToken(token, at: 104)
        collector.outputToken(token, at: 105)

        var snapshot = collector.metricsSnapshot()
        XCTAssertEqual(snapshot.fullPromptTokensTotal, 12)
        XCTAssertEqual(snapshot.computedPromptTokensTotal, 5)
        XCTAssertEqual(snapshot.generatedTokensTotal, 2)
        XCTAssertEqual(snapshot.maximumGeneratedTokens.count, 0)
        XCTAssertEqual(snapshot.maximumOutputTokens.count, 0)

        XCTAssertTrue(
            collector.requestFinished(
                token,
                observation: AFMInferenceRequestFinishObservation(
                    reason: .stop,
                    completedAt: 109,
                    fullPromptTokens: 12,
                    computedPromptTokens: 5,
                    generatedTokens: 2,
                    maximumOutputTokens: 64
                )
            )
        )

        snapshot = collector.metricsSnapshot()
        XCTAssertEqual(snapshot.fullPromptTokensTotal, 12)
        XCTAssertEqual(snapshot.computedPromptTokensTotal, 5)
        XCTAssertEqual(snapshot.generatedTokensTotal, 2)
        XCTAssertEqual(snapshot.maximumGeneratedTokens.count, 1)
        XCTAssertEqual(snapshot.maximumGeneratedTokens.sum, 2)
        XCTAssertEqual(snapshot.maximumOutputTokens.count, 1)
        XCTAssertEqual(snapshot.maximumOutputTokens.sum, 64)

        let repeatedSnapshot = collector.metricsSnapshot()
        XCTAssertEqual(repeatedSnapshot.fullPromptTokensTotal, 12)
        XCTAssertEqual(repeatedSnapshot.computedPromptTokensTotal, 5)
        XCTAssertEqual(repeatedSnapshot.generatedTokensTotal, 2)
    }

    func testFailedRequestRetainsAlreadyProcessedTokenCounters() {
        let collector = InferenceTelemetryCollector(now: { 20 }, wallTime: { 1_000 })
        let token = collector.requestAccepted(at: 10)
        collector.requestStarted(token, at: 11)
        collector.promptTokensProcessed(
            token,
            fullPromptTokens: 7,
            computedPromptTokens: 7,
            at: 12
        )
        collector.outputToken(token, at: 13)

        XCTAssertTrue(collector.requestFailed(token, reason: .inference, at: 14))

        let snapshot = collector.metricsSnapshot()
        XCTAssertEqual(snapshot.fullPromptTokensTotal, 7)
        XCTAssertEqual(snapshot.computedPromptTokensTotal, 7)
        XCTAssertEqual(snapshot.generatedTokensTotal, 1)
        XCTAssertEqual(snapshot.terminalRequestsTotal, 1)
        XCTAssertEqual(snapshot.terminalCounts.first { $0.name == "error" }?.count, 1)
    }

    func testFailureAndIngressWritesHaveDisjointOwnership() {
        let collector = InferenceTelemetryCollector(now: { 12 }, wallTime: { 100 })
        collector.recordRejection(.decode)
        collector.recordRejection(.capacity)
        var snapshot = collector.metricsSnapshot()
        XCTAssertEqual(snapshot.acceptedRequestsTotal, 0)
        XCTAssertEqual(snapshot.terminalRequestsTotal, 0)
        XCTAssertEqual(snapshot.failureCounts.first { $0.name == "decode" }?.count, 1)

        let token = collector.requestAccepted(at: 10)
        XCTAssertTrue(collector.requestFailed(token, reason: .cancelled, at: 12))
        XCTAssertFalse(collector.requestFailed(token, reason: .cancelled, at: 12))
        snapshot = collector.metricsSnapshot()
        XCTAssertEqual(snapshot.acceptedRequestsTotal, 1)
        XCTAssertEqual(snapshot.terminalRequestsTotal, 1)
        XCTAssertEqual(snapshot.failureCounts.first { $0.name == "cancelled" }?.count, 1)
        XCTAssertEqual(snapshot.terminalCounts.first { $0.name == "abort" }?.count, 1)
    }

    func testConnectionCloseIsIdempotentAndPeakIsStable() {
        let collector = InferenceTelemetryCollector()
        let first = collector.connectionOpened()
        let second = collector.connectionOpened()
        collector.connectionClosed(first)
        collector.connectionClosed(first)

        var snapshot = collector.metricsSnapshot()
        XCTAssertEqual(
            snapshot.supplementalIntegerGauges.first { $0.name == "active_connections" }?.value,
            1
        )
        XCTAssertEqual(
            snapshot.supplementalIntegerGauges.first {
                $0.name == "active_connections_peak"
            }?.value,
            2
        )

        collector.connectionClosed(second)
        snapshot = collector.metricsSnapshot()
        XCTAssertEqual(
            snapshot.supplementalIntegerGauges.first { $0.name == "active_connections" }?.value,
            0
        )
    }

    func testRollingRatesUseTenSecondWindowAndExpire() {
        let clock = Clock(100)
        let collector = InferenceTelemetryCollector(
            now: { clock.read() },
            wallTime: { 1_000 }
        )
        collector.legacyAddComputedPromptTokens(50)
        collector.legacyAddGeneratedTokens(20)
        collector.legacyRequestCompleted()

        var snapshot = collector.metricsSnapshot()
        XCTAssertEqual(rate("computed_prompt_throughput", in: snapshot), 5)
        XCTAssertEqual(rate("generation_throughput", in: snapshot), 2)
        XCTAssertEqual(rate("request_throughput", in: snapshot), 0.1)

        clock.set(111)
        snapshot = collector.metricsSnapshot()
        XCTAssertEqual(rate("computed_prompt_throughput", in: snapshot), 0)
        XCTAssertEqual(rate("generation_throughput", in: snapshot), 0)
        XCTAssertEqual(rate("request_throughput", in: snapshot), 0)
    }

    func testProviderStateClampsLogicalAndOptionalCacheGauges() {
        let collector = InferenceTelemetryCollector()
        collector.updateProviderState(
            AFMInferenceProviderState(
                runningRequests: 3,
                waitingRequests: 2,
                activeLogicalCachePositions: 150,
                logicalCacheCapacity: 100,
                memoryCacheUsage: -0.5,
                prefixCacheFill: 1.5
            )
        )

        let snapshot = collector.metricsSnapshot()
        XCTAssertEqual(snapshot.runningRequests, 3)
        XCTAssertEqual(snapshot.waitingRequests, 2)
        XCTAssertEqual(snapshot.peakRunningRequests, 3)
        XCTAssertEqual(snapshot.logicalCacheUsage, 1)
        XCTAssertEqual(snapshot.memoryCacheUsage, 0)
        XCTAssertEqual(snapshot.prefixCacheFill, 1)
    }

    private func rate(_ name: String, in snapshot: AFMInferenceMetricsSnapshot) -> Double? {
        snapshot.supplementalDoubleGauges.first { $0.name == name }?.value
    }
}
