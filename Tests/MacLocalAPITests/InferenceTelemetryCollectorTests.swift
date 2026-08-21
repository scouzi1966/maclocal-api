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

    func testOneTokenRequestRecordsVLLMCompatibleZeroTPOT() {
        let collector = InferenceTelemetryCollector(now: { 20 }, wallTime: { 1_000 })
        let token = collector.requestAccepted(at: 10)
        collector.requestStarted(token, at: 11)
        collector.outputToken(token, at: 12)
        XCTAssertTrue(collector.requestFinished(
            token,
            observation: AFMInferenceRequestFinishObservation(
                reason: .stop,
                completedAt: 13,
                fullPromptTokens: 2,
                computedPromptTokens: 2,
                generatedTokens: 1
            )
        ))

        let tpot = collector.metricsSnapshot().timePerOutputToken
        XCTAssertEqual(tpot.count, 1)
        XCTAssertEqual(tpot.sum, 0)
        XCTAssertEqual(tpot.bucketCounts.first, 1)
    }

    func testExplicitZeroMaximumOutputTokensIsObservedButUnspecifiedIsNot() {
        let collector = InferenceTelemetryCollector(now: { 20 }, wallTime: { 1_000 })
        let explicitZero = collector.requestAccepted(at: 10)
        collector.requestStarted(explicitZero, at: 11)
        XCTAssertTrue(collector.requestFinished(
            explicitZero,
            observation: AFMInferenceRequestFinishObservation(
                reason: .length,
                completedAt: 12,
                fullPromptTokens: 1,
                computedPromptTokens: 1,
                generatedTokens: 0,
                maximumOutputTokens: 0
            )
        ))

        let unspecified = collector.requestAccepted(at: 13)
        collector.requestStarted(unspecified, at: 14)
        XCTAssertTrue(collector.requestFinished(
            unspecified,
            observation: AFMInferenceRequestFinishObservation(
                reason: .stop,
                completedAt: 15,
                fullPromptTokens: 1,
                computedPromptTokens: 1,
                generatedTokens: 0
            )
        ))

        let histogram = collector.metricsSnapshot().maximumOutputTokens
        XCTAssertEqual(histogram.count, 1)
        XCTAssertEqual(histogram.sum, 0)
        XCTAssertEqual(histogram.bucketCounts.first, 1)
    }

    func testProviderEpochTimestampsCannotPinBoundedRollingGauges() {
        let clock = Clock(100)
        let collector = InferenceTelemetryCollector(
            now: { clock.read() },
            wallTime: { 1_000 }
        )
        let providerEpoch = 1_800_000_000.0
        let token = collector.requestAccepted(at: providerEpoch)
        collector.requestStarted(token, at: providerEpoch + 0.1)
        collector.promptTokensProcessed(
            token,
            fullPromptTokens: 4,
            computedPromptTokens: 4,
            at: providerEpoch + 0.2
        )
        for index in 0..<20_000 {
            collector.outputToken(token, at: providerEpoch + 0.3 + Double(index) / 1_000)
        }
        XCTAssertTrue(collector.requestFinished(
            token,
            observation: AFMInferenceRequestFinishObservation(
                reason: .length,
                completedAt: providerEpoch + 21,
                fullPromptTokens: 4,
                computedPromptTokens: 4,
                generatedTokens: 20_000
            )
        ))
        XCTAssertGreaterThan(doubleGauge("generation_throughput", in: collector.metricsSnapshot()), 0)

        clock.set(110.1)
        let decayed = collector.metricsSnapshot()
        XCTAssertEqual(doubleGauge("computed_prompt_throughput", in: decayed), 0)
        XCTAssertEqual(doubleGauge("generation_throughput", in: decayed), 0)
        XCTAssertEqual(doubleGauge("request_throughput", in: decayed), 0)
    }

    private func doubleGauge(
        _ name: String,
        in snapshot: AFMInferenceMetricsSnapshot
    ) -> Double {
        snapshot.supplementalDoubleGauges.first { $0.name == name }?.value ?? -1
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

    func testOutOfOrderRollingCallbacksPreserveNewerBuckets() {
        let clock = Clock(100)
        let collector = InferenceTelemetryCollector(
            now: { clock.read() },
            wallTime: { 1_000 }
        )
        collector.legacyAddGeneratedTokens(10)
        clock.set(101)
        collector.legacyAddGeneratedTokens(20)
        clock.set(100.5)
        collector.legacyAddGeneratedTokens(5)
        clock.set(101)

        XCTAssertEqual(rate("generation_throughput", in: collector.metricsSnapshot()), 3.5)
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

    func testResetDuringActiveRequestPreservesOwnershipAndCountsLaterCallbacks() {
        let clock = Clock(100)
        let collector = InferenceTelemetryCollector(
            now: { clock.read() },
            wallTime: { 1_000 }
        )
        collector.configure(
            modelName: "active-model",
            maximumConcurrentRequests: 4,
            maximumContextTokens: 4_096
        )
        collector.updateProviderState(AFMInferenceProviderState(
            runningRequests: 2,
            waitingRequests: 1,
            activeLogicalCachePositions: 50,
            logicalCacheCapacity: 100,
            memoryCacheUsage: 0.4,
            prefixCacheFill: 0.3
        ))
        let token = collector.requestAccepted(at: 90)
        collector.requestStarted(token, at: 91)
        collector.promptTokensProcessed(
            token,
            fullPromptTokens: 5,
            computedPromptTokens: 4,
            at: 92
        )
        collector.outputToken(token, at: 93)

        collector.reset()

        var snapshot = collector.metricsSnapshot()
        XCTAssertEqual(snapshot.modelName, "active-model")
        XCTAssertEqual(snapshot.maximumConcurrentRequests, 4)
        XCTAssertEqual(snapshot.maximumContextTokens, 4_096)
        XCTAssertEqual(snapshot.runningRequests, 2)
        XCTAssertEqual(snapshot.waitingRequests, 1)
        XCTAssertEqual(snapshot.peakRunningRequests, 2)
        XCTAssertEqual(snapshot.logicalCacheUsage, 0.5)
        XCTAssertEqual(snapshot.memoryCacheUsage, 0.4)
        XCTAssertEqual(snapshot.prefixCacheFill, 0.3)
        XCTAssertEqual(snapshot.acceptedRequestsTotal, 0)
        XCTAssertEqual(snapshot.terminalRequestsTotal, 0)
        XCTAssertEqual(snapshot.fullPromptTokensTotal, 0)
        XCTAssertEqual(snapshot.computedPromptTokensTotal, 0)
        XCTAssertEqual(snapshot.generatedTokensTotal, 0)
        XCTAssertEqual(snapshot.endToEndLatency.count, 0)
        XCTAssertEqual(snapshot.interTokenLatency.count, 0)

        clock.set(101)
        collector.promptTokensProcessed(
            token,
            fullPromptTokens: 7,
            computedPromptTokens: 6,
            at: 101
        )
        collector.outputToken(token, at: 102)
        XCTAssertTrue(collector.requestFinished(
            token,
            observation: AFMInferenceRequestFinishObservation(
                reason: .length,
                completedAt: 103,
                fullPromptTokens: 7,
                computedPromptTokens: 6,
                generatedTokens: 2,
                maximumOutputTokens: 2
            )
        ))

        snapshot = collector.metricsSnapshot()
        XCTAssertEqual(snapshot.acceptedRequestsTotal, 0)
        XCTAssertEqual(snapshot.terminalRequestsTotal, 1)
        XCTAssertEqual(snapshot.fullPromptTokensTotal, 2)
        XCTAssertEqual(snapshot.computedPromptTokensTotal, 2)
        XCTAssertEqual(snapshot.generatedTokensTotal, 1)
        XCTAssertEqual(snapshot.terminalCounts.first { $0.name == "length" }?.count, 1)
        XCTAssertEqual(snapshot.endToEndLatency.count, 1)
        XCTAssertEqual(snapshot.interTokenLatency.count, 1)
        XCTAssertEqual(snapshot.maximumGeneratedTokens.sum, 2)
    }

    func testResetDuringActiveRequestPreservesTokenForLaterFailure() {
        let collector = InferenceTelemetryCollector(now: { 20 }, wallTime: { 1_000 })
        let token = collector.requestAccepted(at: 10)
        collector.requestStarted(token, at: 11)
        collector.outputToken(token, at: 12)

        collector.reset()

        collector.outputToken(token, at: 13)
        XCTAssertTrue(collector.requestFailed(token, reason: .cancelled, at: 14))
        XCTAssertFalse(collector.requestFailed(token, reason: .cancelled, at: 14))

        let snapshot = collector.metricsSnapshot()
        XCTAssertEqual(snapshot.generatedTokensTotal, 1)
        XCTAssertEqual(snapshot.terminalRequestsTotal, 1)
        XCTAssertEqual(snapshot.terminalCounts.first { $0.name == "abort" }?.count, 1)
        XCTAssertEqual(snapshot.failureCounts.first { $0.name == "cancelled" }?.count, 1)
    }

    private func rate(_ name: String, in snapshot: AFMInferenceMetricsSnapshot) -> Double? {
        snapshot.supplementalDoubleGauges.first { $0.name == name }?.value
    }
}
