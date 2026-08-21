import AFMKitCore
import AFMKitServices
import XCTest

final class LegacyInferenceMetricsCompatibilityAdapterTests: XCTestCase {
    private final class Box: @unchecked Sendable {
        let lock = NSLock()
        var adapter: LegacyInferenceMetricsCompatibilityAdapter?
        var calls = 0
        var nestedRunning: Int?
        var running = 0
    }

    func testGaugeCallbacksOverlayOnlyCompatibilitySnapshot() {
        let collector = InferenceTelemetryCollector()
        collector.updateProviderState(
            AFMInferenceProviderState(runningRequests: 2, waitingRequests: 1)
        )
        let adapter = LegacyInferenceMetricsCompatibilityAdapter(collector: collector)
        let box = Box()
        adapter.registerGaugeReaders(
            running: {
                box.lock.withLock { box.calls += 1 }
                return 7
            },
            waiting: { 4 }
        )

        let compatibility = adapter.metricsSnapshotWithLegacyGauges()
        let authoritative = collector.metricsSnapshot()

        XCTAssertEqual(compatibility.runningRequests, 7)
        XCTAssertEqual(compatibility.waitingRequests, 4)
        XCTAssertEqual(compatibility.peakRunningRequests, 7)
        XCTAssertEqual(authoritative.runningRequests, 2)
        XCTAssertEqual(authoritative.waitingRequests, 1)
        XCTAssertEqual(box.lock.withLock { box.calls }, 1)
    }

    func testReentrantSnapshotSkipsCallbacksAndDoesNotDeadlock() {
        let collector = InferenceTelemetryCollector()
        collector.updateProviderState(
            AFMInferenceProviderState(runningRequests: 2, waitingRequests: 0)
        )
        let adapter = LegacyInferenceMetricsCompatibilityAdapter(collector: collector)
        let box = Box()
        box.adapter = adapter
        adapter.registerGaugeReaders(
            running: {
                box.lock.withLock { box.calls += 1 }
                let nested = box.adapter?.metricsSnapshotWithLegacyGauges()
                box.lock.withLock { box.nestedRunning = nested?.runningRequests }
                return 8
            },
            waiting: { 0 }
        )

        let outer = adapter.metricsSnapshotWithLegacyGauges()
        XCTAssertEqual(outer.runningRequests, 8)
        XCTAssertEqual(box.lock.withLock { box.calls }, 1)
        XCTAssertEqual(box.lock.withLock { box.nestedRunning }, 2)
    }

    func testResetClearsCompatibilityPeak() {
        let collector = InferenceTelemetryCollector()
        let adapter = LegacyInferenceMetricsCompatibilityAdapter(collector: collector)
        let box = Box()
        box.running = 5
        adapter.registerGaugeReaders(
            running: { box.lock.withLock { box.running } },
            waiting: { 0 }
        )
        XCTAssertEqual(adapter.metricsSnapshotWithLegacyGauges().peakRunningRequests, 5)

        box.lock.withLock { box.running = 1 }
        XCTAssertEqual(adapter.metricsSnapshotWithLegacyGauges().peakRunningRequests, 5)
        adapter.reset()
        XCTAssertEqual(adapter.metricsSnapshotWithLegacyGauges().peakRunningRequests, 1)
    }

    func testResetClearsLegacyCountersAndHistogramsButPreservesConfiguration() {
        let collector = InferenceTelemetryCollector(now: { 20 }, wallTime: { 1_000 })
        let adapter = LegacyInferenceMetricsCompatibilityAdapter(collector: collector)
        adapter.setModel("reset-model", maxConcurrent: 4)
        adapter.connectionStarted()
        adapter.addGeneratedTokens(3)
        adapter.addComputedPromptTokens(5)
        adapter.requestStarted()
        adapter.requestCompleted()
        adapter.cacheHit()
        adapter.cacheMiss()
        adapter.requestSucceeded(reason: "stop")
        adapter.observeEndToEndLatency(1.5)
        adapter.observeTimeToFirstToken(0.25)
        adapter.observeTimePerOutputToken(0.05)
        adapter.observeComputedPromptTokens(5)
        adapter.observeGeneratedTokens(3)

        adapter.reset()

        let snapshot = adapter.metricsSnapshotWithLegacyGauges()
        XCTAssertEqual(snapshot.modelName, "reset-model")
        XCTAssertEqual(snapshot.maximumConcurrentRequests, 4)
        XCTAssertEqual(
            snapshot.supplementalIntegerGauges.first { $0.name == "active_connections" }?.value,
            1
        )
        XCTAssertEqual(snapshot.generatedTokensTotal, 0)
        XCTAssertEqual(snapshot.computedPromptTokensTotal, 0)
        XCTAssertEqual(snapshot.acceptedRequestsTotal, 0)
        XCTAssertEqual(snapshot.terminalRequestsTotal, 0)
        XCTAssertEqual(snapshot.prefixCacheQueriesTotal, 0)
        XCTAssertEqual(snapshot.prefixCacheHitsTotal, 0)
        XCTAssertNil(snapshot.supplementalCounts.first { $0.name == "legacy_finish:stop" })
        XCTAssertEqual(snapshot.endToEndLatency.count, 0)
        XCTAssertEqual(snapshot.timeToFirstToken.count, 0)
        XCTAssertEqual(snapshot.timePerOutputToken.count, 0)
        XCTAssertEqual(snapshot.computedPromptTokens.count, 0)
        XCTAssertEqual(snapshot.generatedTokens.count, 0)
    }

    func testLegacyRequestsSkipZeroTokenHistogramsAndSingleTokenZeroTPOT() {
        let collector = InferenceTelemetryCollector(now: { 20 }, wallTime: { 1_000 })
        let adapter = LegacyInferenceMetricsCompatibilityAdapter(collector: collector)

        adapter.observeRequest(
            queuedAt: 1,
            startedAt: 2,
            firstTokenAt: nil,
            completedAt: 3,
            promptTokens: 1,
            generationTokens: 0,
            samplingN: 1,
            samplingBestOf: 1
        )

        var snapshot = collector.metricsSnapshot()
        XCTAssertEqual(snapshot.generatedTokens.count, 0)
        XCTAssertEqual(snapshot.maximumGeneratedTokens.count, 0)
        XCTAssertEqual(snapshot.timePerOutputToken.count, 0)

        adapter.observeRequest(
            queuedAt: 4,
            startedAt: 5,
            firstTokenAt: 6,
            completedAt: 7,
            promptTokens: 1,
            generationTokens: 1,
            samplingN: 1,
            samplingBestOf: 1
        )

        snapshot = collector.metricsSnapshot()
        XCTAssertEqual(snapshot.generatedTokens.count, 1)
        XCTAssertEqual(snapshot.generatedTokens.sum, 1)
        XCTAssertEqual(snapshot.maximumGeneratedTokens.count, 1)
        XCTAssertEqual(snapshot.maximumGeneratedTokens.sum, 1)
        XCTAssertEqual(snapshot.timePerOutputToken.count, 0)
    }
}
