import XCTest

@testable import AFMKitMLX

final class StatsAggregatorCompatibilityTests: XCTestCase {
    private final class Box: @unchecked Sendable {
        let lock = NSLock()
        var running = 3
        var nestedRunning: Int?
        var aggregator: StatsAggregator?
    }

    func testDefaultCollectorPreservesLegacyBehaviorAndResetSemantics() {
        let aggregator = StatsAggregator()
        let box = Box()
        aggregator.setModel("legacy-model", maxConcurrent: 4)
        aggregator.registerGaugeReaders(
            running: { box.lock.withLock { box.running } },
            waiting: { 2 }
        )
        aggregator.registerGpuCacheUsageReader { 0.5 }
        aggregator.registerRadixCacheFillReader { 0.25 }
        aggregator.connectionStarted()
        aggregator.addGenTokens(3)
        aggregator.addPromptTokens(5)
        aggregator.requestStarted()
        aggregator.requestCompleted()
        aggregator.cacheHit()
        aggregator.cacheMiss()
        aggregator.requestSucceeded(reason: "tool calls")
        aggregator.observeRequest(.init(
            queuedAt: 1,
            startedAt: 2,
            firstTokenAt: 3,
            completedAt: 5,
            promptTokens: 5,
            generationTokens: 3
        ))

        var snapshot = aggregator.snapshot()
        XCTAssertEqual(snapshot.modelName, "legacy-model")
        XCTAssertEqual(snapshot.maxConcurrent, 4)
        XCTAssertEqual(snapshot.numRunning, 3)
        XCTAssertEqual(snapshot.numWaiting, 2)
        XCTAssertEqual(snapshot.batchSizePeak, 3)
        XCTAssertEqual(snapshot.activeConnections, 1)
        XCTAssertEqual(snapshot.genTokensTotal, 3)
        XCTAssertEqual(snapshot.promptTokensTotal, 5)
        XCTAssertEqual(snapshot.requestsStartedTotal, 1)
        XCTAssertEqual(snapshot.requestsCompletedTotal, 1)
        XCTAssertEqual(snapshot.cacheHitsTotal, 1)
        XCTAssertEqual(snapshot.cacheMissesTotal, 1)
        XCTAssertEqual(snapshot.requestSuccessByReason["tool_calls"], 1)
        XCTAssertEqual(snapshot.e2eLatency.count, 1)

        box.lock.withLock { box.running = 1 }
        aggregator.reset()
        snapshot = aggregator.snapshot()
        XCTAssertEqual(snapshot.modelName, "legacy-model")
        XCTAssertEqual(snapshot.maxConcurrent, 4)
        XCTAssertEqual(snapshot.numRunning, 1)
        XCTAssertEqual(snapshot.batchSizePeak, 1)
        XCTAssertEqual(snapshot.activeConnections, 1)
        XCTAssertEqual(snapshot.activeConnectionsPeak, 1)
        XCTAssertEqual(snapshot.genTokensTotal, 0)
        XCTAssertEqual(snapshot.promptTokensTotal, 0)
        XCTAssertEqual(snapshot.requestsStartedTotal, 0)
        XCTAssertEqual(snapshot.requestsCompletedTotal, 0)
        XCTAssertEqual(snapshot.e2eLatency.count, 0)
    }

    func testDefaultCollectorSnapshotIsReentrantAndLateBindingIsReplacementSafe() {
        let aggregator = StatsAggregator()
        let box = Box()
        box.aggregator = aggregator
        aggregator.registerGaugeReaders(
            running: {
                let nested = box.aggregator?.snapshot()
                box.lock.withLock { box.nestedRunning = nested?.numRunning }
                return 4
            },
            waiting: { 0 }
        )

        XCTAssertEqual(aggregator.snapshot().numRunning, 4)
        XCTAssertEqual(box.lock.withLock { box.nestedRunning }, 0)
        XCTAssertFalse(aggregator.installCompatibilityTarget(
            DefaultStatsAggregatorCompatibilityTarget()
        ))

        let unbound = StatsAggregator()
        unbound.reset()
        XCTAssertTrue(unbound.installCompatibilityTarget(
            DefaultStatsAggregatorCompatibilityTarget()
        ))
    }
}
