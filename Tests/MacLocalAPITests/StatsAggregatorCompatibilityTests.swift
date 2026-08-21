import XCTest

@testable import AFMKit
@testable import AFMKitMLX
@testable import AFMKitServices

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

    func testBoundServicesFacadeResetPreservesLegacyResetContract() {
        let collector = InferenceTelemetryCollector(now: { 20 }, wallTime: { 1_000 })
        let aggregator = StatsAggregator()
        XCTAssertTrue(aggregator.installCompatibilityTarget(
            StatsAggregatorServicesCompatibilityTarget(collector: collector)
        ))

        aggregator.setModel("bound-model", maxConcurrent: 8)
        aggregator.connectionStarted()
        aggregator.addGenTokens(7)
        aggregator.addPromptTokens(11)
        aggregator.requestStarted()
        aggregator.requestCompleted()
        aggregator.cacheHit()
        aggregator.cacheMiss()
        aggregator.requestSucceeded(reason: "stop")
        aggregator.observeRequest(.init(
            queuedAt: 1,
            startedAt: 2,
            firstTokenAt: 3,
            completedAt: 5,
            promptTokens: 11,
            generationTokens: 7
        ))

        aggregator.reset()

        let snapshot = aggregator.snapshot()
        XCTAssertEqual(snapshot.modelName, "bound-model")
        XCTAssertEqual(snapshot.maxConcurrent, 8)
        XCTAssertEqual(snapshot.activeConnections, 1)
        XCTAssertEqual(snapshot.activeConnectionsPeak, 1)
        XCTAssertEqual(snapshot.genTokensTotal, 0)
        XCTAssertEqual(snapshot.promptTokensTotal, 0)
        XCTAssertEqual(snapshot.requestsStartedTotal, 0)
        XCTAssertEqual(snapshot.requestsCompletedTotal, 0)
        XCTAssertEqual(snapshot.cacheHitsTotal, 0)
        XCTAssertEqual(snapshot.cacheMissesTotal, 0)
        XCTAssertTrue(snapshot.requestSuccessByReason.isEmpty)
        XCTAssertEqual(snapshot.e2eLatency.count, 0)
        XCTAssertEqual(snapshot.timeToFirstToken.count, 0)
        XCTAssertEqual(snapshot.promptTokens.count, 0)
        XCTAssertEqual(snapshot.generationTokens.count, 0)
    }

    func testBoundServicesFacadePreservesLegacyHistogramBucketIdentityAndCounts() {
        let collector = InferenceTelemetryCollector(now: { 20 }, wallTime: { 1_000 })
        let aggregator = StatsAggregator()
        XCTAssertTrue(aggregator.installCompatibilityTarget(
            StatsAggregatorServicesCompatibilityTarget(collector: collector)
        ))

        aggregator.observeRequest(.init(
            queuedAt: 1,
            startedAt: 2,
            firstTokenAt: 3,
            completedAt: 125,
            promptTokens: 150_000,
            generationTokens: 150_000
        ))
        aggregator.observeE2eLatency(0.4)
        aggregator.observeTimeToFirstToken(0.05)
        aggregator.observeTimePerOutputToken(0.2)
        aggregator.observePromptTokens(50)
        aggregator.observeGenerationTokens(2)

        let snapshot = aggregator.snapshot()
        assertHistogram(
            snapshot.e2eLatency,
            hasBuckets: StatsAggregator.Buckets.requestLatency
        )
        assertHistogram(snapshot.queueTime, hasBuckets: StatsAggregator.Buckets.requestLatency)
        assertHistogram(
            snapshot.inferenceTime,
            hasBuckets: StatsAggregator.Buckets.requestLatency
        )
        assertHistogram(snapshot.prefillTime, hasBuckets: StatsAggregator.Buckets.requestLatency)
        assertHistogram(snapshot.decodeTime, hasBuckets: StatsAggregator.Buckets.requestLatency)
        assertHistogram(
            snapshot.timeToFirstToken,
            hasBuckets: StatsAggregator.Buckets.timeToFirstToken
        )
        assertHistogram(
            snapshot.timePerOutputToken,
            hasBuckets: StatsAggregator.Buckets.timePerOutputToken
        )
        assertHistogram(snapshot.promptTokens, hasBuckets: StatsAggregator.Buckets.tokenCount)
        assertHistogram(
            snapshot.generationTokens,
            hasBuckets: StatsAggregator.Buckets.tokenCount
        )
        assertHistogram(snapshot.paramsN, hasBuckets: StatsAggregator.Buckets.samplingParam)
        assertHistogram(snapshot.paramsBestOf, hasBuckets: StatsAggregator.Buckets.samplingParam)

        XCTAssertEqual(snapshot.e2eLatency.bucketCounts[1], 1)
        XCTAssertEqual(snapshot.e2eLatency.bucketCounts.last, 2)
        XCTAssertEqual(snapshot.timeToFirstToken.bucketCounts[5], 1)
        XCTAssertEqual(snapshot.timeToFirstToken.bucketCounts.last, 2)
        XCTAssertEqual(snapshot.timePerOutputToken.bucketCounts[6], 2)
        XCTAssertEqual(snapshot.timePerOutputToken.bucketCounts.last, 2)
        XCTAssertEqual(snapshot.promptTokens.bucketCounts[5], 1)
        XCTAssertEqual(snapshot.promptTokens.bucketCounts.last, 2)
        XCTAssertEqual(snapshot.generationTokens.bucketCounts[1], 1)
        XCTAssertEqual(snapshot.generationTokens.bucketCounts.last, 2)
    }

    func testBoundServicesFacadeRebucketsTruncatedCollectorTokenHistograms() throws {
        let collector = InferenceTelemetryCollector(now: { 20 }, wallTime: { 1_000 })
        collector.configure(
            modelName: "small-context",
            maximumConcurrentRequests: 1,
            maximumContextTokens: 4_096
        )
        let aggregator = StatsAggregator()
        XCTAssertTrue(aggregator.installCompatibilityTarget(
            StatsAggregatorServicesCompatibilityTarget(collector: collector)
        ))

        aggregator.observePromptTokens(3_000)
        aggregator.observeGenerationTokens(3_000)

        let snapshot = aggregator.snapshot()
        let fiveThousandIndex = try XCTUnwrap(
            StatsAggregator.Buckets.tokenCount.firstIndex(of: 5_000)
        )
        assertHistogram(snapshot.promptTokens, hasBuckets: StatsAggregator.Buckets.tokenCount)
        assertHistogram(
            snapshot.generationTokens,
            hasBuckets: StatsAggregator.Buckets.tokenCount
        )
        XCTAssertEqual(snapshot.promptTokens.bucketCounts[fiveThousandIndex], 1)
        XCTAssertEqual(snapshot.generationTokens.bucketCounts[fiveThousandIndex], 1)
    }

    private func assertHistogram(
        _ histogram: StatsAggregator.Histogram,
        hasBuckets buckets: [Double],
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(histogram.buckets, buckets, file: file, line: line)
        XCTAssertEqual(
            histogram.bucketCounts.count,
            buckets.count + 1,
            file: file,
            line: line
        )
    }
}
