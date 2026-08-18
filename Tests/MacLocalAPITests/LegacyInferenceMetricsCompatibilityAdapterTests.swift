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
}
