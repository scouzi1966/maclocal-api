@testable import AFMKitMLX
@testable import AFMServer
import XCTest

final class AFMServerOwnershipTests: XCTestCase {
    func testActiveConnectionTrackerMaintainsCurrentAndPeakCounts() {
        let tracker = ActiveConnectionTracker()

        DispatchQueue.concurrentPerform(iterations: 32) { _ in
            tracker.connectionStarted()
        }

        var snapshot = tracker.snapshot()
        XCTAssertEqual(snapshot.activeConnections, 32)
        XCTAssertEqual(snapshot.activeConnectionsPeak, 32)

        DispatchQueue.concurrentPerform(iterations: 32) { _ in
            tracker.connectionEnded()
        }
        tracker.connectionEnded()

        snapshot = tracker.snapshot()
        XCTAssertEqual(snapshot.activeConnections, 0)
        XCTAssertEqual(snapshot.activeConnectionsPeak, 32)
    }

    func testPrometheusRenderingUsesServerOwnedConnectionSnapshot() {
        let tracker = ActiveConnectionTracker()
        tracker.connectionStarted()
        tracker.connectionStarted()

        let body = MetricsController.renderPrometheus(
            StatsAggregator.shared.snapshot(),
            connections: tracker.snapshot()
        )
        let lines = body.split(separator: "\n").map(String.init)

        XCTAssertTrue(lines.contains {
            $0.hasPrefix("afm:num_active_connections{") && $0.hasSuffix("} 2")
        })
        XCTAssertTrue(lines.contains {
            $0.hasPrefix("afm:active_connections_peak{") && $0.hasSuffix("} 2")
        })
    }

    func testConcreteMLXAdapterPreservesProviderServingConfiguration() {
        let service = MLXModelService(resolver: MLXCacheResolver())
        service.maxConcurrent = 8
        service.toolCallParser = "afm_adaptive_xml"
        service.enableGrammarConstraints = true
        service.fixToolArgs = true
        let model = AFMMLXModel(
            modelID: "test/concrete-adapter",
            attachedService: service
        )

        let adapter = AFMKitMLXChatServingAdapter(model: model)

        XCTAssertEqual(adapter.maxConcurrent, 8)
        XCTAssertEqual(adapter.servingConfiguration.toolCallParser, "afm_adaptive_xml")
        XCTAssertTrue(adapter.servingConfiguration.grammarConstraintsEnabled)
        XCTAssertTrue(adapter.servingConfiguration.fixToolArguments)
        XCTAssertTrue(adapter.tryReserveSlot())
        adapter.releaseSlot()
    }
}
