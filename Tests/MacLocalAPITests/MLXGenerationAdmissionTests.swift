import AFMKitCore
import AFMKitServices
import XCTest

@testable import AFMKitMLX

final class MLXGenerationAdmissionTests: XCTestCase {
    func testAbandonedLeaseFinalizesTelemetryExactlyOnce() {
        let collector = InferenceTelemetryCollector()
        let token = collector.requestAccepted(at: 1)
        collector.requestStarted(token, at: 2)
        let lease = AFMGenerationLease(telemetryToken: token) {} onAbandon: {
            _ = collector.requestFailed(token, reason: .internal, at: 3)
        }

        lease.release()
        lease.release()

        let snapshot = collector.metricsSnapshot()
        XCTAssertEqual(snapshot.acceptedRequestsTotal, 1)
        XCTAssertEqual(snapshot.terminalRequestsTotal, 1)
        XCTAssertEqual(snapshot.terminalCounts.first { $0.name == "error" }?.count, 1)
        XCTAssertEqual(snapshot.failureCounts.first { $0.name == "internal" }?.count, 1)
    }

    func testSerialAdmissionQueuesUntilCapacityAndRecordsQueueLatency() async throws {
        let collector = InferenceTelemetryCollector()
        let service = MLXModelService(
            resolver: MLXCacheResolver(),
            telemetryObserver: collector
        )
        let first = try await service.admitGeneration(timeout: .seconds(1))
        let waiter = Task {
            try await service.admitGeneration(timeout: .seconds(1))
        }

        try await waitForState(collector) { snapshot in
            snapshot.runningRequests == 1 && snapshot.waitingRequests == 1
        }
        try await Task.sleep(for: .milliseconds(30))
        first.release()
        let second = try await waiter.value
        second.release()

        let snapshot = collector.metricsSnapshot()
        XCTAssertEqual(snapshot.acceptedRequestsTotal, 2)
        XCTAssertEqual(snapshot.terminalRequestsTotal, 2)
        XCTAssertEqual(snapshot.runningRequests, 0)
        XCTAssertEqual(snapshot.waitingRequests, 0)
        XCTAssertEqual(snapshot.queueLatency.count, 2)
        XCTAssertGreaterThan(snapshot.queueLatency.sum, 0.02)
    }

    func testSerialAdmissionTimeoutIsProviderFailureNotIngressCapacity() async throws {
        let collector = InferenceTelemetryCollector()
        let service = MLXModelService(
            resolver: MLXCacheResolver(),
            telemetryObserver: collector
        )
        let occupied = try await service.admitGeneration(timeout: .seconds(1))

        do {
            _ = try await service.admitGeneration(timeout: .milliseconds(30))
            XCTFail("serial admission should honor its capacity timeout")
        } catch let error as AFMGenerationAdmissionError {
            XCTAssertEqual(error, .timedOut)
        }

        var snapshot = collector.metricsSnapshot()
        XCTAssertEqual(snapshot.acceptedRequestsTotal, 2)
        XCTAssertEqual(snapshot.terminalRequestsTotal, 1)
        XCTAssertEqual(snapshot.runningRequests, 1)
        XCTAssertEqual(snapshot.waitingRequests, 0)
        XCTAssertEqual(snapshot.failureCounts.first { $0.name == "inference" }?.count, 1)
        XCTAssertEqual(snapshot.failureCounts.first { $0.name == "capacity" }?.count, 0)

        occupied.release()
        snapshot = collector.metricsSnapshot()
        XCTAssertEqual(snapshot.terminalRequestsTotal, 2)
    }

    func testSerialAdmissionCancellationMapsToAbort() async throws {
        let collector = InferenceTelemetryCollector()
        let service = MLXModelService(
            resolver: MLXCacheResolver(),
            telemetryObserver: collector
        )
        let occupied = try await service.admitGeneration(timeout: .seconds(1))
        let waiter = Task {
            try await service.admitGeneration(timeout: .seconds(1))
        }

        try await waitForState(collector) { $0.waitingRequests == 1 }
        waiter.cancel()
        do {
            _ = try await waiter.value
            XCTFail("cancelled serial admission must throw")
        } catch let error as AFMGenerationAdmissionError {
            XCTAssertEqual(error, .cancelled)
        }

        let snapshot = collector.metricsSnapshot()
        XCTAssertEqual(snapshot.terminalCounts.first { $0.name == "abort" }?.count, 1)
        XCTAssertEqual(snapshot.failureCounts.first { $0.name == "cancelled" }?.count, 1)
        occupied.release()
    }

    func testSerialAdmissionDuringShutdownIsInternalFailure() async {
        let collector = InferenceTelemetryCollector()
        let service = MLXModelService(
            resolver: MLXCacheResolver(),
            telemetryObserver: collector
        )
        await service.shutdownAndReleaseResources()

        do {
            _ = try await service.admitGeneration(timeout: .seconds(1))
            XCTFail("shutdown service must reject admission")
        } catch let error as AFMGenerationAdmissionError {
            XCTAssertEqual(error, .internalFailure)
        } catch {
            XCTFail("unexpected shutdown admission error: \(error)")
        }

        let snapshot = collector.metricsSnapshot()
        XCTAssertEqual(snapshot.acceptedRequestsTotal, 1)
        XCTAssertEqual(snapshot.terminalRequestsTotal, 1)
        XCTAssertEqual(snapshot.failureCounts.first { $0.name == "internal" }?.count, 1)
    }

    private func waitForState(
        _ collector: InferenceTelemetryCollector,
        predicate: (AFMInferenceMetricsSnapshot) -> Bool
    ) async throws {
        let deadline = ContinuousClock.now + .seconds(1)
        while ContinuousClock.now < deadline {
            if predicate(collector.metricsSnapshot()) { return }
            try await Task.sleep(for: .milliseconds(5))
        }
        XCTFail("telemetry state did not reach expected value")
    }
}
