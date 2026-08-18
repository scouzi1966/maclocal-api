import AFMKitCore
import AFMKitServices

/// Bridges AFMServer's HTTP lifecycle to the process-owned telemetry collector.
/// HTTP classification remains in AFMServer; mutable state remains in Services.
public struct AFMServerTelemetryAdapter:
    AFMInferenceMetricsSnapshotSource,
    AFMIngressTelemetryRecording,
    Sendable
{
    private let snapshotSource: any AFMInferenceMetricsSnapshotSource
    private let ingressRecorder: any AFMIngressTelemetryRecording
    private let configureHandler: @Sendable (String, Int) -> Void

    public init(
        snapshotSource: any AFMInferenceMetricsSnapshotSource,
        ingressRecorder: any AFMIngressTelemetryRecording
    ) {
        self.snapshotSource = snapshotSource
        self.ingressRecorder = ingressRecorder
        self.configureHandler = { _, _ in }
    }

    public init(collector: InferenceTelemetryCollector) {
        self.snapshotSource = collector
        self.ingressRecorder = collector
        self.configureHandler = { modelName, maximumConcurrentRequests in
            collector.configure(
                modelName: modelName,
                maximumConcurrentRequests: maximumConcurrentRequests
            )
        }
    }

    public func configure(modelName: String, maximumConcurrentRequests: Int) {
        configureHandler(modelName, maximumConcurrentRequests)
    }

    public func metricsSnapshot() -> AFMInferenceMetricsSnapshot {
        snapshotSource.metricsSnapshot()
    }

    public func recordRejection(_ reason: AFMIngressRejectionReason) {
        ingressRecorder.recordRejection(reason)
    }

    public func connectionOpened() -> AFMIngressConnectionToken {
        ingressRecorder.connectionOpened()
    }

    public func connectionClosed(_ token: AFMIngressConnectionToken) {
        ingressRecorder.connectionClosed(token)
    }
}
