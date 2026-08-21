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
    private let configureHandler: @Sendable (String, Int, Int) -> Void
    let providerTelemetryObserver: (any AFMInferenceTelemetryObserving)?

    public init(
        snapshotSource: any AFMInferenceMetricsSnapshotSource,
        ingressRecorder: any AFMIngressTelemetryRecording
    ) {
        self.snapshotSource = snapshotSource
        self.ingressRecorder = ingressRecorder
        self.configureHandler = { _, _, _ in }
        self.providerTelemetryObserver = nil
    }

    public init(
        snapshotSource: any AFMInferenceMetricsSnapshotSource,
        ingressRecorder: any AFMIngressTelemetryRecording,
        providerTelemetryObserver: any AFMInferenceTelemetryObserving
    ) {
        self.snapshotSource = snapshotSource
        self.ingressRecorder = ingressRecorder
        self.configureHandler = { _, _, _ in }
        self.providerTelemetryObserver = providerTelemetryObserver
    }

    public init(collector: InferenceTelemetryCollector) {
        self.snapshotSource = collector
        self.ingressRecorder = collector
        self.providerTelemetryObserver = collector
        self.configureHandler = {
            modelName,
            maximumConcurrentRequests,
            maximumContextTokens in
            collector.configure(
                modelName: modelName,
                maximumConcurrentRequests: maximumConcurrentRequests,
                maximumContextTokens: maximumContextTokens
            )
        }
    }

    static func standalone() -> Self {
        Self(collector: InferenceTelemetryCollector())
    }

    public func configure(
        modelName: String,
        maximumConcurrentRequests: Int,
        maximumContextTokens: Int = 0
    ) {
        configureHandler(modelName, maximumConcurrentRequests, maximumContextTokens)
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
