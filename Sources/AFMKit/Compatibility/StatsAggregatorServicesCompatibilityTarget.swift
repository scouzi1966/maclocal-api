import AFMKitCore
import AFMKitMLX
import AFMKitServices

/// Binds the deprecated MLX metrics facade to the process Services collector.
public final class StatsAggregatorServicesCompatibilityTarget:
    StatsAggregatorCompatibilityTarget,
    @unchecked Sendable
{
    public let adapter: LegacyInferenceMetricsCompatibilityAdapter

    public init(adapter: LegacyInferenceMetricsCompatibilityAdapter) {
        self.adapter = adapter
    }

    public convenience init(collector: InferenceTelemetryCollector) {
        self.init(adapter: LegacyInferenceMetricsCompatibilityAdapter(collector: collector))
    }

    public func setModel(_ name: String, maxConcurrent: Int) {
        adapter.setModel(name, maxConcurrent: maxConcurrent)
    }

    public func registerGaugeReaders(
        running: @escaping StatsAggregator.GaugeReader,
        waiting: @escaping StatsAggregator.GaugeReader
    ) {
        adapter.registerGaugeReaders(running: running, waiting: waiting)
    }

    public func registerGpuCacheUsageReader(_ reader: @escaping StatsAggregator.FractionReader) {
        adapter.registerMemoryCacheUsageReader(reader)
    }

    public func registerRadixCacheFillReader(_ reader: @escaping StatsAggregator.FractionReader) {
        adapter.registerPrefixCacheFillReader(reader)
    }

    public func connectionStarted() { adapter.connectionStarted() }
    public func connectionEnded() { adapter.connectionEnded() }
    public func reset() { adapter.reset() }
    public func addGenTokens(_ count: Int) { adapter.addGeneratedTokens(count) }
    public func addPromptTokens(_ count: Int) { adapter.addComputedPromptTokens(count) }
    public func requestStarted() { adapter.requestStarted() }
    public func requestCompleted() { adapter.requestCompleted() }
    public func cacheHit() { adapter.cacheHit() }
    public func cacheMiss() { adapter.cacheMiss() }
    public func requestSucceeded(reason: String) { adapter.requestSucceeded(reason: reason) }

    public func observeRequest(_ observation: StatsAggregator.RequestObservation) {
        adapter.observeRequest(
            queuedAt: observation.queuedAt,
            startedAt: observation.startedAt,
            firstTokenAt: observation.firstTokenAt,
            completedAt: observation.completedAt,
            promptTokens: observation.promptTokens,
            generationTokens: observation.generationTokens,
            samplingN: observation.paramsN,
            samplingBestOf: observation.paramsBestOf
        )
    }

    public func observeE2eLatency(_ seconds: Double) {
        adapter.observeEndToEndLatency(seconds)
    }

    public func observeTimeToFirstToken(_ seconds: Double) {
        adapter.observeTimeToFirstToken(seconds)
    }

    public func observeTimePerOutputToken(_ seconds: Double) {
        adapter.observeTimePerOutputToken(seconds)
    }

    public func observePromptTokens(_ count: Int) {
        adapter.observeComputedPromptTokens(count)
    }

    public func observeGenerationTokens(_ count: Int) {
        adapter.observeGeneratedTokens(count)
    }

    public func metricsSnapshot() -> AFMInferenceMetricsSnapshot {
        adapter.metricsSnapshotWithLegacyGauges()
    }
}
