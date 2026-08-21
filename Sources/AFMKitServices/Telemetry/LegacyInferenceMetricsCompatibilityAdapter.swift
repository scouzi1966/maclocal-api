import AFMKitCore
import Foundation
import os

public final class LegacyInferenceMetricsCompatibilityAdapter: @unchecked Sendable {
    public typealias GaugeReader = @Sendable () -> Int
    public typealias FractionReader = @Sendable () -> Double

    private struct Callbacks {
        var running: GaugeReader?
        var waiting: GaugeReader?
        var memoryCacheUsage: FractionReader?
        var prefixCacheFill: FractionReader?
        var legacyBatchSizePeak = 0
        var connectionTokens: [AFMIngressConnectionToken] = []
    }

    public let collector: InferenceTelemetryCollector
    private let callbacks = OSAllocatedUnfairLock(initialState: Callbacks())

    public init(collector: InferenceTelemetryCollector) {
        self.collector = collector
    }

    public func setModel(_ name: String, maxConcurrent: Int) {
        collector.configure(modelName: name, maximumConcurrentRequests: maxConcurrent)
    }

    public func registerGaugeReaders(
        running: @escaping GaugeReader,
        waiting: @escaping GaugeReader
    ) {
        callbacks.withLock {
            $0.running = running
            $0.waiting = waiting
        }
    }

    public func registerMemoryCacheUsageReader(_ reader: @escaping FractionReader) {
        callbacks.withLock { $0.memoryCacheUsage = reader }
    }

    public func registerPrefixCacheFillReader(_ reader: @escaping FractionReader) {
        callbacks.withLock { $0.prefixCacheFill = reader }
    }

    public func connectionStarted() {
        let token = collector.connectionOpened()
        callbacks.withLock { $0.connectionTokens.append(token) }
    }

    public func connectionEnded() {
        let token = callbacks.withLock { $0.connectionTokens.popLast() }
        if let token { collector.connectionClosed(token) }
    }

    public func reset() {
        // Preserve the public StatsAggregator reset contract for callers bound
        // through the compatibility facade. Collector reset intentionally keeps
        // model configuration and active connections while clearing request
        // counters, token totals, windows, and histograms.
        collector.reset()
        callbacks.withLock { $0.legacyBatchSizePeak = 0 }
    }

    public func addGeneratedTokens(_ count: Int) {
        collector.legacyAddGeneratedTokens(count)
    }

    public func addComputedPromptTokens(_ count: Int) {
        collector.legacyAddComputedPromptTokens(count)
    }

    public func requestStarted() { collector.legacyRequestStarted() }
    public func requestCompleted() { collector.legacyRequestCompleted() }
    public func cacheHit() { collector.legacyCacheHit() }
    public func cacheMiss() { collector.legacyCacheMiss() }
    public func requestSucceeded(reason: String) { collector.legacyRequestSucceeded(reason: reason) }

    public func observeRequest(
        queuedAt: Double,
        startedAt: Double?,
        firstTokenAt: Double?,
        completedAt: Double,
        promptTokens: Int,
        generationTokens: Int,
        samplingN: Int,
        samplingBestOf: Int
    ) {
        collector.legacyObserveRequest(
            queuedAt: queuedAt,
            startedAt: startedAt,
            firstTokenAt: firstTokenAt,
            completedAt: completedAt,
            promptTokens: promptTokens,
            generationTokens: generationTokens,
            samplingN: samplingN,
            samplingBestOf: samplingBestOf
        )
    }

    public func observeEndToEndLatency(_ seconds: Double) {
        collector.legacyObserveEndToEndLatency(seconds)
    }

    public func observeTimeToFirstToken(_ seconds: Double) {
        collector.legacyObserveTimeToFirstToken(seconds)
    }

    public func observeTimePerOutputToken(_ seconds: Double) {
        collector.legacyObserveTimePerOutputToken(seconds)
    }

    public func observeComputedPromptTokens(_ count: Int) {
        collector.legacyObserveComputedPromptTokens(count)
    }

    public func observeGeneratedTokens(_ count: Int) {
        collector.legacyObserveGeneratedTokens(count)
    }

    /// Returns a compatibility-only snapshot with non-atomic callback samples overlaid.
    public func metricsSnapshotWithLegacyGauges() -> AFMInferenceMetricsSnapshot {
        let dictionary = Thread.current.threadDictionary
        let reentryKey =
            "com.maclocal.afm.LegacyInferenceMetricsCompatibilityAdapter.snapshot."
            + String(describing: ObjectIdentifier(self))
        guard dictionary[reentryKey] == nil else {
            return collector.metricsSnapshot()
        }
        dictionary[reentryKey] = true
        defer { dictionary.removeObject(forKey: reentryKey) }

        let copied = callbacks.withLock {
            ($0.running, $0.waiting, $0.memoryCacheUsage, $0.prefixCacheFill)
        }
        let running = copied.0?() ?? 0
        let waiting = copied.1?() ?? 0
        let memory = copied.2?()
        let prefix = copied.3?()
        let base = collector.metricsSnapshot()
        let peak = callbacks.withLock { state in
            state.legacyBatchSizePeak = max(state.legacyBatchSizePeak, running)
            return state.legacyBatchSizePeak
        }
        return Self.overlay(
            base,
            running: running,
            waiting: waiting,
            peak: peak,
            memoryCacheUsage: memory,
            prefixCacheFill: prefix
        )
    }

    private static func overlay(
        _ snapshot: AFMInferenceMetricsSnapshot,
        running: Int,
        waiting: Int,
        peak: Int,
        memoryCacheUsage: Double?,
        prefixCacheFill: Double?
    ) -> AFMInferenceMetricsSnapshot {
        AFMInferenceMetricsSnapshot(
            timestampMilliseconds: snapshot.timestampMilliseconds,
            processStartEpochSeconds: snapshot.processStartEpochSeconds,
            modelName: snapshot.modelName,
            maximumConcurrentRequests: snapshot.maximumConcurrentRequests,
            maximumContextTokens: snapshot.maximumContextTokens,
            runningRequests: max(0, running),
            waitingRequests: max(0, waiting),
            peakRunningRequests: max(0, peak),
            logicalCacheUsage: snapshot.logicalCacheUsage,
            memoryCacheUsage: memoryCacheUsage,
            prefixCacheFill: prefixCacheFill,
            generatedTokensTotal: snapshot.generatedTokensTotal,
            fullPromptTokensTotal: snapshot.fullPromptTokensTotal,
            computedPromptTokensTotal: snapshot.computedPromptTokensTotal,
            acceptedRequestsTotal: snapshot.acceptedRequestsTotal,
            terminalRequestsTotal: snapshot.terminalRequestsTotal,
            preemptionsTotal: snapshot.preemptionsTotal,
            prefixCacheQueriesTotal: snapshot.prefixCacheQueriesTotal,
            prefixCacheHitsTotal: snapshot.prefixCacheHitsTotal,
            speculativeDraftRoundsTotal: snapshot.speculativeDraftRoundsTotal,
            speculativeDraftTokensTotal: snapshot.speculativeDraftTokensTotal,
            speculativeAcceptedTokensTotal: snapshot.speculativeAcceptedTokensTotal,
            terminalCounts: snapshot.terminalCounts,
            failureCounts: snapshot.failureCounts,
            supplementalCounts: snapshot.supplementalCounts,
            supplementalIntegerGauges: snapshot.supplementalIntegerGauges,
            supplementalDoubleGauges: snapshot.supplementalDoubleGauges,
            endToEndLatency: snapshot.endToEndLatency,
            queueLatency: snapshot.queueLatency,
            inferenceLatency: snapshot.inferenceLatency,
            prefillLatency: snapshot.prefillLatency,
            decodeLatency: snapshot.decodeLatency,
            timeToFirstToken: snapshot.timeToFirstToken,
            timePerOutputToken: snapshot.timePerOutputToken,
            interTokenLatency: snapshot.interTokenLatency,
            fullPromptTokens: snapshot.fullPromptTokens,
            computedPromptTokens: snapshot.computedPromptTokens,
            generatedTokens: snapshot.generatedTokens,
            maximumGeneratedTokens: snapshot.maximumGeneratedTokens,
            maximumOutputTokens: snapshot.maximumOutputTokens,
            samplingN: snapshot.samplingN,
            samplingBestOf: snapshot.samplingBestOf
        )
    }
}
