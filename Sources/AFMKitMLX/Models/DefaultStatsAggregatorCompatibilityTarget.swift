import AFMKitCore
import Foundation
import os

final class DefaultStatsAggregatorCompatibilityTarget:
    StatsAggregatorCompatibilityTarget,
    @unchecked Sendable
{
    private struct Histograms: Sendable {
        var endToEnd = StatsAggregator.Histogram(buckets: StatsAggregator.Buckets.requestLatency)
        var queue = StatsAggregator.Histogram(buckets: StatsAggregator.Buckets.requestLatency)
        var inference = StatsAggregator.Histogram(buckets: StatsAggregator.Buckets.requestLatency)
        var prefill = StatsAggregator.Histogram(buckets: StatsAggregator.Buckets.requestLatency)
        var decode = StatsAggregator.Histogram(buckets: StatsAggregator.Buckets.requestLatency)
        var timeToFirstToken = StatsAggregator.Histogram(
            buckets: StatsAggregator.Buckets.timeToFirstToken
        )
        var timePerOutputToken = StatsAggregator.Histogram(
            buckets: StatsAggregator.Buckets.timePerOutputToken
        )
        var promptTokens = StatsAggregator.Histogram(buckets: StatsAggregator.Buckets.tokenCount)
        var generationTokens = StatsAggregator.Histogram(buckets: StatsAggregator.Buckets.tokenCount)
        var samplingN = StatsAggregator.Histogram(buckets: StatsAggregator.Buckets.samplingParam)
        var samplingBestOf = StatsAggregator.Histogram(
            buckets: StatsAggregator.Buckets.samplingParam
        )
    }

    private struct State: Sendable {
        let processStartEpoch = Date().timeIntervalSince1970
        var modelName = ""
        var maximumConcurrentRequests = 0
        var running: StatsAggregator.GaugeReader?
        var waiting: StatsAggregator.GaugeReader?
        var memoryCacheUsage: StatsAggregator.FractionReader?
        var prefixCacheFill: StatsAggregator.FractionReader?
        var peakRunningRequests = 0
        var activeConnections = 0
        var activeConnectionsPeak = 0
        var generatedTokensTotal: UInt64 = 0
        var computedPromptTokensTotal: UInt64 = 0
        var acceptedRequestsTotal: UInt64 = 0
        var terminalRequestsTotal: UInt64 = 0
        var cacheHitsTotal: UInt64 = 0
        var cacheMissesTotal: UInt64 = 0
        var successReasons: [String: UInt64] = [:]
        var histograms = Histograms()
    }

    private let state = OSAllocatedUnfairLock(initialState: State())

    func setModel(_ name: String, maxConcurrent: Int) {
        state.withLock {
            $0.modelName = name
            $0.maximumConcurrentRequests = maxConcurrent
        }
    }

    func registerGaugeReaders(
        running: @escaping StatsAggregator.GaugeReader,
        waiting: @escaping StatsAggregator.GaugeReader
    ) {
        state.withLock {
            $0.running = running
            $0.waiting = waiting
        }
    }

    func registerGpuCacheUsageReader(_ reader: @escaping StatsAggregator.FractionReader) {
        state.withLock { $0.memoryCacheUsage = reader }
    }

    func registerRadixCacheFillReader(_ reader: @escaping StatsAggregator.FractionReader) {
        state.withLock { $0.prefixCacheFill = reader }
    }

    func connectionStarted() {
        state.withLock {
            $0.activeConnections += 1
            $0.activeConnectionsPeak = max($0.activeConnectionsPeak, $0.activeConnections)
        }
    }

    func connectionEnded() {
        state.withLock { $0.activeConnections = max(0, $0.activeConnections - 1) }
    }

    func reset() {
        state.withLock {
            $0.generatedTokensTotal = 0
            $0.computedPromptTokensTotal = 0
            $0.acceptedRequestsTotal = 0
            $0.terminalRequestsTotal = 0
            $0.cacheHitsTotal = 0
            $0.cacheMissesTotal = 0
            $0.successReasons = [:]
            $0.histograms = Histograms()
            $0.peakRunningRequests = 0
        }
    }

    func addGenTokens(_ count: Int) {
        guard count > 0 else { return }
        state.withLock { $0.generatedTokensTotal &+= UInt64(count) }
    }

    func addPromptTokens(_ count: Int) {
        guard count > 0 else { return }
        state.withLock { $0.computedPromptTokensTotal &+= UInt64(count) }
    }

    func requestStarted() { state.withLock { $0.acceptedRequestsTotal &+= 1 } }
    func requestCompleted() { state.withLock { $0.terminalRequestsTotal &+= 1 } }
    func cacheHit() { state.withLock { $0.cacheHitsTotal &+= 1 } }
    func cacheMiss() { state.withLock { $0.cacheMissesTotal &+= 1 } }

    func requestSucceeded(reason: String) {
        let reason = Self.sanitizedReason(reason)
        state.withLock { $0.successReasons[reason, default: 0] &+= 1 }
    }

    func observeRequest(_ observation: StatsAggregator.RequestObservation) {
        let endToEnd = max(0, observation.completedAt - observation.queuedAt)
        let queue = observation.startedAt.map { max(0, $0 - observation.queuedAt) } ?? 0
        let inference = observation.startedAt.map {
            max(0, observation.completedAt - $0)
        } ?? endToEnd
        let prefill = observation.startedAt.flatMap { start in
            observation.firstTokenAt.map { max(0, $0 - start) }
        }
        let decode = observation.firstTokenAt.map {
            max(0, observation.completedAt - $0)
        }
        let timeToFirstToken = observation.firstTokenAt.map {
            max(0, $0 - observation.queuedAt)
        }

        state.withLock { state in
            state.histograms.endToEnd.observe(endToEnd)
            state.histograms.queue.observe(queue)
            state.histograms.inference.observe(inference)
            if let prefill { state.histograms.prefill.observe(prefill) }
            if let decode {
                state.histograms.decode.observe(decode)
                if observation.generationTokens > 1 {
                    state.histograms.timePerOutputToken.observe(
                        decode / Double(observation.generationTokens - 1)
                    )
                }
            }
            if let timeToFirstToken {
                state.histograms.timeToFirstToken.observe(timeToFirstToken)
            }
            if observation.promptTokens > 0 {
                state.histograms.promptTokens.observe(Double(observation.promptTokens))
            }
            if observation.generationTokens > 0 {
                state.histograms.generationTokens.observe(Double(observation.generationTokens))
            }
            state.histograms.samplingN.observe(Double(max(1, observation.paramsN)))
            state.histograms.samplingBestOf.observe(Double(max(1, observation.paramsBestOf)))
        }
    }

    func observeE2eLatency(_ seconds: Double) {
        state.withLock { $0.histograms.endToEnd.observe(seconds) }
    }

    func observeTimeToFirstToken(_ seconds: Double) {
        state.withLock { $0.histograms.timeToFirstToken.observe(seconds) }
    }

    func observeTimePerOutputToken(_ seconds: Double) {
        state.withLock { $0.histograms.timePerOutputToken.observe(seconds) }
    }

    func observePromptTokens(_ count: Int) {
        guard count > 0 else { return }
        state.withLock { $0.histograms.promptTokens.observe(Double(count)) }
    }

    func observeGenerationTokens(_ count: Int) {
        guard count >= 0 else { return }
        state.withLock { $0.histograms.generationTokens.observe(Double(count)) }
    }

    func metricsSnapshot() -> AFMInferenceMetricsSnapshot {
        let dictionary = Thread.current.threadDictionary
        let reentryKey = "com.maclocal.afm.DefaultStatsAggregatorCompatibilityTarget.snapshot."
            + String(describing: ObjectIdentifier(self))
        let reentered = dictionary[reentryKey] != nil
        if !reentered { dictionary[reentryKey] = true }
        defer {
            if !reentered { dictionary.removeObject(forKey: reentryKey) }
        }

        let copied = state.withLock { $0 }
        let running = reentered ? 0 : copied.running?() ?? 0
        let waiting = reentered ? 0 : copied.waiting?() ?? 0
        let memoryCacheUsage = reentered ? nil : copied.memoryCacheUsage?()
        let prefixCacheFill = reentered ? nil : copied.prefixCacheFill?()
        let peak = state.withLock { state in
            state.peakRunningRequests = max(state.peakRunningRequests, running)
            return state.peakRunningRequests
        }

        let histograms = copied.histograms
        let emptyTokens = StatsAggregator.Histogram(buckets: StatsAggregator.Buckets.tokenCount)
        return AFMInferenceMetricsSnapshot(
            timestampMilliseconds: Int64(Date().timeIntervalSince1970 * 1_000),
            processStartEpochSeconds: copied.processStartEpoch,
            modelName: copied.modelName,
            maximumConcurrentRequests: copied.maximumConcurrentRequests,
            runningRequests: max(0, running),
            waitingRequests: max(0, waiting),
            peakRunningRequests: max(0, peak),
            memoryCacheUsage: memoryCacheUsage,
            prefixCacheFill: prefixCacheFill,
            generatedTokensTotal: copied.generatedTokensTotal,
            computedPromptTokensTotal: copied.computedPromptTokensTotal,
            acceptedRequestsTotal: copied.acceptedRequestsTotal,
            terminalRequestsTotal: copied.terminalRequestsTotal,
            supplementalCounts: [
                AFMNamedCount(name: "legacy_cache_hits", count: copied.cacheHitsTotal),
                AFMNamedCount(name: "legacy_cache_misses", count: copied.cacheMissesTotal),
            ] + copied.successReasons.keys.sorted().map {
                AFMNamedCount(
                    name: "legacy_finish:\($0)",
                    count: copied.successReasons[$0, default: 0]
                )
            },
            supplementalIntegerGauges: [
                AFMNamedIntegerGauge(name: "active_connections", value: copied.activeConnections),
                AFMNamedIntegerGauge(
                    name: "active_connections_peak",
                    value: copied.activeConnectionsPeak
                ),
            ],
            endToEndLatency: Self.snapshot(histograms.endToEnd),
            queueLatency: Self.snapshot(histograms.queue),
            inferenceLatency: Self.snapshot(histograms.inference),
            prefillLatency: Self.snapshot(histograms.prefill),
            decodeLatency: Self.snapshot(histograms.decode),
            timeToFirstToken: Self.snapshot(histograms.timeToFirstToken),
            timePerOutputToken: Self.snapshot(histograms.timePerOutputToken),
            interTokenLatency: Self.snapshot(histograms.timePerOutputToken),
            fullPromptTokens: Self.snapshot(emptyTokens),
            computedPromptTokens: Self.snapshot(histograms.promptTokens),
            generatedTokens: Self.snapshot(histograms.generationTokens),
            samplingN: Self.snapshot(histograms.samplingN),
            samplingBestOf: Self.snapshot(histograms.samplingBestOf)
        )
    }

    private static func snapshot(
        _ histogram: StatsAggregator.Histogram
    ) -> AFMHistogramSnapshot {
        AFMHistogramSnapshot(
            buckets: histogram.buckets,
            bucketCounts: histogram.bucketCounts,
            sum: histogram.sum,
            count: histogram.count
        )
    }

    private static func sanitizedReason(_ reason: String) -> String {
        let trimmed = reason.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !trimmed.isEmpty else { return "unknown" }
        return String(trimmed.unicodeScalars.map { scalar in
            ((scalar >= "a" && scalar <= "z") || (scalar >= "0" && scalar <= "9"))
                ? Character(String(scalar))
                : "_"
        })
    }
}
