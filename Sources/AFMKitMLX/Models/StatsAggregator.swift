import AFMKitCore
import Foundation
import os

/// AFMKitMLX-local forwarding contract used by the deprecated compatibility facade.
public protocol StatsAggregatorCompatibilityTarget: AnyObject, Sendable {
    func setModel(_ name: String, maxConcurrent: Int)
    func registerGaugeReaders(
        running: @escaping StatsAggregator.GaugeReader,
        waiting: @escaping StatsAggregator.GaugeReader
    )
    func registerGpuCacheUsageReader(_ reader: @escaping StatsAggregator.FractionReader)
    func registerRadixCacheFillReader(_ reader: @escaping StatsAggregator.FractionReader)
    func connectionStarted()
    func connectionEnded()
    func reset()
    func addGenTokens(_ count: Int)
    func addPromptTokens(_ count: Int)
    func requestStarted()
    func requestCompleted()
    func cacheHit()
    func cacheMiss()
    func requestSucceeded(reason: String)
    func observeRequest(_ observation: StatsAggregator.RequestObservation)
    func observeE2eLatency(_ seconds: Double)
    func observeTimeToFirstToken(_ seconds: Double)
    func observeTimePerOutputToken(_ seconds: Double)
    func observePromptTokens(_ count: Int)
    func observeGenerationTokens(_ count: Int)
    func metricsSnapshot() -> AFMInferenceMetricsSnapshot
}

@available(
    *,
    deprecated,
    message: "Inject AFMKitCore telemetry protocols and use InferenceTelemetryCollector from AFMKitServices."
)
public final class StatsAggregator: @unchecked Sendable {
    public static let shared = StatsAggregator()

    public typealias GaugeReader = @Sendable () -> Int
    public typealias FractionReader = @Sendable () -> Double

    public enum Buckets {
        public static let requestLatency: [Double] = [
            0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0,
            10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0,
        ]
        public static let timeToFirstToken: [Double] = [
            0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1,
            0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0,
        ]
        public static let timePerOutputToken: [Double] = [
            0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2,
            0.3, 0.4, 0.5, 0.75, 1.0, 2.5,
        ]
        public static let tokenCount: [Double] = [
            1, 2, 5, 10, 20, 50, 100, 200, 500, 1000,
            2000, 5000, 10000, 20000, 50000, 100000,
        ]
        public static let samplingParam: [Double] = [1, 2, 5, 10, 20]
    }

    public struct Histogram: Sendable {
        public let buckets: [Double]
        public var bucketCounts: [UInt64]
        public var sum: Double
        public var count: UInt64

        public init(buckets: [Double]) {
            self.buckets = buckets
            self.bucketCounts = Array(repeating: 0, count: buckets.count + 1)
            self.sum = 0
            self.count = 0
        }

        public mutating func observe(_ value: Double) {
            guard value.isFinite, value >= 0 else { return }
            sum += value
            count &+= 1
            for index in buckets.indices where value <= buckets[index] {
                bucketCounts[index] &+= 1
            }
            bucketCounts[buckets.count] &+= 1
        }

        init(_ snapshot: AFMHistogramSnapshot) {
            buckets = snapshot.buckets
            bucketCounts = snapshot.bucketCounts
            sum = snapshot.sum
            count = snapshot.count
        }

        init(_ snapshot: AFMHistogramSnapshot, rebucketedTo compatibleBuckets: [Double]) {
            buckets = compatibleBuckets
            bucketCounts = compatibleBuckets.map { boundary in
                if let exactIndex = snapshot.buckets.firstIndex(of: boundary),
                   exactIndex < snapshot.bucketCounts.count {
                    return snapshot.bucketCounts[exactIndex]
                }
                if let lastBoundary = snapshot.buckets.last, boundary > lastBoundary {
                    return snapshot.count
                }
                guard let lowerIndex = snapshot.buckets.lastIndex(where: { $0 < boundary }),
                      lowerIndex < snapshot.bucketCounts.count else {
                    return 0
                }
                return snapshot.bucketCounts[lowerIndex]
            }
            bucketCounts.append(snapshot.count)
            sum = snapshot.sum
            count = snapshot.count
        }
    }

    public struct RequestObservation: Sendable {
        public var queuedAt: Double
        public var startedAt: Double?
        public var firstTokenAt: Double?
        public var completedAt: Double
        public var promptTokens: Int
        public var generationTokens: Int
        public var paramsN: Int
        public var paramsBestOf: Int

        public init(
            queuedAt: Double,
            startedAt: Double?,
            firstTokenAt: Double?,
            completedAt: Double,
            promptTokens: Int,
            generationTokens: Int,
            paramsN: Int = 1,
            paramsBestOf: Int = 1
        ) {
            self.queuedAt = queuedAt
            self.startedAt = startedAt
            self.firstTokenAt = firstTokenAt
            self.completedAt = completedAt
            self.promptTokens = promptTokens
            self.generationTokens = generationTokens
            self.paramsN = paramsN
            self.paramsBestOf = paramsBestOf
        }
    }

    public struct Snapshot: Sendable {
        public let timestampMs: Int64
        public let processStartEpoch: Double
        public let modelName: String
        public let maxConcurrent: Int
        public let numRunning: Int
        public let numWaiting: Int
        public let batchSizePeak: Int
        public let activeConnections: Int
        public let activeConnectionsPeak: Int
        public let gpuCacheUsage: Double?
        public let radixCacheFill: Double?
        public let genTokensTotal: UInt64
        public let promptTokensTotal: UInt64
        public let requestsStartedTotal: UInt64
        public let requestsCompletedTotal: UInt64
        public let cacheHitsTotal: UInt64
        public let cacheMissesTotal: UInt64
        public let requestSuccessByReason: [String: UInt64]
        public let e2eLatency: Histogram
        public let queueTime: Histogram
        public let inferenceTime: Histogram
        public let prefillTime: Histogram
        public let decodeTime: Histogram
        public let timeToFirstToken: Histogram
        public let timePerOutputToken: Histogram
        public let promptTokens: Histogram
        public let generationTokens: Histogram
        public let paramsN: Histogram
        public let paramsBestOf: Histogram
    }

    private struct Binding {
        var target: any StatsAggregatorCompatibilityTarget
        var identity: ObjectIdentifier?
        var used = false
    }

    private let binding: OSAllocatedUnfairLock<Binding>

    init(
        compatibilityTarget: any StatsAggregatorCompatibilityTarget =
            DefaultStatsAggregatorCompatibilityTarget()
    ) {
        binding = OSAllocatedUnfairLock(
            initialState: Binding(target: compatibilityTarget)
        )
    }

    /// Installs the Services-backed target used by this compatibility facade.
    /// Reinstalling the same object is harmless. A different target cannot replace
    /// a binding after the facade has forwarded its first operation.
    @discardableResult
    public static func installCompatibilityTarget(
        _ target: any StatsAggregatorCompatibilityTarget
    ) -> Bool {
        shared.installCompatibilityTarget(target)
    }

    @discardableResult
    func installCompatibilityTarget(
        _ target: any StatsAggregatorCompatibilityTarget
    ) -> Bool {
        binding.withLock { binding in
            let identity = ObjectIdentifier(target)
            if binding.identity == identity { return true }
            guard !binding.used else { return false }
            binding.target = target
            binding.identity = identity
            return true
        }
    }

    private func target(markUsed: Bool = true) -> any StatsAggregatorCompatibilityTarget {
        binding.withLock { binding in
            if markUsed || binding.identity != nil {
                binding.used = true
            }
            return binding.target
        }
    }

    public func setModel(_ name: String, maxConcurrent: Int) {
        target().setModel(name, maxConcurrent: maxConcurrent)
    }

    public func registerGaugeReaders(
        running: @escaping GaugeReader,
        waiting: @escaping GaugeReader
    ) {
        target().registerGaugeReaders(running: running, waiting: waiting)
    }

    public func registerGpuCacheUsageReader(_ reader: @escaping FractionReader) {
        target().registerGpuCacheUsageReader(reader)
    }

    public func registerRadixCacheFillReader(_ reader: @escaping FractionReader) {
        target().registerRadixCacheFillReader(reader)
    }

    public func connectionStarted() { target().connectionStarted() }
    public func connectionEnded() { target().connectionEnded() }
    public func reset() {
        // Resetting an untouched fallback leaves no state to migrate, so it must
        // not prevent the composition root from installing its shared collector.
        target(markUsed: false).reset()
    }

    public func addGenTokens(_ n: Int = 1) { target().addGenTokens(n) }
    public func addPromptTokens(_ n: Int) { target().addPromptTokens(n) }
    public func requestStarted() { target().requestStarted() }
    public func requestCompleted() { target().requestCompleted() }
    public func cacheHit() { target().cacheHit() }
    public func cacheMiss() { target().cacheMiss() }
    public func requestSucceeded(reason: String) { target().requestSucceeded(reason: reason) }
    public func observeRequest(_ observation: RequestObservation) {
        target().observeRequest(observation)
    }
    public func observeE2eLatency(_ seconds: Double) {
        target().observeE2eLatency(seconds)
    }
    public func observeTimeToFirstToken(_ seconds: Double) {
        target().observeTimeToFirstToken(seconds)
    }
    public func observeTimePerOutputToken(_ seconds: Double) {
        target().observeTimePerOutputToken(seconds)
    }
    public func observePromptTokens(_ n: Int) { target().observePromptTokens(n) }
    public func observeGenerationTokens(_ n: Int) { target().observeGenerationTokens(n) }

    public func snapshot() -> Snapshot {
        let snapshot = target().metricsSnapshot()
        let supplementalCounts = Dictionary(
            uniqueKeysWithValues: snapshot.supplementalCounts.map { ($0.name, $0.count) }
        )
        let integerGauges = Dictionary(
            uniqueKeysWithValues: snapshot.supplementalIntegerGauges.map { ($0.name, $0.value) }
        )
        let legacyReasons = snapshot.supplementalCounts.compactMap { metric -> (String, UInt64)? in
            let prefix = "legacy_finish:"
            guard metric.name.hasPrefix(prefix) else { return nil }
            return (String(metric.name.dropFirst(prefix.count)), metric.count)
        }
        let reasons = legacyReasons.isEmpty
            ? Dictionary(uniqueKeysWithValues: snapshot.terminalCounts.compactMap { metric in
                metric.count > 0 ? (metric.name, metric.count) : nil
            })
            : Dictionary(uniqueKeysWithValues: legacyReasons)
        return Snapshot(
            timestampMs: snapshot.timestampMilliseconds,
            processStartEpoch: snapshot.processStartEpochSeconds,
            modelName: snapshot.modelName,
            maxConcurrent: snapshot.maximumConcurrentRequests,
            numRunning: snapshot.runningRequests,
            numWaiting: snapshot.waitingRequests,
            batchSizePeak: snapshot.peakRunningRequests,
            activeConnections: integerGauges["active_connections", default: 0],
            activeConnectionsPeak: integerGauges["active_connections_peak", default: 0],
            gpuCacheUsage: snapshot.memoryCacheUsage,
            radixCacheFill: snapshot.prefixCacheFill,
            genTokensTotal: snapshot.generatedTokensTotal,
            promptTokensTotal: snapshot.computedPromptTokensTotal,
            requestsStartedTotal: snapshot.acceptedRequestsTotal,
            requestsCompletedTotal: snapshot.terminalRequestsTotal,
            cacheHitsTotal: supplementalCounts["legacy_cache_hits", default: 0],
            cacheMissesTotal: supplementalCounts["legacy_cache_misses", default: 0],
            requestSuccessByReason: reasons,
            e2eLatency: Histogram(
                snapshot.endToEndLatency,
                rebucketedTo: Buckets.requestLatency
            ),
            queueTime: Histogram(snapshot.queueLatency, rebucketedTo: Buckets.requestLatency),
            inferenceTime: Histogram(
                snapshot.inferenceLatency,
                rebucketedTo: Buckets.requestLatency
            ),
            prefillTime: Histogram(
                snapshot.prefillLatency,
                rebucketedTo: Buckets.requestLatency
            ),
            decodeTime: Histogram(snapshot.decodeLatency, rebucketedTo: Buckets.requestLatency),
            timeToFirstToken: Histogram(
                snapshot.timeToFirstToken,
                rebucketedTo: Buckets.timeToFirstToken
            ),
            timePerOutputToken: Histogram(
                snapshot.timePerOutputToken,
                rebucketedTo: Buckets.timePerOutputToken
            ),
            promptTokens: Histogram(
                snapshot.computedPromptTokens,
                rebucketedTo: Buckets.tokenCount
            ),
            generationTokens: Histogram(
                snapshot.generatedTokens,
                rebucketedTo: Buckets.tokenCount
            ),
            paramsN: Histogram(snapshot.samplingN, rebucketedTo: Buckets.samplingParam),
            paramsBestOf: Histogram(
                snapshot.samplingBestOf,
                rebucketedTo: Buckets.samplingParam
            )
        )
    }
}
