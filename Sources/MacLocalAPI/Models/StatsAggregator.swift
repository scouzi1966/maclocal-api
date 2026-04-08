import Foundation
import os

/// Global thread-safe aggregator for AFM server metrics, exposed via
/// `GET /metrics` in Prometheus text exposition format.
///
/// Modelled after vLLM's `vllm/engine/metrics.py`:
///
///   - counters (monotonic)     — total tokens, total requests, cache events
///   - gauges (instantaneous)   — running / waiting / peak batch size
///
/// Both the batched code path (`BatchScheduler`) and the serial
/// single-sequence path (`MLXModelService.generate*`) call into the
/// same singleton so `/metrics` always reflects the complete server
/// activity regardless of which path served the request.
///
/// Gauges that describe live state (inflight, queue depth) are read
/// through closures registered at startup by whichever component owns
/// the state. This lets the metrics endpoint pull live values from
/// existing nonisolated locks without hopping through any actor.
public final class StatsAggregator: @unchecked Sendable {

    /// Process-wide singleton. All increment calls are O(1) under a
    /// single unfair lock; contention is negligible vs the Metal
    /// decode step.
    public static let shared = StatsAggregator()

    public typealias GaugeReader = @Sendable () -> Int

    // MARK: - Storage

    private struct Counters {
        var genTokensTotal: UInt64 = 0
        var promptTokensTotal: UInt64 = 0
        var requestsStartedTotal: UInt64 = 0
        var requestsCompletedTotal: UInt64 = 0
        var cacheHitsTotal: UInt64 = 0
        var cacheMissesTotal: UInt64 = 0
    }

    private struct Meta {
        var modelName: String = ""
        var maxConcurrent: Int = 0
        var processStartEpoch: Double
    }

    private struct GaugeState {
        var running: GaugeReader?
        var waiting: GaugeReader?
        var batchSizePeak: Int = 0
    }

    private let counters = OSAllocatedUnfairLock(initialState: Counters())
    private let meta: OSAllocatedUnfairLock<Meta>
    private let gauges = OSAllocatedUnfairLock(initialState: GaugeState())

    private init() {
        self.meta = OSAllocatedUnfairLock(
            initialState: Meta(processStartEpoch: Date().timeIntervalSince1970)
        )
    }

    // MARK: - Configuration

    /// Called once at server startup by `MLXModelService` (or the serial
    /// path equivalent) with the resolved model id and the configured
    /// `--concurrent` capacity. Appears in every response as a label.
    public func setModel(_ name: String, maxConcurrent: Int) {
        meta.withLock { m in
            m.modelName = name
            m.maxConcurrent = maxConcurrent
        }
    }

    /// Register live-gauge providers. `running` returns the number of
    /// requests currently generating on the GPU (active batch size).
    /// `waiting` returns the number of requests queued behind the
    /// `--concurrent` cap. Both are polled once per `/metrics` request.
    public func registerGaugeReaders(
        running: @escaping GaugeReader,
        waiting: @escaping GaugeReader
    ) {
        gauges.withLock { g in
            g.running = running
            g.waiting = waiting
        }
    }

    /// Reset counters (for long-running processes that want to rebaseline).
    /// Gauge readers and metadata are preserved.
    public func reset() {
        counters.withLock { $0 = Counters() }
        gauges.withLock { $0.batchSizePeak = 0 }
    }

    // MARK: - Increments

    public func addGenTokens(_ n: Int = 1) {
        guard n > 0 else { return }
        counters.withLock { $0.genTokensTotal &+= UInt64(n) }
    }

    public func addPromptTokens(_ n: Int) {
        guard n > 0 else { return }
        counters.withLock { $0.promptTokensTotal &+= UInt64(n) }
    }

    public func requestStarted() {
        counters.withLock { $0.requestsStartedTotal &+= 1 }
    }

    public func requestCompleted() {
        counters.withLock { $0.requestsCompletedTotal &+= 1 }
    }

    public func cacheHit() {
        counters.withLock { $0.cacheHitsTotal &+= 1 }
    }

    public func cacheMiss() {
        counters.withLock { $0.cacheMissesTotal &+= 1 }
    }

    // MARK: - Snapshot

    public struct Snapshot: Sendable {
        public let timestampMs: Int64
        public let processStartEpoch: Double
        public let modelName: String
        public let maxConcurrent: Int
        public let numRunning: Int
        public let numWaiting: Int
        public let batchSizePeak: Int
        public let genTokensTotal: UInt64
        public let promptTokensTotal: UInt64
        public let requestsStartedTotal: UInt64
        public let requestsCompletedTotal: UInt64
        public let cacheHitsTotal: UInt64
        public let cacheMissesTotal: UInt64
    }

    /// Build a single-point-in-time snapshot of every metric. Cheap —
    /// three lock acquisitions, one call to each gauge reader.
    public func snapshot() -> Snapshot {
        let c = counters.withLock { $0 }
        let m = meta.withLock { $0 }
        let (running, waiting, peak) = gauges.withLock { g -> (Int, Int, Int) in
            let r = g.running?() ?? 0
            let w = g.waiting?() ?? 0
            if r > g.batchSizePeak { g.batchSizePeak = r }
            return (r, w, g.batchSizePeak)
        }
        return Snapshot(
            timestampMs: Int64(Date().timeIntervalSince1970 * 1000),
            processStartEpoch: m.processStartEpoch,
            modelName: m.modelName,
            maxConcurrent: m.maxConcurrent,
            numRunning: running,
            numWaiting: waiting,
            batchSizePeak: peak,
            genTokensTotal: c.genTokensTotal,
            promptTokensTotal: c.promptTokensTotal,
            requestsStartedTotal: c.requestsStartedTotal,
            requestsCompletedTotal: c.requestsCompletedTotal,
            cacheHitsTotal: c.cacheHitsTotal,
            cacheMissesTotal: c.cacheMissesTotal
        )
    }
}
